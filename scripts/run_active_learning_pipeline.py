#!/usr/bin/env python3

"""
One-click MDKG active-learning + DPO demo.

Pipeline overview (multi-round AL):
  1. (Optional) create GPU-ready Conda env and install dependencies.
  2. Download MDIEC annotations from Hugging Face if missing.
  3. Convert `.ann/.txt` files into SynSpERT JSON and split counts
     (default: 300 train / 150 eval / remainder pool).
  4. Train a baseline SynSpERT model (SFT) on the base training split and
     evaluate on the held-out validation/test set.
  5. For each active-learning round:
       a. Expose a batch of pool samples, run eval with AL tensors,并挑出
          `--al-sample-count` 个低置信度样本。
       b. 把新样本并入训练集，以上一轮 checkpoint warm-start 做 SFT 微调，
          同时在固定测试集和该批新样本上各评估一次。
       c. 用“人工标注 vs 最新 SFT 推理”构造偏好对（仅保留仍有差异的样本），
          可选跑一段 DPO，然后在测试集评估。
       d. 记录当轮所有 artifact，并把未被选中的样本放回池子，继续下一轮。

Artifacts are stored under `NER&RE_model/InputsAndOutputs/data/active_learning/`.

Example (`python scripts/run_active_learning_pipeline.py`):
    python scripts/run_active_learning_pipeline.py \
        --env-name mdkgspert_gpu \
        --setup-env \
        --bert-model GanjinZero/coder_eng_pp \
        --config NER&RE_model/SynSpERT/configs/config-coder.json \
        --al-sample-count 50 \
        --pool-release-size 100 \
        --max-rounds 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
NER_ROOT = REPO_ROOT / "NER&RE_model"
IO_ROOT = NER_ROOT / "InputsAndOutputs"
RAW_DATA_DIR = IO_ROOT / "data" / "dataset" / "MDIEC"
DATASET_JSON = IO_ROOT / "data" / "dataset" / "MDIEC.json"
SPLIT_DIR = IO_ROOT / "data" / "datasets"
LOG_DIR = IO_ROOT / "data" / "log"
SAVE_DIR = IO_ROOT / "data" / "save"
CACHE_DIR = IO_ROOT / "data" / "cache"
MODEL_CACHE = IO_ROOT / "models"
ACTIVE_DIR = IO_ROOT / "data" / "active_learning"

DEFAULT_PREFIX = "diabetes"
DEFAULT_RUN_LABEL = "diabetes_small_run"


class PipelineError(RuntimeError):
    """Custom runtime error for pipeline failures."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate MDKG active-learning + DPO loop.")

    env = parser.add_argument_group("Environment")
    env.add_argument("--setup-env", action="store_true", help="Create Conda env and install GPU deps.")
    env.add_argument("--env-name", default="mdkgspert_gpu", help="Name of the Conda environment to create.")
    env.add_argument(
        "--requirements",
        default="requirements_gpu.txt",
        help="Requirements file (relative to repo root) used when --setup-env is specified.",
    )

    data = parser.add_argument_group("Data preparation")
    data.add_argument("--skip-download", action="store_true", help="Skip fetching MDIEC data from Hugging Face.")
    data.add_argument(
        "--train-count",
        type=int,
        default=300,
        help="Number of documents used for the initial supervised training split.",
    )
    data.add_argument(
        "--valid-count",
        type=int,
        default=150,
        help="Number of documents reserved as the fixed validation/test set.",
    )
    data.add_argument("--prefix", default=DEFAULT_PREFIX, help="Dataset prefix (defaults to SynSpERT's diabetes).")

    train = parser.add_argument_group("Training")
    train.add_argument("--bert-model", default="bert-base-uncased", help="Backbone encoder (HF model path or local).")
    train.add_argument(
        "--config",
        default=str(NER_ROOT / "SynSpERT" / "configs" / "config.json"),
        help="SynSpERT config JSON passed via --config_override.",
    )
    train.add_argument("--seed", type=int, default=11, help="Random seed forwarded to main.py.")
    train.add_argument("--epochs", type=int, default=10, help="Epochs for baseline SFT training.")
    train.add_argument("--dpo-epochs", type=int, default=4, help="Epochs for DPO fine-tuning.")
    train.add_argument("--train-batch-size", type=int, default=4, help="Train batch size for both runs.")
    train.add_argument("--eval-batch-size", type=int, default=4, help="Eval batch size for both runs.")
    train.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for baseline training.")
    train.add_argument("--sft-ft-epochs", type=int, default=3, help="Epochs for SFT fine-tuning after sample promotion.")
    train.add_argument(
        "--sft-ft-learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate for SFT fine-tuning after sample promotion.",
    )
    train.add_argument("--dpo-learning-rate", type=float, default=3e-5, help="Learning rate during DPO fine-tuning.")
    train.add_argument("--dpo-beta", type=float, default=0.1, help="DPO beta temperature.")
    train.add_argument("--dpo-lambda", type=float, default=0.1, help="Weight applied to the DPO loss term.")
    train.add_argument("--dpo-negatives", type=int, default=4, help="Negative classes sampled per example for DPO.")

    al = parser.add_argument_group("Active learning")
    al.add_argument(
        "--al-sample-count",
        type=int,
        default=50,
        help="Number of low-confidence pool documents promoted for annotation / DPO per round.",
    )
    al.add_argument(
        "--pool-release-size",
        type=int,
        default=100,
        help="Pool batch size exposed for selection each round.",
    )
    al.add_argument(
        "--max-rounds",
        type=int,
        default=1,
        help="Maximum number of active-learning rounds to execute.",
    )
    al.add_argument(
        "--al-reinit-each-round",
        action="store_true",
        help="Reinitialize each AL SFT run from the original --bert-model instead of the previous checkpoint.",
    )
    al.add_argument(
        "--selection-script",
        default=str(REPO_ROOT / "Active_learning.py"),
        help="Path to Active_learning.py used for selecting informative samples.",
    )
    al.add_argument("--selection-top-k", type=int, default=20, help="How many clusters to retain (Active_learning.py).")
    al.add_argument(
        "--selection-sample-per-group",
        type=int,
        default=10,
        help="Samples taken per cluster when calling Active_learning.py.",
    )
    al.add_argument("--selection-beta", type=float, default=0.1, help="beta parameter passed to Active_learning.py.")
    al.add_argument("--selection-gamma", type=float, default=0.1, help="gamma parameter passed to Active_learning.py.")
    al.add_argument("--selection-ncentroids", type=int, default=20, help="Cluster count for Active_learning.py.")
    al.add_argument(
        "--selection-use-weights",
        action="store_true",
        help="Forward --use_weights to Active_learning.py for weighted clustering.",
    )

    misc = parser.add_argument_group("Misc")
    misc.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete intermediate AL temp files (selected subsets, preference JSON).",
    )

    return parser.parse_args()


def run_command(cmd: Sequence[str], cwd: Path | None = None) -> None:
    """Run a subprocess with live stdout/stderr streaming."""
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_conda_available() -> None:
    if shutil.which("conda") is None:
        raise PipelineError("Conda is required for --setup-env but was not found on PATH.")


def setup_environment(args: argparse.Namespace) -> None:
    ensure_conda_available()
    env_name = args.env_name
    req_path = REPO_ROOT / args.requirements
    if not req_path.exists():
        raise PipelineError(f"Requirements file not found: {req_path}")

    run_command(["conda", "create", "-y", "-n", env_name, "python=3.10"])
    run_command(["conda", "run", "-n", env_name, "pip", "install", "-r", str(req_path)])
    # spaCy / NLTK resources needed by SynSpERT utilities
    run_command(["conda", "run", "-n", env_name, "python", "-m", "spacy", "download", "en_core_web_sm"])
    run_command(["conda", "run", "-n", env_name, "python", "-m", "spacy", "download", "en_core_sci_sm"])
    run_command(["conda", "run", "-n", env_name, "python", "-m", "nltk.downloader", "punkt", "punkt_tab"])
    print(f"[ENV] Conda environment '{env_name}' is ready.")


def download_data() -> None:
    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.glob("*.ann")):
        print("[DATA] MDIEC annotations already present; skipping download.")
        return
    print("[DATA] Downloading MDIEC annotations from Hugging Face.")
    run_command(["python", str(REPO_ROOT / "scripts" / "fetch_official_data.py"), "--overwrite"], cwd=REPO_ROOT)


def generate_synspert_inputs(train_count: int, valid_count: int, prefix: str, seed: int) -> None:
    print("[DATA] Converting Brat annotations to SynSpERT JSON.")
    run_command(
        [
            "python",
            str(NER_ROOT / "SynSpERT" / "generate_input.py"),
            "--input_dir",
            str(RAW_DATA_DIR),
            "--output_json",
            str(DATASET_JSON),
        ],
        cwd=REPO_ROOT,
    )

    print("[DATA] Shuffling corpus and creating custom splits.")
    docs = load_json(DATASET_JSON)
    total = len(docs)
    if train_count + valid_count >= total:
        raise PipelineError(
            f"Requested train ({train_count}) + valid ({valid_count}) consumes entire dataset of {total} documents."
        )

    rng = random.Random(seed)
    rng.shuffle(docs)

    train_docs = docs[:train_count]
    valid_docs = docs[train_count : train_count + valid_count]
    pool_docs = docs[train_count + valid_count :]

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    dump_json(SPLIT_DIR / f"{prefix}_train.json", train_docs)
    dump_json(SPLIT_DIR / f"{prefix}_valid.json", valid_docs)
    dump_json(SPLIT_DIR / f"{prefix}_test.json", pool_docs)

    print(
        f"[DATA] Split complete: train={len(train_docs)}, valid/test={len(valid_docs)}, pool={len(pool_docs)} "
        f"(seed={seed})."
    )


def dataset_paths(prefix: str) -> tuple[Path, Path, Path]:
    train = SPLIT_DIR / f"{prefix}_train.json"
    valid = SPLIT_DIR / f"{prefix}_valid.json"
    test = SPLIT_DIR / f"{prefix}_test.json"
    if not train.exists() or not valid.exists() or not test.exists():
        raise PipelineError(f"Expected SynSpERT splits not found for prefix '{prefix}'.")
    return train, valid, test


def latest_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise PipelineError(f"No runs found under {base_dir}")
    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise PipelineError(f"No run subdirectories under {base_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def relocate_training_run(new_label: str) -> tuple[Path, Path]:
    """
    Move the most recent training artifacts from the default label directory
    to a custom label, keeping timestamps intact. Returns (log_dir, save_dir).
    """
    default_label_dir_log = LOG_DIR / DEFAULT_RUN_LABEL
    default_label_dir_save = SAVE_DIR / DEFAULT_RUN_LABEL

    log_run = latest_run_dir(default_label_dir_log)
    save_run = latest_run_dir(default_label_dir_save)

    when = log_run.name
    target_log_root = LOG_DIR / new_label
    target_save_root = SAVE_DIR / new_label
    target_log_root.mkdir(parents=True, exist_ok=True)
    target_save_root.mkdir(parents=True, exist_ok=True)

    moved_log = target_log_root / when
    moved_save = target_save_root / when

    shutil.move(str(log_run), moved_log)
    shutil.move(str(save_run), moved_save)

    if not any(default_label_dir_log.iterdir()):
        default_label_dir_log.rmdir()
    if not any(default_label_dir_save.iterdir()):
        default_label_dir_save.rmdir()

    return moved_log, moved_save


def run_training(
    label_suffix: str,
    bert_model: str,
    config_path: str,
    seed: int,
    epochs: int,
    lr: float,
    train_batch: int,
    eval_batch: int,
    ft_mode: str = "sft",
    dpo_beta: float = 0.1,
    dpo_lambda: float = 0.1,
    dpo_negatives: int = 4,
    dpo_reference: str | None = None,
) -> tuple[Path, Path]:
    print(f"[TRAIN] Starting training run '{label_suffix}'.")
    cmd = [
        "python",
        str(NER_ROOT / "SynSpERT" / "main.py"),
        "--mode",
        "train",
        "--bert_model",
        bert_model,
        "--config_override",
        config_path,
        "--run_seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--train_batch_size",
        str(train_batch),
        "--eval_batch_size",
        str(eval_batch),
        "--download_dir",
        str(MODEL_CACHE),
        "--log_path",
        str(LOG_DIR),
        "--save_path",
        str(SAVE_DIR),
        "--ft_mode",
        ft_mode,
    ]
    if ft_mode == "dpo":
        cmd.extend(
            [
                "--dpo_beta",
                str(dpo_beta),
                "--dpo_lambda",
                str(dpo_lambda),
                "--dpo_negatives",
                str(dpo_negatives),
            ]
        )
        if dpo_reference:
            cmd.extend(["--dpo_reference", dpo_reference])
    run_command(cmd, cwd=REPO_ROOT)
    log_dir, save_dir = relocate_training_run(label_suffix)
    print(f"[TRAIN] Artifacts moved to {save_dir}")
    return log_dir, save_dir


def final_model_path(save_dir: Path) -> Path:
    final_model = save_dir / "final_model"
    if final_model.exists():
        return final_model

    best_model = save_dir / "best_model"
    if best_model.exists():
        print(f"[WARN] final_model not found in {save_dir}. Falling back to best_model.")
        return best_model

    raise PipelineError(
        f"Neither final_model nor best_model directories found under {save_dir}. "
        "Check the training logs for errors or adjust training configuration."
    )


def run_evaluation(
    model_dir: Path,
    dataset_path: Path,
    label: str,
    eval_batch: int,
    al_dump_dir: Path | None = None,
) -> Path:
    cmd = [
        "python",
        str(NER_ROOT / "SynSpERT" / "main.py"),
        "--mode",
        "eval",
        "--model_dir",
        str(model_dir),
        "--dataset_path",
        str(dataset_path),
        "--label",
        label,
        "--eval_batch_size",
        str(eval_batch),
        "--log_path",
        str(LOG_DIR),
        "--save_path",
        str(SAVE_DIR),
    ]
    if al_dump_dir is not None:
        cmd.extend(["--al_dump_dir", str(al_dump_dir)])
    run_command(cmd, cwd=REPO_ROOT)
    eval_root = LOG_DIR / label
    return latest_run_dir(eval_root)


def load_json(path: Path) -> list:
    return json.loads(path.read_text())


def dump_json(path: Path, data: Iterable, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(data), ensure_ascii=False, indent=indent))


def parse_eval_metrics(log_dir: Path) -> dict:
    log_path = log_dir / "all.log"
    if not log_path.exists():
        raise PipelineError(f"Evaluation log not found: {log_path}")

    metrics: dict[str, dict[str, float]] = {
        "ner_micro": None,
        "relations_micro": None,
        "relations_nec_micro": None,
    }
    section: str | None = None
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("--- Entities"):
                section = "ner"
                continue
            if line.startswith("--- Relations"):
                section = None
                continue
            if line.startswith("Without named entity classification"):
                section = "relations"
                continue
            if line.startswith("With named entity classification"):
                section = "relations_nec"
                continue

            if section and line.startswith("micro"):
                parts = line.split()
                if len(parts) >= 5:
                    metric = {
                        "precision": float(parts[1]),
                        "recall": float(parts[2]),
                        "f1": float(parts[3]),
                        "support": int(parts[4]),
                    }
                    if section == "ner":
                        metrics["ner_micro"] = metric
                    elif section == "relations":
                        metrics["relations_micro"] = metric
                    elif section == "relations_nec":
                        metrics["relations_nec_micro"] = metric
                section = None

    missing = [key for key, value in metrics.items() if value is None]
    if missing:
        raise PipelineError(f"Failed to parse metrics {missing} from {log_path}")

    return metrics


def build_dpo_preferences(human_docs: list, model_docs: list, output_jsonl: Path) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for doc, pred in zip(human_docs, model_docs):
            tokens = doc.get("tokens", [])
            prompt = "Extract entities and relations from the sentence:\n" + " ".join(tokens)

            chosen = {
                "entities": sorted(
                    doc.get("entities", []), key=lambda e: (e["start"], e["end"], e.get("type", ""))
                ),
                "relations": sorted(
                    doc.get("relations", []), key=lambda r: (r["head"], r["tail"], r.get("type", ""))
                ),
            }
            rejected = {
                "entities": sorted(
                    pred.get("entities", []), key=lambda e: (e["start"], e["end"], e.get("type", ""))
                ),
                "relations": sorted(
                    pred.get("relations", []), key=lambda r: (r["head"], r["tail"], r.get("type", ""))
                ),
            }

            if chosen == rejected:
                skipped += 1
                continue

            record = {
                "prompt": prompt,
                "chosen": json.dumps(chosen, ensure_ascii=False, sort_keys=True),
                "rejected": json.dumps(rejected, ensure_ascii=False, sort_keys=True),
                "orig_id": doc.get("orig_id"),
            }
            fout.write(json.dumps(record, ensure_ascii=False))
            fout.write("\n")
            kept += 1

    if kept == 0:
        print(
            "[WARN] All selected samples perfectly matched the latest SFT predictions; "
            "no DPO preference pairs were generated."
        )
    elif skipped:
        print(f"[INFO] Skipped {skipped} preference pairs where model output already matched the ground truth.")
    return kept


def main() -> None:
    args = parse_args()

    if args.setup_env:
        setup_environment(args)

    if not args.skip_download:
        download_data()

    generate_synspert_inputs(args.train_count, args.valid_count, args.prefix, args.seed)
    train_path, valid_path, pool_path = dataset_paths(args.prefix)

    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    round0_dir = ACTIVE_DIR / "round0"
    round0_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backup initial splits for reference.
    shutil.copy2(train_path, round0_dir / f"{args.prefix}_train_round0_{timestamp}.json")
    shutil.copy2(valid_path, round0_dir / f"{args.prefix}_test_round0_{timestamp}.json")
    shutil.copy2(pool_path, round0_dir / f"{args.prefix}_pool_round0_{timestamp}.json")

    # Baseline SFT training.
    base_log_dir, base_save_dir = run_training(
        label_suffix="active_learning_sft_round0",
        bert_model=args.bert_model,
        config_path=args.config,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.learning_rate,
        train_batch=args.train_batch_size,
        eval_batch=args.eval_batch_size,
        ft_mode="sft",
    )
    base_model = final_model_path(base_save_dir)

    # Evaluate baseline on held-out test split.
    test_eval_dir = run_evaluation(
        model_dir=base_model,
        dataset_path=valid_path,
        label="active_learning_eval_round0",
        eval_batch=args.eval_batch_size,
    )

    baseline_summary = {
        "baseline_save_dir": str(base_save_dir),
        "baseline_log_dir": str(base_log_dir),
        "baseline_test_eval": str(test_eval_dir),
    }

    train_docs_cache = load_json(train_path)
    pool_docs_cache = load_json(pool_path)
    round_summaries: list[dict] = []
    current_model = base_model

    for round_idx in range(1, args.max_rounds + 1):
        if not pool_docs_cache:
            print(f"[AL] Pool exhausted before round {round_idx}; stopping.")
            break

        release_size = min(args.pool_release_size, len(pool_docs_cache))
        release_docs = pool_docs_cache[:release_size]
        pool_docs_cache = pool_docs_cache[release_size:]

        round_dir = ACTIVE_DIR / f"round{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_pool_path = round_dir / f"{args.prefix}_pool_round{round_idx}.json"
        dump_json(round_pool_path, release_docs)
        al_dump_dir = round_dir / "uncertainty"

        pool_eval_label = f"active_learning_pool_round{round_idx}"
        pool_eval_dir = run_evaluation(
            model_dir=current_model,
            dataset_path=round_pool_path,
            label=pool_eval_label,
            eval_batch=args.eval_batch_size,
            al_dump_dir=al_dump_dir,
        )

        predictions_path = pool_eval_dir / "predictions_test_epoch_0.json"
        entropy_entities = al_dump_dir / "entropy_entities.pt"
        entropy_relations = al_dump_dir / "entropy_relation.pt"

        if not predictions_path.exists():
            raise PipelineError(f"Prediction JSON not found at {predictions_path}")
        if not entropy_entities.exists() or not entropy_relations.exists():
            raise PipelineError("Entropy tensors missing; ensure --al_dump_dir evaluation succeeded.")

        actual_sample_count = min(args.al_sample_count, len(release_docs))
        if actual_sample_count == 0:
            print(f"[AL] Round {round_idx}: no candidates available; stopping.")
            break

        selected_doc_path = round_dir / f"selected_docs_round{round_idx}.json"
        sft_pred_path = round_dir / f"selected_predictions_round{round_idx}_sft.json"
        baseline_pred_path = round_dir / f"selected_predictions_round{round_idx}_baseline.json"
        remaining_pool_path = round_dir / f"pool_remaining_round{round_idx}.json"
        selection_meta_path = round_dir / f"selection_round{round_idx}.json"

        predictions = load_json(predictions_path)
        entities_entropy_tensor = torch.load(entropy_entities)
        relations_entropy_tensor = torch.load(entropy_relations)
        entities_entropy = entities_entropy_tensor.view(-1).tolist()
        relations_entropy = relations_entropy_tensor.view(-1).tolist()

        if not (len(release_docs) == len(predictions) == len(entities_entropy) == len(relations_entropy)):
            raise PipelineError("Pool docs, predictions, and entropy tensors misaligned in length.")

        label_prediction_path = al_dump_dir / "label_prediction.pt"
        pooler_output_path = al_dump_dir / "pooler_output.pt"
        for required in (label_prediction_path, pooler_output_path):
            if not required.exists():
                raise PipelineError(f"Required AL tensor missing: {required}")

        selection_prefix = f"round{round_idx}"
        selection_script = Path(args.selection_script)
        if not selection_script.exists():
            raise PipelineError(f"Selection script not found: {selection_script}")

        selection_cmd = [
            "python",
            str(selection_script),
            "--dump_dir",
            str(al_dump_dir),
            "--unlabeled_json",
            str(round_pool_path),
            "--output_prefix",
            selection_prefix,
            "--top_k",
            str(args.selection_top_k),
            "--sample_per_group",
            str(args.selection_sample_per_group),
            "--beta",
            str(args.selection_beta),
            "--gamma",
            str(args.selection_gamma),
            "--ncentroids",
            str(args.selection_ncentroids),
        ]
        if args.selection_use_weights:
            selection_cmd.append("--use_weights")
        run_command(selection_cmd, cwd=REPO_ROOT)

        selection_indices_path = al_dump_dir / f"selected_indices_{selection_prefix}.pt"
        sampling_json_path = al_dump_dir / f"sampling_json_{selection_prefix}.json"
        sampling_text_path = al_dump_dir / f"sampling_text_{selection_prefix}.txt"
        if not selection_indices_path.exists():
            raise PipelineError(f"Selection indices not found: {selection_indices_path}")

        all_selected_indices_tensor = torch.load(selection_indices_path)
        all_selected_indices = [int(idx) for idx in all_selected_indices_tensor.tolist()]
        if not all_selected_indices:
            print(f"[WARN] Round {round_idx}: selection script returned no candidates; stopping.")
            break

        limited_indices = all_selected_indices[:actual_sample_count]
        if not limited_indices:
            print(f"[WARN] Round {round_idx}: no candidates selected after limiting; stopping.")
            break

        selected_docs = [release_docs[i] for i in limited_indices]
        selected_predictions = [predictions[i] for i in limited_indices]
        selected_entity_entropy = [entities_entropy[i] for i in limited_indices]
        selected_relation_entropy = [relations_entropy[i] for i in limited_indices]

        selected_idx_set = set(limited_indices)
        remaining_docs = [doc for idx, doc in enumerate(release_docs) if idx not in selected_idx_set]

        dump_json(selected_doc_path, selected_docs)
        dump_json(baseline_pred_path, selected_predictions)
        dump_json(remaining_pool_path, remaining_docs)
        dump_json(
            selection_meta_path,
            [
                {
                    "index": idx,
                    "orig_id": selected_docs[i].get("orig_id"),
                    "entity_entropy": selected_entity_entropy[i],
                    "relation_entropy": selected_relation_entropy[i],
                }
                for i, idx in enumerate(limited_indices)
            ],
        )

        sampling_json_copied = None
        sampling_text_copied = None
        if sampling_json_path.exists():
            sampling_json_copied = round_dir / sampling_json_path.name
            shutil.copy2(sampling_json_path, sampling_json_copied)
        if sampling_text_path.exists():
            sampling_text_copied = round_dir / sampling_text_path.name
            shutil.copy2(sampling_text_path, sampling_text_copied)

        # Reinsert unselected docs to the front of the remaining pool and persist split.
        pool_docs_cache = remaining_docs + pool_docs_cache
        dump_json(pool_path, pool_docs_cache)

        # Merge selected samples into the supervised training set.
        train_docs_cache = train_docs_cache + selected_docs
        dump_json(train_path, train_docs_cache)

        # SFT fine-tuning on the expanded training set.
        sft_ft_seed = args.seed + 51 + round_idx
        sft_label_suffix = f"active_learning_sft_round{round_idx}"
        if args.al_reinit_each_round:
            sft_init_model = args.bert_model
            print(f"[AL] Round {round_idx}: reinitializing SFT from '{sft_init_model}'.")
        else:
            sft_init_model = str(current_model)

        sft_ft_log_dir, sft_ft_save_dir = run_training(
            label_suffix=sft_label_suffix,
            bert_model=sft_init_model,
            config_path=args.config,
            seed=sft_ft_seed,
            epochs=args.sft_ft_epochs,
            lr=args.sft_ft_learning_rate,
            train_batch=args.train_batch_size,
            eval_batch=args.eval_batch_size,
            ft_mode="sft",
        )
        sft_ft_model = final_model_path(sft_ft_save_dir)
        current_model = sft_ft_model

        sft_eval_label = f"active_learning_eval_sft_round{round_idx}"
        sft_ft_eval_dir = run_evaluation(
            model_dir=sft_ft_model,
            dataset_path=valid_path,
            label=sft_eval_label,
            eval_batch=args.eval_batch_size,
        )
        selected_eval_label = f"active_learning_selected_sft_round{round_idx}"
        selected_eval_dir = run_evaluation(
            model_dir=sft_ft_model,
            dataset_path=selected_doc_path,
            label=selected_eval_label,
            eval_batch=args.eval_batch_size,
        )
        sft_selected_pred_path = selected_eval_dir / "predictions_test_epoch_0.json"
        if not sft_selected_pred_path.exists():
            raise PipelineError(f"SFT predictions for selected docs not found: {sft_selected_pred_path}")
        sft_selected_predictions = load_json(sft_selected_pred_path)
        dump_json(sft_pred_path, sft_selected_predictions)

        preference_jsonl = round_dir / f"dpo_preferences_round{round_idx}.jsonl"
        preference_count = build_dpo_preferences(selection["selected_docs"], sft_selected_predictions, preference_jsonl)

        dpo_save_dir = dpo_log_dir = dpo_eval_dir = None
        dpo_model = None
        if preference_count > 0 and args.dpo_epochs > 0:
            dpo_seed = args.seed + 101 + round_idx
            dpo_label_suffix = f"active_learning_dpo_round{round_idx}"
            dpo_log_dir, dpo_save_dir = run_training(
                label_suffix=dpo_label_suffix,
                bert_model=str(sft_ft_model),
                config_path=args.config,
                seed=dpo_seed,
                epochs=args.dpo_epochs,
                lr=args.dpo_learning_rate,
                train_batch=args.train_batch_size,
                eval_batch=args.eval_batch_size,
                ft_mode="dpo",
                dpo_beta=args.dpo_beta,
                dpo_lambda=args.dpo_lambda,
                dpo_negatives=args.dpo_negatives,
                dpo_reference=str(sft_ft_model),
            )
            dpo_model = final_model_path(dpo_save_dir)
            dpo_eval_label = f"active_learning_eval_dpo_round{round_idx}"
            dpo_eval_dir = run_evaluation(
                model_dir=dpo_model,
                dataset_path=valid_path,
                label=dpo_eval_label,
                eval_batch=args.eval_batch_size,
            )
        elif preference_count == 0:
            print(f"[WARN] Round {round_idx}: no DPO pairs; skipping DPO fine-tuning.")

        if not args.keep_temp:
            selected_doc_path.unlink(missing_ok=True)
            sft_pred_path.unlink(missing_ok=True)

        round_summaries.append(
            {
                "round": round_idx,
                "released": release_size,
                "selected_count": len(selected_docs),
                "selection_script": str(selection_script),
                "selection_prefix": selection_prefix,
                "selection_sampling_json": str(sampling_json_copied or sampling_json_path),
                "selection_sampling_text": str(sampling_text_copied or sampling_text_path),
                "pool_eval_dir": str(pool_eval_dir),
                "selection_meta": str(selection_meta_path),
                "baseline_selection_predictions": str(baseline_pred_path),
                "sft_ft_save_dir": str(sft_ft_save_dir),
                "sft_ft_log_dir": str(sft_ft_log_dir),
                "sft_ft_test_eval": str(sft_ft_eval_dir),
                "sft_selected_eval": str(selected_eval_dir),
                "sft_selected_predictions": str(sft_pred_path),
                "dpo_save_dir": str(dpo_save_dir) if dpo_save_dir else None,
                "dpo_log_dir": str(dpo_log_dir) if dpo_log_dir else None,
                "dpo_test_eval": str(dpo_eval_dir) if dpo_eval_dir else None,
                "preference_jsonl": str(preference_jsonl),
                "workspace": str(round_dir),
            }
        )

    summary = {
        "baseline": baseline_summary,
        "rounds": round_summaries,
        "final_train_size": len(train_docs_cache),
        "remaining_pool": len(pool_docs_cache),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "pool_path": str(pool_path),
    }

    summary_path = ACTIVE_DIR / f"active_learning_summary_{timestamp}.json"
    dump_json(summary_path, [summary], indent=2)

    print("\n=== Pipeline complete ===")
    print(f"Baseline checkpoint: {baseline_summary['baseline_save_dir']}")
    for round_summary in round_summaries:
        print(
            f"Round {round_summary['round']}: selected {round_summary['selected_count']} "
            f"(released {round_summary['released']}); "
            f"SFT eval log: {round_summary['sft_ft_test_eval']}"
        )
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"Subprocess failed (exit code {exc.returncode}): {' '.join(exc.cmd)}") from exc
    except PipelineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
