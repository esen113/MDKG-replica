#!/usr/bin/env python3

"""
One-click MDKG active-learning + DPO demo.

Pipeline overview (single AL round):
  1. (Optional) create GPU-ready Conda env and install dependencies.
  2. Download MDIEC annotations from Hugging Face if missing.
  3. Convert `.ann/.txt` files into SynSpERT JSON and split 0.6/0.1/0.3.
  4. Train a baseline SynSpERT model (SFT) on the 60% training split.
  5. Evaluate the baseline on the held-out test set (10%).
  6. Treat the 30% split as unlabeled pool: run eval with AL tensors and
     select low-confidence samples.
  7. Merge selected samples into the training set, then fine-tune once with
     standard SFT and once with `--ft_mode dpo` (same starting checkpoint).
  8. Re-evaluate both SFT-FT and DPO-FT models on the test split.

Artifacts are stored under `NER&RE_model/InputsAndOutputs/data/active_learning/`.

Example (`python scripts/run_active_learning_pipeline.py`):
    python scripts/run_active_learning_pipeline.py \
        --env-name mdkgspert_gpu \
        --setup-env \
        --bert-model GanjinZero/coder_eng_pp \
        --config NER&RE_model/SynSpERT/configs/config-coder.json \
        --al-sample-count 50
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

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
    data.add_argument("--train-ratio", type=float, default=0.6, help="Train split ratio for generate_augmented_input.py.")
    data.add_argument("--valid-ratio", type=float, default=0.1, help="Validation (test) split ratio.")
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
        help="Number of low-confidence pool documents promoted for annotation / DPO.",
    )
    al.add_argument(
        "--selection-metric",
        choices=("entropy_sum", "entity_entropy", "relation_entropy"),
        default="entropy_sum",
        help="Metric used to rank pool samples.",
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


def generate_synspert_inputs(train_ratio: float, valid_ratio: float, prefix: str) -> None:
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

    print("[DATA] Generating augmented splits.")
    run_command(
        [
            "python",
            str(NER_ROOT / "SynSpERT" / "generate_augmented_input.py"),
            "--input",
            str(DATASET_JSON),
            "--output_dir",
            str(SPLIT_DIR),
            "--prefix",
            prefix,
            "--train_ratio",
            str(train_ratio),
            "--valid_ratio",
            str(valid_ratio),
        ],
        cwd=REPO_ROOT,
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
    micro_pattern = re.compile(r"\s*micro\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)")

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("--- Entities"):
                section = "ner"
                continue
            if line.startswith("--- Relations"):
                # keep section as None until specific relation block headers appear
                section = None
                continue
            if line.startswith("Without named entity classification"):
                section = "relations"
                continue
            if line.startswith("With named entity classification"):
                section = "relations_nec"
                continue

            match = micro_pattern.match(raw_line)
            if match and section:
                precision, recall, f1, support = match.groups()
                metric = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "support": int(support),
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


def select_low_confidence_samples(
    pool_path: Path,
    predictions_path: Path,
    entropy_entities: Path,
    entropy_relations: Path,
    count: int,
    metric: str,
) -> dict:
    pool_docs = load_json(pool_path)
    preds = load_json(predictions_path)
    ent = torch.load(entropy_entities)
    rel = torch.load(entropy_relations)

    ent = ent.tolist()
    rel = rel.tolist()

    if not (len(pool_docs) == len(preds) == len(ent) == len(rel)):
        raise PipelineError("Pool docs, predictions, and entropy tensors misaligned in length.")

    def score(idx: int) -> float:
        if metric == "entropy_sum":
            return ent[idx] + rel[idx]
        if metric == "entity_entropy":
            return ent[idx]
        return rel[idx]

    scored = sorted(range(len(pool_docs)), key=score, reverse=True)
    selected_indices = sorted(scored[: min(count, len(scored))])

    selected_docs = [pool_docs[i] for i in selected_indices]
    selected_preds = [preds[i] for i in selected_indices]
    remaining_docs = [doc for idx, doc in enumerate(pool_docs) if idx not in set(selected_indices)]

    return {
        "selected_indices": selected_indices,
        "selected_docs": selected_docs,
        "selected_predictions": selected_preds,
        "remaining_docs": remaining_docs,
        "entity_entropy": [ent[i] for i in selected_indices],
        "relation_entropy": [rel[i] for i in selected_indices],
    }


def build_dpo_preferences(human_docs: list, model_docs: list, output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
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

            record = {
                "prompt": prompt,
                "chosen": json.dumps(chosen, ensure_ascii=False, sort_keys=True),
                "rejected": json.dumps(rejected, ensure_ascii=False, sort_keys=True),
                "orig_id": doc.get("orig_id"),
            }
            fout.write(json.dumps(record, ensure_ascii=False))
            fout.write("\n")


def main() -> None:
    args = parse_args()

    if args.setup_env:
        setup_environment(args)

    if not args.skip_download:
        download_data()

    generate_synspert_inputs(args.train_ratio, args.valid_ratio, args.prefix)
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
    baseline_metrics = parse_eval_metrics(test_eval_dir)

    # Evaluate on pool to obtain AL tensors.
    al_dump_dir = ACTIVE_DIR / "round1" / "uncertainty"
    pool_eval_dir = run_evaluation(
        model_dir=base_model,
        dataset_path=pool_path,
        label="active_learning_pool_round0",
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

    selection = select_low_confidence_samples(
        pool_path=pool_path,
        predictions_path=predictions_path,
        entropy_entities=entropy_entities,
        entropy_relations=entropy_relations,
        count=args.al_sample_count,
        metric=args.selection_metric,
    )

    round1_dir = ACTIVE_DIR / "round1"
    round1_dir.mkdir(parents=True, exist_ok=True)
    selected_doc_path = round1_dir / "selected_docs_round1.json"
    selected_pred_path = round1_dir / "selected_predictions_round1.json"
    remaining_pool_path = round1_dir / "pool_remaining_round1.json"
    selection_meta_path = round1_dir / "selection_round1.json"

    dump_json(selected_doc_path, selection["selected_docs"])
    dump_json(selected_pred_path, selection["selected_predictions"])
    baseline_pred_path = round1_dir / "selected_predictions_round1_baseline.json"
    dump_json(baseline_pred_path, selection["selected_predictions"])
    dump_json(remaining_pool_path, selection["remaining_docs"])
    dump_json(
        selection_meta_path,
        [
            {
                "index": idx,
                "orig_id": selection["selected_docs"][i].get("orig_id"),
                "entity_entropy": selection["entity_entropy"][i],
                "relation_entropy": selection["relation_entropy"][i],
            }
            for i, idx in enumerate(selection["selected_indices"])
        ],
    )

    # Update pool file by removing selected samples.
    dump_json(pool_path, selection["remaining_docs"])

    # Merge selected docs into training set and overwrite canonical train JSON.
    base_train_docs = load_json(train_path)
    merged_train_docs = base_train_docs + selection["selected_docs"]
    dump_json(train_path, merged_train_docs)

    # SFT fine-tuning on the expanded training set.
    sft_ft_seed = args.seed + 51
    sft_ft_log_dir, sft_ft_save_dir = run_training(
        label_suffix="active_learning_sft_round1",
        bert_model=str(base_model),
        config_path=args.config,
        seed=sft_ft_seed,
        epochs=args.sft_ft_epochs,
        lr=args.sft_ft_learning_rate,
        train_batch=args.train_batch_size,
        eval_batch=args.eval_batch_size,
        ft_mode="sft",
    )
    sft_ft_model = final_model_path(sft_ft_save_dir)

    sft_ft_eval_dir = run_evaluation(
        model_dir=sft_ft_model,
        dataset_path=valid_path,
        label="active_learning_eval_sft_round1",
        eval_batch=args.eval_batch_size,
    )
    sft_metrics = parse_eval_metrics(sft_ft_eval_dir)
    preference_jsonl = round1_dir / "dpo_preferences_round1.jsonl"
    build_dpo_preferences(selection["selected_docs"], selection["selected_predictions"], preference_jsonl)

    # DPO fine-tuning (starting from the SFT-finetuned model).
    dpo_seed = args.seed + 101
    dpo_log_dir, dpo_save_dir = run_training(
        label_suffix="active_learning_dpo_round1",
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

    # Evaluate DPO model on the held-out test split.
    dpo_eval_dir = run_evaluation(
        model_dir=dpo_model,
        dataset_path=valid_path,
        label="active_learning_eval_dpo_round1",
        eval_batch=args.eval_batch_size,
    )
    dpo_metrics = parse_eval_metrics(dpo_eval_dir)

    if not args.keep_temp:
        selected_doc_path.unlink(missing_ok=True)
        selected_pred_path.unlink(missing_ok=True)

    metrics_summary = {
        "baseline": baseline_metrics,
        "sft_round1": sft_metrics,
        "dpo_round1": dpo_metrics,
    }
    metrics_path = ACTIVE_DIR / f"active_learning_metrics_{timestamp}.json"
    metrics_path.write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2))

    summary = {
        "baseline_save_dir": str(base_save_dir),
        "baseline_log_dir": str(base_log_dir),
        "baseline_test_eval": str(test_eval_dir),
        "pool_eval_dir": str(pool_eval_dir),
        "selected_count": len(selection["selected_docs"]),
        "selection_metric": args.selection_metric,
        "baseline_selection_predictions": str(baseline_pred_path),
        "sft_ft_save_dir": str(sft_ft_save_dir),
        "sft_ft_log_dir": str(sft_ft_log_dir),
        "sft_ft_test_eval": str(sft_ft_eval_dir),
        "dpo_save_dir": str(dpo_save_dir),
        "dpo_log_dir": str(dpo_log_dir),
        "dpo_test_eval": str(dpo_eval_dir),
        "preference_jsonl": str(preference_jsonl),
        "active_learning_workspace": str(round1_dir),
        "metrics_json": str(metrics_path),
    }

    summary_path = ACTIVE_DIR / f"active_learning_summary_{timestamp}.json"
    dump_json(summary_path, [summary], indent=2)

    print("\n=== Pipeline complete ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"Subprocess failed (exit code {exc.returncode}): {' '.join(exc.cmd)}") from exc
    except PipelineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
