#!/usr/bin/env python3

"""
One-click MDKG active-learning + DPO demo.

Pipeline overview (multi-round AL with fixed SFT base):
  1. (Optional) create GPU-ready Conda env and install dependencies.
  2. Download MDIEC annotations from Hugging Face if missing.
  3. Convert `.ann/.txt` files into SynSpERT JSON and split the corpus into
     (i) SFT base subset, (ii) reserved DPO seed subset, (iii) validation/test,
     and (iv) an unlabeled pool released gradually.
  4. Train a baseline SynSpERT model (SFT) on the base subset and evaluate on
     the held-out validation/test split.
  5. Build an initial DPO preference set by contrasting the frozen SFT base
     predictions against the reserved DPO seed subset, then fine-tune via DPO
     to obtain Active-Chat (AC) model #1.
  6. For each active-learning round:
       a. Expose a batch of pool samples, score with the latest AC model, and
          retain `--al-sample-count` informative samples via clustering.
       b. Append the newly “labeled” samples to the DPO dataset, regenerate
          preference pairs with the frozen SFT base predictions, and keep an
          ever-growing JSONL archive.
       c. Run a fresh DPO fine-tune (default: re-init from the round0 SFT base)
          on the accumulated preferences to produce the next AC model.
       d. Log all artifacts, return unselected samples to the pool, and repeat.

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

from prepare_dpo_preferences import DEFAULT_PROMPT_TEMPLATE, prepare_preference_records


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
    data.add_argument(
        "--initial-dpo-count",
        type=int,
        default=300,
        help="Number of documents held out from SFT to seed the first DPO fine-tune.",
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
    train.add_argument(
        "--dpo-train-batch-size",
        type=int,
        default=None,
        help="Override batch size used during DPO fine-tuning (defaults to --train-batch-size).",
    )
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
    train.add_argument(
        "--dpo-init-mode",
        choices=("base", "latest"),
        default="base",
        help="Checkpoint used to initialize/reference each DPO run ('base' uses the round0 SFT model).",
    )
    train.add_argument(
        "--dpo-preference-format",
        choices=("doc", "triple"),
        default="triple",
        help="Schema used for generated DPO preferences ('triple' enables candidate-level supervision).",
    )
    train.add_argument(
        "--dpo-max-entity-prefs",
        type=int,
        default=50,
        help="Maximum entity preference pairs per example when building triple-format preferences.",
    )
    train.add_argument(
        "--dpo-max-relation-prefs",
        type=int,
        default=50,
        help="Maximum relation preference pairs per example when building triple-format preferences.",
    )
    train.add_argument(
        "--dpo-entity-none-label",
        default="None",
        help="Placeholder label written for missing entities in triple-format preferences.",
    )
    train.add_argument(
        "--dpo-relation-none-label",
        default="None",
        help="Placeholder label written for missing relations in triple-format preferences.",
    )

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


def generate_synspert_inputs(
    train_count: int,
    valid_count: int,
    prefix: str,
    seed: int,
    initial_dpo_count: int,
) -> None:
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
    if train_count + valid_count + initial_dpo_count >= total:
        raise PipelineError(
            "Requested train ({train_count}) + valid ({valid_count}) + initial_dpo ({initial_dpo_count}) "
            f"consumes entire dataset of {total} documents.".format(
                train_count=train_count, valid_count=valid_count, initial_dpo_count=initial_dpo_count
            )
        )

    rng = random.Random(seed)
    rng.shuffle(docs)

    train_docs = docs[:train_count]
    dpo_seed_docs = docs[train_count : train_count + initial_dpo_count]
    valid_docs = docs[train_count + initial_dpo_count : train_count + initial_dpo_count + valid_count]
    pool_docs = docs[train_count + initial_dpo_count + valid_count :]

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    dump_json(SPLIT_DIR / f"{prefix}_train.json", train_docs)
    dump_json(SPLIT_DIR / f"{prefix}_valid.json", valid_docs)
    dump_json(SPLIT_DIR / f"{prefix}_test.json", pool_docs)
    dump_json(SPLIT_DIR / f"{prefix}_dpo_seed.json", dpo_seed_docs)
    dump_json(SPLIT_DIR / f"{prefix}_dpo_all.json", dpo_seed_docs)

    print(
        "[DATA] Split complete: train={train}, initial_dpo={dpo}, valid/test={valid}, pool={pool} "
        "(seed={seed}).".format(
            train=len(train_docs),
            dpo=len(dpo_seed_docs),
            valid=len(valid_docs),
            pool=len(pool_docs),
            seed=seed,
        )
    )


def dataset_paths(prefix: str) -> tuple[Path, Path, Path, Path, Path]:
    train = SPLIT_DIR / f"{prefix}_train.json"
    valid = SPLIT_DIR / f"{prefix}_valid.json"
    test = SPLIT_DIR / f"{prefix}_test.json"
    dpo_seed = SPLIT_DIR / f"{prefix}_dpo_seed.json"
    dpo_all = SPLIT_DIR / f"{prefix}_dpo_all.json"
    missing = [path for path in (train, valid, test, dpo_seed, dpo_all) if not path.exists()]
    if missing:
        raise PipelineError(
            f"Expected SynSpERT splits not found for prefix '{prefix}': "
            + ", ".join(str(path) for path in missing)
        )
    return train, valid, test, dpo_seed, dpo_all


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
    dpo_preferences: str | None = None,
    dpo_train_batch: int | None = None,
    dpo_format: str = "doc",
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
                "--dpo_format",
                dpo_format,
            ]
        )
        if dpo_train_batch is not None:
            cmd.extend(["--dpo_train_batch_size", str(dpo_train_batch)])
        cmd.extend(
            [
                "--neg_entity_count",
                "0",
                "--neg_relation_count",
                "0",
            ]
        )
        if dpo_reference:
            cmd.extend(["--dpo_reference", dpo_reference])
        if dpo_preferences:
            cmd.extend(["--dpo_preferences", dpo_preferences])
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


def append_jsonl(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dest.open("a", encoding="utf-8") as fout:
        for raw_line in fin:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            fout.write(line + "\n")


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


def build_dpo_preferences(
    human_docs: list,
    model_docs: list,
    output_jsonl: Path,
    *,
    preference_format: str,
    max_entity_prefs: int,
    max_relation_prefs: int,
    entity_none_label: str,
    relation_none_label: str,
) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records, skipped = prepare_preference_records(
        human_docs,
        model_docs,
        DEFAULT_PROMPT_TEMPLATE,
        preference_format,
        max_entity_prefs,
        max_relation_prefs,
        entity_none_label,
        relation_none_label,
    )

    with output_jsonl.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False))
            fout.write("\n")

    kept = len(records)
    if kept == 0:
        print(
            "[WARN] All selected samples perfectly matched the latest SFT predictions; "
            "no DPO preference pairs were generated."
        )
    elif skipped:
        print(f"[INFO] Skipped {skipped} preference pairs where model output already matched the ground truth.")
    return kept


def resolve_dpo_init(
    init_mode: str,
    base_model: Path,
    current_model: Path,
) -> tuple[str, str]:
    if init_mode == "base":
        reference = base_model
    elif init_mode == "latest":
        reference = current_model
    else:
        raise PipelineError(f"Unsupported --dpo-init-mode value: {init_mode}")
    return str(reference), str(reference)


def main() -> None:
    args = parse_args()

    if args.setup_env:
        setup_environment(args)

    if not args.skip_download:
        download_data()

    generate_synspert_inputs(
        args.train_count,
        args.valid_count,
        args.prefix,
        args.seed,
        args.initial_dpo_count,
    )
    train_path, valid_path, pool_path, dpo_seed_path, dpo_all_path = dataset_paths(args.prefix)

    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    round0_dir = ACTIVE_DIR / "round0"
    round0_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    shutil.copy2(train_path, round0_dir / f"{args.prefix}_train_round0_{timestamp}.json")
    shutil.copy2(valid_path, round0_dir / f"{args.prefix}_test_round0_{timestamp}.json")
    shutil.copy2(pool_path, round0_dir / f"{args.prefix}_pool_round0_{timestamp}.json")
    shutil.copy2(dpo_seed_path, round0_dir / f"{args.prefix}_dpo_seed_round0_{timestamp}.json")
    shutil.copy2(dpo_all_path, round0_dir / f"{args.prefix}_dpo_all_round0_{timestamp}.json")

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
    dpo_seed_docs = load_json(dpo_seed_path)
    dpo_docs_cache = load_json(dpo_all_path)
    round_summaries: list[dict] = []

    preference_archive = ACTIVE_DIR / f"dpo_preferences_accum_{timestamp}.jsonl"
    total_preference_count = 0
    current_model = base_model

    seed_pref_path = round0_dir / f"dpo_preferences_seed_round0_{timestamp}.jsonl"
    seed_pref_count = 0
    if dpo_seed_docs:
        seed_eval_label = "active_learning_dpo_seed_round0_baseline"
        seed_eval_dir = run_evaluation(
            model_dir=base_model,
            dataset_path=dpo_seed_path,
            label=seed_eval_label,
            eval_batch=args.eval_batch_size,
        )
        seed_pred_path = seed_eval_dir / "predictions_test_epoch_0.json"
        seed_predictions = load_json(seed_pred_path)
        seed_pref_count = build_dpo_preferences(
            dpo_seed_docs,
            seed_predictions,
            seed_pref_path,
            preference_format=args.dpo_preference_format,
            max_entity_prefs=args.dpo_max_entity_prefs,
            max_relation_prefs=args.dpo_max_relation_prefs,
            entity_none_label=args.dpo_entity_none_label,
            relation_none_label=args.dpo_relation_none_label,
        )
    else:
        seed_pref_path.touch()

    if seed_pref_count > 0:
        shutil.copy2(seed_pref_path, preference_archive)
        total_preference_count = seed_pref_count
    else:
        preference_archive.touch(exist_ok=True)

    initial_dpo_log_dir = initial_dpo_save_dir = initial_dpo_eval_dir = None
    if seed_pref_count > 0 and args.dpo_epochs > 0:
        dpo_init_model, dpo_reference_model = resolve_dpo_init(
            args.dpo_init_mode, base_model, current_model
        )
        dpo_seed_label_suffix = "active_learning_dpo_seed_round0"
        initial_dpo_log_dir, initial_dpo_save_dir = run_training(
            label_suffix=dpo_seed_label_suffix,
            bert_model=dpo_init_model,
            config_path=args.config,
            seed=args.seed + 101,
            epochs=args.dpo_epochs,
            lr=args.dpo_learning_rate,
            train_batch=args.train_batch_size,
            eval_batch=args.eval_batch_size,
            ft_mode="dpo",
            dpo_beta=args.dpo_beta,
            dpo_lambda=args.dpo_lambda,
            dpo_negatives=args.dpo_negatives,
            dpo_reference=dpo_reference_model,
            dpo_preferences=str(preference_archive),
            dpo_train_batch=args.dpo_train_batch_size,
            dpo_format=args.dpo_preference_format,
        )
        current_model = final_model_path(initial_dpo_save_dir)
        initial_dpo_eval_dir = run_evaluation(
            model_dir=current_model,
            dataset_path=valid_path,
            label="active_learning_eval_dpo_seed_round0",
            eval_batch=args.eval_batch_size,
        )
    elif seed_pref_count == 0:
        print("[WARN] Initial DPO stage skipped: no preference pairs from the seed subset.")
    else:
        print("[WARN] Initial DPO stage skipped because --dpo-epochs=0.")

    initial_dpo_summary = {
        "preference_jsonl": str(seed_pref_path),
        "preference_archive": str(preference_archive),
        "preference_count": seed_pref_count,
        "dpo_log_dir": str(initial_dpo_log_dir) if initial_dpo_log_dir else None,
        "dpo_save_dir": str(initial_dpo_save_dir) if initial_dpo_save_dir else None,
        "dpo_eval_dir": str(initial_dpo_eval_dir) if initial_dpo_eval_dir else None,
    }

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

        pool_docs_cache = remaining_docs + pool_docs_cache
        dump_json(pool_path, pool_docs_cache)

        dpo_docs_cache.extend(selected_docs)
        dump_json(dpo_all_path, dpo_docs_cache)

        baseline_selected_eval_label = f"active_learning_baseline_selected_round{round_idx}"
        baseline_selected_eval_dir = run_evaluation(
            model_dir=base_model,
            dataset_path=selected_doc_path,
            label=baseline_selected_eval_label,
            eval_batch=args.eval_batch_size,
        )
        baseline_selected_pred_path = baseline_selected_eval_dir / "predictions_test_epoch_0.json"
        baseline_selected_predictions = load_json(baseline_selected_pred_path)
        dump_json(sft_pred_path, baseline_selected_predictions)

        preference_jsonl = round_dir / f"dpo_preferences_round{round_idx}.jsonl"
        preference_count = build_dpo_preferences(
            selected_docs,
            baseline_selected_predictions,
            preference_jsonl,
            preference_format=args.dpo_preference_format,
            max_entity_prefs=args.dpo_max_entity_prefs,
            max_relation_prefs=args.dpo_max_relation_prefs,
            entity_none_label=args.dpo_entity_none_label,
            relation_none_label=args.dpo_relation_none_label,
        )
        if preference_count > 0:
            append_jsonl(preference_jsonl, preference_archive)
            total_preference_count += preference_count
        else:
            print(f"[WARN] Round {round_idx}: no preference pairs generated from selected docs.")

        dpo_log_dir = dpo_save_dir = dpo_eval_dir = None
        if preference_count > 0 and args.dpo_epochs > 0:
            dpo_init_model, dpo_reference_model = resolve_dpo_init(
                args.dpo_init_mode, base_model, current_model
            )
            dpo_label_suffix = f"active_learning_dpo_round{round_idx}"
            dpo_log_dir, dpo_save_dir = run_training(
                label_suffix=dpo_label_suffix,
                bert_model=dpo_init_model,
                config_path=args.config,
                seed=args.seed + 101 + round_idx,
                epochs=args.dpo_epochs,
                lr=args.dpo_learning_rate,
                train_batch=args.train_batch_size,
                eval_batch=args.eval_batch_size,
                ft_mode="dpo",
                dpo_beta=args.dpo_beta,
                dpo_lambda=args.dpo_lambda,
                dpo_negatives=args.dpo_negatives,
                dpo_reference=dpo_reference_model,
                dpo_preferences=str(preference_archive),
                dpo_train_batch=args.dpo_train_batch_size,
                dpo_format=args.dpo_preference_format,
            )
            current_model = final_model_path(dpo_save_dir)
            dpo_eval_label = f"active_learning_eval_dpo_round{round_idx}"
            dpo_eval_dir = run_evaluation(
                model_dir=current_model,
                dataset_path=valid_path,
                label=dpo_eval_label,
                eval_batch=args.eval_batch_size,
            )
        elif preference_count == 0:
            pass
        else:
            print(f"[WARN] Round {round_idx}: DPO epochs set to 0; skipping fine-tuning.")

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
                "ac_selection_predictions": str(baseline_pred_path),
                "base_selection_eval": str(baseline_selected_eval_dir),
                "base_selection_predictions": str(sft_pred_path),
                "preference_jsonl": str(preference_jsonl),
                "preference_count": preference_count,
                "preference_archive": str(preference_archive),
                "dpo_log_dir": str(dpo_log_dir) if dpo_log_dir else None,
                "dpo_save_dir": str(dpo_save_dir) if dpo_save_dir else None,
                "dpo_eval_dir": str(dpo_eval_dir) if dpo_eval_dir else None,
                "workspace": str(round_dir),
            }
        )

    summary = {
        "baseline": baseline_summary,
        "initial_dpo": initial_dpo_summary,
        "rounds": round_summaries,
        "final_train_size": len(train_docs_cache),
        "dpo_dataset_size": len(dpo_docs_cache),
        "remaining_pool": len(pool_docs_cache),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "pool_path": str(pool_path),
        "dpo_all_path": str(dpo_all_path),
        "preference_archive": str(preference_archive),
        "total_preferences": total_preference_count,
    }

    summary_path = ACTIVE_DIR / f"active_learning_summary_{timestamp}.json"
    dump_json(summary_path, [summary], indent=2)

    print("\n=== Pipeline complete ===")
    print(f"Baseline checkpoint: {baseline_summary['baseline_save_dir']}")
    if initial_dpo_summary.get("dpo_save_dir"):
        print(f"Initial AC checkpoint: {initial_dpo_summary['dpo_save_dir']}")
    for round_summary in round_summaries:
        print(
            f"Round {round_summary['round']}: selected {round_summary['selected_count']} "
            f"(released {round_summary['released']}); "
            f"DPO eval log: {round_summary.get('dpo_eval_dir')}"
        )
    print(f"Preference archive ({total_preference_count} pairs): {preference_archive}")
    print(f"Summary written to: {summary_path}")
if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"Subprocess failed (exit code {exc.returncode}): {' '.join(exc.cmd)}") from exc
    except PipelineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
