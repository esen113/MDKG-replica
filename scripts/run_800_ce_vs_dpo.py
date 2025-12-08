#!/usr/bin/env python3
"""
Pure DPO sanity check on an 800/100 split.

Flow:
1) Ensure MDIEC data is present (reuse pipeline fetch/generate).
2) Sample 900 docs -> 800 train / 100 test (tmp_800/).
3) Swap tiny split into SynSpERT default train/valid paths (backup original).
4) Run base model on 800 to get predictions; build triple DPO prefs (no truncation, no background).
5) Train pure DPO (policy/ref = same base) on 800, eval on 100.
6) Restore original datasets and print summary.

All artifacts live under tmp_800/.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
NER_ROOT = REPO_ROOT / "NER&RE_model"
IO_ROOT = NER_ROOT / "InputsAndOutputs"
RAW_DATA_DIR = IO_ROOT / "data" / "dataset" / "MDIEC"
DATASET_JSON = IO_ROOT / "data" / "dataset" / "MDIEC.json"
TMP_ROOT = REPO_ROOT / "tmp_800"


class RunnerError(RuntimeError):
    pass


def run(cmd: List[str], cwd: Path | None = None) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_data() -> None:
    if DATASET_JSON.exists():
        print(f"[DATA] Found dataset JSON: {DATASET_JSON}")
        return
    print("[DATA] dataset JSON missing; fetching and generating input.")
    run(["python", str(REPO_ROOT / "scripts" / "fetch_official_data.py"), "--overwrite"], cwd=REPO_ROOT)
    run(
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
    if not DATASET_JSON.exists():
        raise RunnerError("Failed to create MDIEC.json")


def sample_split(seed: int = 11, train_count: int = 800, test_count: int = 100) -> Tuple[Path, Path]:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATASET_JSON.read_text())
    if len(data) < train_count + test_count:
        raise RunnerError(f"Dataset too small ({len(data)}) for requested split.")
    rng = random.Random(seed)
    rng.shuffle(data)
    train_docs = data[:train_count]
    test_docs = data[train_count : train_count + test_count]
    train_path = TMP_ROOT / "train_800.json"
    test_path = TMP_ROOT / "test_100.json"
    train_path.write_text(json.dumps(train_docs, ensure_ascii=False, indent=2))
    test_path.write_text(json.dumps(test_docs, ensure_ascii=False, indent=2))
    print(f"[DATA] Split written: train={train_path} ({len(train_docs)}), test={test_path} ({len(test_docs)})")
    return train_path, test_path


def latest_run_dir(root: Path) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        raise RunnerError(f"No runs under {root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def parse_eval_csv(path: Path) -> Dict[str, float]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        row = next(reader)
    return {
        "ner_prec": float(row["ner_prec_micro"]),
        "ner_rec": float(row["ner_rec_micro"]),
        "ner_f1": float(row["ner_f1_micro"]),
        "rel_prec": float(row["rel_prec_micro"]),
        "rel_rec": float(row["rel_rec_micro"]),
        "rel_f1": float(row["rel_f1_micro"]),
        "rel_nec_prec": float(row["rel_nec_prec_micro"]),
        "rel_nec_rec": float(row["rel_nec_rec_micro"]),
        "rel_nec_f1": float(row["rel_nec_f1_micro"]),
    }


def backup_and_swap(train_src: Path, valid_src: Path) -> Tuple[Path, Path]:
    datasets_dir = IO_ROOT / "data" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    orig_train = datasets_dir / "diabetes_train.json"
    orig_valid = datasets_dir / "diabetes_valid.json"
    bak_dir = TMP_ROOT / "bak"
    bak_dir.mkdir(parents=True, exist_ok=True)
    bak_train = bak_dir / "diabetes_train.json.bak"
    bak_valid = bak_dir / "diabetes_valid.json.bak"
    if orig_train.exists():
        shutil.copy2(orig_train, bak_train)
    if orig_valid.exists():
        shutil.copy2(orig_valid, bak_valid)
    shutil.copy2(train_src, orig_train)
    shutil.copy2(valid_src, orig_valid)
    print(f"[DATA] Swapped in tiny split -> {orig_train}, {orig_valid}")
    return bak_train, bak_valid


def restore_dataset(backups: Tuple[Path, Path]) -> None:
    datasets_dir = IO_ROOT / "data" / "datasets"
    orig_train = datasets_dir / "diabetes_train.json"
    orig_valid = datasets_dir / "diabetes_valid.json"
    bak_train, bak_valid = backups
    if bak_train.exists():
        shutil.copy2(bak_train, orig_train)
    if bak_valid.exists():
        shutil.copy2(bak_valid, orig_valid)
    print("[DATA] Restored original datasets")


def train_ce(bert_model: str, epochs: int, batch_size: int, eval_batch: int, seed: int) -> Path:
    log_path = TMP_ROOT / "log_ce"
    save_path = TMP_ROOT / "save_ce"
    shutil.rmtree(log_path, ignore_errors=True)
    shutil.rmtree(save_path, ignore_errors=True)
    run(
        [
            "python",
            str(NER_ROOT / "SynSpERT" / "main.py"),
            "--mode",
            "train",
            "--bert_model",
            bert_model,
            "--config_override",
            str(NER_ROOT / "SynSpERT" / "configs" / "config-coder.json"),
            "--run_seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--train_batch_size",
            str(batch_size),
            "--eval_batch_size",
            str(eval_batch),
            "--lr",
            "5e-5",
            "--log_path",
            str(log_path),
            "--save_path",
            str(save_path),
            "--ft_mode",
            "sft",
            "--rel_filter_threshold",
            "0.3",
            "--entity_filter_threshold",
            "0.3",
        ],
        cwd=REPO_ROOT,
    )
    model_dir = latest_run_dir(save_path / "diabetes_small_run") / "best_model"
    print(f"[CE] best_model: {model_dir}")
    return model_dir


def eval_model(model_dir: Path, dataset: Path, label: str, eval_batch: int, seed: int) -> Dict[str, float]:
    log_root = TMP_ROOT / f"log_{label}"
    save_root = TMP_ROOT / f"save_{label}"
    shutil.rmtree(log_root, ignore_errors=True)
    shutil.rmtree(save_root, ignore_errors=True)
    config_path = NER_ROOT / "SynSpERT" / "configs" / "config-coder.json"
    run(
        [
            "python",
            str(NER_ROOT / "SynSpERT" / "main.py"),
            "--mode",
            "eval",
            "--model_dir",
            str(model_dir),
            "--dataset_path",
            str(dataset),
            "--label",
            label,
            "--config_override",
            str(config_path),
            "--eval_batch_size",
            str(eval_batch),
            "--log_path",
            str(log_root),
            "--save_path",
            str(save_root),
            "--rel_filter_threshold",
            "0.3",
            "--entity_filter_threshold",
            "0.3",
            "--run_seed",
            str(seed),
        ],
        cwd=REPO_ROOT,
    )
    eval_dir = latest_run_dir(log_root / label)
    metrics = parse_eval_csv(eval_dir / "eval_test.csv")
    print(
        f"[EVAL {label}] NER F1 {metrics['ner_f1']:.2f}, REL F1 {metrics['rel_f1']:.2f}, "
        f"REL+NEC F1 {metrics['rel_nec_f1']:.2f}"
    )
    return metrics


def predict_base(train_path: Path, bert_model: str, eval_batch: int, seed: int) -> Path:
    log_root = TMP_ROOT / "log_pred_base"
    save_root = TMP_ROOT / "save_pred_base"
    shutil.rmtree(log_root, ignore_errors=True)
    shutil.rmtree(save_root, ignore_errors=True)
    config_path = NER_ROOT / "SynSpERT" / "configs" / "config-coder.json"
    run(
        [
            "python",
            str(NER_ROOT / "SynSpERT" / "main.py"),
            "--mode",
            "eval",
            "--model_dir",
            bert_model,
            "--dataset_path",
            str(train_path),
            "--label",
            "pred_base_800",
            "--config_override",
            str(config_path),
            "--eval_batch_size",
            str(eval_batch),
            "--log_path",
            str(log_root),
            "--save_path",
            str(save_root),
            "--rel_filter_threshold",
            "0.3",
            "--entity_filter_threshold",
            "0.3",
            "--run_seed",
            str(seed),
        ],
        cwd=REPO_ROOT,
    )
    eval_dir = latest_run_dir(log_root / "pred_base_800")
    pred_path = eval_dir / "predictions_test_epoch_0.json"
    if not pred_path.exists():
        raise RunnerError("Prediction JSON not found.")
    print(f"[PRED] base predictions: {pred_path}")
    return pred_path


def build_prefs(train_path: Path, pred_path: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "python",
            str(REPO_ROOT / "scripts" / "prepare_dpo_preferences.py"),
            "--human-json",
            str(train_path),
            "--prediction-json",
            str(pred_path),
            "--output-jsonl",
            str(output),
            "--format",
            "triple",
            "--max-entity-prefs",
            "0",
            "--max-relation-prefs",
            "0",
            "--dpo-entity-bg-ratio",
            "0",
            "--dpo-relation-bg-ratio",
            "0",
            "--max-span-size",
            "10",
            "--entity-none-label",
            "None",
            "--relation-none-label",
            "None",
        ],
        cwd=REPO_ROOT,
    )
    print(f"[PREF] written to {output}")


def train_dpo(
    bert_model: str,
    prefs: Path,
    epochs: int,
    batch_size: int,
    eval_batch: int,
    seed: int,
) -> Path:
    log_path = TMP_ROOT / "log_dpo"
    save_path = TMP_ROOT / "save_dpo"
    shutil.rmtree(log_path, ignore_errors=True)
    shutil.rmtree(save_path, ignore_errors=True)
    run(
        [
            "python",
            str(NER_ROOT / "SynSpERT" / "main.py"),
            "--mode",
            "train",
            "--bert_model",
            bert_model,
            "--config_override",
            str(NER_ROOT / "SynSpERT" / "configs" / "config-coder.json"),
            "--run_seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--train_batch_size",
            str(batch_size),
            "--eval_batch_size",
            str(eval_batch),
            "--lr",
            "3e-5",
            "--log_path",
            str(log_path),
            "--save_path",
            str(save_path),
            "--ft_mode",
            "dpo",
            "--dpo_beta",
            "1.0",
            "--dpo_lambda",
            "1.0",
            "--dpo_negatives",
            "2",
            "--dpo_format",
            "triple",
            "--dpo_train_batch_size",
            str(batch_size),
            "--dpo_reference",
            bert_model,
            "--dpo_preferences",
            str(prefs),
            "--rel_filter_threshold",
            "0.3",
            "--entity_filter_threshold",
            "0.3",
            "--neg_entity_count",
            "0",
            "--neg_relation_count",
            "0",
        ],
        cwd=REPO_ROOT,
    )
    model_dir = latest_run_dir(save_path / "diabetes_small_run") / "best_model"
    print(f"[DPO] best_model: {model_dir}")
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="CE vs DPO on 800/100 split (base predictions).")
    parser.add_argument("--bert-model", default="GanjinZero/coder_eng_pp", help="Backbone encoder path or HF id.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dpo-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    args = parser.parse_args()

    ensure_data()
    train_path, test_path = sample_split(seed=args.seed)

    backups = backup_and_swap(train_path, test_path)
    try:
        pred_path = predict_base(train_path, args.bert_model, args.eval_batch_size, args.seed)
        prefs_path = TMP_ROOT / "prefs_800.jsonl"
        build_prefs(train_path, pred_path, prefs_path)

        dpo_model = train_dpo(
            args.bert_model,
            prefs_path,
            args.dpo_epochs,
            args.batch_size,
            args.eval_batch_size,
            args.seed,
        )
        dpo_metrics = eval_model(dpo_model, test_path, "dpo_eval_800", args.eval_batch_size, args.seed)

        print("\n=== Summary (800/100 split, DPO-only) ===")
        print(
            f"DPO : NER F1 {dpo_metrics['ner_f1']:.2f}, REL F1 {dpo_metrics['rel_f1']:.2f}, "
            f"REL+NEC F1 {dpo_metrics['rel_nec_f1']:.2f}"
        )
        print("Artifacts under tmp_800/.")
    finally:
        restore_dataset(backups)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise RunnerError(f"Subprocess failed (exit {exc.returncode}): {' '.join(exc.cmd)}") from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")
        raise
