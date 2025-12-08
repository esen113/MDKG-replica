#!/usr/bin/env python3
"""
Tiny CE vs DPO sanity check on a 50-doc subset.

What it does:
1) Ensure MDIEC data is available (reuse pipeline fetch/generate if missing).
2) Sample 50 docs -> 40 train / 10 valid (tmp_tiny/).
3) Train CE on the tiny set (overfit expected).
4) Generate base-model predictions on the tiny train set for DPO prefs.
5) Build triple-format DPO preferences (no truncation, no background).
6) Train pure DPO (policy/ref = same base encoder) on the tiny set.
7) Evaluate CE vs DPO on the tiny valid set and print metrics.

This script avoids touching your main runs; everything is under tmp_tiny/.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
NER_ROOT = REPO_ROOT / "NER&RE_model"
IO_ROOT = NER_ROOT / "InputsAndOutputs"
RAW_DATA_DIR = IO_ROOT / "data" / "dataset" / "MDIEC"
DATASET_JSON = IO_ROOT / "data" / "dataset" / "MDIEC.json"
TMP_ROOT = REPO_ROOT / "tmp_tiny"


class SanityError(RuntimeError):
    pass


def run(cmd: List[str], cwd: Path | None = None) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_data() -> None:
    """Fetch official data and build MDIEC.json if missing."""
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
        raise SanityError("Failed to create MDIEC.json")


def sample_tiny(seed: int = 11, train_count: int = 40, valid_count: int = 10) -> tuple[Path, Path]:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATASET_JSON.read_text())
    if len(data) < train_count + valid_count:
        raise SanityError(f"Dataset too small ({len(data)}) for requested tiny split.")
    rng = random.Random(seed)
    rng.shuffle(data)
    tiny_train = data[:train_count]
    tiny_valid = data[train_count : train_count + valid_count]
    train_path = TMP_ROOT / "train.json"
    valid_path = TMP_ROOT / "valid.json"
    train_path.write_text(json.dumps(tiny_train, ensure_ascii=False, indent=2))
    valid_path.write_text(json.dumps(tiny_valid, ensure_ascii=False, indent=2))
    print(f"[DATA] Tiny split written: train={train_path} ({len(tiny_train)} docs), "
          f"valid={valid_path} ({len(tiny_valid)} docs)")
    return train_path, valid_path


def latest_run_dir(root: Path) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        raise SanityError(f"No runs under {root}")
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


def train_ce(train_path: Path, valid_path: Path, bert_model: str) -> Path:
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
            "11",
            "--epochs",
            "20",
            "--train_batch_size",
            "2",
            "--eval_batch_size",
            "2",
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


def eval_model(model_dir: Path, dataset: Path, label: str) -> Dict[str, float]:
    log_root = TMP_ROOT / f"log_{label}"
    save_root = TMP_ROOT / f"save_{label}"
    shutil.rmtree(log_root, ignore_errors=True)
    shutil.rmtree(save_root, ignore_errors=True)
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
            "--eval_batch_size",
            "2",
            "--log_path",
            str(log_root),
            "--save_path",
            str(save_root),
            "--rel_filter_threshold",
            "0.3",
            "--entity_filter_threshold",
            "0.3",
        ],
        cwd=REPO_ROOT,
    )
    eval_dir = latest_run_dir(log_root / label)
    metrics = parse_eval_csv(eval_dir / "eval_test.csv")
    print(f"[EVAL] {label} metrics: NER F1 {metrics['ner_f1']:.2f}, REL F1 {metrics['rel_f1']:.2f}, "
          f"REL+NEC F1 {metrics['rel_nec_f1']:.2f}")
    return metrics


def predict_base(train_path: Path, bert_model: str) -> Path:
    log_root = TMP_ROOT / "log_pred_base"
    save_root = TMP_ROOT / "save_pred_base"
    shutil.rmtree(log_root, ignore_errors=True)
    shutil.rmtree(save_root, ignore_errors=True)
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
            "tiny_pred_base",
            "--eval_batch_size",
            "2",
            "--log_path",
            str(log_root),
            "--save_path",
            str(save_root),
            "--rel_filter_threshold",
            "0.3",
            "--entity_filter_threshold",
            "0.3",
        ],
        cwd=REPO_ROOT,
    )
    eval_dir = latest_run_dir(log_root / "tiny_pred_base")
    pred_path = eval_dir / "predictions_test_epoch_0.json"
    if not pred_path.exists():
        raise SanityError("Prediction JSON not found.")
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


def train_dpo(train_path: Path, valid_path: Path, prefs: Path, bert_model: str) -> Path:
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
            "11",
            "--epochs",
            "20",
            "--train_batch_size",
            "2",
            "--eval_batch_size",
            "2",
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
            "2",
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


def backup_and_swap_dataset(train_src: Path, valid_src: Path) -> tuple[Path, Path]:
    """Backup original train/valid JSON and swap in tiny split for training/eval."""
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
    print(f"[DATA] Swapped datasets: {orig_train} <- tiny train, {orig_valid} <- tiny valid")
    return bak_train, bak_valid


def restore_dataset(backups: tuple[Path, Path]) -> None:
    datasets_dir = IO_ROOT / "data" / "datasets"
    orig_train = datasets_dir / "diabetes_train.json"
    orig_valid = datasets_dir / "diabetes_valid.json"
    bak_train, bak_valid = backups
    if bak_train.exists():
        shutil.copy2(bak_train, orig_train)
    if bak_valid.exists():
        shutil.copy2(bak_valid, orig_valid)
    print("[DATA] Restored original datasets")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny CE vs DPO sanity check.")
    parser.add_argument("--bert-model", default="GanjinZero/coder_eng_pp", help="Backbone encoder path or HF id.")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    ensure_data()
    train_path, valid_path = sample_tiny(seed=args.seed)

    backups = backup_and_swap_dataset(train_path, valid_path)
    try:
        ce_model = train_ce(train_path, valid_path, args.bert_model)
        ce_metrics = eval_model(ce_model, valid_path, "tiny_ce_eval")

        pred_path = predict_base(train_path, args.bert_model)
        prefs_path = TMP_ROOT / "prefs.jsonl"
        build_prefs(train_path, pred_path, prefs_path)

        dpo_model = train_dpo(train_path, valid_path, prefs_path, args.bert_model)
        dpo_metrics = eval_model(dpo_model, valid_path, "tiny_dpo_eval")

        print("\n=== Summary (tiny set) ===")
        print(f"CE  : NER F1 {ce_metrics['ner_f1']:.2f}, REL F1 {ce_metrics['rel_f1']:.2f}, "
              f"REL+NEC F1 {ce_metrics['rel_nec_f1']:.2f}")
        print(f"DPO : NER F1 {dpo_metrics['ner_f1']:.2f}, REL F1 {dpo_metrics['rel_f1']:.2f}, "
              f"REL+NEC F1 {dpo_metrics['rel_nec_f1']:.2f}")
        print("Artifacts under tmp_tiny/.")
    finally:
        restore_dataset(backups)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SanityError(f"Subprocess failed (exit {exc.returncode}): {' '.join(exc.cmd)}") from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")
        raise
