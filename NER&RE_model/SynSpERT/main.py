import argparse
from pathlib import Path

import Runner


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = (Path(__file__).resolve().parent / "../InputsAndOutputs").resolve()
DATA_DIR = BASE_DIR / "data" / "datasets"

BERT_MODEL = "bert-base-uncased"
MODEL_TYPE = "syn_spert"

LOG_PATH = (BASE_DIR / "data" / "log").resolve()
SAVE_PATH = (BASE_DIR / "data" / "save").resolve()
CACHE_PATH = (BASE_DIR / "data" / "cache").resolve()

TRAIN_PATH = (DATA_DIR / "diabetes_train.json").resolve()
VALID_PATH = (DATA_DIR / "diabetes_valid.json").resolve()
TEST_PATH = (DATA_DIR / "diabetes_test.json").resolve()
TYPES_PATH = (REPO_ROOT / "nutrition_diabetes_types.json").resolve()

RUN_LABEL = "diabetes_small_run"
SEED = 11


def ensure_directories():
    for path in [LOG_PATH, SAVE_PATH, CACHE_PATH]:
        path.mkdir(parents=True, exist_ok=True)


def build_train_args() -> list:
    return [
        "train",
        "--model_type",
        MODEL_TYPE,
        "--label",
        RUN_LABEL,
        "--model_path",
        BERT_MODEL,
        "--tokenizer_path",
        BERT_MODEL,
        "--train_path",
        str(TRAIN_PATH),
        "--valid_path",
        str(VALID_PATH),
        "--types_path",
        str(TYPES_PATH),
        "--cache_path",
        str(CACHE_PATH),
        "--size_embedding",
        "25",
        "--train_batch_size",
        "2",
        "--use_pos",
        "--pos_embedding",
        "25",
        "--use_entity_clf",
        "logits",
        "--eval_batch_size",
        "2",
        "--epochs",
        "5",
        "--lr",
        "5e-5",
        "--lr_warmup",
        "0.1",
        "--weight_decay",
        "0.01",
        "--max_grad_norm",
        "1.0",
        "--prop_drop",
        "0.1",
        "--neg_entity_count",
        "20",
        "--neg_relation_count",
        "20",
        "--max_span_size",
        "10",
        "--rel_filter_threshold",
        "0.4",
        "--max_pairs",
        "1000",
        "--sampling_processes",
        "0",
        "--sampling_limit",
        "100",
        "--store_predictions",
        "--store_examples",
        "--skip_eval",
        "--log_path",
        str(LOG_PATH),
        "--save_path",
        str(SAVE_PATH),
        "--max_seq_length",
        "512",
        "--config_path",
        BERT_MODEL,
        "--seed",
        str(SEED),
    ]


def build_eval_args(model_dir: Path) -> list:
    return [
        "eval",
        "--model_type",
        MODEL_TYPE,
        "--label",
        f"{RUN_LABEL}_eval",
        "--model_path",
        str(model_dir),
        "--tokenizer_path",
        str(model_dir),
        "--dataset_path",
        str(TEST_PATH),
        "--types_path",
        str(TYPES_PATH),
        "--cache_path",
        str(CACHE_PATH),
        "--size_embedding",
        "25",
        "--use_pos",
        "--pos_embedding",
        "25",
        "--use_entity_clf",
        "logits",
        "--eval_batch_size",
        "2",
        "--lr",
        "5e-5",
        "--lr_warmup",
        "0.1",
        "--weight_decay",
        "0.01",
        "--max_grad_norm",
        "1.0",
        "--prop_drop",
        "0.1",
        "--max_span_size",
        "10",
        "--rel_filter_threshold",
        "0.4",
        "--max_pairs",
        "1000",
        "--sampling_processes",
        "0",
        "--sampling_limit",
        "100",
        "--store_predictions",
        "--store_examples",
        "--log_path",
        str(LOG_PATH),
        "--save_path",
        str(SAVE_PATH),
        "--max_seq_length",
        "512",
        "--config_path",
        BERT_MODEL,
        "--seed",
        str(SEED),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utility entrypoint for SynSpERT experiments.")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=SAVE_PATH,
        help="Model directory to use when running in eval mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_directories()

    if args.mode == "train":
        run_args = build_train_args()
    else:
        run_args = build_eval_args(args.model_dir)

    print("*** Commandline:", " ".join(run_args))
    runner = Runner.Runner()
    runner.run(run_args)


if __name__ == "__main__":
    main()
