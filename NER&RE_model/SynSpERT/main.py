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

DEFAULT_EVAL_LABEL = f"{RUN_LABEL}_eval"


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


def build_eval_args(cli_args: argparse.Namespace) -> list:
    dataset_path = cli_args.dataset_path
    label = cli_args.label or DEFAULT_EVAL_LABEL
    model_dir = cli_args.model_dir

    args = [
        "eval",
        "--model_type",
        MODEL_TYPE,
        "--label",
        label,
        "--model_path",
        str(model_dir),
        "--tokenizer_path",
        str(model_dir),
        "--dataset_path",
        str(dataset_path),
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

    if cli_args.al_dump_dir:
        args.extend(["--al_dump_dir", str(cli_args.al_dump_dir)])
    if cli_args.store_predictions:
        args.append("--store_predictions")

    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utility entrypoint for SynSpERT experiments.")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=SAVE_PATH,
        help="Model directory to use when running in eval mode.",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=TEST_PATH,
        help="Dataset file to evaluate when running in eval mode.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=DEFAULT_EVAL_LABEL,
        help="Run label used for evaluation logs and checkpoints.",
    )
    parser.add_argument(
        "--al_dump_dir",
        type=Path,
        default=None,
        help="If set, dump active-learning features to this directory during eval.",
    )
    parser.add_argument(
        "--store_predictions",
        dest="store_predictions",
        action="store_true",
        default=True,
        help="Store prediction JSON during evaluation (enabled by default).",
    )
    parser.add_argument(
        "--no_store_predictions",
        dest="store_predictions",
        action="store_false",
        help="Disable storing predictions during evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_directories()

    if args.mode == "train":
        run_args = build_train_args()
    else:
        run_args = build_eval_args(args)

    print("*** Commandline:", " ".join(run_args))
    runner = Runner.Runner()
    runner.run(run_args)


if __name__ == "__main__":
    main()
