import argparse
import re
from pathlib import Path

import Runner
try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - optional dependency
    snapshot_download = None


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = (Path(__file__).resolve().parent / "../InputsAndOutputs").resolve()
DATA_DIR = BASE_DIR / "data" / "datasets"

BERT_MODEL_DEFAULT = "bert-base-uncased"
BERT_MODEL = BERT_MODEL_DEFAULT
CONFIG_PATH = BERT_MODEL_DEFAULT
CONFIG_OVERRIDE_SET = False
MODEL_TYPE = "syn_spert"

LOG_PATH = (BASE_DIR / "data" / "log").resolve()
SAVE_PATH = (BASE_DIR / "data" / "save").resolve()
CACHE_PATH = (BASE_DIR / "data" / "cache").resolve()
MODEL_DOWNLOAD_DIR = (BASE_DIR / "models").resolve()

TRAIN_PATH = (DATA_DIR / "diabetes_train.json").resolve()
VALID_PATH = (DATA_DIR / "diabetes_valid.json").resolve()
TEST_PATH = (DATA_DIR / "diabetes_test.json").resolve()
TYPES_PATH = (REPO_ROOT / "nutrition_diabetes_types.json").resolve()

RUN_LABEL = "diabetes_small_run"
SEED = 11

DEFAULT_EVAL_LABEL = f"{RUN_LABEL}_eval"


def ensure_directories(extra_dir: Path | None = None):
    paths = [LOG_PATH, SAVE_PATH, CACHE_PATH]
    if extra_dir:
        paths.append(extra_dir)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _sanitize_repo_id(repo_id: str) -> str:
    return re.sub(r"[^\w.-]", "_", repo_id)


def ensure_pretrained_model(model_id: str, download_dir: Path | None) -> str:
    candidate_path = Path(model_id).expanduser()
    if candidate_path.exists():
        return str(candidate_path.resolve())

    if download_dir is None:
        return model_id

    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is required to auto-download pretrained checkpoints. "
            "Install it or provide a local path via --bert_model."
        )

    safe_name = _sanitize_repo_id(model_id)
    target_dir = (download_dir / safe_name).resolve()
    required_files = ["config.json", "pytorch_model.bin"]

    if target_dir.exists() and all((target_dir / fname).exists() for fname in required_files):
        return str(target_dir)

    download_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to download pretrained model '{model_id}': {exc}") from exc

    missing = [fname for fname in required_files if not (target_dir / fname).exists()]
    if missing:
        raise RuntimeError(
            f"Checkpoint '{model_id}' downloaded to '{target_dir}' but missing files: {missing}"
        )

    tokenizer_exists = any(
        (target_dir / fname).exists() for fname in ("tokenizer.json", "vocab.txt")
    )
    if not tokenizer_exists:
        raise RuntimeError(
            f"Checkpoint '{model_id}' at '{target_dir}' does not contain tokenizer files."
        )

    return str(target_dir)


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
        str(BERT_MODEL),
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
        "20",
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
        "100",
        "--neg_relation_count",
        "100",
        "--max_span_size",
        "10",
        "--rel_filter_threshold",
        "0.5",
        "--max_pairs",
        "1000",
        "--sampling_processes",
        "4",
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
        str(CONFIG_PATH),
        "--seed",
        str(SEED),
    ]


def build_eval_args(cli_args: argparse.Namespace) -> list:
    dataset_path = cli_args.dataset_path
    label = cli_args.label or DEFAULT_EVAL_LABEL
    model_dir = cli_args.model_dir

    config_arg = CONFIG_PATH if CONFIG_OVERRIDE_SET else str(model_dir)

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
        "0.5",
        "--max_pairs",
        "1000",
        "--sampling_processes",
        "4",
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
        config_arg,
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
    parser.add_argument(
        "--bert_model",
        type=str,
        default=BERT_MODEL_DEFAULT,
        help="Pretrained model name or path for encoder weights and tokenizer.",
    )
    parser.add_argument(
        "--config_override",
        type=Path,
        default=None,
        help="Optional path to a SynSpERT config JSON. Defaults to the model path.",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=str(MODEL_DOWNLOAD_DIR),
        help="Directory used to cache Hugging Face checkpoints locally. "
        "Set to empty string to disable auto-download.",
    )
    parser.add_argument(
        "--run_seed",
        type=int,
        default=None,
        help="Override training/evaluation seed. Defaults to script constant.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    download_dir = Path(args.download_dir).expanduser().resolve() if args.download_dir else None
    ensure_directories(download_dir)

    global BERT_MODEL, CONFIG_PATH, CONFIG_OVERRIDE_SET, SEED
    selected_model = args.bert_model or BERT_MODEL_DEFAULT
    BERT_MODEL = ensure_pretrained_model(selected_model, download_dir)
    CONFIG_OVERRIDE_SET = args.config_override is not None
    CONFIG_PATH = (
        str(args.config_override.resolve())
        if CONFIG_OVERRIDE_SET
        else BERT_MODEL
    )
    if args.run_seed is not None:
        SEED = args.run_seed

    if args.mode == "train":
        run_args = build_train_args()
    else:
        run_args = build_eval_args(args)

    print("*** Commandline:", " ".join(run_args))
    runner = Runner.Runner()
    runner.run(run_args)


if __name__ == "__main__":
    main()
