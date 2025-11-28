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

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 5e-5
LR_WARMUP = 0.1
WEIGHT_DECAY = 0.01
TRAIN_EPOCHS = 40
NOISE_LAMBDA = 0.15
NEG_ENTITY_COUNT = 100
NEG_RELATION_COUNT = 100
TRAIN_LOG_ITER = 1
SAVE_OPTIMIZER_ENABLED = False
FINAL_EVAL_ENABLED = False
REL_FILTER_THRESHOLD = 0.1
ENTITY_FILTER_THRESHOLD = 0.05
FT_MODE = "sft"
DPO_TRAIN_BATCH_SIZE: int | None = None
DPO_BETA = 0.1
DPO_LAMBDA = 0.1
DPO_NEGATIVES = 4
DPO_REFERENCE: str | None = None
DPO_PREFERENCES: str | None = None
DPO_FORMAT = "doc"
DPO_LAMBDA_ENTITY = 1.0
DPO_LAMBDA_RELATION = 1.0


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
    args_list = [
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
        str(TRAIN_BATCH_SIZE),
        "--use_pos",
        "--pos_embedding",
        "25",
        "--use_entity_clf",
        "logits",
        "--eval_batch_size",
        str(EVAL_BATCH_SIZE),
        "--epochs",
        str(TRAIN_EPOCHS),
        "--lr",
        str(LEARNING_RATE),
        "--lr_warmup",
        str(LR_WARMUP),
        "--weight_decay",
        str(WEIGHT_DECAY),
        "--max_grad_norm",
        "1.0",
        "--prop_drop",
        "0.1",
        "--neg_entity_count",
        str(NEG_ENTITY_COUNT),
        "--neg_relation_count",
        str(NEG_RELATION_COUNT),
        "--max_span_size",
        "10",
        "--rel_filter_threshold",
        str(REL_FILTER_THRESHOLD),
        "--entity_filter_threshold",
        str(ENTITY_FILTER_THRESHOLD),
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
    args_list.extend(["--ft_mode", FT_MODE])
    if FT_MODE == "dpo":
        args_list.extend(["--dpo_beta", str(DPO_BETA)])
        args_list.extend(["--dpo_lambda", str(DPO_LAMBDA)])
        args_list.extend(["--dpo_negatives", str(DPO_NEGATIVES)])
        args_list.extend(["--dpo_format", DPO_FORMAT])
        args_list.extend(["--dpo_lambda_entity", str(DPO_LAMBDA_ENTITY)])
        args_list.extend(["--dpo_lambda_relation", str(DPO_LAMBDA_RELATION)])
        if DPO_TRAIN_BATCH_SIZE is not None:
            args_list.extend(["--dpo_train_batch_size", str(DPO_TRAIN_BATCH_SIZE)])
        if DPO_REFERENCE:
            args_list.extend(["--dpo_reference", str(DPO_REFERENCE)])
        if DPO_PREFERENCES:
            args_list.extend(["--dpo_preferences", str(DPO_PREFERENCES)])
    if NOISE_LAMBDA is not None:
        args_list.extend(["--noise_lambda", str(NOISE_LAMBDA)])
    if TRAIN_LOG_ITER is not None:
        args_list.extend(["--train_log_iter", str(TRAIN_LOG_ITER)])
    if SAVE_OPTIMIZER_ENABLED:
        args_list.append("--save_optimizer")
    if FINAL_EVAL_ENABLED:
        args_list.append("--final_eval")
    return args_list


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
        str(REL_FILTER_THRESHOLD),
        "--entity_filter_threshold",
        str(ENTITY_FILTER_THRESHOLD),
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
    parser.add_argument(
        "--log_path",
        type=Path,
        default=LOG_PATH,
        help="Directory for log outputs when running in train mode.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=SAVE_PATH,
        help="Directory for saving checkpoints when running in train mode.",
    )
    parser.add_argument("--train_batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval_batch_size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--lr_warmup", type=float, default=LR_WARMUP)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--noise_lambda", type=float, default=NOISE_LAMBDA)
    parser.add_argument("--neg_entity_count", type=int, default=NEG_ENTITY_COUNT)
    parser.add_argument("--neg_relation_count", type=int, default=NEG_RELATION_COUNT)
    parser.add_argument("--train_log_iter", type=int, default=TRAIN_LOG_ITER)
    parser.add_argument("--save_optimizer", action="store_true", default=SAVE_OPTIMIZER_ENABLED)
    parser.add_argument("--final_eval", action="store_true", default=FINAL_EVAL_ENABLED)
    parser.add_argument("--rel_filter_threshold", type=float, default=REL_FILTER_THRESHOLD)
    parser.add_argument("--entity_filter_threshold", type=float, default=ENTITY_FILTER_THRESHOLD)
    parser.add_argument("--ft_mode", choices=["sft", "dpo"], default=FT_MODE)
    parser.add_argument("--dpo_train_batch_size", type=int, default=None,
                        help="Batch size used for DPO preference loader; defaults to --train_batch_size.")
    parser.add_argument("--dpo_beta", type=float, default=DPO_BETA)
    parser.add_argument("--dpo_lambda", type=float, default=DPO_LAMBDA)
    parser.add_argument("--dpo_negatives", type=int, default=DPO_NEGATIVES)
    parser.add_argument("--dpo_reference", type=str, default=DPO_REFERENCE)
    parser.add_argument("--dpo_preferences", type=str, default=None)
    parser.add_argument("--dpo_format", choices=["doc", "triple"], default=DPO_FORMAT)
    parser.add_argument("--dpo_lambda_entity", type=float, default=DPO_LAMBDA_ENTITY)
    parser.add_argument("--dpo_lambda_relation", type=float, default=DPO_LAMBDA_RELATION)
    return parser.parse_args()


def main():
    args = parse_args()
    global BERT_MODEL, CONFIG_PATH, CONFIG_OVERRIDE_SET, SEED, LOG_PATH, SAVE_PATH
    global TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, LEARNING_RATE, LR_WARMUP, WEIGHT_DECAY
    global TRAIN_EPOCHS, NOISE_LAMBDA, NEG_ENTITY_COUNT, NEG_RELATION_COUNT, TRAIN_LOG_ITER
    global FT_MODE, DPO_BETA, DPO_LAMBDA, DPO_NEGATIVES, DPO_REFERENCE, DPO_PREFERENCES, DPO_TRAIN_BATCH_SIZE
    global DPO_FORMAT, DPO_LAMBDA_ENTITY, DPO_LAMBDA_RELATION
    global SAVE_OPTIMIZER_ENABLED, FINAL_EVAL_ENABLED, REL_FILTER_THRESHOLD, ENTITY_FILTER_THRESHOLD

    LOG_PATH = Path(args.log_path).expanduser().resolve()
    SAVE_PATH = Path(args.save_path).expanduser().resolve()
    download_dir = Path(args.download_dir).expanduser().resolve() if args.download_dir else None
    ensure_directories(download_dir)

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
    TRAIN_BATCH_SIZE = args.train_batch_size
    EVAL_BATCH_SIZE = args.eval_batch_size
    LEARNING_RATE = args.lr
    LR_WARMUP = args.lr_warmup
    WEIGHT_DECAY = args.weight_decay
    TRAIN_EPOCHS = args.epochs
    NOISE_LAMBDA = args.noise_lambda
    NEG_ENTITY_COUNT = args.neg_entity_count
    NEG_RELATION_COUNT = args.neg_relation_count
    TRAIN_LOG_ITER = args.train_log_iter
    SAVE_OPTIMIZER_ENABLED = args.save_optimizer
    FINAL_EVAL_ENABLED = args.final_eval
    REL_FILTER_THRESHOLD = args.rel_filter_threshold
    ENTITY_FILTER_THRESHOLD = args.entity_filter_threshold
    FT_MODE = args.ft_mode
    DPO_TRAIN_BATCH_SIZE = args.dpo_train_batch_size
    DPO_BETA = args.dpo_beta
    DPO_LAMBDA = args.dpo_lambda
    DPO_NEGATIVES = args.dpo_negatives
    DPO_REFERENCE = args.dpo_reference
    DPO_PREFERENCES = args.dpo_preferences
    DPO_FORMAT = args.dpo_format
    DPO_LAMBDA_ENTITY = args.dpo_lambda_entity
    DPO_LAMBDA_RELATION = args.dpo_lambda_relation

    if args.mode == "train":
        run_args = build_train_args()
    else:
        run_args = build_eval_args(args)

    print("*** Commandline:", " ".join(run_args))
    runner = Runner.Runner()
    runner.run(run_args)


if __name__ == "__main__":
    main()
