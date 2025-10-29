#!/usr/bin/env python3
"""
Download the MDKG data archive from Hugging Face and extract it locally.

Default behaviour:
    - downloads YF0808/MDKG_data :: data/MDIEC.zip
    - extracts contents into NER&RE_model/InputsAndOutputs/data/dataset/

Usage:
    python scripts/fetch_official_data.py

Optional flags:
    --repo-id          Hugging Face dataset repository id (default: YF0808/MDKG_data)
    --filename         Path to the ZIP file inside the repo (default: data/MDIEC.zip)
    --output-dir       Destination directory for extracted files
    --overwrite        Remove existing destination folder before extraction
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

DEFAULT_REPO_ID = "YF0808/MDKG_data"
DEFAULT_FILENAME = "data/MDIEC.zip"
DEFAULT_OUTPUT = Path("NER&RE_model/InputsAndOutputs/data/dataset").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and extract MDKG data from Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repository id.")
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help="File path inside the repository to download (ZIP archive).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory to extract the dataset into.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output directory before extraction.",
    )
    return parser.parse_args()


def ensure_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError:
        sys.exit("huggingface-hub is required. Install it with `pip install huggingface-hub` and retry.")


def download_zip(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return Path(local_path)


def extract_zip(zip_path: Path, target_dir: Path, overwrite: bool = False) -> None:
    if overwrite and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)


def main() -> None:
    args = parse_args()
    ensure_huggingface_hub()

    zip_path = download_zip(args.repo_id, args.filename)
    extract_zip(zip_path, args.output_dir.resolve(), args.overwrite)
    print(f"Dataset extracted to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
