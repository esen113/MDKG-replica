"""
Convert raw Brat-style annotations into a single SynSpERT-compatible JSON file.

Example usage (from repository root):

    python 'NER&RE_model/SynSpERT/generate_input.py' \
        --input_dir 'NER&RE_model/InputsAndOutputs/data/dataset/MDIEC' \
        --output_json 'NER&RE_model/InputsAndOutputs/data/dataset/MDIEC.json'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nltk

from ann2json_updated import Annotation


def ensure_nltk_punkt() -> None:
    """Ensure the Punkt tokenizer required by ann2json is available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def convert_directory(input_dir: Path) -> list[dict]:
    """Load all .ann/.txt files within input_dir into SynSpERT JSON docs."""
    docs: list[dict] = []
    ann_files = sorted(input_dir.glob("*.ann"))

    if not ann_files:
        raise FileNotFoundError(f"No .ann files found in {input_dir}")

    for ann_path in ann_files:
        pmid = ann_path.stem
        annotation = Annotation(str(input_dir), pmid)
        sentences = annotation.obtain_annotations()
        for idx, sent in enumerate(sentences):
            sent["orig_id"] = f"{pmid}_{idx}"
            docs.append(sent)

    return docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SynSpERT-ready JSON from Brat-style .ann/.txt files."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Directory containing paired .ann/.txt files (e.g. MDIEC).",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("NER&RE_model/InputsAndOutputs/data/dataset/MDIEC.json"),
        help="Destination JSON file. Defaults to MDIEC.json within the dataset directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    ensure_nltk_punkt()

    docs = convert_directory(input_dir)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(docs, ensure_ascii=False, indent=2))

    print(f"Wrote {len(docs)} sentences to {output_json}")


if __name__ == "__main__":
    main()
