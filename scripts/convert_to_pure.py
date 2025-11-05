#!/usr/bin/env python3

"""
Produce PURE-style NER and RE datasets from SynSpERT JSON splits.

For each input split (train/valid/test), the script emits two JSONL files:
  - pure_ner_<split>.jsonl   : tokens + BIO labels for NER training.
  - pure_re_<split>.jsonl    : tokens + entity mentions + relation labels.

Example usage:
    python scripts/convert_to_pure.py \
        --input-dir NER&RE_model/InputsAndOutputs/data/datasets \
        --prefix diabetes \
        --output-dir Outputs/pure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SynSpERT JSON splits to PURE-friendly JSONL files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing <prefix>_{train,valid,test}.json files.",
    )
    parser.add_argument(
        "--prefix",
        default="diabetes",
        help="Filename prefix of the SynSpERT JSON splits (default: diabetes).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store the generated PURE JSONL files.",
    )
    return parser.parse_args()


def load_split(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of documents in {path}")
    return data


def build_bio_labels(token_count: int, entities: List[Dict]) -> List[str]:
    labels = ["O"] * token_count
    for ent in entities:
        start = int(ent.get("start", 0))
        end = int(ent.get("end", start))
        ent_type = ent.get("type", "Entity")
        if start < 0 or end <= start or start >= token_count:
            continue
        adjusted_end = min(end, token_count)
        labels[start] = f"B-{ent_type}"
        for idx in range(start + 1, adjusted_end):
            labels[idx] = f"I-{ent_type}"
    return labels


def entity_record(doc_id: str, ent_idx: int, entity: Dict) -> Dict:
    start = int(entity.get("start", 0))
    end = int(entity.get("end", start))
    return {
        "id": f"{doc_id}-e{ent_idx}",
        "start": start,
        "end": end,
        "label": entity.get("type", "Entity"),
        "text": entity.get("text"),
    }


def relation_record(doc_id: str, rel_idx: int, head_id: str, tail_id: str, relation: Dict) -> Dict:
    return {
        "id": f"{doc_id}-r{rel_idx}",
        "head": head_id,
        "tail": tail_id,
        "label": relation.get("type", "None"),
    }


def convert_split(split_name: str, docs: List[Dict], output_dir: Path) -> None:
    ner_path = output_dir / f"pure_ner_{split_name}.jsonl"
    re_path = output_dir / f"pure_re_{split_name}.jsonl"

    with ner_path.open("w", encoding="utf-8") as ner_file, re_path.open(
        "w", encoding="utf-8"
    ) as re_file:
        for idx, doc in enumerate(docs):
            tokens = doc.get("tokens", [])
            doc_id = str(doc.get("orig_id", f"{split_name}_{idx:05d}"))
            entities = doc.get("entities", [])
            relations = doc.get("relations", [])

            ner_example = {
                "doc_id": doc_id,
                "tokens": tokens,
                "labels": build_bio_labels(len(tokens), entities),
            }
            ner_file.write(json.dumps(ner_example, ensure_ascii=False) + "\n")

            entity_entries = [
                entity_record(doc_id, ent_idx, ent) for ent_idx, ent in enumerate(entities)
            ]
            relations_map = []
            for rel_idx, rel in enumerate(relations):
                head_idx = int(rel.get("head", -1))
                tail_idx = int(rel.get("tail", -1))
                if head_idx < 0 or tail_idx < 0:
                    continue
                if head_idx >= len(entity_entries) or tail_idx >= len(entity_entries):
                    continue
                head_id = entity_entries[head_idx]["id"]
                tail_id = entity_entries[tail_idx]["id"]
                relations_map.append(relation_record(doc_id, rel_idx, head_id, tail_id, rel))

            re_example = {
                "doc_id": doc_id,
                "tokens": tokens,
                "entity_mentions": entity_entries,
                "relations": relations_map,
            }
            re_file.write(json.dumps(re_example, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_path = input_dir / f"{args.prefix}_{split}.json"
        docs = load_split(split_path)
        convert_split(split, docs, output_dir)
        print(f"Wrote PURE NER/RE files for split '{split}' to {output_dir}")


if __name__ == "__main__":
    main()
