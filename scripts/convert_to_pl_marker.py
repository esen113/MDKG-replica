#!/usr/bin/env python3

"""
Generate PL-Marker style relation instances (with entity markers) from SynSpERT JSON.

The script produces a JSONL file where each line corresponds to an entity pair
augmented with special markers surrounding the subject/object spans. Positive
relations use their gold label; optional negative examples are labelled as "None".

Example usage:
    python scripts/convert_to_pl_marker.py \
        --input-json NER&RE_model/InputsAndOutputs/data/datasets/diabetes_train.json \
        --output-jsonl Outputs/pl_marker/diabetes_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PL-Marker style datasets with entity markers."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to a SynSpERT JSON split (e.g. diabetes_train.json).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Destination JSONL to store pair-wise marked sentences.",
    )
    parser.add_argument(
        "--include-negative",
        action="store_true",
        help="Whether to add negative (None) relation pairs.",
    )
    parser.add_argument(
        "--negative-label",
        default="None",
        help="Label assigned to negative pairs when --include-negative is set.",
    )
    return parser.parse_args()


def load_docs(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    docs = json.loads(path.read_text())
    if not isinstance(docs, list):
        raise ValueError(f"Expected list of documents inside {path}")
    return docs


def prepare_entities(doc_id: str, entities: List[Dict]) -> List[Dict]:
    prepared = []
    for idx, ent in enumerate(entities):
        start = int(ent.get("start", 0))
        end = int(ent.get("end", start))
        if end <= start:
            continue
        prepared.append(
            {
                "id": f"{doc_id}-e{idx}",
                "start": start,
                "end": end,
                "label": ent.get("type", "Entity"),
                "text": ent.get("text"),
            }
        )
    return prepared


def relation_lookup(relations: Iterable[Dict]) -> Dict[Tuple[int, int], str]:
    lookup: Dict[Tuple[int, int], str] = {}
    for rel in relations:
        head_idx = int(rel.get("head", -1))
        tail_idx = int(rel.get("tail", -1))
        if head_idx < 0 or tail_idx < 0:
            continue
        label = rel.get("type", "None")
        lookup[(head_idx, tail_idx)] = label
    return lookup


def insert_markers(tokens: List[str], head_span: Tuple[int, int], tail_span: Tuple[int, int]) -> List[str]:
    markers_start = {}
    markers_end = {}

    head_start, head_end = head_span
    tail_start, tail_end = tail_span

    markers_start.setdefault(head_start, []).append("<e1>")
    markers_end.setdefault(head_end, []).append("</e1>")
    markers_start.setdefault(tail_start, []).append("<e2>")
    markers_end.setdefault(tail_end, []).append("</e2>")

    marked_tokens: List[str] = []
    token_count = len(tokens)

    for idx, token in enumerate(tokens):
        if idx in markers_start:
            marked_tokens.extend(markers_start[idx])
        marked_tokens.append(token)
        if (idx + 1) in markers_end:
            marked_tokens.extend(markers_end[idx + 1])

    if token_count in markers_end:
        marked_tokens.extend(markers_end[token_count])

    return marked_tokens


def create_pair_records(
    doc: Dict,
    doc_index: int,
    include_negative: bool,
    negative_label: str,
) -> Iterable[Dict]:
    doc_id = str(doc.get("orig_id", f"doc_{doc_index:05d}"))
    tokens = doc.get("tokens", [])
    raw_entities = doc.get("entities", [])
    if not tokens or not raw_entities:
        return []

    entities = prepare_entities(doc_id, raw_entities)
    if not entities:
        return []

    relation_map = relation_lookup(doc.get("relations", []))
    records: List[Dict] = []

    for head_idx, tail_idx in product(range(len(entities)), repeat=2):
        if head_idx == tail_idx:
            continue
        label = relation_map.get((head_idx, tail_idx), negative_label)
        if label == negative_label and not include_negative:
            continue

        head_ent = entities[head_idx]
        tail_ent = entities[tail_idx]
        marked_tokens = insert_markers(
            tokens,
            (head_ent["start"], head_ent["end"]),
            (tail_ent["start"], tail_ent["end"]),
        )

        records.append(
            {
                "doc_id": doc_id,
                "pair_id": f"{doc_id}-h{head_idx}-t{tail_idx}",
                "tokens": tokens,
                "marked_tokens": marked_tokens,
                "marked_text": " ".join(marked_tokens),
                "head": head_ent,
                "tail": tail_ent,
                "label": label,
            }
        )

    return records


def main() -> None:
    args = parse_args()
    input_path = args.input_json.expanduser().resolve()
    output_path = args.output_jsonl.expanduser().resolve()
    docs = load_docs(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as outfile:
        for doc_idx, doc in enumerate(docs):
            pair_records = create_pair_records(
                doc,
                doc_idx,
                args.include_negative,
                args.negative_label,
            )
            for record in pair_records:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} PL-Marker style instances to {output_path}")


if __name__ == "__main__":
    main()
