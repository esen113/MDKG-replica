#!/usr/bin/env python3

"""
Convert SynSpERT-style sentence JSON into DyGIE++ document JSONL.

The script expects the input JSON to be a list of sentence-level examples
with fields:
    - tokens: list[str]
    - entities: list[{"type", "start", "end", "text"}]
    - relations: list[{"type", "head", "tail"}]
    - orig_id: "{doc_id}_{sent_idx}" (optional)

Example usage:
    python scripts/convert_to_dygie.py \
        --input-json NER&RE_model/InputsAndOutputs/data/datasets/diabetes_all.json \
        --output-jsonl Outputs/dygie/diabetes.jsonl \
        --dataset-name mdkg_diabetes
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SynSpERT JSON into DyGIE++ JSONL format."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to the sentence-level SynSpERT JSON (e.g. diabetes_all.json).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Destination DyGIE++ JSONL file.",
    )
    parser.add_argument(
        "--dataset-name",
        default="mdkg",
        help="Value to populate the 'dataset' field inside DyGIE++ docs.",
    )
    return parser.parse_args()


def _extract_ids(orig_id: str | None, fallback_idx: int) -> Tuple[str, int]:
    if not orig_id:
        return f"doc_{fallback_idx:05d}", fallback_idx

    parts = str(orig_id).split("_")
    if len(parts) == 1:
        return parts[0], fallback_idx

    doc_id = "_".join(parts[:-1]) or parts[0]
    try:
        sent_idx = int(parts[-1])
    except ValueError:
        sent_idx = fallback_idx
    return doc_id, sent_idx


def _ensure_lists(size: int) -> Tuple[List[List[Any]], List[List[Any]]]:
    return [[] for _ in range(size)], [[] for _ in range(size)]


def _entity_bounds(entity: Dict[str, Any]) -> Tuple[int, int]:
    start = int(entity.get("start", 0))
    end = int(entity.get("end", start))
    if end <= start:
        raise ValueError(f"Invalid entity span: {entity}")
    # DyGIE++ expects end to be inclusive
    return start, end - 1


def convert(sentences: Iterable[Tuple[int, Dict[str, Any]]], dataset_name: str, doc_key: str) -> Dict[str, Any]:
    sorted_sents = sorted(sentences, key=lambda item: item[0])
    sentence_tokens: List[List[str]] = []
    ner_per_sentence, rels_per_sentence = _ensure_lists(len(sorted_sents))

    for sent_position, (sent_idx, sent_doc) in enumerate(sorted_sents):
        tokens = sent_doc.get("tokens", [])
        sentence_tokens.append(tokens)

        entity_info: List[Tuple[int, int, str]] = []
        for ent in sent_doc.get("entities", []):
            start, end = _entity_bounds(ent)
            label = ent.get("type", "None")
            if not tokens:
                continue
            if start >= len(tokens) or end >= len(tokens):
                raise ValueError(f"Entity span out of range for doc '{doc_key}': {ent}")
            ner_per_sentence[sent_position].append([start, end, label])
            entity_info.append((start, end, label))

        for rel in sent_doc.get("relations", []):
            head_idx = int(rel.get("head", -1))
            tail_idx = int(rel.get("tail", -1))
            if head_idx < 0 or tail_idx < 0:
                continue
            if head_idx >= len(entity_info) or tail_idx >= len(entity_info):
                raise ValueError(
                    f"Relation references invalid entity index in doc '{doc_key}': {rel}"
                )
            head_start, head_end, _ = entity_info[head_idx]
            tail_start, tail_end, _ = entity_info[tail_idx]
            label = rel.get("type", "None")
            rels_per_sentence[sent_position].append(
                [head_start, head_end, tail_start, tail_end, label]
            )

    return {
        "doc_key": doc_key,
        "dataset": dataset_name,
        "sentences": sentence_tokens,
        "ner": ner_per_sentence,
        "relations": rels_per_sentence,
        "clusters": [],
    }


def main() -> None:
    args = parse_args()
    input_path = args.input_json.expanduser().resolve()
    output_path = args.output_jsonl.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    data = json.loads(input_path.read_text())
    if not isinstance(data, list):
        raise ValueError("Expected the input JSON to contain a list of documents.")

    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, sentence in enumerate(data):
        doc_id, sent_idx = _extract_ids(sentence.get("orig_id"), idx)
        grouped[doc_id].append((sent_idx, sentence))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for doc_id, sentences in sorted(grouped.items()):
            dygie_doc = convert(sentences, args.dataset_name, doc_id)
            outfile.write(json.dumps(dygie_doc, ensure_ascii=False))
            outfile.write("\n")

    print(f"Wrote {len(grouped)} DyGIE++ documents to {output_path}")


if __name__ == "__main__":
    main()
