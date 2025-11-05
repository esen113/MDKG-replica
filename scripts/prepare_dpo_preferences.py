#!/usr/bin/env python3

"""
Build a DPO preference dataset from SynSpERT-formatted gold labels and model predictions.

Each preference sample captures the same token sequence with two annotations:
    - chosen: human-labelled entities / relations
    - rejected: model-predicted entities / relations

The output JSONL can be consumed by downstream DPO fine-tuning pipelines.

Example:
    python scripts/prepare_dpo_preferences.py \
        --human-json NER&RE_model/InputsAndOutputs/data/datasets/diabetes_train.json \
        --prediction-json NER&RE_model/InputsAndOutputs/data/log/run_label/predictions_train_epoch_0.json \
        --output-jsonl Outputs/dpo/diabetes_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare prompt/chosen/rejected triplets for DPO fine-tuning."
    )
    parser.add_argument(
        "--human-json",
        type=Path,
        required=True,
        help="Path to SynSpERT JSON with human annotations (e.g. diabetes_train.json).",
    )
    parser.add_argument(
        "--prediction-json",
        type=Path,
        required=True,
        help="Path to model predictions JSON produced by --store_predictions.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Destination JSONL file storing DPO preference records.",
    )
    parser.add_argument(
        "--prompt-template",
        default="Extract entities and relations from the sentence:",
        help="Instruction prefix prepended to each prompt.",
    )
    return parser.parse_args()


def load_json_list(path: Path, field: str) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{field} file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {field} file: {path}")
    return data


def serialise_annotation(tokens: List[str], entities: Iterable[Dict[str, Any]], relations: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    def span_text(start: int, end: int) -> str:
        return " ".join(tokens[start:end])

    serialised_entities: List[Dict[str, Any]] = []
    for ent in entities:
        start = int(ent.get("start", 0))
        end = int(ent.get("end", start))
        ent_type = ent.get("type", "Entity")
        serialised_entities.append(
            {
                "type": ent_type,
                "start": start,
                "end": end,
                "text": ent.get("text") or span_text(start, end),
            }
        )

    serialised_entities.sort(key=lambda item: (item["start"], item["end"], item["type"]))

    serialised_relations: List[Dict[str, Any]] = []
    for rel in relations:
        head = int(rel.get("head", -1))
        tail = int(rel.get("tail", -1))
        rel_type = rel.get("type", "relation")
        serialised_relations.append(
            {
                "type": rel_type,
                "head": head,
                "tail": tail,
            }
        )

    serialised_relations.sort(key=lambda item: (item["head"], item["tail"], item["type"]))

    return {
        "entities": serialised_entities,
        "relations": serialised_relations,
    }


def ensure_alignment(human_doc: Dict[str, Any], model_doc: Dict[str, Any], idx: int) -> None:
    human_tokens = human_doc.get("tokens", [])
    model_tokens = model_doc.get("tokens", [])
    if human_tokens != model_tokens:
        raise ValueError(
            f"Token mismatch at index {idx}: human vs model predictions differ.\n"
            f"Human: {human_tokens}\nModel: {model_tokens}"
        )


def build_prompt(tokens: List[str], template: str) -> str:
    sentence = " ".join(tokens)
    return f"{template}\n{sentence}"


def main() -> None:
    args = parse_args()

    human_docs = load_json_list(args.human_json.expanduser().resolve(), "human")
    model_docs = load_json_list(args.prediction_json.expanduser().resolve(), "prediction")

    if len(human_docs) != len(model_docs):
        raise ValueError(
            f"Human annotations ({len(human_docs)}) and predictions ({len(model_docs)}) "
            "must have identical lengths."
        )

    output_path = args.output_jsonl.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, (human_doc, model_doc) in enumerate(zip(human_docs, model_docs)):
            ensure_alignment(human_doc, model_doc, idx)
            tokens = human_doc.get("tokens", [])

            prompt = build_prompt(tokens, args.prompt_template)
            chosen = serialise_annotation(tokens, human_doc.get("entities", []), human_doc.get("relations", []))
            rejected = serialise_annotation(tokens, model_doc.get("entities", []), model_doc.get("relations", []))

            record = {
                "prompt": prompt,
                "chosen": json.dumps(chosen, ensure_ascii=False, sort_keys=True),
                "rejected": json.dumps(rejected, ensure_ascii=False, sort_keys=True),
                "orig_id": human_doc.get("orig_id"),
            }
            fout.write(json.dumps(record, ensure_ascii=False))
            fout.write("\n")
            written += 1

    print(f"Wrote {written} preference pairs to {output_path}")


if __name__ == "__main__":
    main()

