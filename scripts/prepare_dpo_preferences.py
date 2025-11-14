#!/usr/bin/env python3

"""
Build a DPO preference dataset from SynSpERT-formatted gold labels and model predictions.

Two formats are supported:
    - doc: legacy prompt/chosen/rejected pairs (doc-level).
    - triple: new candidate/pairwise preferences aligned on shared candidate pools.

Example:
    python scripts/prepare_dpo_preferences.py \
        --human-json NER&RE_model/InputsAndOutputs/data/datasets/diabetes_train.json \
        --prediction-json NER&RE_model/InputsAndOutputs/data/log/run_label/predictions_train_epoch_0.json \
        --output-jsonl Outputs/dpo/diabetes_train.jsonl \
        --format triple
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Span = Tuple[int, int]
EntityTriplet = Tuple[int, int, str]
RelationTriplet = Tuple[Span, Span, str]
PreferencePairs = List[Tuple[EntityTriplet | RelationTriplet, EntityTriplet | RelationTriplet]]

DEFAULT_PROMPT_TEMPLATE = "Extract entities and relations from the sentence:"
DEFAULT_ENTITY_NONE_LABEL = "None"
DEFAULT_RELATION_NONE_LABEL = "None"

__all__ = [
    "build_doc_preference_records",
    "build_triple_preference_records",
    "prepare_preference_records",
]


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
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Instruction prefix prepended to each prompt.",
    )
    parser.add_argument(
        "--format",
        choices=("doc", "triple"),
        default="triple",
        help="Preference format to emit ('doc' for legacy chosen/rejected, 'triple' for candidate pools).",
    )
    parser.add_argument(
        "--max-entity-prefs",
        type=int,
        default=50,
        help="Maximum entity-level preference pairs to retain per document (0 = unlimited).",
    )
    parser.add_argument(
        "--max-relation-prefs",
        type=int,
        default=50,
        help="Maximum relation-level preference pairs to retain per document (0 = unlimited).",
    )
    parser.add_argument(
        "--entity-none-label",
        default=DEFAULT_ENTITY_NONE_LABEL,
        help="Label name representing the absence of an entity (defaults to 'None').",
    )
    parser.add_argument(
        "--relation-none-label",
        default=DEFAULT_RELATION_NONE_LABEL,
        help="Label name representing the absence of a relation (defaults to 'None').",
    )
    return parser.parse_args()


def load_json_list(path: Path, field: str) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{field} file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {field} file: {path}")
    return data


def ensure_alignment(human_doc: Dict[str, Any], model_doc: Dict[str, Any], idx: int) -> None:
    human_tokens = human_doc.get("tokens", [])
    model_tokens = model_doc.get("tokens", [])
    if human_tokens != model_tokens:
        raise ValueError(
            f"Token mismatch at index {idx}: human vs model predictions differ.\n"
            f"Human: {human_tokens}\nModel: {model_tokens}"
        )


def build_prompt(tokens: Sequence[str], template: str) -> str:
    sentence = " ".join(tokens)
    return f"{template}\n{sentence}"


def build_doc_preference_records(
    human_docs: Sequence[Dict[str, Any]],
    model_docs: Sequence[Dict[str, Any]],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    skipped = 0

    for idx, (human_doc, model_doc) in enumerate(zip(human_docs, model_docs)):
        ensure_alignment(human_doc, model_doc, idx)
        tokens = human_doc.get("tokens", [])
        prompt = build_prompt(tokens, prompt_template)

        chosen_entities = sorted(
            human_doc.get("entities", []), key=lambda e: (e.get("start", 0), e.get("end", 0), e.get("type", ""))
        )
        chosen_relations = sorted(
            human_doc.get("relations", []), key=lambda r: (r.get("head", 0), r.get("tail", 0), r.get("type", ""))
        )
        rejected_entities = sorted(
            model_doc.get("entities", []), key=lambda e: (e.get("start", 0), e.get("end", 0), e.get("type", ""))
        )
        rejected_relations = sorted(
            model_doc.get("relations", []), key=lambda r: (r.get("head", 0), r.get("tail", 0), r.get("type", ""))
        )

        if chosen_entities == rejected_entities and chosen_relations == rejected_relations:
            skipped += 1
            continue

        record = {
            "prompt": prompt,
            "doc": copy.deepcopy(human_doc),
            "rejected_entities": rejected_entities,
            "rejected_relations": rejected_relations,
            "orig_id": human_doc.get("orig_id"),
            "preference_format": "doc",
        }
        records.append(record)

    return records, skipped


def _extract_gold_entities(doc: Dict[str, Any]) -> List[EntityTriplet]:
    entities = []
    for ent in doc.get("entities", []):
        start = int(ent.get("start", 0))
        end = int(ent.get("end", start))
        entities.append((start, end, ent.get("type", DEFAULT_ENTITY_NONE_LABEL)))
    return entities


def _extract_gold_relations(doc: Dict[str, Any]) -> List[RelationTriplet]:
    relations = []
    entities = doc.get("entities", [])
    spans = [(int(ent.get("start", 0)), int(ent.get("end", ent.get("start", 0)))) for ent in entities]
    for rel in doc.get("relations", []):
        head_idx = int(rel.get("head", -1))
        tail_idx = int(rel.get("tail", -1))
        if 0 <= head_idx < len(spans):
            head_span = (
                int(rel.get("head_start", spans[head_idx][0])),
                int(rel.get("head_end", spans[head_idx][1])),
            )
        else:
            head_span = (
                int(rel.get("head_start", 0)),
                int(rel.get("head_end", 0)),
            )
        if 0 <= tail_idx < len(spans):
            tail_span = (
                int(rel.get("tail_start", spans[tail_idx][0])),
                int(rel.get("tail_end", spans[tail_idx][1])),
            )
        else:
            tail_span = (
                int(rel.get("tail_start", 0)),
                int(rel.get("tail_end", 0)),
            )
        relations.append((head_span, tail_span, rel.get("type", DEFAULT_RELATION_NONE_LABEL)))
    return relations


def _extract_pred_entities(pred_doc: Dict[str, Any]) -> List[EntityTriplet]:
    entities = []
    for ent in pred_doc.get("entities", []):
        start = int(ent.get("start", 0))
        end = int(ent.get("end", start))
        entities.append((start, end, ent.get("type", DEFAULT_ENTITY_NONE_LABEL)))
    return entities


def _extract_pred_relations(pred_doc: Dict[str, Any]) -> List[RelationTriplet]:
    relations = []
    pred_entities = pred_doc.get("entities", [])
    spans = [(int(ent.get("start", 0)), int(ent.get("end", ent.get("start", 0)))) for ent in pred_entities]
    for rel in pred_doc.get("relations", []):
        head_idx = int(rel.get("head", -1))
        tail_idx = int(rel.get("tail", -1))
        if head_idx < 0 or tail_idx < 0:
            continue
        if head_idx >= len(spans) or tail_idx >= len(spans):
            continue
        head_span = spans[head_idx]
        tail_span = spans[tail_idx]
        relations.append((head_span, tail_span, rel.get("type", DEFAULT_RELATION_NONE_LABEL)))
    return relations


def _entity_sort_key(entity: EntityTriplet) -> Tuple[int, int, str]:
    return entity[0], entity[1], entity[2]


def _relation_sort_key(relation: RelationTriplet) -> Tuple[int, int, int, int, str]:
    head, tail, rel_type = relation
    return head[0], head[1], tail[0], tail[1], rel_type


def _limit_preferences(preferences: List[Dict[str, int]], limit: int) -> List[Dict[str, int]]:
    if limit and len(preferences) > limit:
        return preferences[:limit]
    return preferences


def _entity_candidates_and_preferences(
    gold_entities: Sequence[EntityTriplet],
    pred_entities: Sequence[EntityTriplet],
    none_label: str,
) -> Tuple[List[Dict[str, Any]], PreferencePairs]:
    gold_set = set(gold_entities)
    pred_set = set(pred_entities)
    candidates = set(gold_set | pred_set)
    prefs: PreferencePairs = []

    gold_by_span: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    pred_by_span: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    for start, end, label in gold_set:
        gold_by_span[(start, end)].add(label)
    for start, end, label in pred_set:
        pred_by_span[(start, end)].add(label)

    for span in sorted(gold_by_span.keys()):
        gold_types = sorted(gold_by_span[span])
        pred_types = sorted(pred_by_span.get(span, set()))
        if not pred_types:
            none_candidate = (span[0], span[1], none_label)
            candidates.add(none_candidate)
            for label in gold_types:
                prefs.append(((span[0], span[1], label), none_candidate))
        else:
            for label in gold_types:
                for pred_label in pred_types:
                    if label != pred_label:
                        prefs.append(((span[0], span[1], label), (span[0], span[1], pred_label)))

    for hallucination in sorted(pred_set - gold_set, key=_entity_sort_key):
        span = (hallucination[0], hallucination[1])
        none_candidate = (span[0], span[1], none_label)
        candidates.add(none_candidate)
        prefs.append((none_candidate, hallucination))

    ordered_candidates = sorted(candidates, key=_entity_sort_key)
    candidate_map = {cand: idx for idx, cand in enumerate(ordered_candidates)}
    candidate_payload = [{"span": [start, end], "type": ent_type} for start, end, ent_type in ordered_candidates]

    indexed_prefs: PreferencePairs = []
    for pos, neg in prefs:
        if pos not in candidate_map or neg not in candidate_map:
            continue
        indexed_prefs.append((pos, neg))

    return candidate_payload, indexed_prefs


def _relation_candidates_and_preferences(
    gold_relations: Sequence[RelationTriplet],
    pred_relations: Sequence[RelationTriplet],
    none_label: str,
) -> Tuple[List[Dict[str, Any]], PreferencePairs]:
    gold_set = set(gold_relations)
    pred_set = set(pred_relations)
    candidates = set(gold_set | pred_set)
    prefs: PreferencePairs = []

    gold_by_pair: Dict[Tuple[Span, Span], set[str]] = defaultdict(set)
    pred_by_pair: Dict[Tuple[Span, Span], set[str]] = defaultdict(set)
    for head, tail, label in gold_set:
        gold_by_pair[(head, tail)].add(label)
    for head, tail, label in pred_set:
        pred_by_pair[(head, tail)].add(label)

    for pair in sorted(gold_by_pair.keys(), key=lambda p: (p[0][0], p[0][1], p[1][0], p[1][1])):
        gold_types = sorted(gold_by_pair[pair])
        pred_types = sorted(pred_by_pair.get(pair, set()))
        if not pred_types:
            none_candidate = (pair[0], pair[1], none_label)
            candidates.add(none_candidate)
            for label in gold_types:
                prefs.append(((pair[0], pair[1], label), none_candidate))
            continue
        for label in gold_types:
            for pred_label in pred_types:
                if label != pred_label:
                    prefs.append(((pair[0], pair[1], label), (pair[0], pair[1], pred_label)))

    for hallucination in sorted(pred_set - gold_set, key=_relation_sort_key):
        none_candidate = (hallucination[0], hallucination[1], none_label)
        candidates.add(none_candidate)
        prefs.append((none_candidate, hallucination))

    ordered_candidates = sorted(candidates, key=_relation_sort_key)
    candidate_map = {cand: idx for idx, cand in enumerate(ordered_candidates)}
    candidate_payload = [
        {"h": [head[0], head[1]], "t": [tail[0], tail[1]], "type": rel_type} for head, tail, rel_type in ordered_candidates
    ]

    indexed_prefs: PreferencePairs = []
    for pos, neg in prefs:
        if pos not in candidate_map or neg not in candidate_map:
            continue
        indexed_prefs.append((pos, neg))

    return candidate_payload, indexed_prefs


def _serialise_preferences(
    prefs: PreferencePairs,
    candidate_map: Dict[EntityTriplet | RelationTriplet, int],
    limit: int,
) -> List[Dict[str, int]]:
    serialised = []
    for pos, neg in prefs:
        serialised.append({"pos": candidate_map[pos], "neg": candidate_map[neg]})
    return _limit_preferences(serialised, limit)


def build_triple_preference_records(
    human_docs: Sequence[Dict[str, Any]],
    model_docs: Sequence[Dict[str, Any]],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_entity_prefs: int = 50,
    max_relation_prefs: int = 50,
    entity_none_label: str = DEFAULT_ENTITY_NONE_LABEL,
    relation_none_label: str = DEFAULT_RELATION_NONE_LABEL,
) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    skipped = 0

    for idx, (human_doc, model_doc) in enumerate(zip(human_docs, model_docs)):
        ensure_alignment(human_doc, model_doc, idx)
        tokens = human_doc.get("tokens", [])
        prompt = build_prompt(tokens, prompt_template)

        gold_entities = _extract_gold_entities(human_doc)
        pred_entities = _extract_pred_entities(model_doc)
        entity_candidates, entity_pref_pairs = _entity_candidates_and_preferences(
            gold_entities, pred_entities, entity_none_label
        )

        gold_relations = _extract_gold_relations(human_doc)
        pred_relations = _extract_pred_relations(model_doc)
        relation_candidates, relation_pref_pairs = _relation_candidates_and_preferences(
            gold_relations, pred_relations, relation_none_label
        )

        if not entity_pref_pairs and not relation_pref_pairs:
            skipped += 1
            continue

        entity_map = {
            (cand["span"][0], cand["span"][1], cand["type"]): idx for idx, cand in enumerate(entity_candidates)
        }
        relation_map = {
            ((cand["h"][0], cand["h"][1]), (cand["t"][0], cand["t"][1]), cand["type"]): idx
            for idx, cand in enumerate(relation_candidates)
        }

        entity_preferences = _limit_preferences(
            [{"pos": entity_map[pos], "neg": entity_map[neg]} for pos, neg in entity_pref_pairs],
            max_entity_prefs,
        )
        relation_preferences = _limit_preferences(
            [{"pos": relation_map[pos], "neg": relation_map[neg]} for pos, neg in relation_pref_pairs],
            max_relation_prefs,
        )

        if not entity_preferences and not relation_preferences:
            skipped += 1
            continue

        record = {
            "prompt": prompt,
            "doc": copy.deepcopy(human_doc),
            "orig_id": human_doc.get("orig_id"),
            "entity_candidates": entity_candidates,
            "relation_candidates": relation_candidates,
            "entity_preferences": entity_preferences,
            "relation_preferences": relation_preferences,
            "entity_none_label": entity_none_label,
            "relation_none_label": relation_none_label,
            "preference_format": "triple",
        }
        records.append(record)

    return records, skipped


def prepare_preference_records(
    human_docs: Sequence[Dict[str, Any]],
    model_docs: Sequence[Dict[str, Any]],
    prompt_template: str,
    fmt: str,
    max_entity_prefs: int,
    max_relation_prefs: int,
    entity_none_label: str,
    relation_none_label: str,
) -> Tuple[List[Dict[str, Any]], int]:
    if fmt == "doc":
        return build_doc_preference_records(human_docs, model_docs, prompt_template)
    return build_triple_preference_records(
        human_docs,
        model_docs,
        prompt_template,
        max_entity_prefs=max_entity_prefs,
        max_relation_prefs=max_relation_prefs,
        entity_none_label=entity_none_label,
        relation_none_label=relation_none_label,
    )


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

    records, skipped = prepare_preference_records(
        human_docs,
        model_docs,
        args.prompt_template,
        args.format,
        args.max_entity_prefs,
        args.max_relation_prefs,
        args.entity_none_label,
        args.relation_none_label,
    )

    with output_path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False))
            fout.write("\n")

    print(f"Wrote {len(records)} preference samples to {output_path}")
    if skipped:
        print(f"Skipped {skipped} documents with no informative preference pairs.")


if __name__ == "__main__":
    main()
