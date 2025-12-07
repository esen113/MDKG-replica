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
import random
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
        default=200,
        help="Maximum entity-level preference pairs to retain per document (0 = unlimited).",
    )
    parser.add_argument(
        "--max-relation-prefs",
        type=int,
        default=200,
        help="Maximum relation-level preference pairs to retain per document (0 = unlimited).",
    )
    parser.add_argument(
        "--max-span-size",
        type=int,
        default=10,
        help="Maximum span size considered when sampling background entity spans.",
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
    parser.add_argument(
        "--gold-only-candidates",
        action="store_true",
        help="Restrict triple-format candidates to gold spans/pairs only (no hallucinated spans).",
    )
    parser.add_argument(
        "--dpo-entity-bg-ratio",
        type=float,
        default=0.0,
        help="Ratio of entity background prefs (None>fake) relative to (gold+hallucination) prefs per doc.",
    )
    parser.add_argument(
        "--dpo-relation-bg-ratio",
        type=float,
        default=0.0,
        help="Ratio of relation background prefs (None>fake) relative to (gold+hallucination) prefs per doc.",
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


def _extract_pred_relations(
    pred_doc: Dict[str, Any],
    reference_entities: Sequence[EntityTriplet] | None = None,
) -> List[RelationTriplet]:
    relations = []
    pred_entities = pred_doc.get("entities", [])
    spans = [(int(ent.get("start", 0)), int(ent.get("end", ent.get("start", 0)))) for ent in pred_entities]

    reference_spans = None
    if reference_entities:
        reference_spans = {(start, end) for start, end, _ in reference_entities}

    for rel in pred_doc.get("relations", []):
        head_idx = int(rel.get("head", -1))
        tail_idx = int(rel.get("tail", -1))
        if head_idx < 0 or tail_idx < 0:
            continue
        if head_idx >= len(spans) or tail_idx >= len(spans):
            continue
        head_span = spans[head_idx]
        tail_span = spans[tail_idx]

        if reference_spans is not None:
            if head_span not in reference_spans or tail_span not in reference_spans:
                # Skip relations whose entities do not align to any gold entity span.
                continue

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


def sample_background_entity_spans(
    tokens: Sequence[str],
    gold_entities: Sequence[EntityTriplet],
    pred_entities: Sequence[EntityTriplet],
    max_span_size: int,
    max_bg: int,
) -> List[Span]:
    """
    Sample background spans that are neither gold nor predicted entities.
    """
    if max_bg <= 0:
        return []

    total_tokens = len(tokens)
    gold_spans = {(s, e) for (s, e, _) in gold_entities}
    pred_spans = {(s, e) for (s, e, _) in pred_entities}

    bg_spans: List[Span] = []
    for size in range(1, max_span_size + 1):
        for i in range(0, total_tokens - size + 1):
            span = (i, i + size)
            if span in gold_spans or span in pred_spans:
                continue
            bg_spans.append(span)

    if not bg_spans:
        return []

    random.shuffle(bg_spans)
    return bg_spans[:max_bg]


def sample_background_relation_pairs(
    entity_spans: Sequence[Span],
    gold_relations: Sequence[RelationTriplet],
    pred_relations: Sequence[RelationTriplet],
    max_bg: int,
) -> List[Tuple[Span, Span]]:
    """
    Sample background relation pairs from existing entity spans, excluding gold/pred pairs.
    """
    if max_bg <= 0:
        return []

    gold_pairs = {(h, t) for (h, t, _) in gold_relations}
    pred_pairs = {(h, t) for (h, t, _) in pred_relations}

    unique_spans = list({(s, e) for (s, e) in entity_spans})
    bg_pairs: List[Tuple[Span, Span]] = []
    for h_span in unique_spans:
        for t_span in unique_spans:
            if h_span == t_span:
                continue
            pair = (h_span, t_span)
            if pair in gold_pairs or pair in pred_pairs:
                continue
            bg_pairs.append(pair)

    if not bg_pairs:
        return []

    random.shuffle(bg_pairs)
    return bg_pairs[:max_bg]


def _entity_candidates_and_preferences(
    gold_entities: Sequence[EntityTriplet],
    pred_entities: Sequence[EntityTriplet],
    none_label: str,
    gold_only: bool = False,
) -> Tuple[List[Dict[str, Any]], PreferencePairs]:
    gold_by_span: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    pred_by_span: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    for start, end, label in gold_entities:
        gold_by_span[(start, end)].add(label)
    gold_spans = list(gold_by_span.keys())
    for start, end, label in pred_entities:
        pred_by_span[(start, end)].add(label)

    candidates: set[EntityTriplet] = set()
    prefs: PreferencePairs = []

    def _add_candidate(span: Tuple[int, int], label: str) -> EntityTriplet:
        candidate = (span[0], span[1], label)
        candidates.add(candidate)
        return candidate

    for span in sorted(gold_by_span.keys()):
        gold_types = sorted(gold_by_span[span])
        pred_types = sorted(pred_by_span.get(span, set()))

        for label in gold_types:
            _add_candidate(span, label)
        if pred_types:
            # 输入一致，输出不一致：gold > pred（不含与 gold 相同的标签）
            for label in gold_types:
                for pred_label in pred_types:
                    if pred_label != label:
                        _add_candidate(span, pred_label)
                        prefs.append(((span[0], span[1], label), (span[0], span[1], pred_label)))
            # gold 标签未被输出：gold > None
            none_candidate = _add_candidate(span, none_label)
            for label in gold_types:
                if label not in pred_types:
                    prefs.append(((span[0], span[1], label), none_candidate))
        else:
            # 输入一致，模型未输出：gold > None
            none_candidate = _add_candidate(span, none_label)
            for label in gold_types:
                prefs.append(((span[0], span[1], label), none_candidate))

        pred_by_span.pop(span, None)

    if not gold_only:
        for span in sorted(pred_by_span.keys()):
            # 与任一 gold span 重叠则跳过，避免相似跨度被当成幻觉
            if any(not (span[1] <= g_start or span[0] >= g_end) for g_start, g_end in gold_spans):
                continue
            pred_types = sorted(pred_by_span[span])
            none_candidate = _add_candidate(span, none_label)
            for label in pred_types:
                hallucination = _add_candidate(span, label)
                # 非 gold 输入：None 作为正，模型预测为负
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
    gold_only: bool = False,
) -> Tuple[List[Dict[str, Any]], PreferencePairs]:
    gold_by_pair: Dict[Tuple[Span, Span], set[str]] = defaultdict(set)
    pred_by_pair: Dict[Tuple[Span, Span], set[str]] = defaultdict(set)
    for head, tail, label in gold_relations:
        gold_by_pair[(head, tail)].add(label)
    for head, tail, label in pred_relations:
        pred_by_pair[(head, tail)].add(label)

    candidates: set[RelationTriplet] = set()
    prefs: PreferencePairs = []

    def _add_candidate(pair: Tuple[Span, Span], label: str) -> RelationTriplet:
        candidate = (pair[0], pair[1], label)
        candidates.add(candidate)
        return candidate

    for pair in sorted(gold_by_pair.keys(), key=lambda p: (p[0][0], p[0][1], p[1][0], p[1][1])):
        gold_types = sorted(gold_by_pair[pair])
        pred_types = sorted(pred_by_pair.get(pair, set()))

        for label in gold_types:
            _add_candidate(pair, label)
        if pred_types:
            for label in gold_types:
                for pred_label in pred_types:
                    if pred_label != label:
                        _add_candidate(pair, pred_label)
                        prefs.append(((pair[0], pair[1], label), (pair[0], pair[1], pred_label)))
            none_candidate = _add_candidate(pair, none_label)
            for label in gold_types:
                if label not in pred_types:
                    prefs.append(((pair[0], pair[1], label), none_candidate))
        else:
            none_candidate = _add_candidate(pair, none_label)
            for label in gold_types:
                prefs.append(((pair[0], pair[1], label), none_candidate))

        pred_by_pair.pop(pair, None)

    if not gold_only:
        for pair in sorted(pred_by_pair.keys(), key=lambda p: (p[0][0], p[0][1], p[1][0], p[1][1])):
            pred_types = sorted(pred_by_pair[pair])
            none_candidate = _add_candidate(pair, none_label)
            for label in pred_types:
                hallucination = _add_candidate(pair, label)
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
    max_entity_prefs: int = 200,
    max_relation_prefs: int = 200,
    entity_none_label: str = DEFAULT_ENTITY_NONE_LABEL,
    relation_none_label: str = DEFAULT_RELATION_NONE_LABEL,
    gold_only_candidates: bool = False,
    entity_bg_ratio: float = 0.0,
    relation_bg_ratio: float = 0.0,
    max_span_size: int = 10,
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
            gold_entities, pred_entities, entity_none_label, gold_only=gold_only_candidates
        )

        gold_relations = _extract_gold_relations(human_doc)
        # 只接受 head/tail 对齐到 gold span 的预测关系
        pred_relations = _extract_pred_relations(model_doc, reference_entities=gold_entities)
        relation_candidates, relation_pref_pairs = _relation_candidates_and_preferences(
            gold_relations, pred_relations, relation_none_label, gold_only=gold_only_candidates
        )

        # Track existing candidates for de-duplication when adding backgrounds
        entity_candidate_map = {
            (cand["span"][0], cand["span"][1], cand["type"]): idx for idx, cand in enumerate(entity_candidates)
        }
        relation_candidate_map = {
            ((cand["h"][0], cand["h"][1]), (cand["t"][0], cand["t"][1]), cand["type"]): idx
            for idx, cand in enumerate(relation_candidates)
        }

        def _ensure_entity_candidate(span: Span, label: str) -> int:
            key = (span[0], span[1], label)
            if key in entity_candidate_map:
                return entity_candidate_map[key]
            idx = len(entity_candidates)
            entity_candidates.append({"span": [span[0], span[1]], "type": label})
            entity_candidate_map[key] = idx
            return idx

        def _ensure_relation_candidate(head_span: Span, tail_span: Span, label: str) -> int:
            key = (head_span, tail_span, label)
            if key in relation_candidate_map:
                return relation_candidate_map[key]
            idx = len(relation_candidates)
            relation_candidates.append(
                {"h": [head_span[0], head_span[1]], "t": [tail_span[0], tail_span[1]], "type": label}
            )
            relation_candidate_map[key] = idx
            return idx

        # --------------------
        # Entity background prefs: None > fake_label
        # --------------------
        if entity_bg_ratio > 0.0:
            base_entity_pref_count = len(entity_pref_pairs)
            target_bg_entities = int(round(entity_bg_ratio * max(base_entity_pref_count, 1)))
            if target_bg_entities > 0:
                bg_spans = sample_background_entity_spans(
                    tokens,
                    gold_entities,
                    pred_entities,
                    max_span_size=max_span_size,
                    max_bg=target_bg_entities,
                )
                observed_labels = {
                    lab for (_, _, lab) in gold_entities + pred_entities if lab != entity_none_label
                }
                observed_labels = sorted(observed_labels)

                for span in bg_spans:
                    if not observed_labels:
                        break
                    fake_label = random.choice(observed_labels)
                    _ensure_entity_candidate(span, entity_none_label)
                    _ensure_entity_candidate(span, fake_label)
                    entity_pref_pairs.append(((span[0], span[1], entity_none_label), (span[0], span[1], fake_label)))

        # --------------------
        # Relation background prefs: None > fake_relation_type
        # --------------------
        if relation_bg_ratio > 0.0:
            base_relation_pref_count = len(relation_pref_pairs)
            target_bg_relations = int(round(relation_bg_ratio * max(base_relation_pref_count, 1)))
            if target_bg_relations > 0:
                entity_spans = [(s, e) for (s, e, _) in gold_entities]
                bg_pairs = sample_background_relation_pairs(
                    entity_spans,
                    gold_relations,
                    pred_relations,
                    max_bg=target_bg_relations,
                )
                observed_rel_labels = {
                    lab for (_, _, lab) in gold_relations + pred_relations if lab != relation_none_label
                }
                observed_rel_labels = sorted(observed_rel_labels)

                for h_span, t_span in bg_pairs:
                    if not observed_rel_labels:
                        break
                    fake_rel = random.choice(observed_rel_labels)
                    _ensure_relation_candidate(h_span, t_span, relation_none_label)
                    _ensure_relation_candidate(h_span, t_span, fake_rel)
                    relation_pref_pairs.append(((h_span, t_span, relation_none_label), (h_span, t_span, fake_rel)))

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
    gold_only_candidates: bool = False,
    entity_bg_ratio: float = 0.0,
    relation_bg_ratio: float = 0.0,
    max_span_size: int = 10,
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
        gold_only_candidates=gold_only_candidates,
        entity_bg_ratio=entity_bg_ratio,
        relation_bg_ratio=relation_bg_ratio,
        max_span_size=max_span_size,
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
        gold_only_candidates=args.gold_only_candidates,
        entity_bg_ratio=args.dpo_entity_bg_ratio,
        relation_bg_ratio=args.dpo_relation_bg_ratio,
        max_span_size=args.max_span_size,
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
