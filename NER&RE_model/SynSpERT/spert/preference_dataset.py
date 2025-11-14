import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset as TorchDataset

from spert import sampling, util
from spert.entities import Dataset as SpERTDataset


class PreferenceDataset(TorchDataset):
    """Dataset that pairs human annotations with model predictions for DPO training."""

    DOC_FORMAT = "doc"
    TRIPLE_FORMAT = "triple"

    def __init__(
        self,
        label: str,
        preference_path: str,
        input_reader,
        neg_entity_count: int,
        neg_relation_count: int,
        max_span_size: int,
        dpo_format: str = "doc",
    ):
        self.label = label
        self._format = dpo_format or self.DOC_FORMAT
        self._neg_entity_count = neg_entity_count
        self._neg_relation_count = neg_relation_count
        self._max_span_size = max_span_size
        self._relation_type_count = input_reader.relation_type_count
        self._input_reader = input_reader

        rel_types = list(input_reader.relation_types.values())
        entity_types = list(input_reader.entity_types.values())

        if self._format == self.DOC_FORMAT:
            self._positive_dataset = SpERTDataset(
                f"{label}_chosen", rel_types, entity_types, neg_entity_count, neg_relation_count, max_span_size
            )
            self._positive_dataset._input_reader = input_reader  # type: ignore[attr-defined]

            self._negative_dataset = SpERTDataset(
                f"{label}_rejected", rel_types, entity_types, neg_entity_count, neg_relation_count, max_span_size
            )
            self._negative_dataset._input_reader = input_reader  # type: ignore[attr-defined]

            self._load_doc_preferences(Path(preference_path))
            self._positive_docs = self._positive_dataset.documents
            self._negative_docs = self._negative_dataset.documents

            self._blueprints: List[Dict[str, Any]] = []
            dpo_max_entities = getattr(input_reader, "dpo_max_entities", 0)
            dpo_max_relations = getattr(input_reader, "dpo_max_relations", 0)
            for chosen_doc, rejected_doc in zip(self._positive_docs, self._negative_docs):
                bp = sampling.build_preference_blueprint(
                    chosen_doc,
                    rejected_doc,
                    max_span_size=self._max_span_size,
                    relation_type_count=self._relation_type_count,
                    max_entities=dpo_max_entities,
                    max_relations=dpo_max_relations,
                )
                self._blueprints.append(bp)

        else:
            self._triple_dataset = SpERTDataset(
                f"{label}_triple", rel_types, entity_types, neg_entity_count, neg_relation_count, max_span_size
            )
            self._triple_dataset._input_reader = input_reader  # type: ignore[attr-defined]
            self._triple_records: List[Dict[str, Any]] = []
            self._load_triple_preferences(Path(preference_path))

    def _load_doc_preferences(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"DPO preference file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                base_doc = record.get("doc")
                if not base_doc:
                    raise ValueError("Preference record missing 'doc' field.")
                rejected_entities = record.get("rejected_entities", [])
                rejected_relations = record.get("rejected_relations", [])

                negative_doc = copy.deepcopy(base_doc)
                negative_doc["entities"] = rejected_entities
                negative_doc["relations"] = rejected_relations

                self._positive_dataset.input_reader._parse_document(  # type: ignore[attr-defined]
                    base_doc, self._positive_dataset
                )
                self._negative_dataset.input_reader._parse_document(  # type: ignore[attr-defined]
                    negative_doc, self._negative_dataset
                )

        if len(self._positive_dataset.documents) != len(self._negative_dataset.documents):
            raise RuntimeError("Preference datasets for chosen/rejected annotations are misaligned.")

    def _load_triple_preferences(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"DPO preference file not found: {path}")

        dpo_max_entities = getattr(self._input_reader, "dpo_max_entities", 0)
        dpo_max_relations = getattr(self._input_reader, "dpo_max_relations", 0)

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                base_doc = record.get("doc")
                if not base_doc:
                    raise ValueError("Triple-format preference missing 'doc'.")
                entity_candidates = record.get("entity_candidates", [])
                relation_candidates = record.get("relation_candidates", [])
                entity_prefs = record.get("entity_preferences", [])
                relation_prefs = record.get("relation_preferences", [])
                if not entity_prefs and not relation_prefs:
                    continue

                self._triple_dataset.input_reader._parse_document(base_doc, self._triple_dataset)  # type: ignore[attr-defined]
                doc_obj = self._triple_dataset.documents[-1]
                blueprint = sampling.build_candidate_blueprint_from_candidates(
                    doc_obj,
                    entity_candidates,
                    relation_candidates,
                    max_span_size=self._max_span_size,
                    relation_type_count=self._relation_type_count,
                    max_entities=dpo_max_entities,
                    max_relations=dpo_max_relations,
                )

                entity_type_ids = self._map_entity_type_ids(
                    entity_candidates, record.get("entity_none_label")
                )
                relation_type_ids = self._map_relation_type_ids(
                    relation_candidates, record.get("relation_none_label")
                )
                relation_pair_indices = self._map_relation_pair_indices(
                    blueprint, entity_candidates, relation_candidates
                )

                entity_pref_pairs = self._tensor_from_preferences(entity_prefs, len(entity_candidates))
                relation_pref_pairs = self._tensor_from_preferences(relation_prefs, len(relation_candidates))

                triple_record = {
                    "blueprint": blueprint,
                    "entity_type_ids": entity_type_ids,
                    "relation_type_ids": relation_type_ids,
                    "relation_pair_indices": relation_pair_indices,
                    "entity_pref_pairs": entity_pref_pairs,
                    "relation_pref_pairs": relation_pref_pairs,
                }
                self._triple_records.append(triple_record)

    def _map_entity_type_ids(self, candidates: Sequence[Dict[str, Any]], none_label: str | None) -> torch.Tensor:
        label = none_label or self._input_reader.NONE_ENTITY
        ids: List[int] = []
        fallback = self._input_reader.entity_types[label].index
        for cand in candidates:
            type_name = cand.get("type", label)
            ent_meta = self._input_reader.entity_types.get(type_name)
            if ent_meta is None:
                ent_meta = self._input_reader.entity_types.get(label)
            if ent_meta is None:
                raise ValueError(f"Unknown entity type '{type_name}' in preference data.")
            ids.append(ent_meta.index)
        if not ids:
            ids = [fallback]
        return torch.tensor(ids, dtype=torch.long)

    def _map_relation_type_ids(self, candidates: Sequence[Dict[str, Any]], none_label: str | None) -> torch.Tensor:
        label = none_label or self._input_reader.NONE_RELATION
        fallback = self._input_reader.relation_types[label].index
        ids: List[int] = []
        for cand in candidates:
            type_name = cand.get("type", label)
            rel_meta = self._input_reader.relation_types.get(type_name)
            if rel_meta is None:
                rel_meta = self._input_reader.relation_types.get(label)
            if rel_meta is None:
                raise ValueError(f"Unknown relation type '{type_name}' in preference data.")
            ids.append(rel_meta.index)
        if not ids:
            ids = [fallback]
        return torch.tensor(ids, dtype=torch.long)

    def _map_relation_pair_indices(
        self,
        blueprint: Dict[str, Any],
        entity_candidates: Sequence[Dict[str, Any]],
        relation_candidates: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        entity_spans = blueprint["entity_spans"]
        if not torch.is_tensor(entity_spans):
            entity_spans = torch.as_tensor(entity_spans, dtype=torch.long)
        span_lookup = {tuple(span.tolist()): idx for idx, span in enumerate(entity_spans)}

        rel_indices = blueprint["rels"]
        rel_masks = blueprint["rel_sample_masks"]
        if not torch.is_tensor(rel_indices):
            rel_indices = torch.as_tensor(rel_indices, dtype=torch.long)
        if not torch.is_tensor(rel_masks):
            rel_masks = torch.as_tensor(rel_masks, dtype=torch.bool)
        pair_lookup = {}
        for idx, mask in enumerate(rel_masks.tolist()):
            if not mask:
                continue
            pair = tuple(rel_indices[idx].tolist())
            pair_lookup[pair] = idx

        indices: List[int] = []
        for cand in relation_candidates:
            h_span = tuple(map(int, cand.get("h", [])))
            t_span = tuple(map(int, cand.get("t", [])))
            head_idx = span_lookup.get(h_span)
            tail_idx = span_lookup.get(t_span)
            if head_idx is None or tail_idx is None:
                indices.append(0)
                continue
            pair_idx = pair_lookup.get((head_idx, tail_idx))
            if pair_idx is None:
                raise ValueError(f"Relation pair {(head_idx, tail_idx)} missing from blueprint.")
            indices.append(pair_idx)

        if not indices:
            indices = [0]
        return torch.tensor(indices, dtype=torch.long)

    @staticmethod
    def _tensor_from_preferences(prefs: Sequence[Dict[str, int]], candidate_count: int) -> torch.Tensor:
        if not prefs:
            return torch.zeros((0, 2), dtype=torch.long)
        pairs = []
        for pref in prefs:
            pos = int(pref.get("pos", -1))
            neg = int(pref.get("neg", -1))
            if pos < 0 or neg < 0:
                continue
            if pos >= candidate_count or neg >= candidate_count:
                continue
            pairs.append((pos, neg))
        if not pairs:
            return torch.zeros((0, 2), dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long)

    def __len__(self) -> int:
        if self._format == self.DOC_FORMAT:
            return len(self._positive_docs)
        return len(self._triple_records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._format == self.DOC_FORMAT:
            blueprint = self._blueprints[idx]
            chosen_doc = self._positive_docs[idx]
            rejected_doc = self._negative_docs[idx]

            pos_ent, pos_rel = sampling.attach_labels_to_blueprint(
                blueprint, chosen_doc, self._input_reader, self._relation_type_count
            )
            neg_ent, neg_rel = sampling.attach_labels_to_blueprint(
                blueprint, rejected_doc, self._input_reader, self._relation_type_count
            )

            def assemble(entity_types, rel_types):
                sample = {k: v for k, v in blueprint.items()}
                sample["entity_types"] = entity_types
                sample["rel_types"] = rel_types
                return sample

            return {"chosen": assemble(pos_ent, pos_rel), "rejected": assemble(neg_ent, neg_rel)}

        triple_record = self._triple_records[idx]
        bp_copy = {k: v for k, v in triple_record["blueprint"].items()}
        return {
            "blueprint": bp_copy,
            "entity_type_ids": triple_record["entity_type_ids"],
            "relation_type_ids": triple_record["relation_type_ids"],
            "relation_pair_indices": triple_record["relation_pair_indices"],
            "entity_pref_pairs": triple_record["entity_pref_pairs"],
            "relation_pref_pairs": triple_record["relation_pref_pairs"],
        }


def _stack_pref_pairs(batch: List[Dict[str, Any]], field: str) -> torch.Tensor:
    pieces = []
    for batch_idx, item in enumerate(batch):
        tensor = item[field]
        if tensor.numel() == 0:
            continue
        batch_column = torch.full((tensor.shape[0], 1), batch_idx, dtype=torch.long)
        pieces.append(torch.cat([batch_column, tensor], dim=1))
    if not pieces:
        return torch.zeros((0, 3), dtype=torch.long)
    return torch.cat(pieces, dim=0)


def preference_collate_fn(batch: List[Dict[str, Any]]) -> Any:
    first = batch[0]
    if "chosen" in first:
        chosen_batch = [item["chosen"] for item in batch]
        rejected_batch = [item["rejected"] for item in batch]
        return sampling.collate_fn_padding(chosen_batch), sampling.collate_fn_padding(rejected_batch)

    blueprints = [item["blueprint"] for item in batch]
    blueprint_batch = sampling.collate_fn_padding(blueprints)
    entity_type_ids = util.padded_stack([item["entity_type_ids"] for item in batch])
    relation_type_ids = util.padded_stack([item["relation_type_ids"] for item in batch])
    relation_pair_indices = util.padded_stack([item["relation_pair_indices"] for item in batch])

    entity_pref_pairs = _stack_pref_pairs(batch, "entity_pref_pairs")
    relation_pref_pairs = _stack_pref_pairs(batch, "relation_pref_pairs")

    return {
        "blueprint": blueprint_batch,
        "entity_type_ids": entity_type_ids,
        "relation_type_ids": relation_type_ids,
        "relation_pair_indices": relation_pair_indices,
        "entity_pref_pairs": entity_pref_pairs,
        "relation_pref_pairs": relation_pref_pairs,
    }
