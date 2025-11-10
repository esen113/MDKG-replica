import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from torch.utils.data import Dataset as TorchDataset

from spert import sampling
from spert.entities import Dataset as SpERTDataset


class PreferenceDataset(TorchDataset):
    """Dataset that pairs human annotations with model predictions for DPO training."""

    def __init__(
        self,
        label: str,
        preference_path: str,
        input_reader,
        neg_entity_count: int,
        neg_relation_count: int,
        max_span_size: int,
    ):
        self.label = label
        self._neg_entity_count = neg_entity_count
        self._neg_relation_count = neg_relation_count
        self._max_span_size = max_span_size
        self._relation_type_count = input_reader.relation_type_count

        rel_types = list(input_reader.relation_types.values())
        entity_types = list(input_reader.entity_types.values())

        self._positive_dataset = SpERTDataset(
            f"{label}_chosen", rel_types, entity_types, neg_entity_count, neg_relation_count, max_span_size
        )
        self._positive_dataset._input_reader = input_reader  # type: ignore[attr-defined]

        self._negative_dataset = SpERTDataset(
            f"{label}_rejected", rel_types, entity_types, neg_entity_count, neg_relation_count, max_span_size
        )
        self._negative_dataset._input_reader = input_reader  # type: ignore[attr-defined]

        self._load_preferences(Path(preference_path))
        self._positive_docs = self._positive_dataset.documents
        self._negative_docs = self._negative_dataset.documents

    def _load_preferences(self, path: Path) -> None:
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

                # Rely on JsonInputReader internals to attach documents.
                self._positive_dataset.input_reader._parse_document(  # type: ignore[attr-defined]
                    base_doc, self._positive_dataset
                )
                self._negative_dataset.input_reader._parse_document(  # type: ignore[attr-defined]
                    negative_doc, self._negative_dataset
                )

        if len(self._positive_dataset.documents) != len(self._negative_dataset.documents):
            raise RuntimeError("Preference datasets for chosen/rejected annotations are misaligned.")

    def __len__(self) -> int:
        return len(self._positive_docs)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        chosen_doc = self._positive_docs[idx]
        rejected_doc = self._negative_docs[idx]

        chosen_sample = sampling.create_train_sample(
            chosen_doc, self._neg_entity_count, self._neg_relation_count, self._max_span_size, self._relation_type_count
        )
        rejected_sample = sampling.create_train_sample(
            rejected_doc,
            self._neg_entity_count,
            self._neg_relation_count,
            self._max_span_size,
            self._relation_type_count,
        )

        return {"chosen": chosen_sample, "rejected": rejected_sample}


def preference_collate_fn(batch: List[Dict[str, Dict[str, Any]]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    chosen_batch = [item["chosen"] for item in batch]
    rejected_batch = [item["rejected"] for item in batch]
    return sampling.collate_fn_padding(chosen_batch), sampling.collate_fn_padding(rejected_batch)
