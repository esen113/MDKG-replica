import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from transformers import PreTrainedTokenizer

from spert.entities import Dataset, EntityType, RelationType


class BaseInputReader:
    """Base class that loads entity/relation types and datasets."""

    NONE_RELATION = "None"
    NONE_ENTITY = "None"

    def __init__(
        self,
        types_path: str,
        tokenizer: PreTrainedTokenizer,
        neg_entity_count: int,
        neg_relation_count: int,
        max_span_size: int,
        logger=None,
        max_seq_length: Optional[int] = None,
        use_gold_eval_spans: bool = False,
    ):
        self._types_path = types_path
        self._tokenizer = tokenizer
        self._neg_entity_count = neg_entity_count
        self._neg_relation_count = neg_relation_count
        self._max_span_size = max_span_size
        self._logger = logger
        self._use_gold_eval_spans = use_gold_eval_spans

        self._datasets: Dict[str, Dataset] = OrderedDict()
        self._entity_types: Dict[str, EntityType] = OrderedDict()
        self._relation_types: Dict[str, RelationType] = OrderedDict()
        self._max_context_size = 0

        model_max = getattr(tokenizer, "model_max_length", None)
        self._max_seq_length = max_seq_length or model_max or 512

        self._read_types(types_path)

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #

    @property
    def datasets(self) -> Dict[str, Dataset]:
        return self._datasets

    def get_dataset(self, label: str) -> Dataset:
        return self._datasets[label]

    @property
    def entity_types(self) -> Dict[str, EntityType]:
        return self._entity_types

    def get_entity_type(self, index: int) -> EntityType:
        return list(self._entity_types.values())[index]

    @property
    def relation_types(self) -> Dict[str, RelationType]:
        return self._relation_types

    def get_relation_type(self, index: int) -> RelationType:
        return list(self._relation_types.values())[index]

    @property
    def entity_type_count(self) -> int:
        return len(self._entity_types)

    @property
    def relation_type_count(self) -> int:
        return len(self._relation_types)

    @property
    def context_size(self) -> int:
        return self._max_context_size

    # ------------------------------------------------------------------ #

    def read(self, dataset_paths: Dict[str, str]) -> None:
        for label, path in dataset_paths.items():
            if path is None:
                continue
            dataset_path = Path(path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

            with dataset_path.open("r") as f:
                data = json.load(f)

            dataset = Dataset(
                label=label,
                rel_types=list(self._relation_types.values()),
                entity_types=list(self._entity_types.values()),
                neg_entity_count=self._neg_entity_count,
                neg_rel_count=self._neg_relation_count,
                max_span_size=self._max_span_size,
                use_gold_eval_spans=self._use_gold_eval_spans,
            )
            dataset._input_reader = self  # type: ignore[attr-defined]

            self._parse_dataset(data, dataset)
            self._datasets[label] = dataset

    # ------------------------------------------------------------------ #

    def _read_types(self, types_path: str) -> None:
        types_data = json.loads(Path(types_path).read_text())

        # Entity types
        none_entity = EntityType(
            identifier=self.NONE_ENTITY,
            index=0,
            short_name=self.NONE_ENTITY,
            verbose_name=self.NONE_ENTITY,
        )
        self._entity_types[none_entity.identifier] = none_entity
        for idx, (identifier, meta) in enumerate(types_data.get("entities", {}).items(), start=1):
            short = meta.get("short", identifier)
            verbose = meta.get("verbose", identifier)
            entity_type = EntityType(identifier, idx, short, verbose)
            self._entity_types[identifier] = entity_type

        # Relation types
        none_relation = RelationType(
            identifier=self.NONE_RELATION,
            index=0,
            short_name=self.NONE_RELATION,
            verbose_name=self.NONE_RELATION,
            symmetric=False,
        )
        self._relation_types[none_relation.identifier] = none_relation
        for idx, (identifier, meta) in enumerate(types_data.get("relations", {}).items(), start=1):
            short = meta.get("short", identifier)
            verbose = meta.get("verbose", identifier)
            symmetric = bool(meta.get("symmetric", False))
            relation_type = RelationType(identifier, idx, short, verbose, symmetric=symmetric)
            self._relation_types[identifier] = relation_type

    # ------------------------------------------------------------------ #
    # Hooks implemented in subclasses
    # ------------------------------------------------------------------ #

    def _parse_dataset(self, data: Iterable[dict], dataset: Dataset) -> None:
        raise NotImplementedError()


class JsonInputReader(BaseInputReader):
    """Reads datasets stored as JSON documents."""

    POS_FALLBACK = "NN"
    DEP_FALLBACK = "dep"
    ROOT_DEP = "ROOT"

    def __init__(
        self,
        types_path: str,
        tokenizer: PreTrainedTokenizer,
        neg_entity_count: int = 0,
        neg_relation_count: int = 0,
        max_span_size: int = 10,
        logger=None,
        use_gold_eval_spans: bool = False,
    ):
        super().__init__(
            types_path=types_path,
            tokenizer=tokenizer,
            neg_entity_count=neg_entity_count,
            neg_relation_count=neg_relation_count,
            max_span_size=max_span_size,
            logger=logger,
            use_gold_eval_spans=use_gold_eval_spans,
        )

        self._cls_token_id = tokenizer.cls_token_id
        self._sep_token_id = tokenizer.sep_token_id
        if self._cls_token_id is None or self._sep_token_id is None:
            raise ValueError("Tokenizer must define CLS and SEP token ids.")

    # ------------------------------------------------------------------ #

    def _parse_dataset(self, data: Iterable[dict], dataset: Dataset) -> None:
        for doc in data:
            self._parse_document(doc, dataset)

    # ------------------------------------------------------------------ #

    def _parse_document(self, doc: dict, dataset: Dataset) -> None:
        tokens = doc.get("tokens", [])
        if not tokens:
            return

        pos_tags = self._normalise_sequence(doc, ["pos_tags", "pos", "pos_tag"], len(tokens), self.POS_FALLBACK)
        dep_labels = self._normalise_sequence(doc, ["dep_label", "deplabel", "dependency_labels"], len(tokens), self.DEP_FALLBACK)
        dep_heads = self._normalise_sequence(doc, ["dep_head", "dephead"], len(tokens), 0)
        verb_indicator = self._normalise_sequence(doc, ["verb_indicator", "verb"], len(tokens), 0)

        encoding: List[int] = [self._cls_token_id]
        token_objects = []

        for idx, token_phrase in enumerate(tokens):
            word_pieces = self._tokenizer.tokenize(token_phrase)
            if not word_pieces:
                word_pieces = [self._tokenizer.unk_token]

            span_start = len(encoding)
            for wp in word_pieces:
                encoding.append(self._tokenizer.convert_tokens_to_ids(wp))
            span_end = len(encoding)

            token_objects.append(dataset.create_token(idx, span_start, span_end, token_phrase))

        encoding.append(self._sep_token_id)
        self._max_context_size = max(self._max_context_size, len(encoding))

        entity_mentions = []
        for ent in doc.get("entities", []):
            ent_type_key = ent.get("type")
            if ent_type_key not in self._entity_types:
                raise ValueError(f"Unknown entity type '{ent_type_key}' in document.")

            entity_type = self._entity_types[ent_type_key]
            start = int(ent.get("start"))
            end = int(ent.get("end"))
            if start < 0 or end > len(token_objects) or start >= end:
                raise ValueError(f"Invalid entity span [{start}, {end}) for document.")

            entity_tokens = token_objects[start:end]
            phrase = ent.get("text", " ".join([t.phrase for t in entity_tokens]))
            entity_mentions.append(dataset.create_entity(entity_type, entity_tokens, phrase))

        relations = []
        for rel in doc.get("relations", []):
            rel_type_key = rel.get("type")
            if rel_type_key not in self._relation_types:
                raise ValueError(f"Unknown relation type '{rel_type_key}' in document.")

            head_idx = int(rel.get("head"))
            tail_idx = int(rel.get("tail"))
            if head_idx >= len(entity_mentions) or tail_idx >= len(entity_mentions):
                raise ValueError("Relation references out-of-range entity indices.")

            relation_type = self._relation_types[rel_type_key]
            reverse = bool(rel.get("reverse", False))
            relation = dataset.create_relation(
                relation_type,
                head_entity=entity_mentions[head_idx],
                tail_entity=entity_mentions[tail_idx],
                reverse=reverse,
            )
            relations.append(relation)

        dataset.create_document(
            tokens=token_objects,
            entity_mentions=entity_mentions,
            relations=relations,
            doc_encoding=encoding,
            pos=pos_tags,
            deplabel=dep_labels,
            verb=verb_indicator,
            dephead=[int(h) for h in dep_heads],
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_sequence(doc: dict, keys: Iterable[str], expected_len: int, default_value):
        for key in keys:
            if key in doc:
                seq = doc[key]
                if len(seq) != expected_len:
                    raise ValueError(f"Field '{key}' expected length {expected_len}, got {len(seq)}.")
                return seq
        return [default_value] * expected_len
