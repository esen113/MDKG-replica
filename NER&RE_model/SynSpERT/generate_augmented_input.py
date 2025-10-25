import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _load_nlp(model_candidates: Sequence[str]) -> Optional[Tuple[object, object]]:
    """Try loading a spaCy model from the given list of candidates."""
    try:
        import spacy
        from spacy.tokens import Doc
    except ImportError:
        return None

    for name in model_candidates:
        try:
            nlp = spacy.load(name)
            break
        except OSError:
            nlp = None

    if nlp is None:
        return None

    def custom_tokenizer(text: str) -> Doc:
        tokens = text.split(" ")
        # Doc expects a list of words. Setting spaces to True preserves delimiters.
        return Doc(nlp.vocab, words=tokens)

    nlp.tokenizer = custom_tokenizer
    return nlp, Doc


def _simple_pos(token: str) -> str:
    """Heuristic POS tagging compatible with constant.POS_TO_ID."""
    lowered = token.lower()
    if token.startswith("##"):
        return "NN"
    if lowered in {"is", "are", "was", "were", "be", "being", "am"}:
        return "VB"
    if lowered.endswith("ing"):
        return "VBG"
    if lowered.endswith("ed"):
        return "VBD"
    if lowered.endswith("ly"):
        return "RB"
    if lowered in {"and", "or"}:
        return "CC"
    if lowered in {"to"}:
        return "TO"
    if lowered in {"in", "on", "for", "of", "with", "by"}:
        return "IN"
    if token.isdigit():
        return "CD"
    return "NN"


def _fallback_features(tokens: List[str]) -> Dict[str, List]:
    pos_tags = []
    dep_labels = []
    dep_heads = []
    verb_indicator = []

    for idx, tok in enumerate(tokens):
        pos = _simple_pos(tok)
        if tok.startswith("##") and pos_tags:
            pos = pos_tags[-1]

        pos_tags.append(pos)
        dep_labels.append("ROOT" if idx == 0 else "dep")
        dep_heads.append(0 if idx == 0 else idx)
        verb_indicator.append(1 if pos.startswith("VB") else 0)

    return dict(
        pos_tags=pos_tags,
        dep_label=dep_labels,
        dep_head=dep_heads,
        verb_indicator=verb_indicator,
    )


def _spacy_features(nlp_bundle: Tuple[object, object], tokens: List[str]) -> Optional[Dict[str, List]]:
    if nlp_bundle is None:
        return None

    nlp, Doc = nlp_bundle
    text = " ".join(tokens)

    try:
        doc = nlp(text)
    except Exception:
        return None

    pos_tags = []
    dep_labels = []
    dep_heads = []
    verb_indicator = []

    for token in doc:  # type: ignore[attr-defined]
        pos = token.tag_ if token.tag_ else _simple_pos(token.text)
        dep = token.dep_ if token.dep_ else "dep"

        pos_tags.append(pos)
        dep_labels.append(dep)
        if token.head == token:
            dep_heads.append(0)
        else:
            dep_heads.append(token.head.i + 1)
        verb_indicator.append(1 if token.tag_.startswith("VB") else 0)

    if len(pos_tags) != len(tokens):
        # spaCy may collapse tokens despite the custom tokenizer; fallback if so.
        return None

    return dict(
        pos_tags=pos_tags,
        dep_label=dep_labels,
        dep_head=dep_heads,
        verb_indicator=verb_indicator,
    )


def _augment_document(nlp_bundle, doc: Dict) -> Dict:
    tokens = doc["tokens"]
    features = _spacy_features(nlp_bundle, tokens)
    if features is None:
        features = _fallback_features(tokens)

    augmented = dict(doc)
    augmented["pos_tags"] = features["pos_tags"]
    augmented["dep_label"] = features["dep_label"]
    augmented["dep_head"] = features["dep_head"]
    augmented["verb_indicator"] = features["verb_indicator"]
    augmented.setdefault("sents", " ".join(tokens))
    return augmented


def _split_indices(count: int, train_ratio: float, valid_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)

    train_end = int(math.floor(train_ratio * count))
    valid_end = train_end + int(math.floor(valid_ratio * count))

    train_idx = indices[:train_end] or indices[:1]
    valid_idx = indices[train_end:valid_end] or indices[train_end:train_end + 1]
    test_idx = indices[valid_end:] or indices[-1:]

    return train_idx, valid_idx, test_idx


def _select_docs(all_docs: List[Dict], indices: List[int]) -> List[Dict]:
    return [all_docs[i] for i in indices]


def _write_dataset(path: Path, docs: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(docs, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate augmented SynSpERT inputs for the MDKG pipeline.")
    parser.add_argument("--input", required=True, help="Path to the base JSON dataset.")
    parser.add_argument("--output_dir", required=True, help="Directory to store augmented splits.")
    parser.add_argument("--prefix", default="diabetes", help="Filename prefix for generated splits.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of samples for the training split.")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="Proportion of samples for the validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument("--spacy_models", nargs="*", default=("en_core_sci_sm", "en_core_web_sm"),
                        help="spaCy models to try when extracting linguistic features.")
    return parser.parse_args()


def main():
    args = parse_args()

    base_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = json.loads(base_path.read_text())

    nlp_bundle = _load_nlp(args.spacy_models)

    augmented_docs = [_augment_document(nlp_bundle, doc) for doc in docs]

    train_idx, valid_idx, test_idx = _split_indices(len(augmented_docs), args.train_ratio, args.valid_ratio, args.seed)

    splits = {
        f"{args.prefix}_train.json": _select_docs(augmented_docs, train_idx),
        f"{args.prefix}_valid.json": _select_docs(augmented_docs, valid_idx),
        f"{args.prefix}_test.json": _select_docs(augmented_docs, test_idx),
        f"{args.prefix}_all.json": augmented_docs,
    }

    for name, split_docs in splits.items():
        _write_dataset(output_dir / name, split_docs)

    print(f"Generated augmented datasets in {output_dir}")


if __name__ == "__main__":
    main()
