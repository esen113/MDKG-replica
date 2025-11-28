#!/usr/bin/env python3
"""
Convert SynSpERT-style JSON (tokens/entities/relations) to TPLinker+ format.

Output structure per sample:
{
  "id": str | int,
  "text": "tokenized sentence joined by spaces",
  "entity_list": [
      {"text": "...", "type": "disease", "tok_span": [start, end], "char_span": [cs, ce]}
  ],
  "relation_list": [
      {
          "subject": "...",
          "object": "...",
          "predicate": "associated_with",
          "subj_tok_span": [h_start, h_end],
          "obj_tok_span": [t_start, t_end],
          "subj_char_span": [h_cs, h_ce],
          "obj_char_span": [t_cs, t_ce],
      }
  ]
}

Also emits ent2id.json / rel2id.json alongside the converted data.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert SynSpERT JSON to TPLinker+ format.")
    ap.add_argument("--input", required=True, help="Path to SynSpERT JSON (tokens/entities/relations).")
    ap.add_argument("--output", required=True, help="Destination path for TPLinker JSON (or train split).")
    ap.add_argument("--valid-output", help="Optional path to write validation split JSON.")
    ap.add_argument("--test-output", help="Optional path to write test split JSON (uses remaining samples).")
    ap.add_argument("--train-count", type=int, default=None, help="Number of samples for train split.")
    ap.add_argument("--valid-count", type=int, default=None, help="Number of samples for valid split.")
    ap.add_argument("--seed", type=int, default=11, help="Shuffle seed when splitting.")
    ap.add_argument("--ent2id", required=True, help="Path to write ent2id.json.")
    ap.add_argument("--rel2id", required=True, help="Path to write rel2id.json.")
    ap.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length (wordpiece) for downstream config.")
    ap.add_argument("--bert-model", default="bert-base-uncased", help="BERT path (only recorded in metadata).")
    return ap.parse_args()


def build_char_spans(tokens: List[str]) -> List[Tuple[int, int]]:
    spans = []
    cursor = 0
    for tok in tokens:
        start = cursor
        end = start + len(tok)
        spans.append((start, end))
        cursor = end + 1  # add space
    return spans


def convert_sample(doc: Dict, ent2id: Dict[str, int], rel2id: Dict[str, int]) -> Dict:
    tokens = doc.get("tokens", [])
    text = " ".join(tokens)
    char_spans = build_char_spans(tokens)

    entities_out = []
    for ent in doc.get("entities", []):
        start = int(ent.get("start", 0))
        end = int(ent.get("end", start))
        ent_type = ent.get("type", "None")
        cs, ce = char_spans[start][0], char_spans[end - 1][1]
        entities_out.append(
            {
                "text": " ".join(tokens[start:end]),
                "type": ent_type,
                "tok_span": [start, end],
                "char_span": [cs, ce],
            }
        )
        ent2id.setdefault(ent_type, len(ent2id))

    relations_out = []
    ents = doc.get("entities", [])
    for rel in doc.get("relations", []):
        h_idx = int(rel.get("head", -1))
        t_idx = int(rel.get("tail", -1))
        if h_idx < 0 or t_idx < 0 or h_idx >= len(ents) or t_idx >= len(ents):
            continue
        head = ents[h_idx]
        tail = ents[t_idx]
        hs, he = int(head.get("start", 0)), int(head.get("end", 0))
        ts, te = int(tail.get("start", 0)), int(tail.get("end", 0))
        rel_type = rel.get("type", "None")
        rel2id.setdefault(rel_type, len(rel2id))
        subj_text = " ".join(tokens[hs:he])
        obj_text = " ".join(tokens[ts:te])
        relations_out.append(
            {
                "subject": subj_text,
                "object": obj_text,
                "predicate": rel_type,
                "subj_tok_span": [hs, he],
                "obj_tok_span": [ts, te],
                "subj_char_span": [char_spans[hs][0], char_spans[he - 1][1]],
                "obj_char_span": [char_spans[ts][0], char_spans[te - 1][1]],
            }
        )

    return {
        "id": doc.get("orig_id") or doc.get("id"),
        "text": text,
        "entity_list": entities_out,
        "relation_list": relations_out,
    }


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    valid_out = Path(args.valid_output) if args.valid_output else None
    test_out = Path(args.test_output) if args.test_output else None
    ent2id_path = Path(args.ent2id)
    rel2id_path = Path(args.rel2id)

    data = json.loads(inp.read_text())
    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}

    converted = [convert_sample(doc, ent2id, rel2id) for doc in data]

    def _dump(path: Path, samples: List[Dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(samples, ensure_ascii=False, indent=2))

    if valid_out or test_out:
        rng = random.Random(args.seed)
        rng.shuffle(converted)
        train_count = args.train_count or 0
        valid_count = args.valid_count or 0
        if train_count + valid_count > len(converted):
            raise ValueError("train_count + valid_count exceeds dataset size.")
        train_split = converted[:train_count] if train_count else []
        valid_split = converted[train_count : train_count + valid_count] if valid_count else []
        test_split = converted[train_count + valid_count :]
        _dump(out, train_split or converted)
        if valid_out:
            _dump(valid_out, valid_split)
        if test_out:
            _dump(test_out, test_split)
        meta_counts = {
            "train": len(train_split or converted),
            "valid": len(valid_split),
            "test": len(test_split) if test_out else 0,
        }
    else:
        _dump(out, converted)
        meta_counts = {"train": len(converted), "valid": 0, "test": 0}

    ent2id_path.parent.mkdir(parents=True, exist_ok=True)
    rel2id_path.parent.mkdir(parents=True, exist_ok=True)
    ent2id_path.write_text(json.dumps(ent2id, ensure_ascii=False, indent=2))
    rel2id_path.write_text(json.dumps(rel2id, ensure_ascii=False, indent=2))

    meta = {
        "source": str(inp),
        "samples": len(converted),
        "ent_types": len(ent2id),
        "rel_types": len(rel2id),
        "bert_model": args.bert_model,
        "max_seq_len": args.max_seq_len,
        "splits": meta_counts,
    }
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
