import random

import torch

from spert import util  ##DKS
from spert import constant

import sys

_UNKNOWN_POS_TAGS = set()


def add_syntax_info(doc, context_size: int):
    
    wp_dephead, wp_deplabel = [-1]*context_size, [-1]*context_size
    for idx, j in enumerate(doc.tokens):
      dh = doc.dephead[idx]  #dependency head
      if (dh == 0): #root
          root = idx
      else:
          dh = doc.tokens[doc.dephead[idx]-1].span_start
      wp_dephead[j.span_start] = dh
        
      #print(idx, "ddd", j,  j.span_start, j.span_end)
      ##wp_dephead[j.span_start]= doc.tokens[doc.dephead[idx]-1].span_start
      ##if(doc.dephead[idx]==0):
        ##root = idx     #idx+1
        ##wp_dephead[j.span_start]= 0
      try: 
         wp_deplabel[j.span_start]=constant.DEPREL_TO_ID[doc.deplabel[idx]]
      except KeyError:
         print("### Keyerror for key = ", doc.deplabel[idx])
         wp_deplabel[j.span_start]=constant.DEPREL_TO_ID[constant.UNK_TOKEN]
      for i in range(j.span_start+1,j.span_end):
        wp_dephead[i]= j.span_start  #edge from start subtoken to each remaining subtoken
        wp_deplabel[i]=constant.DEPREL_TO_ID['subtokens'] #or constant.DEPREL_TO_ID[doc.deplabel[idx]]
      
    wp_deplabel[0] = constant.DEPREL_TO_ID['special_rel']
    wp_dephead[0] = root 
    wp_deplabel[-1] = constant.DEPREL_TO_ID['special_rel']
    wp_dephead[-1] = root 

    #pos tags
    wp_pos = [-1]*context_size
    for idx, j in enumerate(doc.tokens):
      #print(idx, "ddd", j,  j.span_start, j.span_end)
      pos_tag = doc.pos[idx]
      pos_id = constant.POS_TO_ID.get(pos_tag)
      if pos_id is None:
        if pos_tag not in _UNKNOWN_POS_TAGS:
          print(f"### Unknown POS tag encountered: {pos_tag}. Using UNK fallback.")
          _UNKNOWN_POS_TAGS.add(pos_tag)
        pos_id = constant.POS_TO_ID[constant.UNK_TOKEN]
      wp_pos[j.span_start]=pos_id
      for i in range(j.span_start+1,j.span_end):
        wp_pos[i]=pos_id
    
    wp_pos[0] = constant.POS_TO_ID['special_token']
    wp_pos[-1] = constant.POS_TO_ID['special_token']

    assert(-1 not in wp_deplabel and -1 not in wp_dephead and -1 not in wp_pos)
    dephead = torch.tensor(wp_dephead)
    deplabel = torch.tensor(wp_deplabel)
    pos = torch.tensor(wp_pos)

    return dephead, deplabel, pos


def create_train_sample(doc, neg_entity_count: int, neg_rel_count: int, max_span_size: int, rel_type_count: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)
    
    dephead, deplabel, pos = add_syntax_info(doc, context_size)  #DKS, TYSS
    

    # positive entities
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = [], [], [], []
    for e in doc.entities:
        #print("############ span = ", e.span)
        pos_entity_spans.append(e.span)
        pos_entity_types.append(e.entity_type.index)
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))
        pos_entity_sizes.append(len(e.tokens))

    # positive relations
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks = [], [], [], []
    for rel in doc.relations:
        #print("############ rel = ", rel)
        s1, s2 = rel.head_entity.span, rel.tail_entity.span
        #print("############ s1 = ", s1, ", s2 = ", s2)
        pos_rels.append((pos_entity_spans.index(s1), pos_entity_spans.index(s2)))
        #print("########### pos_rels = ", pos_rels)
        pos_rel_spans.append((s1, s2))
        pos_rel_types.append(rel.relation_type)
        pos_rel_masks.append(create_rel_mask(s1, s2, context_size))

    # negative entities
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span  #tokens = objs of class Token
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)

    # sample negative entities
    neg_entity_samples = random.sample(list(zip(neg_entity_spans, neg_entity_sizes)),
                                       min(len(neg_entity_spans), neg_entity_count))
    neg_entity_spans, neg_entity_sizes = zip(*neg_entity_samples) if neg_entity_samples else ([], [])

    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    neg_entity_types = [0] * len(neg_entity_spans)  #None type repeated 'len(neg_entity_spans)' times.

    # negative relations
    # use only strong negative relations, i.e. pairs of actual (labeled) entities that are not related
    neg_rel_spans = []

    for i1, s1 in enumerate(pos_entity_spans):
        for i2, s2 in enumerate(pos_entity_spans):
            rev = (s2, s1)
            rev_symmetric = rev in pos_rel_spans and pos_rel_types[pos_rel_spans.index(rev)].symmetric

            # do not add as negative relation sample:
            # neg. relations from an entity to itself
            # entity pairs that are related according to gt
            # entity pairs whose reverse exists as a symmetric relation in gt
            if s1 != s2 and (s1, s2) not in pos_rel_spans and not rev_symmetric:
                neg_rel_spans.append((s1, s2))

    # sample negative relations
    neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))

    neg_rels = [(pos_entity_spans.index(s1), pos_entity_spans.index(s2)) for s1, s2 in neg_rel_spans]
    neg_rel_masks = [create_rel_mask(*spans, context_size) for spans in neg_rel_spans]
    neg_rel_types = [0] * len(neg_rel_spans)

    # merge
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)

    rels = pos_rels + neg_rels
    rel_types = [r.index for r in pos_rel_types] + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)
    assert len(rels) == len(rel_masks) == len(rel_types)

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_types = torch.tensor(rel_types, dtype=torch.long)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1], dtype=torch.long)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    # relation types to one-hot encoding
    rel_types_onehot = torch.zeros([rel_types.shape[0], rel_type_count], dtype=torch.float32)
    rel_types_onehot.scatter_(1, rel_types.unsqueeze(1), 1)
    rel_types_onehot = rel_types_onehot[:, 1:]  # all zeros for 'none' relation

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_types=entity_types,
                rels=rels, rel_masks=rel_masks, rel_types=rel_types_onehot,
                entity_sample_masks=entity_sample_masks, rel_sample_masks=rel_sample_masks,
                dephead= dephead, deplabel=deplabel, pos =pos)


def create_eval_sample(doc, max_span_size: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)
    
    dephead, deplabel, pos = add_syntax_info(doc, context_size)  #DKS, TYSS

    
    # create entity candidates
    entity_spans = []
    entity_masks = []
    entity_sizes = []

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_sizes.append(size)

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # entities
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_spans=entity_spans, entity_sample_masks=entity_sample_masks,
                dephead= dephead, deplabel=deplabel, pos =pos)


def build_candidate_blueprint(doc, input_reader, max_span_size: int, relation_type_count: int):
    """
    Build a deterministic candidate set per document without attaching labels.
    """
    encodings = torch.tensor(doc.encoding, dtype=torch.long)
    context_size = len(doc.encoding)
    context_masks = torch.ones(context_size, dtype=torch.bool)

    dephead, deplabel, pos = add_syntax_info(doc, context_size)

    entity_spans = []
    entity_masks = []
    entity_sizes = []

    token_count = len(doc.tokens)
    for size in range(1, max_span_size + 1):
        limit = token_count - size + 1
        if limit <= 0:
            continue
        for i in range(limit):
            span = doc.tokens[i:i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_sizes.append(size)

    if entity_masks:
        entity_masks_tensor = torch.stack(entity_masks)
        entity_sizes_tensor = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans_tensor = torch.tensor(entity_spans, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks_tensor.shape[0]], dtype=torch.bool)
    else:
        entity_masks_tensor = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes_tensor = torch.zeros([1], dtype=torch.long)
        entity_spans_tensor = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)
        entity_spans = [(0, 0)]

    rels = []
    rel_masks = []
    span_count = len(entity_spans)
    if span_count > 1:
        for i, s1 in enumerate(entity_spans):
            for j, s2 in enumerate(entity_spans):
                if i == j:
                    continue
                rels.append((i, j))
                rel_masks.append(create_rel_mask(s1, s2, context_size))

    if rels:
        rels_tensor = torch.tensor(rels, dtype=torch.long)
        rel_masks_tensor = torch.stack(rel_masks)
        rel_sample_masks = torch.ones([rels_tensor.shape[0]], dtype=torch.bool)
    else:
        rels_tensor = torch.zeros([1, 2], dtype=torch.long)
        rel_masks_tensor = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(
        encodings=encodings,
        context_masks=context_masks,
        entity_masks=entity_masks_tensor,
        entity_sizes=entity_sizes_tensor,
        entity_spans=entity_spans_tensor,
        rels=rels_tensor,
        rel_masks=rel_masks_tensor,
        entity_sample_masks=entity_sample_masks,
        rel_sample_masks=rel_sample_masks,
        dephead=dephead,
        deplabel=deplabel,
        pos=pos,
    )


def _extract_entity_spans(doc):
    if hasattr(doc, "entities"):
        for entity in getattr(doc, "entities", []):
            yield (entity.span_start, entity.span_end)
    else:
        for entity in doc.get("entities", []):
            start = entity.get("start", entity.get("span_start"))
            end = entity.get("end", entity.get("span_end"))
            if start is None or end is None:
                continue
            yield (start, end)


def _extract_relation_spans(doc):
    if hasattr(doc, "relations"):
        for rel in getattr(doc, "relations", []):
            head_span = (rel.head_entity.span_start, rel.head_entity.span_end)
            tail_span = (rel.tail_entity.span_start, rel.tail_entity.span_end)
            yield head_span, tail_span
    else:
        for rel in doc.get("relations", []):
            head_span = (rel.get("head_start"), rel.get("head_end"))
            tail_span = (rel.get("tail_start"), rel.get("tail_end"))
            if head_span[0] is None or head_span[1] is None:
                continue
            if tail_span[0] is None or tail_span[1] is None:
                continue
            yield head_span, tail_span


def build_preference_blueprint(
    chosen_doc,
    rejected_doc,
    max_span_size: int,
    relation_type_count: int,
    max_entities: int = 0,
    max_relations: int = 0,
):
    """
    Construct a compact candidate blueprint for DPO by reusing spans (and relation pairs)
    that appear in either the chosen or rejected annotations.
    """
    # max_span_size and relation_type_count are accepted for API parity with build_candidate_blueprint.
    _ = (max_span_size, relation_type_count)

    encodings = torch.tensor(chosen_doc.encoding, dtype=torch.long)
    context_size = len(chosen_doc.encoding)
    context_masks = torch.ones(context_size, dtype=torch.bool)

    dephead, deplabel, pos = add_syntax_info(chosen_doc, context_size)

    chosen_spans = list(_extract_entity_spans(chosen_doc))
    rejected_spans = list(_extract_entity_spans(rejected_doc))

    span_set = {span: span for span in chosen_spans}
    for span in rejected_spans:
        span_set.setdefault(span, span)

    spans = sorted(span_set.values(), key=lambda s: (s[0], s[1]))
    gt_spans = set(chosen_spans)
    if max_entities > 0 and len(spans) > max_entities:
        kept = [span for span in spans if span in gt_spans]
        if len(kept) >= max_entities:
            spans = kept[:max_entities]
        else:
            extras_needed = max_entities - len(kept)
            extras = [span for span in spans if span not in gt_spans][:extras_needed]
            spans = kept + extras

    if spans:
        entity_masks = [create_entity_mask(span[0], span[1], context_size) for span in spans]
        entity_sizes = [span[1] - span[0] for span in spans]
        entity_masks_tensor = torch.stack(entity_masks)
        entity_sizes_tensor = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans_tensor = torch.tensor(spans, dtype=torch.long)
        entity_sample_masks = torch.ones(len(spans), dtype=torch.bool)
    else:
        entity_masks_tensor = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes_tensor = torch.zeros([1], dtype=torch.long)
        entity_spans_tensor = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)
        spans = [(0, 0)]

    span2idx = {span: idx for idx, span in enumerate(spans)}
    rel_pairs = []
    rel_pair_set = set()

    def _add_relations(doc):
        for head_span, tail_span in _extract_relation_spans(doc):
            if head_span == tail_span:
                continue
            if head_span not in span2idx or tail_span not in span2idx:
                continue
            pair = (span2idx[head_span], span2idx[tail_span])
            if pair in rel_pair_set:
                continue
            rel_pair_set.add(pair)
            rel_pairs.append(pair)

    _add_relations(chosen_doc)
    _add_relations(rejected_doc)
    if max_relations > 0 and len(rel_pairs) > max_relations:
        rel_pairs = rel_pairs[:max_relations]

    if rel_pairs:
        rels_tensor = torch.tensor(rel_pairs, dtype=torch.long)
        rel_masks = [
            create_rel_mask(spans[h_idx], spans[t_idx], context_size) for h_idx, t_idx in rel_pairs
        ]
        rel_masks_tensor = torch.stack(rel_masks)
        rel_sample_masks = torch.ones(len(rel_pairs), dtype=torch.bool)
    else:
        rels_tensor = torch.zeros([1, 2], dtype=torch.long)
        rel_masks_tensor = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(
        encodings=encodings,
        context_masks=context_masks,
        entity_masks=entity_masks_tensor,
        entity_sizes=entity_sizes_tensor,
        entity_spans=entity_spans_tensor,
        rels=rels_tensor,
        rel_masks=rel_masks_tensor,
        entity_sample_masks=entity_sample_masks,
        rel_sample_masks=rel_sample_masks,
        dephead=dephead,
        deplabel=deplabel,
        pos=pos,
    )


def build_candidate_blueprint_from_candidates(
    doc,
    entity_candidates,
    relation_candidates,
    max_span_size: int,
    relation_type_count: int,
    max_entities: int = 0,
    max_relations: int = 0,
):
    """
    Construct a blueprint directly from explicit candidate pools.
    """
    _ = (max_span_size, relation_type_count, max_entities, max_relations)

    encodings = torch.tensor(doc.encoding, dtype=torch.long)
    context_size = len(doc.encoding)
    context_masks = torch.ones(context_size, dtype=torch.bool)
    dephead, deplabel, pos = add_syntax_info(doc, context_size)

    spans = []
    seen_spans = set()
    for cand in entity_candidates:
        span = cand.get("span")
        if not span or len(span) != 2:
            continue
        start, end = int(span[0]), int(span[1])
        tup = (start, end)
        if tup in seen_spans:
            continue
        seen_spans.add(tup)
        spans.append(tup)

    if spans:
        entity_masks = [create_entity_mask(start, end, context_size) for start, end in spans]
        entity_sizes = [end - start for start, end in spans]
        entity_masks_tensor = torch.stack(entity_masks)
        entity_sizes_tensor = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans_tensor = torch.tensor(spans, dtype=torch.long)
        entity_sample_masks = torch.ones(len(spans), dtype=torch.bool)
    else:
        entity_masks_tensor = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes_tensor = torch.zeros([1], dtype=torch.long)
        entity_spans_tensor = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)
        spans = [(0, 0)]

    span2idx = {span: idx for idx, span in enumerate(spans)}
    rel_pairs = []
    rel_masks = []
    rel_sample_masks = []
    seen_pairs = set()

    for cand in relation_candidates:
        h_span_raw = cand.get("h")
        t_span_raw = cand.get("t")
        if not h_span_raw or not t_span_raw:
            continue
        head_span = (int(h_span_raw[0]), int(h_span_raw[1]))
        tail_span = (int(t_span_raw[0]), int(t_span_raw[1]))
        if head_span not in span2idx or tail_span not in span2idx:
            continue
        pair = (span2idx[head_span], span2idx[tail_span])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        rel_pairs.append(pair)
        rel_masks.append(create_rel_mask(head_span, tail_span, context_size))
        rel_sample_masks.append(1)

    if rel_pairs:
        rels_tensor = torch.tensor(rel_pairs, dtype=torch.long)
        rel_masks_tensor = torch.stack(rel_masks)
        rel_sample_masks_tensor = torch.tensor(rel_sample_masks, dtype=torch.bool)
    else:
        rels_tensor = torch.zeros([1, 2], dtype=torch.long)
        rel_masks_tensor = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks_tensor = torch.zeros([1], dtype=torch.bool)

    return dict(
        encodings=encodings,
        context_masks=context_masks,
        entity_masks=entity_masks_tensor,
        entity_sizes=entity_sizes_tensor,
        entity_spans=entity_spans_tensor,
        rels=rels_tensor,
        rel_masks=rel_masks_tensor,
        entity_sample_masks=entity_sample_masks,
        rel_sample_masks=rel_sample_masks_tensor,
        dephead=dephead,
        deplabel=deplabel,
        pos=pos,
    )


def attach_labels_to_blueprint(bp, doc, input_reader, relation_type_count: int):
    """
    Attach entity/relation labels to a pre-computed blueprint.
    """
    entity_spans = bp["entity_spans"]
    if not torch.is_tensor(entity_spans):
        entity_spans = torch.as_tensor(entity_spans, dtype=torch.long)
    rel_indices = bp["rels"]
    if not torch.is_tensor(rel_indices):
        rel_indices = torch.as_tensor(rel_indices, dtype=torch.long)
    num_spans = entity_spans.shape[0]
    entity_types = torch.zeros(num_spans, dtype=torch.long)
    span_lookup = {tuple(span): idx for idx, span in enumerate(entity_spans.tolist())}

    if hasattr(doc, "entities"):
        for entity in getattr(doc, "entities", []):
            idx = span_lookup.get((entity.span_start, entity.span_end))
            if idx is not None:
                entity_types[idx] = entity.entity_type.index
    else:
        for entity in doc.get("entities", []):
            idx = span_lookup.get((entity["start"], entity["end"]))
            if idx is not None:
                ent_type = input_reader.entity_types[entity["type"]].index
                entity_types[idx] = ent_type

    rel_type_dim = max(relation_type_count - 1, 0)
    rel_types = torch.zeros(rel_indices.shape[0], rel_type_dim, dtype=torch.float32)
    if rel_type_dim == 0 or int(bp["entity_sample_masks"].sum()) == 0 or rel_indices.shape[0] == 0:
        return entity_types, rel_types

    pair_lookup = {}
    rel_list = rel_indices.tolist() if rel_indices.numel() else []
    for idx, (head_idx, tail_idx) in enumerate(rel_list):
        pair_lookup[(head_idx, tail_idx)] = idx

    if hasattr(doc, "relations"):
        for rel in getattr(doc, "relations", []):
            head_idx = span_lookup.get((rel.head_entity.span_start, rel.head_entity.span_end))
            tail_idx = span_lookup.get((rel.tail_entity.span_start, rel.tail_entity.span_end))
            if head_idx is None or tail_idx is None:
                continue
            pair_idx = pair_lookup.get((head_idx, tail_idx))
            if pair_idx is None or pair_idx >= rel_types.shape[0]:
                continue
            rel_type_idx = rel.relation_type.index - 1
            if 0 <= rel_type_idx < rel_type_dim:
                rel_types[pair_idx, rel_type_idx] = 1.0
    else:
        for rel in doc.get("relations", []):
            head_idx = span_lookup.get((rel["head_start"], rel["head_end"]))
            tail_idx = span_lookup.get((rel["tail_start"], rel["tail_end"]))
            if head_idx is None or tail_idx is None:
                continue
            pair_idx = pair_lookup.get((head_idx, tail_idx))
            if pair_idx is None or pair_idx >= rel_types.shape[0]:
                continue
            rel_meta = input_reader.relation_types[rel["type"]]
            rel_type_idx = rel_meta.index - 1
            if 0 <= rel_type_idx < rel_type_dim:
                rel_types[pair_idx, rel_type_idx] = 1.0

    return entity_types, rel_types


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
