from abc import ABC
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


def _sample_negatives(logits: torch.Tensor, positives: torch.Tensor, k: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if k <= 0 or logits.shape[1] <= 1:
        return None, None

    masked_logits = logits.clone()
    masked_logits[torch.arange(masked_logits.shape[0], device=logits.device), positives] = float("-inf")

    k_eff = min(k, logits.shape[1] - 1)
    if k_eff == 0:
        return None, None

    values, indices = torch.topk(masked_logits, k_eff, dim=-1)
    valid_mask = torch.isfinite(values)
    if not torch.any(valid_mask):
        return None, None

    return indices, valid_mask


def _dpo_multi_neg_loss(
    logits_theta: torch.Tensor,
    logits_ref: torch.Tensor,
    y_pos: torch.Tensor,
    y_negs: torch.Tensor,
    beta: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if y_negs is None or y_negs.numel() == 0:
        return torch.zeros(1, device=logits_theta.device, dtype=logits_theta.dtype).squeeze()

    pos_t = logits_theta.gather(1, y_pos.unsqueeze(1))
    pos_r = logits_ref.gather(1, y_pos.unsqueeze(1))

    pos_t = pos_t.expand_as(y_negs)
    pos_r = pos_r.expand_as(y_negs)

    neg_t = logits_theta.gather(1, y_negs)
    neg_r = logits_ref.gather(1, y_negs)

    d = beta * ((pos_t - neg_t) - (pos_r - neg_r))
    loss_matrix = F.softplus(-d)

    if valid_mask is not None:
        loss_matrix = loss_matrix.masked_select(valid_mask)

    if loss_matrix.numel() == 0:
        return torch.zeros(1, device=logits_theta.device, dtype=logits_theta.dtype).squeeze()

    return loss_matrix.mean()


class SpERTLoss(Loss):
    def __init__(
        self,
        rel_criterion,
        entity_criterion,
        model,
        optimizer,
        scheduler,
        max_grad_norm,
        ft_mode: str = "sft",
        dpo_beta: float = 0.1,
        dpo_lambda: float = 0.1,
        dpo_negatives: int = 4,
    ):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self._ft_mode = ft_mode
        self._dpo_beta = dpo_beta
        self._dpo_lambda = dpo_lambda
        self._dpo_negatives = dpo_negatives

    def compute(
        self,
        entity_logits,
        rel_logits,
        entity_types,
        rel_types,
        entity_sample_masks,
        rel_sample_masks,
        ref_entity_logits=None,
        ref_rel_logits=None,
    ):
        # Cache original tensors for DPO before flattening
        entity_logits_raw = entity_logits
        entity_types_raw = entity_types
        entity_masks_raw = entity_sample_masks
        rel_logits_raw = rel_logits
        rel_types_raw = rel_types
        rel_masks_raw = rel_sample_masks

        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        rel_loss = torch.zeros_like(entity_loss)
        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss_raw = self._rel_criterion(rel_logits, rel_types)
            rel_loss_raw = rel_loss_raw.sum(-1) / rel_loss_raw.shape[-1]
            rel_loss = (rel_loss_raw * rel_sample_masks).sum() / rel_count

        train_loss = 1.2 * entity_loss + rel_loss

        if self._ft_mode == "dpo" and ref_entity_logits is not None:
            dpo_loss = self._compute_dpo_loss(
                entity_logits_raw,
                entity_types_raw,
                entity_masks_raw,
                ref_entity_logits,
                rel_logits_raw,
                rel_types_raw,
                rel_masks_raw,
                ref_rel_logits,
            )
            if dpo_loss is not None:
                train_loss = train_loss + self._dpo_lambda * dpo_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()

    def _compute_dpo_loss(
        self,
        entity_logits,
        entity_types,
        entity_masks,
        ref_entity_logits,
        rel_logits,
        rel_types,
        rel_masks,
        ref_rel_logits,
    ) -> Optional[torch.Tensor]:
        losses: list[torch.Tensor] = []

        entity_loss = self._compute_entity_dpo(
            entity_logits, entity_types, entity_masks, ref_entity_logits
        )
        if entity_loss is not None:
            losses.append(entity_loss)

        rel_loss = self._compute_relation_dpo(
            rel_logits, rel_types, rel_masks, ref_rel_logits
        )
        if rel_loss is not None:
            losses.append(rel_loss)

        if not losses:
            return None

        return torch.stack(losses).sum()

    def _compute_entity_dpo(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        mask_flat = masks.view(-1).bool()
        if mask_flat.sum() == 0:
            return None

        logits_flat = logits.view(-1, logits.shape[-1])[mask_flat]
        ref_logits_flat = ref_logits.view(-1, ref_logits.shape[-1])[mask_flat]
        labels_flat = labels.view(-1)[mask_flat]

        negatives, valid_mask = _sample_negatives(logits_flat, labels_flat, self._dpo_negatives)
        if negatives is None:
            return None

        return _dpo_multi_neg_loss(
            logits_flat,
            ref_logits_flat,
            labels_flat,
            negatives,
            self._dpo_beta,
            valid_mask,
        )

    def _compute_relation_dpo(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        ref_logits: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if ref_logits is None or logits is None:
            return None

        if logits.shape[-1] <= 1:
            return None

        mask_flat = masks.view(-1).bool()
        labels_flat = labels.view(-1, labels.shape[-1])
        keep = mask_flat & (labels_flat.sum(-1) > 0)
        if keep.sum() == 0:
            return None

        logits_flat = logits.view(-1, logits.shape[-1])[keep]
        ref_logits_flat = ref_logits.view(-1, ref_logits.shape[-1])[keep]
        pos_mask = labels_flat[keep] > 0.5

        masked_for_neg = logits_flat.masked_fill(pos_mask, float("-inf"))
        k = min(self._dpo_negatives, logits_flat.size(1) - 1)
        if k <= 0:
            return None

        neg_vals, y_negs = torch.topk(masked_for_neg, k, dim=-1)
        valid_neg_mask = torch.isfinite(neg_vals)

        row_idx, pos_idx = torch.where(pos_mask)
        if row_idx.numel() == 0:
            return None

        pos_t = logits_flat[row_idx, pos_idx].unsqueeze(1)
        pos_r = ref_logits_flat[row_idx, pos_idx].unsqueeze(1)

        counts = pos_mask.sum(dim=1).to(torch.long)
        if counts.sum() == 0:
            return None

        rep_idx = torch.repeat_interleave(
            torch.arange(pos_mask.size(0), device=logits_flat.device),
            counts,
        )

        neg_idx_exp = y_negs[rep_idx]
        valid_exp = valid_neg_mask[rep_idx]
        neg_t = logits_flat.gather(1, neg_idx_exp)
        neg_r = ref_logits_flat.gather(1, neg_idx_exp)

        d = self._dpo_beta * ((pos_t - neg_t) - (pos_r - neg_r))
        loss_mat = F.softplus(-d)
        loss_mat = loss_mat.masked_select(valid_exp)
        if loss_mat.numel() == 0:
            return None
        return loss_mat.mean()
