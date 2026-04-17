from __future__ import annotations

import torch
import torch.nn.functional as F

from cape_det.models.cape.types import CapePredictions
from cape_det.utils.nms import box_iou

from .matching import match_hypotheses


def _human_targets(target: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    boxes = target["boxes"].to(device)
    labels = target["labels"].to(device)
    ignore = target.get("ignore", torch.zeros_like(labels, dtype=torch.bool)).to(device)
    keep = ~ignore
    return boxes[keep], labels[keep]


def _target_scale(target: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    size = target.get("size", target.get("orig_size"))
    if size is None:
        raise KeyError("CAPE normalized losses require target['size'] or target['orig_size']")
    if isinstance(size, torch.Tensor):
        values = size.detach().to(device=device, dtype=dtype).flatten()
        height = values[0]
        width = values[1]
    else:
        height = torch.tensor(float(size[0]), device=device, dtype=dtype)
        width = torch.tensor(float(size[1]), device=device, dtype=dtype)
    return torch.stack([width, height, width, height]).clamp(min=1.0)


def _normalize_boxes(boxes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (boxes / scale).clamp(0.0, 1.0)


def _box_centers_and_sizes(boxes_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centers = (boxes_norm[..., :2] + boxes_norm[..., 2:]) * 0.5
    sizes = (boxes_norm[..., 2:] - boxes_norm[..., :2]).clamp(min=0.0)
    return centers, sizes


def footprint_consistency_loss(footprints: torch.Tensor, evidence_maps: torch.Tensor) -> torch.Tensor:
    """Train rendered CAPE footprints toward detached feature-derived evidence."""

    evidence_target = evidence_maps.sigmoid().detach()
    return F.l1_loss(footprints, evidence_target)


def _gather_compatibility(compatibility: torch.Tensor | None, batch_idx: int, pred_inds: torch.Tensor) -> torch.Tensor | None:
    if compatibility is None:
        return None
    return compatibility[batch_idx, pred_inds].reshape(-1)


def refinement_improvement_targets(cape: CapePredictions, targets: list[dict]) -> torch.Tensor:
    """Detached seed-to-final usefulness target for the CAPE value head."""

    seed_readout = getattr(cape.internals, "seed_readout", None)
    final_readout = getattr(cape.internals, "final_readout", None)
    if seed_readout is None or final_readout is None:
        return torch.zeros_like(cape.scores).detach()

    device = final_readout.scores.device
    value_targets = torch.zeros_like(final_readout.scores)
    seed_compatibility = getattr(cape.internals, "seed_compatibility", None)
    final_compatibility = getattr(cape.internals, "final_compatibility", None)

    for bi, target in enumerate(targets):
        gt_boxes, _ = _human_targets(target, device)
        pred_inds, gt_inds = match_hypotheses(final_readout.boxes[bi], final_readout.scores[bi], gt_boxes)
        if pred_inds.numel() == 0:
            continue

        matched_gt = gt_boxes[gt_inds]
        seed_iou = box_iou(seed_readout.boxes[bi, pred_inds], matched_gt).diag().detach()
        final_iou = box_iou(final_readout.boxes[bi, pred_inds], matched_gt).diag().detach()
        components = [(0.75, (final_iou - seed_iou).clamp(min=0.0))]

        score_gain = final_readout.scores[bi, pred_inds].detach() - seed_readout.scores[bi, pred_inds].detach()
        components.append((0.15, score_gain.clamp(min=0.0)))

        seed_compat = _gather_compatibility(seed_compatibility, bi, pred_inds)
        final_compat = _gather_compatibility(final_compatibility, bi, pred_inds)
        if seed_compat is not None and final_compat is not None:
            components.append((0.10, (final_compat.detach() - seed_compat.detach()).clamp(min=0.0)))

        weight_sum = sum(weight for weight, _ in components)
        improvement = sum(weight * value for weight, value in components) / max(weight_sum, 1e-6)
        value_targets[bi, pred_inds] = improvement.clamp(0.0, 1.0)

    return value_targets.detach()


def cape_hypothesis_loss(cape: CapePredictions, targets: list[dict], weights: dict | None = None) -> dict[str, torch.Tensor]:
    weights = weights or {}
    device = cape.boxes.device
    box_loss = torch.tensor(0.0, device=device)
    center_loss = torch.tensor(0.0, device=device)
    size_loss = torch.tensor(0.0, device=device)
    cls_loss = torch.tensor(0.0, device=device)
    conf_loss = torch.tensor(0.0, device=device)
    matched_count = torch.tensor(0.0, device=device)
    final_readout = cape.internals.final_readout

    for bi, target in enumerate(targets):
        gt_boxes, gt_labels = _human_targets(target, device)
        pred_boxes = final_readout.boxes[bi]
        pred_scores = final_readout.scores[bi]
        pred_inds, gt_inds = match_hypotheses(pred_boxes, pred_scores, gt_boxes)
        conf_target = torch.zeros_like(pred_scores)
        if pred_inds.numel() > 0:
            scale = _target_scale(target, device, pred_boxes.dtype)
            matched_gt = gt_boxes[gt_inds]
            matched_gt_norm = _normalize_boxes(matched_gt, scale)
            gt_centers, gt_sizes = _box_centers_and_sizes(matched_gt_norm)
            matched_pred_norm = final_readout.boxes_norm[bi, pred_inds]
            box_loss = box_loss + F.l1_loss(matched_pred_norm, matched_gt_norm, reduction="sum") / 4.0
            center_loss = center_loss + F.l1_loss(final_readout.centers_norm[bi, pred_inds], gt_centers, reduction="sum") / 2.0
            size_loss = size_loss + F.l1_loss(final_readout.sizes_norm[bi, pred_inds], gt_sizes, reduction="sum") / 2.0
            conf_target[pred_inds] = 1.0
            logits = final_readout.class_logits[bi, pred_inds]
            if logits.shape[-1] == 1:
                cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1),
                    torch.ones_like(logits.squeeze(-1)),
                    reduction="sum",
                )
            else:
                cls_loss = cls_loss + F.cross_entropy(logits, gt_labels[gt_inds].clamp(0, logits.shape[-1] - 1), reduction="sum")
            matched_count = matched_count + pred_inds.numel()
        conf_loss = conf_loss + F.binary_cross_entropy_with_logits(final_readout.conf_logits[bi], conf_target)

    denom = matched_count.clamp(min=1.0)
    box_loss = box_loss / denom
    center_loss = center_loss / denom
    size_loss = size_loss / denom
    cls_loss = cls_loss / denom
    conf_loss = conf_loss / max(len(targets), 1)

    footprint_loss = torch.tensor(0.0, device=device)
    if cape.internals.footprints is not None and cape.internals.evidence_maps is not None:
        footprint_loss = footprint_consistency_loss(cape.internals.footprints, cape.internals.evidence_maps)

    sparsity_loss = torch.tensor(0.0, device=device)
    if weights.get("sparsity", 0.0) > 0:
        for bi in range(cape.boxes.shape[0]):
            ious = box_iou(cape.boxes[bi], cape.boxes[bi])
            off_diag = ious * (1.0 - torch.eye(ious.shape[0], device=device))
            conf_pair = cape.scores[bi][:, None] * cape.scores[bi][None, :]
            sparsity_loss = sparsity_loss + (off_diag * conf_pair).mean()
        sparsity_loss = sparsity_loss / max(cape.boxes.shape[0], 1)

    value_loss = torch.tensor(0.0, device=device)
    if cape.internals.value_logits is not None and weights.get("value", 0.0) > 0:
        value_targets = refinement_improvement_targets(cape, targets)
        value_loss = F.mse_loss(cape.internals.value_logits.sigmoid(), value_targets)

    total = (
        weights.get("cape_box", 5.0) * box_loss
        + weights.get("cape_center", 1.0) * center_loss
        + weights.get("cape_size", 1.0) * size_loss
        + weights.get("cape_cls", 1.0) * cls_loss
        + weights.get("cape_conf", 1.0) * conf_loss
        + weights.get("footprint", 0.2) * footprint_loss
        + weights.get("sparsity", 0.0) * sparsity_loss
        + weights.get("value", 0.0) * value_loss
    )
    return {
        "cape_box_loss": box_loss,
        "cape_center_loss": center_loss,
        "cape_size_loss": size_loss,
        "cape_cls_loss": cls_loss,
        "cape_conf_loss": conf_loss,
        "footprint_loss": footprint_loss,
        "sparsity_loss": sparsity_loss,
        "value_loss": value_loss,
        "cape_loss": total,
    }
