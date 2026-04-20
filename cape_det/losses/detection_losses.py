from __future__ import annotations

import torch
import torch.nn.functional as F

from cape_det.models.heads.decode import DEFAULT_STRIDES


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    if valid_mask is not None:
        valid = valid_mask.expand_as(loss).to(loss.dtype)
        return (loss * valid).sum() / valid.sum().clamp(min=1.0)
    return loss.mean()


def assign_anchor_free_targets(
    outputs: dict[str, dict[str, torch.Tensor]],
    targets: list[dict],
    num_classes: int,
    strides: dict[str, int] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    strides = strides or DEFAULT_STRIDES
    assigned: dict[str, dict[str, torch.Tensor]] = {}
    for level, out in outputs.items():
        b, _, h, w = out["class_logits"].shape
        device = out["class_logits"].device
        cls_t = torch.zeros((b, num_classes, h, w), device=device)
        obj_t = torch.zeros((b, 1, h, w), device=device)
        box_t = torch.zeros((b, 4, h, w), device=device)
        pos = torch.zeros((b, 1, h, w), dtype=torch.bool, device=device)
        ignore_mask = torch.zeros((b, 1, h, w), dtype=torch.bool, device=device)
        stride = strides[level]
        for bi, target in enumerate(targets):
            boxes = target["boxes"].to(device)
            labels = target["labels"].to(device)
            ignore = target.get("ignore", torch.zeros_like(labels, dtype=torch.bool)).to(device)
            for box, label, ign in zip(boxes, labels, ignore):
                if bool(ign):
                    x1 = int(torch.floor(box[0] / stride).long().clamp(0, w - 1).item())
                    y1 = int(torch.floor(box[1] / stride).long().clamp(0, h - 1).item())
                    x2 = int(torch.ceil(box[2] / stride).long().clamp(0, w - 1).item())
                    y2 = int(torch.ceil(box[3] / stride).long().clamp(0, h - 1).item())
                    ignore_mask[bi, 0, y1 : y2 + 1, x1 : x2 + 1] = True
                    continue
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                area = bw * bh
                if (level == "p3" and area >= 32 * 32) or (level == "p4" and (area < 16 * 16 or area >= 96 * 96)) or (level == "p5" and area < 64 * 64):
                    continue
                cx = ((box[0] + box[2]) * 0.5 / stride).floor().long().clamp(0, w - 1)
                cy = ((box[1] + box[3]) * 0.5 / stride).floor().long().clamp(0, h - 1)
                cls_t[bi, label.clamp(0, num_classes - 1), cy, cx] = 1.0
                obj_t[bi, 0, cy, cx] = 1.0
                center_x = (cx.to(box.dtype) + 0.5) * stride
                center_y = (cy.to(box.dtype) + 0.5) * stride
                box_t[bi, :, cy, cx] = torch.stack(
                    [center_x - box[0], center_y - box[1], box[2] - center_x, box[3] - center_y]
                ).clamp(min=0.0) / stride
                pos[bi, 0, cy, cx] = True
        assigned[level] = {"class": cls_t, "objectness": obj_t, "box": box_t, "positive": pos, "ignore": ignore_mask}
    return assigned


def global_detection_loss(
    outputs: dict[str, dict[str, torch.Tensor]],
    targets: list[dict],
    num_classes: int,
    weights: dict | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or {}
    assigned = assign_anchor_free_targets(outputs, targets, num_classes)
    device = next(iter(outputs.values()))["class_logits"].device
    cls_loss = torch.tensor(0.0, device=device)
    obj_loss = torch.tensor(0.0, device=device)
    box_loss = torch.tensor(0.0, device=device)
    pos_count = torch.tensor(0.0, device=device)
    for level, out in outputs.items():
        tgt = assigned[level]
        valid = (~tgt["ignore"]) | tgt["positive"]
        cls_loss = cls_loss + sigmoid_focal_loss(out["class_logits"], tgt["class"], valid_mask=valid)
        if "objectness" in out:
            obj_per_cell = F.binary_cross_entropy_with_logits(out["objectness"], tgt["objectness"], reduction="none")
            obj_loss = obj_loss + (obj_per_cell * valid.to(obj_per_cell.dtype)).sum() / valid.float().sum().clamp(min=1.0)
        pos = tgt["positive"].expand_as(out["box_reg"])
        if pos.any():
            box_loss = box_loss + F.l1_loss(out["box_reg"][pos], tgt["box"][pos], reduction="sum")
            pos_count = pos_count + pos.float().sum() / 4.0
    box_loss = box_loss / pos_count.clamp(min=1.0)
    total = (
        weights.get("global_cls", 1.0) * cls_loss
        + weights.get("global_box", 5.0) * box_loss
        + weights.get("global_obj", 1.0) * obj_loss
    )
    return {"global_cls_loss": cls_loss, "global_box_loss": box_loss, "global_obj_loss": obj_loss, "global_loss": total}
