from __future__ import annotations

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    return inter / (area1[:, None] + area2[None, :] - inter).clamp(min=1e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        idx = order[0]
        keep.append(idx)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[idx].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= threshold]
    return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    keep_all = []
    for label in labels.unique():
        inds = torch.nonzero(labels == label, as_tuple=False).flatten()
        keep_all.append(inds[nms(boxes[inds], scores[inds], threshold)])
    keep = torch.cat(keep_all) if keep_all else torch.empty((0,), dtype=torch.long, device=boxes.device)
    return keep[scores[keep].argsort(descending=True)]
