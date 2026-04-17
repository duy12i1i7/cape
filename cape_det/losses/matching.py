from __future__ import annotations

import numpy as np
import torch

from cape_det.utils.nms import box_iou


def match_hypotheses(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_weight: float = 2.0,
    center_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        device = pred_boxes.device
        return torch.empty((0,), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device)
    ious = box_iou(pred_boxes, gt_boxes)
    pred_center = (pred_boxes[:, :2] + pred_boxes[:, 2:]) * 0.5
    gt_center = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
    norm = (gt_boxes[:, 2:] - gt_boxes[:, :2]).amax(dim=1).clamp(min=1.0)
    center_cost = torch.cdist(pred_center, gt_center) / norm.unsqueeze(0)
    score_cost = -pred_scores[:, None].clamp(1e-4, 1.0).log()
    cost = -iou_weight * ious + center_weight * center_cost + 0.2 * score_cost
    try:
        from scipy.optimize import linear_sum_assignment

        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        return torch.as_tensor(row, dtype=torch.long, device=pred_boxes.device), torch.as_tensor(col, dtype=torch.long, device=pred_boxes.device)
    except Exception:
        remaining_pred = set(range(pred_boxes.shape[0]))
        remaining_gt = set(range(gt_boxes.shape[0]))
        matches: list[tuple[int, int]] = []
        cost_np = cost.detach().cpu().numpy()
        while remaining_pred and remaining_gt:
            best = min(((cost_np[p, g], p, g) for p in remaining_pred for g in remaining_gt), key=lambda x: x[0])
            _, p, g = best
            matches.append((p, g))
            remaining_pred.remove(p)
            remaining_gt.remove(g)
        if not matches:
            device = pred_boxes.device
            return torch.empty((0,), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device)
        row, col = zip(*matches)
        return torch.tensor(row, dtype=torch.long, device=pred_boxes.device), torch.tensor(col, dtype=torch.long, device=pred_boxes.device)
