from __future__ import annotations

import numpy as np

from .box_ops import box_iou_np, ignored_region_mask, to_numpy


def compute_recall_fp(
    predictions: list[dict],
    targets: list[dict],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    max_detections: int | None = None,
) -> dict[str, float]:
    total_gt = 0
    detected_gt = 0
    false_pos = 0
    for pred, tgt in zip(predictions, targets):
        boxes = to_numpy(pred.get("boxes")).reshape(-1, 4)
        scores = to_numpy(pred.get("scores")).reshape(-1)
        labels = to_numpy(pred.get("labels"), dtype=np.int64).reshape(-1) if len(scores) else np.zeros((0,), dtype=np.int64)
        keep = scores >= score_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        order = np.argsort(-scores)
        if max_detections is not None:
            order = order[:max_detections]
        boxes, labels = boxes[order], labels[order]

        gt_boxes = to_numpy(tgt.get("boxes")).reshape(-1, 4)
        gt_labels = to_numpy(tgt.get("labels"), dtype=np.int64).reshape(-1) if gt_boxes.size else np.zeros((0,), dtype=np.int64)
        ignore = to_numpy(tgt.get("ignore"), dtype=bool).reshape(-1) if gt_boxes.size else np.zeros((0,), dtype=bool)
        if gt_boxes.size and ignore.size != gt_boxes.shape[0]:
            ignore = np.zeros((gt_boxes.shape[0],), dtype=bool)
        eval_gt = ~ignore
        total_gt += int(eval_gt.sum())
        matched = np.zeros(gt_boxes.shape[0], dtype=bool)
        for box, label in zip(boxes, labels):
            ignored_candidates = np.where(ignored_region_mask(ignore, gt_labels, int(label)))[0]
            if len(ignored_candidates):
                if box_iou_np(box[None, :], gt_boxes[ignored_candidates]).max(initial=0.0) >= iou_threshold:
                    continue
            candidates = np.where(eval_gt & (gt_labels == label) & ~matched)[0]
            if len(candidates) == 0:
                false_pos += 1
                continue
            ious = box_iou_np(box[None, :], gt_boxes[candidates]).reshape(-1)
            best_pos = int(np.argmax(ious))
            if ious[best_pos] >= iou_threshold:
                matched[candidates[best_pos]] = True
                detected_gt += 1
            else:
                false_pos += 1
    recall = detected_gt / max(total_gt, 1)
    return {
        "Pd": recall,
        "MissRate": 1.0 - recall,
        "FP_per_image": false_pos / max(len(predictions), 1),
        "Recall": recall,
        "TotalGT": float(total_gt),
        "DetectedGT": float(detected_gt),
        "FalsePositives": float(false_pos),
    }


def compute_sar_metrics(
    predictions: list[dict],
    targets: list[dict],
    score_threshold: float = 0.05,
    latency_ms: float = float("nan"),
    fps: float = float("nan"),
    energy_per_image: float = float("nan"),
    avg_active_hypotheses: float = 0.0,
    avg_refinement_budget_used: float = 0.0,
) -> dict[str, float]:
    recall03 = compute_recall_fp(predictions, targets, 0.3, score_threshold)
    recall05 = compute_recall_fp(predictions, targets, 0.5, score_threshold)
    return {
        "Pd": recall03["Pd"],
        "MissRate": recall03["MissRate"],
        "FP_per_image": recall03["FP_per_image"],
        "Recall_IoU_0_3": recall03["Recall"],
        "Recall_IoU_0_5": recall05["Recall"],
        "Latency_ms": latency_ms,
        "FPS": fps,
        "Energy_per_image": energy_per_image,
        "AvgActiveHypotheses": avg_active_hypotheses,
        "AvgRefinementBudgetUsed": avg_refinement_budget_used,
    }
