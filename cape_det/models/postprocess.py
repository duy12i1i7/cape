from __future__ import annotations

import torch

from cape_det.utils.nms import batched_nms


def postprocess_predictions(
    predictions: list[dict[str, torch.Tensor]],
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> list[dict[str, torch.Tensor]]:
    processed = []
    for pred in predictions:
        boxes, scores, labels = pred["boxes"], pred["scores"], pred["labels"]
        keep = scores >= score_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if boxes.numel() > 0:
            keep_idx = batched_nms(boxes, scores, labels, nms_threshold)
            keep_idx = keep_idx[:max_detections]
            boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]
        processed.append({"boxes": boxes, "scores": scores, "labels": labels})
    return processed


def merge_prediction_lists(
    first: list[dict[str, torch.Tensor]],
    second: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    merged = []
    for a, b in zip(first, second):
        if a["boxes"].numel() == 0:
            merged.append(b)
        elif b["boxes"].numel() == 0:
            merged.append(a)
        else:
            merged.append(
                {
                    "boxes": torch.cat([a["boxes"], b["boxes"]], dim=0),
                    "scores": torch.cat([a["scores"], b["scores"]], dim=0),
                    "labels": torch.cat([a["labels"], b["labels"]], dim=0),
                }
            )
    return merged
