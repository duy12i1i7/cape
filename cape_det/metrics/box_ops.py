from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(x: Any, dtype=np.float32) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=dtype)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def box_area(boxes: np.ndarray) -> np.ndarray:
    boxes = to_numpy(boxes)
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    boxes1 = to_numpy(boxes1)
    boxes2 = to_numpy(boxes2)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[..., 0] * wh[..., 1]
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    return inter / np.maximum(area1[:, None] + area2[None, :] - inter, 1e-6)


def size_bin_mask(boxes: np.ndarray, bin_name: str, bins: dict[str, tuple[float, float | None]]) -> np.ndarray:
    area = box_area(boxes)
    lo, hi = bins[bin_name]
    mask = area >= lo
    if hi is not None:
        mask &= area < hi
    return mask


def ignored_region_mask(ignore: np.ndarray, gt_labels: np.ndarray, pred_label: int) -> np.ndarray:
    """Ignored GT with a negative label suppresses predictions of any class."""

    return ignore & ((gt_labels == pred_label) | (gt_labels < 0))
