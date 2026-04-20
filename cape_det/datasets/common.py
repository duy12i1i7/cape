from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Annotation:
    box_xyxy: tuple[float, float, float, float]
    category_id: int | str
    category_name: str
    ignore: bool = False
    raw: dict[str, Any] | None = None


@dataclass
class DatasetRecord:
    image_path: Path
    image_id: str
    width: int | None
    height: int | None
    annotations: list[Annotation]
    dataset_name: str


def xywh_to_xyxy(box: list[float] | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in box]
    return x, y, x + w, y + h


def clip_boxes_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = boxes.astype("float32", copy=True)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return boxes


def valid_box_mask(boxes: np.ndarray, min_size: float = 1.0) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)
    return ((boxes[:, 2] - boxes[:, 0]) >= min_size) & ((boxes[:, 3] - boxes[:, 1]) >= min_size)


def target_from_annotations(
    annotations: list[Annotation],
    mapper,
    width: int,
    height: int,
    image_id: str,
    dataset_name: str,
):
    import torch

    boxes: list[tuple[float, float, float, float]] = []
    labels: list[int] = []
    ignore_flags: list[bool] = []
    source_categories: list[str] = []

    for ann in annotations:
        mapped = mapper.map_annotation(dataset_name, ann.category_id, ann.category_name, ann.ignore)
        if mapped is None:
            continue
        label, ignore = mapped
        boxes.append(ann.box_xyxy)
        labels.append(label)
        ignore_flags.append(bool(ignore))
        source_categories.append(str(ann.category_name))

    if boxes:
        box_arr = clip_boxes_xyxy(np.asarray(boxes, dtype="float32"), width, height)
        mask = valid_box_mask(box_arr)
        box_arr = box_arr[mask]
        labels = [label for label, keep in zip(labels, mask.tolist()) if keep]
        ignore_flags = [flag for flag, keep in zip(ignore_flags, mask.tolist()) if keep]
        source_categories = [cat for cat, keep in zip(source_categories, mask.tolist()) if keep]
    else:
        box_arr = np.zeros((0, 4), dtype="float32")

    return {
        "boxes": torch.as_tensor(box_arr, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.long),
        "ignore": torch.as_tensor(ignore_flags, dtype=torch.bool),
        "image_id": image_id,
        "orig_size": (int(height), int(width)),
        "size": (int(height), int(width)),
        "dataset": dataset_name,
        "source_categories": source_categories,
    }


def summarize_records(records: list[DatasetRecord]) -> dict[str, Any]:
    num_annotations = sum(len(r.annotations) for r in records)
    num_empty = sum(1 for r in records if not r.annotations)
    class_hist: dict[str, int] = {}
    ignored = 0
    invalid = 0
    size_bins = {"tiny": 0, "small": 0, "medium_plus": 0}
    for record in records:
        for ann in record.annotations:
            class_hist[ann.category_name] = class_hist.get(ann.category_name, 0) + 1
            ignored += int(ann.ignore)
            x1, y1, x2, y2 = ann.box_xyxy
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h
            invalid += int(w < 1 or h < 1)
            if area < 16 * 16:
                size_bins["tiny"] += 1
            elif area < 32 * 32:
                size_bins["small"] += 1
            else:
                size_bins["medium_plus"] += 1
    return {
        "images": len(records),
        "annotations": num_annotations,
        "empty_images": num_empty,
        "class_histogram": class_hist,
        "ignored_annotations": ignored,
        "invalid_boxes": invalid,
        "size_bins": size_bins,
    }
