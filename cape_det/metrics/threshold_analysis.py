from __future__ import annotations

import numpy as np

from .sar_metrics import compute_recall_fp


def make_threshold_grid(config: dict) -> np.ndarray:
    grid_cfg = config.get("eval", config).get("threshold_grid", {"start": 0.0, "stop": 1.0, "step": 0.01})
    start = float(grid_cfg.get("start", 0.0))
    stop = float(grid_cfg.get("stop", 1.0))
    step = float(grid_cfg.get("step", 0.01))
    return np.round(np.arange(start, stop + step * 0.5, step), 6)


def threshold_sweep(
    predictions: list[dict],
    targets: list[dict],
    thresholds: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for thr in thresholds:
        stats = compute_recall_fp(predictions, targets, iou_threshold=iou_threshold, score_threshold=float(thr))
        tp = stats["DetectedGT"]
        fp = stats["FalsePositives"]
        total_gt = stats["TotalGT"]
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(total_gt, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fp_per_image": float(stats["FP_per_image"]),
                "miss_rate": float(1.0 - recall),
                "pd": float(recall),
            }
        )
    return rows


def operating_points(
    sweep_rows: list[dict[str, float]],
    high_recall_target: float = 0.90,
    low_fp_target: float = 1.0,
) -> dict[str, float]:
    if not sweep_rows:
        return {}
    best_f1 = max(sweep_rows, key=lambda r: (r["f1"], r["recall"], -r["fp_per_image"]))
    high_recall_candidates = [r for r in sweep_rows if r["recall"] >= high_recall_target]
    high_recall = min(high_recall_candidates, key=lambda r: (r["fp_per_image"], -r["threshold"])) if high_recall_candidates else max(sweep_rows, key=lambda r: r["recall"])
    low_fp_candidates = [r for r in sweep_rows if r["fp_per_image"] <= low_fp_target]
    low_fp = max(low_fp_candidates, key=lambda r: (r["recall"], r["precision"])) if low_fp_candidates else min(sweep_rows, key=lambda r: r["fp_per_image"])
    return {
        "Threshold_BestF1": best_f1["threshold"],
        "Precision_BestF1": best_f1["precision"],
        "Recall_BestF1": best_f1["recall"],
        "F1_BestF1": best_f1["f1"],
        "Threshold_HighRecall": high_recall["threshold"],
        "Precision_HighRecall": high_recall["precision"],
        "Recall_HighRecall": high_recall["recall"],
        "FP_per_image_HighRecall": high_recall["fp_per_image"],
        "Threshold_LowFP": low_fp["threshold"],
        "Precision_LowFP": low_fp["precision"],
        "Recall_LowFP": low_fp["recall"],
        "FP_per_image_LowFP": low_fp["fp_per_image"],
    }
