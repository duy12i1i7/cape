from __future__ import annotations

import numpy as np

from .box_ops import box_iou_np, ignored_region_mask, size_bin_mask, to_numpy
from .sar_metrics import compute_sar_metrics
from .threshold_analysis import make_threshold_grid, operating_points, threshold_sweep


def _normalized_bins(config: dict) -> dict[str, tuple[float, float | None]]:
    raw = config.get("eval", config).get("size_bins", {})
    if not raw:
        return {"tiny": (0, 256), "small": (256, 1024), "medium_plus": (1024, None)}
    return {k: (float(v[0]), None if v[1] is None else float(v[1])) for k, v in raw.items()}


def _budget_metadata(config: dict) -> dict[str, int | str]:
    model_cfg = config.get("model", {})
    cape_cfg = model_cfg.get("cape", {})
    cape_enabled = bool(cape_cfg.get("enabled", model_cfg.get("mode") == "cape"))
    if not cape_enabled:
        return {
            "BudgetMode": "baseline",
            "MaxActiveHypotheses": 0,
            "MaxRefinementSteps": 0,
        }
    return {
        "BudgetMode": str(config.get("budget_mode", model_cfg.get("mode", "cape"))),
        "MaxActiveHypotheses": int(cape_cfg.get("max_active_hypotheses", 0)),
        "MaxRefinementSteps": int(cape_cfg.get("max_refinement_steps", 0)),
    }


def _prepare_image(pred: dict, tgt: dict, score_threshold: float, max_dets: int | None = None):
    boxes = to_numpy(pred.get("boxes")).reshape(-1, 4)
    scores = to_numpy(pred.get("scores")).reshape(-1)
    labels = to_numpy(pred.get("labels"), dtype=np.int64).reshape(-1) if scores.size else np.zeros((0,), dtype=np.int64)
    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    order = np.argsort(-scores)
    if max_dets is not None:
        order = order[:max_dets]
    boxes, scores, labels = boxes[order], scores[order], labels[order]

    gt_boxes = to_numpy(tgt.get("boxes")).reshape(-1, 4)
    gt_labels = to_numpy(tgt.get("labels"), dtype=np.int64).reshape(-1) if gt_boxes.size else np.zeros((0,), dtype=np.int64)
    ignore = to_numpy(tgt.get("ignore"), dtype=bool).reshape(-1) if gt_boxes.size else np.zeros((0,), dtype=bool)
    if gt_boxes.size and ignore.size != gt_boxes.shape[0]:
        ignore = np.zeros((gt_boxes.shape[0],), dtype=bool)
    return boxes, scores, labels, gt_boxes, gt_labels, ignore


def pr_at_iou(
    predictions: list[dict],
    targets: list[dict],
    iou_threshold: float,
    score_threshold: float = 0.0,
    size_bin: str | None = None,
    bins: dict[str, tuple[float, float | None]] | None = None,
    max_dets: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    all_scores: list[float] = []
    all_tp: list[float] = []
    all_fp: list[float] = []
    total_gt = 0
    bins = bins or {"tiny": (0, 256), "small": (256, 1024), "medium_plus": (1024, None)}
    for pred, tgt in zip(predictions, targets):
        boxes, scores, labels, gt_boxes, gt_labels, ignore = _prepare_image(pred, tgt, score_threshold, max_dets)
        eval_gt = ~ignore
        if size_bin is not None and gt_boxes.size:
            eval_gt &= size_bin_mask(gt_boxes, size_bin, bins)
        total_gt += int(eval_gt.sum())
        matched = np.zeros(gt_boxes.shape[0], dtype=bool)
        for box, score, label in zip(boxes, scores, labels):
            ignored_candidates = np.where(ignored_region_mask(ignore, gt_labels, int(label)))[0]
            if len(ignored_candidates):
                if box_iou_np(box[None, :], gt_boxes[ignored_candidates]).max(initial=0.0) >= iou_threshold:
                    continue
            candidates = np.where(eval_gt & (gt_labels == label) & ~matched)[0]
            all_scores.append(float(score))
            if len(candidates) == 0:
                all_tp.append(0.0)
                all_fp.append(1.0)
                continue
            ious = box_iou_np(box[None, :], gt_boxes[candidates]).reshape(-1)
            best = int(np.argmax(ious))
            if ious[best] >= iou_threshold:
                matched[candidates[best]] = True
                all_tp.append(1.0)
                all_fp.append(0.0)
            else:
                all_tp.append(0.0)
                all_fp.append(1.0)
    if not all_scores:
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), total_gt
    order = np.argsort(-np.asarray(all_scores))
    scores = np.asarray(all_scores)[order]
    tp = np.asarray(all_tp)[order]
    fp = np.asarray(all_fp)[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(total_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return precision, recall, scores, total_gt


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    if precision.size == 0:
        return 0.0
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    recall_points = np.linspace(0, 1, 101)
    values = [mpre[mrec >= r].max(initial=0.0) for r in recall_points]
    return float(np.mean(values))


def recall_at_limit(predictions: list[dict], targets: list[dict], iou_thresholds: list[float], max_dets: int) -> float:
    recalls = []
    for thr in iou_thresholds:
        _, recall, _, total = pr_at_iou(predictions, targets, thr, max_dets=max_dets)
        recalls.append(float(recall[-1]) if recall.size and total > 0 else 0.0)
    return float(np.mean(recalls)) if recalls else 0.0


def curve_points(precision: np.ndarray, recall: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    return [
        {"precision": float(p), "recall": float(r), "score": float(s)}
        for p, r, s in zip(precision.tolist(), recall.tolist(), scores.tolist())
    ]


class UnifiedEvaluator:
    def __init__(self, config: dict, dataset_name: str = "unknown", eval_mode: str = "human_unified_single") -> None:
        self.config = config
        self.dataset_name = dataset_name
        self.eval_mode = eval_mode
        self.predictions: list[dict] = []
        self.targets: list[dict] = []
        self.bins = _normalized_bins(config)

    def update(self, predictions: list[dict], targets: list[dict]) -> None:
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    def compute(
        self,
        params: float = float("nan"),
        flops: float = float("nan"),
        latency_ms: float = float("nan"),
        fps: float = float("nan"),
        energy_per_image: float = float("nan"),
        avg_active_hypotheses: float = 0.0,
        avg_refinement_budget_used: float = 0.0,
    ) -> dict:
        eval_cfg = self.config.get("eval", self.config)
        iou_thresholds = [float(v) for v in eval_cfg.get("iou_thresholds", np.arange(0.5, 1.0, 0.05).round(2).tolist())]
        default_threshold = float(eval_cfg.get("default_threshold", 0.05))

        ap_by_iou = {}
        for thr in iou_thresholds:
            p, r, _, _ = pr_at_iou(self.predictions, self.targets, thr, bins=self.bins)
            ap_by_iou[thr] = average_precision(p, r)
        p50, r50, scores50, _ = pr_at_iou(self.predictions, self.targets, 0.5, bins=self.bins)
        p75, r75, _, _ = pr_at_iou(self.predictions, self.targets, 0.75, bins=self.bins)

        threshold_stats = threshold_sweep(self.predictions, self.targets, np.asarray([default_threshold]), 0.5)[0]
        sweep = threshold_sweep(self.predictions, self.targets, make_threshold_grid(self.config), 0.5)
        op = operating_points(
            sweep,
            high_recall_target=float(eval_cfg.get("high_recall_target", 0.90)),
            low_fp_target=float(eval_cfg.get("low_fp_target", 1.0)),
        )

        size_metrics = {}
        pr_by_size = {}
        for bin_name in ["tiny", "small", "medium_plus"]:
            precision, recall, scores, total_size_gt = pr_at_iou(self.predictions, self.targets, 0.5, size_bin=bin_name, bins=self.bins)
            size_metrics[f"AP_{bin_name}"] = average_precision(precision, recall)
            size_metrics[f"Recall_{bin_name}"] = float(recall[-1]) if recall.size else 0.0
            pr_by_size[bin_name] = {
                "total_gt": int(total_size_gt),
                "points": curve_points(precision, recall, scores),
            }

        sar = compute_sar_metrics(
            self.predictions,
            self.targets,
            default_threshold,
            latency_ms,
            fps,
            energy_per_image,
            avg_active_hypotheses,
            avg_refinement_budget_used,
        )
        metrics = {
            "Dataset": self.dataset_name,
            "EvalMode": self.eval_mode,
            **_budget_metadata(self.config),
            "AP50": ap_by_iou.get(0.5, average_precision(p50, r50)),
            "AP50_95": float(np.mean(list(ap_by_iou.values()))) if ap_by_iou else 0.0,
            "AP75": ap_by_iou.get(0.75, average_precision(p75, r75)),
            "Precision": threshold_stats["precision"],
            "Recall": threshold_stats["recall"],
            "F1": threshold_stats["f1"],
            "AR1": recall_at_limit(self.predictions, self.targets, iou_thresholds, 1),
            "AR10": recall_at_limit(self.predictions, self.targets, iou_thresholds, 10),
            "AR100": recall_at_limit(self.predictions, self.targets, iou_thresholds, 100),
            "AP_tiny": size_metrics["AP_tiny"],
            "AP_small": size_metrics["AP_small"],
            "AP_medium_plus": size_metrics["AP_medium_plus"],
            "Recall_tiny": size_metrics["Recall_tiny"],
            "Recall_small": size_metrics["Recall_small"],
            "Params": params,
            "FLOPs": flops,
            "Latency_ms": latency_ms,
            "FPS": fps,
            **sar,
            "threshold_sweep": sweep,
            "operating_points": op,
            "pr_curve": curve_points(p50, r50, scores50),
            "pr_by_size": pr_by_size,
        }
        return metrics
