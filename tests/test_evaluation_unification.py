from cape_det.metrics.sar_metrics import compute_recall_fp
from cape_det.metrics.unified_evaluator import pr_at_iou


def test_ignored_region_suppresses_false_positive_class_agnostically():
    predictions = [
        {
            "boxes": [[10.0, 10.0, 20.0, 20.0]],
            "scores": [0.9],
            "labels": [1],
        }
    ]
    targets = [
        {
            "boxes": [[10.0, 10.0, 20.0, 20.0]],
            "labels": [-1],
            "ignore": [True],
        }
    ]

    stats = compute_recall_fp(predictions, targets, iou_threshold=0.5, score_threshold=0.05)
    precision, recall, scores, total_gt = pr_at_iou(predictions, targets, iou_threshold=0.5)

    assert stats["TotalGT"] == 0.0
    assert stats["FalsePositives"] == 0.0
    assert stats["FP_per_image"] == 0.0
    assert total_gt == 0
    assert precision.size == 0
    assert recall.size == 0
    assert scores.size == 0


def test_ignored_human_class_suppresses_only_matching_class():
    predictions = [
        {
            "boxes": [[10.0, 10.0, 20.0, 20.0], [10.0, 10.0, 20.0, 20.0]],
            "scores": [0.9, 0.8],
            "labels": [0, 1],
        }
    ]
    targets = [
        {
            "boxes": [[10.0, 10.0, 20.0, 20.0]],
            "labels": [0],
            "ignore": [True],
        }
    ]

    stats = compute_recall_fp(predictions, targets, iou_threshold=0.5, score_threshold=0.05)

    assert stats["TotalGT"] == 0.0
    assert stats["FalsePositives"] == 1.0
    assert stats["FP_per_image"] == 1.0
