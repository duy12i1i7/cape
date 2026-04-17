from pathlib import Path

from cape_det.metrics.reporting import TABLE1_COLUMNS, TABLE2_COLUMNS, TABLE3_COLUMNS, TABLE4_COLUMNS, write_all_tables


def test_reporting_writes_exact_four_tables(tmp_path: Path):
    metrics = {
        "Dataset": "VisDrone",
        "EvalMode": "baseline",
        "AP50": 0.1,
        "AP50_95": 0.05,
        "AP75": 0.02,
        "Precision": 0.3,
        "Recall": 0.4,
        "F1": 0.34,
        "AR1": 0.1,
        "AR10": 0.2,
        "AR100": 0.3,
        "AP_tiny": 0.01,
        "AP_small": 0.02,
        "AP_medium_plus": 0.03,
        "Recall_tiny": 0.04,
        "Recall_small": 0.05,
        "Params": 1,
        "FLOPs": float("nan"),
        "Latency_ms": 1.0,
        "FPS": 1000.0,
        "Pd": 0.4,
        "MissRate": 0.6,
        "FP_per_image": 1.0,
        "Recall_IoU_0_3": 0.5,
        "Recall_IoU_0_5": 0.4,
        "Energy_per_image": float("nan"),
        "AvgActiveHypotheses": 0,
        "AvgRefinementBudgetUsed": 0,
        "operating_points": {},
    }
    paths = write_all_tables([metrics], tmp_path)
    assert len(paths) == 8
    assert (tmp_path / "table1_unified_detection.csv").read_text().splitlines()[0] == ",".join(TABLE1_COLUMNS)
    assert (tmp_path / "table2_search_and_rescue.csv").read_text().splitlines()[0] == ",".join(TABLE2_COLUMNS)
    assert (tmp_path / "table3_operating_points.csv").read_text().splitlines()[0] == ",".join(TABLE3_COLUMNS)
    assert (tmp_path / "table4_budget_cape_ablation.csv").read_text().splitlines()[0] == ",".join(TABLE4_COLUMNS)
