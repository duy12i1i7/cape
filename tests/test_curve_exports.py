from pathlib import Path

from cape_det.metrics.reporting import plot_required_figures, write_optional_curve_exports


def test_plot_required_figures_exports_png_and_csv(tmp_path: Path):
    metrics = {
        "Dataset": "VisDrone",
        "EvalMode": "baseline",
        "pr_curve": [{"precision": 1.0, "recall": 0.0, "score": 0.9}, {"precision": 0.5, "recall": 1.0, "score": 0.1}],
        "threshold_sweep": [
            {"threshold": 0.0, "precision": 0.5, "recall": 1.0, "f1": 0.66, "fp_per_image": 1.0},
            {"threshold": 0.5, "precision": 1.0, "recall": 0.5, "f1": 0.66, "fp_per_image": 0.0},
        ],
        "operating_points": {"Threshold_BestF1": 0.5, "Threshold_HighRecall": 0.0, "Threshold_LowFP": 0.5},
    }
    paths = plot_required_figures([metrics], tmp_path)
    assert len(paths) == 3
    assert (tmp_path / "fig1_precision_recall.csv").exists()
    assert (tmp_path / "fig2_recall_vs_fp_per_image.csv").exists()
    assert (tmp_path / "fig3_confidence_threshold.csv").exists()


def test_optional_curve_exports_write_protocol_csvs(tmp_path: Path):
    metrics = {
        "Dataset": "VisDrone",
        "EvalMode": "cape",
        "BudgetMode": "cape_A16_T2",
        "MaxActiveHypotheses": 16,
        "MaxRefinementSteps": 2,
        "AvgActiveHypotheses": 12,
        "AvgRefinementBudgetUsed": 24,
        "pr_curve": [{"precision": 1.0, "recall": 0.5, "score": 0.9}],
        "pr_by_size": {
            "tiny": {"total_gt": 1, "points": [{"precision": 1.0, "recall": 1.0, "score": 0.9}]},
            "small": {"total_gt": 0, "points": []},
            "medium_plus": {"total_gt": 0, "points": []},
        },
        "threshold_sweep": [
            {
                "threshold": 0.5,
                "precision": 1.0,
                "recall": 0.5,
                "f1": 0.66,
                "fp_per_image": 0.0,
                "miss_rate": 0.5,
                "pd": 0.5,
            }
        ],
    }
    paths = write_optional_curve_exports([metrics], tmp_path)

    assert set(paths) == {"pr_by_size", "miss_rate_vs_fp_per_image", "pr_under_budget"}
    assert (tmp_path / "pr_by_size.csv").exists()
    assert (tmp_path / "miss_rate_vs_fp_per_image.csv").exists()
    assert (tmp_path / "pr_under_budget.csv").exists()
