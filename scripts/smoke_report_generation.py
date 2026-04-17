#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cape_det.metrics.reporting import (
    TABLE_SPECS,
    table1_rows,
    table2_rows,
    table3_rows,
    table4_rows,
    validate_metrics_rows,
    verify_report_files,
    write_all_reports,
)
from cape_det.metrics.unified_evaluator import UnifiedEvaluator
from cape_det.utils.config import load_config
from cape_det.utils.io import write_json


def tiny_predictions_and_targets() -> tuple[list[dict], list[dict]]:
    predictions = [
        {
            "boxes": [[0, 0, 10, 10], [20, 20, 28, 30], [60, 60, 80, 82]],
            "scores": [0.95, 0.70, 0.20],
            "labels": [0, 0, 0],
        },
        {
            "boxes": [[5, 5, 17, 17], [40, 40, 50, 50]],
            "scores": [0.90, 0.15],
            "labels": [0, 0],
        },
    ]
    targets = [
        {
            "boxes": [[0, 0, 10, 10], [20, 20, 28, 30]],
            "labels": [0, 0],
            "ignore": [False, False],
        },
        {
            "boxes": [[5, 5, 17, 17], [70, 70, 110, 110]],
            "labels": [0, 0],
            "ignore": [False, False],
        },
    ]
    return predictions, targets


def make_eval_config(base_config: dict, mode: str) -> dict:
    config = dict(base_config)
    config["model"] = dict(base_config.get("model", {}))
    config["model"]["mode"] = mode
    config["model"]["cape"] = dict(base_config.get("model", {}).get("cape", {}))
    if mode == "baseline":
        config["model"]["cape"]["enabled"] = False
    else:
        config["model"]["cape"]["enabled"] = True
        config["model"]["cape"]["num_hypotheses"] = 16
        config["model"]["cape"]["max_active_hypotheses"] = 4
        config["model"]["cape"]["max_refinement_steps"] = 1
        config["budget_mode"] = "smoke_cape_budget_4x1"
    return config


def compute_smoke_metrics(config_path: str) -> list[dict]:
    base_config = load_config(config_path)
    predictions, targets = tiny_predictions_and_targets()
    rows = []
    for mode in ["baseline", "cape"]:
        config = make_eval_config(base_config, mode)
        evaluator = UnifiedEvaluator(config, dataset_name="SmokeTiny", eval_mode=mode)
        evaluator.update(predictions, targets)
        rows.append(
            evaluator.compute(
                params=12345 if mode == "baseline" else 23456,
                flops=float("nan"),
                latency_ms=4.2 if mode == "baseline" else 6.8,
                fps=238.095 if mode == "baseline" else 147.059,
                energy_per_image=float("nan"),
                avg_active_hypotheses=0.0 if mode == "baseline" else 4.0,
                avg_refinement_budget_used=0.0 if mode == "baseline" else 4.0,
            )
        )
    validate_metrics_rows(rows)
    return rows


def verify_table_headers(metrics_rows: list[dict]) -> None:
    table_builders = {
        "table1_unified_detection": table1_rows,
        "table2_search_and_rescue": table2_rows,
        "table3_operating_points": table3_rows,
        "table4_budget_cape_ablation": table4_rows,
    }
    for table_name, builder in table_builders.items():
        rows = builder(metrics_rows)
        expected = TABLE_SPECS[table_name]
        for row in rows:
            missing = [col for col in expected if col not in row]
            if missing:
                raise AssertionError(f"{table_name} missing columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/defaults.yaml")
    parser.add_argument("--output-dir", default="outputs/smoke_reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    metrics_rows = compute_smoke_metrics(args.config)
    verify_table_headers(metrics_rows)
    paths = write_all_reports(metrics_rows, reports_dir, figures_dir, optional_curves=True)
    write_json(metrics_rows, output_dir / "metrics_rows.json")
    expected = verify_report_files(reports_dir, figures_dir)
    for name in ["pr_by_size", "miss_rate_vs_fp_per_image", "pr_under_budget"]:
        path = paths[name]
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty optional curve export: {path}")

    print("smoke_report_generation_ok")
    print(f"metrics: {output_dir / 'metrics_rows.json'}")
    for path in expected:
        print(path)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
