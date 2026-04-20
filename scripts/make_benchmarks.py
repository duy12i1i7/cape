#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cape_det.datasets import default_experiment_config
from cape_det.metrics.reporting import verify_report_files, write_all_reports
from cape_det.metrics.unified_evaluator import UnifiedEvaluator
from cape_det.utils.config import load_config
from cape_det.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--eval-mode", default=None)
    parser.add_argument("--reports-dir", default="outputs/reports")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--metrics-output", default="outputs/reports/metrics.json")
    parser.add_argument("--metrics-rows-output", default=None)
    parser.add_argument("--export-optional-curves", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else None
    if config_path is None and args.dataset in {"visdrone", "tinyperson"}:
        config_path = default_experiment_config(args.dataset, "cape")
    config = load_config(config_path or "configs/defaults.yaml")
    evaluator = UnifiedEvaluator(
        config,
        args.dataset or config.get("dataset", {}).get("name", "unknown"),
        args.eval_mode or config.get("dataset", {}).get("label_mode", "human_unified_single"),
    )
    evaluator.update(read_json(args.predictions), read_json(args.targets))
    metrics = evaluator.compute()
    write_json(metrics, args.metrics_output)
    if args.metrics_rows_output:
        write_json([metrics], args.metrics_rows_output)
    paths = write_all_reports([metrics], args.reports_dir, args.figures_dir, optional_curves=args.export_optional_curves)
    verify_report_files(args.reports_dir, args.figures_dir)
    paths["metrics_json"] = Path(args.metrics_output)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
