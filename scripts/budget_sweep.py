#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cape_det.datasets import default_experiment_config, resolve_dataset_config
from cape_det.metrics.reporting import verify_report_files, write_all_reports
from cape_det.utils.config import deep_merge, load_config
from cape_det.utils.io import write_json


def _sweep_values(config: dict, key: str, override: list[int] | None) -> list[int]:
    if override:
        return [int(value) for value in override]
    values = config.get("budget_sweep", {}).get(key, [])
    if not values:
        raise ValueError(f"Missing budget_sweep.{key}; provide it in config or via CLI")
    return [int(value) for value in values]


def make_budget_config(base_config: dict, max_active: int, max_steps: int) -> dict:
    config = deepcopy(base_config)
    model_cfg = config.setdefault("model", {})
    model_cfg["mode"] = "cape"
    cape_cfg = model_cfg.setdefault("cape", {})
    cape_cfg["enabled"] = True
    cape_cfg["max_active_hypotheses"] = int(max_active)
    cape_cfg["max_refinement_steps"] = int(max_steps)
    cape_cfg["enable_refinement"] = int(max_steps) > 0
    config["budget_mode"] = f"cape_A{int(max_active)}_T{int(max_steps)}"
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CAPE budget settings and populate Table 4.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None, choices=["visdrone", "tinyperson"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-active", nargs="*", type=int, default=None)
    parser.add_argument("--max-steps", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/budget_sweep")
    parser.add_argument("--reports-dir", default="outputs/reports")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--export-optional-curves", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--no-measure-latency", action="store_true")
    parser.add_argument("--latency-warmup-iters", type=int, default=None)
    parser.add_argument("--latency-timed-iters", type=int, default=None)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    if args.config:
        base_config = load_config(args.config)
        if args.dataset:
            dataset_base = load_config(default_experiment_config(args.dataset, "cape"))
            base_config = deep_merge(dataset_base, {"budget_sweep": base_config.get("budget_sweep", {})})
    else:
        dataset_name = args.dataset or "visdrone"
        base_config = load_config(default_experiment_config(dataset_name, "cape"))
        base_config["budget_sweep"] = {
            "max_active_hypotheses": [16, 32, 64, 128],
            "max_refinement_steps": [0, 1, 2, 3],
        }
    base_config = resolve_dataset_config(
        base_config,
        dataset_name=args.dataset,
        raw_root=args.raw_root,
        prepared_root=args.prepared_root,
        allow_download=not args.no_download,
        force_prepare=args.force_prepare,
    )
    active_values = _sweep_values(base_config, "max_active_hypotheses", args.max_active)
    step_values = _sweep_values(base_config, "max_refinement_steps", args.max_steps)

    output_dir = Path(args.output_dir)
    metrics_rows = []
    from scripts.evaluate import evaluate_config

    for max_active in active_values:
        for max_steps in step_values:
            run_config = make_budget_config(base_config, max_active, max_steps)
            budget_name = run_config["budget_mode"]
            metrics, predictions, targets = evaluate_config(
                run_config,
                checkpoint=args.checkpoint,
                split=args.split,
                limit_batches=args.limit_batches,
                measure_latency=not args.no_measure_latency,
                latency_warmup_iters=args.latency_warmup_iters,
                latency_timed_iters=args.latency_timed_iters,
            )
            metrics_rows.append(metrics)
            write_json(metrics, output_dir / f"{budget_name}_metrics.json")
            write_json(predictions, output_dir / f"{budget_name}_predictions.json")
            write_json(targets, output_dir / f"{budget_name}_targets.json")

    metrics_output = Path(args.metrics_output) if args.metrics_output else output_dir / "metrics_rows.json"
    write_json(metrics_rows, metrics_output)
    paths = write_all_reports(
        metrics_rows,
        args.reports_dir,
        args.figures_dir,
        optional_curves=args.export_optional_curves,
    )
    verify_report_files(args.reports_dir, args.figures_dir)
    paths["metrics_rows_json"] = metrics_output
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
