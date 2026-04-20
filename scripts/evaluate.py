#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterator
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cape_det.datasets import default_experiment_config, resolve_dataset_config, validate_label_mapper_num_classes
from cape_det.metrics.latency import benchmark_latency
from cape_det.metrics.reporting import write_all_reports
from cape_det.metrics.unified_evaluator import UnifiedEvaluator
from cape_det.utils.config import load_config
from cape_det.utils.io import write_json
from cape_det.utils.profiler import count_parameters, estimate_flops


def tensor_dict_to_jsonable(item: dict) -> dict:
    out = {}
    for key, value in item.items():
        if hasattr(value, "detach"):
            out[key] = value.detach().cpu().tolist()
        else:
            out[key] = value
    return out


def _iter_limited(loader, limit_batches: int | None) -> Iterator[tuple[Any, Any]]:
    for idx, batch in enumerate(loader):
        if limit_batches is not None and idx >= limit_batches:
            break
        yield batch


def evaluate_config(
    config: dict,
    checkpoint: str | Path | None = None,
    split: str | None = None,
    limit_batches: int | None = None,
    measure_latency: bool = False,
    latency_warmup_iters: int | None = None,
    latency_timed_iters: int | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    import torch
    from cape_det.models import build_model
    from cape_det.trainers.build import build_dataloader
    from cape_det.trainers.checkpoint import load_checkpoint
    from cape_det.trainers.loops import targets_to_device

    validate_label_mapper_num_classes(config)
    split = split or config.get("dataset", {}).get("val_split", "val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device).eval()
    if checkpoint:
        load_checkpoint(checkpoint, model, map_location=device)
    loader = build_dataloader(config, split=split, shuffle=False)
    evaluator = UnifiedEvaluator(
        config,
        config.get("dataset", {}).get("name", "unknown"),
        config.get("dataset", {}).get("label_mode", config.get("eval", {}).get("mode", "human_unified_single")),
    )

    all_predictions = []
    all_targets = []
    active = []
    budget = []
    latency_example = None
    with torch.no_grad():
        for images, targets in _iter_limited(loader, limit_batches):
            images = images.to(device)
            if latency_example is None:
                latency_example = images[:1].detach()
            outputs = model(images, targets_to_device(targets, device))
            preds = [{k: v.detach().cpu() for k, v in pred.items()} for pred in outputs["predictions"]]
            evaluator.update(preds, targets)
            all_predictions.extend(tensor_dict_to_jsonable(p) for p in preds)
            all_targets.extend(tensor_dict_to_jsonable(t) for t in targets)
            if outputs.get("cape") is not None:
                active.append(float(outputs["cape"].internals.avg_active_hypotheses.detach().cpu()))
                budget.append(float(outputs["cape"].internals.avg_refinement_budget_used.detach().cpu()))

    latency_stats = {"Latency_ms": float("nan"), "FPS": float("nan")}
    if measure_latency and latency_example is not None:
        bench_cfg = config.get("benchmark", {})
        latency_stats = benchmark_latency(
            model,
            latency_example,
            warmup_iters=int(latency_warmup_iters if latency_warmup_iters is not None else bench_cfg.get("warmup_iters", 10)),
            timed_iters=int(latency_timed_iters if latency_timed_iters is not None else bench_cfg.get("timed_iters", 50)),
        )

    metrics = evaluator.compute(
        params=count_parameters(model),
        flops=estimate_flops(model),
        latency_ms=latency_stats["Latency_ms"],
        fps=latency_stats["FPS"],
        avg_active_hypotheses=sum(active) / max(len(active), 1) if active else 0.0,
        avg_refinement_budget_used=sum(budget) / max(len(budget), 1) if budget else 0.0,
    )
    return metrics, all_predictions, all_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CAPE-Det and emit the required benchmark reports.")
    parser.add_argument("--config", default=None, help="Experiment YAML. Optional when --dataset is provided.")
    parser.add_argument("--dataset", default=None, choices=["visdrone", "tinyperson"], help="Resolve and prepare a dataset by name.")
    parser.add_argument("--model-mode", default="cape", choices=["cape", "baseline"], help="Default experiment family used with --dataset.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--metrics-rows-output", default=None)
    parser.add_argument("--export-optional-curves", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--measure-latency", action="store_true")
    parser.add_argument("--latency-warmup-iters", type=int, default=None)
    parser.add_argument("--latency-timed-iters", type=int, default=None)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    if config_path is None:
        if args.dataset is None:
            parser.error("Either --config or --dataset is required.")
        config_path = default_experiment_config(args.dataset, args.model_mode)

    config = load_config(config_path)
    config = resolve_dataset_config(
        config,
        dataset_name=args.dataset,
        raw_root=args.raw_root,
        prepared_root=args.prepared_root,
        allow_download=not args.no_download,
        force_prepare=args.force_prepare,
    )
    metrics, all_predictions, all_targets = evaluate_config(
        config,
        checkpoint=args.checkpoint,
        split=args.split,
        limit_batches=args.limit_batches,
        measure_latency=args.measure_latency,
        latency_warmup_iters=args.latency_warmup_iters,
        latency_timed_iters=args.latency_timed_iters,
    )
    output_dir = Path(args.output_dir)
    write_json(all_predictions, output_dir / "predictions.json")
    write_json(all_targets, output_dir / "targets.json")
    metrics_output = Path(args.metrics_output) if args.metrics_output else output_dir / "metrics.json"
    metrics_rows_output = Path(args.metrics_rows_output) if args.metrics_rows_output else output_dir / "metrics_rows.json"
    write_json(metrics, metrics_output)
    write_json([metrics], metrics_rows_output)

    root_output_dir = Path(config.get("output_dir", "outputs"))
    reports_dir = Path(args.reports_dir) if args.reports_dir else root_output_dir / "reports"
    figures_dir = Path(args.figures_dir) if args.figures_dir else root_output_dir / "figures"
    paths = write_all_reports([metrics], reports_dir, figures_dir, optional_curves=args.export_optional_curves)
    paths["metrics_json"] = metrics_output
    paths["metrics_rows_json"] = metrics_rows_output
    paths["predictions_json"] = output_dir / "predictions.json"
    paths["targets_json"] = output_dir / "targets.json"
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
