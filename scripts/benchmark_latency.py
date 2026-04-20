#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cape_det.datasets import default_experiment_config, resolve_dataset_config
from cape_det.metrics.latency import benchmark_latency
from cape_det.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None, choices=["visdrone", "tinyperson"])
    parser.add_argument("--model-mode", default="cape", choices=["cape", "baseline"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--no-download", action="store_true")
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
    )
    import torch
    from cape_det.models import build_model
    from cape_det.trainers.checkpoint import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device)
    batch = torch.randn(1, 3, args.height, args.width, device=device)
    bench_cfg = config.get("benchmark", {})
    print(
        benchmark_latency(
            model,
            batch,
            warmup_iters=int(bench_cfg.get("warmup_iters", 10)),
            timed_iters=int(bench_cfg.get("timed_iters", 50)),
        )
    )


if __name__ == "__main__":
    main()
