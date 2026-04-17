#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cape_det.metrics.latency import benchmark_latency
from cape_det.models import build_model
from cape_det.trainers.checkpoint import load_checkpoint
from cape_det.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    config = load_config(args.config)
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
