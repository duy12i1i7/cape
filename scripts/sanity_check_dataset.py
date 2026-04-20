#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cape_det.datasets import build_dataset, default_experiment_config, resolve_dataset_config
from cape_det.datasets.cli import add_tinyperson_manual_args, apply_tinyperson_manual_paths
from cape_det.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and summarize a CAPE dataset split.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None, choices=["visdrone", "tinyperson"])
    parser.add_argument("--split", default=None)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    add_tinyperson_manual_args(parser)
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else None
    if config_path is None:
        if args.dataset is None:
            parser.error("Either --config or --dataset is required.")
        config_path = default_experiment_config(args.dataset, "cape")
    config = load_config(config_path)
    config = apply_tinyperson_manual_paths(config, args)
    config = resolve_dataset_config(
        config,
        dataset_name=args.dataset,
        raw_root=args.raw_root,
        prepared_root=args.prepared_root,
        allow_download=not args.no_download,
        force_prepare=args.force_prepare,
    )
    split = args.split or config.get("dataset", {}).get("train_split", "train")
    dataset = build_dataset(config, split=split)
    print(dataset.summarize())


if __name__ == "__main__":
    main()
