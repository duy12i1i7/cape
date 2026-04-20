#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cape_det.datasets import default_experiment_config, resolve_dataset_config
from cape_det.datasets.cli import add_tinyperson_manual_args, apply_tinyperson_manual_paths
from cape_det.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CAPE-Det on VisDrone or TinyPerson.")
    parser.add_argument("--config", default=None, help="Experiment YAML. Optional when --dataset is provided.")
    parser.add_argument("--dataset", default=None, choices=["visdrone", "tinyperson"], help="Resolve and prepare a dataset by name.")
    parser.add_argument("--model-mode", default="cape", choices=["cape", "baseline"], help="Default experiment family used with --dataset.")
    parser.add_argument("--raw-root", default=None, help="Manual raw dataset root override.")
    parser.add_argument("--prepared-root", default=None, help="Prepared dataset root override.")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt dataset download when raw data is missing.")
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild the prepared dataset even if validation passes.")
    add_tinyperson_manual_args(parser)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Use tiny training settings for a fast end-to-end check.")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    if config_path is None:
        if args.dataset is None:
            parser.error("Either --config or --dataset is required.")
        config_path = default_experiment_config(args.dataset, args.model_mode)

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
    train_cfg = config.setdefault("train", {})
    dataset_cfg = config.setdefault("dataset", {})
    if args.smoke:
        train_cfg.update({"epochs": 1, "batch_size": 1, "limit_train_batches": 1, "limit_val_batches": 1})
        dataset_cfg.update({"num_workers": 0, "max_size": min(int(dataset_cfg.get("max_size", 1024)), 256)})
        cape_cfg = config.setdefault("model", {}).setdefault("cape", {})
        cape_cfg.update(
            {
                "num_hypotheses": min(int(cape_cfg.get("num_hypotheses", 128)), 16),
                "max_active_hypotheses": min(int(cape_cfg.get("max_active_hypotheses", 64)), 8),
                "max_refinement_steps": min(int(cape_cfg.get("max_refinement_steps", 3)), 1),
            }
        )
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.limit_train_batches is not None:
        train_cfg["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        train_cfg["limit_val_batches"] = args.limit_val_batches
    if args.num_workers is not None:
        dataset_cfg["num_workers"] = args.num_workers
    if args.max_size is not None:
        dataset_cfg["max_size"] = args.max_size
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    from cape_det.trainers.trainer import Trainer

    Trainer(config).fit()


if __name__ == "__main__":
    main()
