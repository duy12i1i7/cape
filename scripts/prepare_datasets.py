#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cape_det.datasets import default_experiment_config, ensure_prepared_dataset
from cape_det.datasets.cli import add_tinyperson_manual_args, apply_tinyperson_manual_paths
from cape_det.utils.config import load_config
from cape_det.utils.io import write_json


def _prepare_one(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config) if args.config else default_experiment_config(dataset, args.model_mode)
    config = load_config(config_path)
    config = apply_tinyperson_manual_paths(config, args)
    return ensure_prepared_dataset(
        dataset,
        config=config,
        raw_root=args.raw_root,
        prepared_root=args.prepared_root,
        allow_download=not args.no_download,
        force_prepare=args.force_prepare,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-download, prepare, and validate CAPE datasets.")
    parser.add_argument("--dataset", required=True, choices=["visdrone", "tinyperson", "both"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-mode", default="cape", choices=["cape", "baseline"])
    parser.add_argument("--raw-root", default=None, help="Raw root override for a single dataset run.")
    parser.add_argument("--prepared-root", default=None, help="Prepared root override for a single dataset run.")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--summary-output", default=None)
    add_tinyperson_manual_args(parser)
    args = parser.parse_args()

    datasets = ["visdrone", "tinyperson"] if args.dataset == "both" else [args.dataset]
    results = {}
    for dataset in datasets:
        result = _prepare_one(dataset, args)
        results[dataset] = {
            "config_path": str(result["config_path"]),
            "prepared_root": str(result.get("prepared_root", "")),
            "summary": result.get("summary", {}),
        }
        print(f"{dataset}: {result['config_path']}")

    if args.summary_output:
        write_json(results, args.summary_output)


if __name__ == "__main__":
    main()
