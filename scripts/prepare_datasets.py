#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cape_det.datasets import default_experiment_config, ensure_prepared_dataset
from cape_det.utils.config import load_config
from cape_det.utils.io import write_json


def _manual_paths(args: argparse.Namespace) -> dict[str, Any]:
    manual: dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        image_dir = getattr(args, f"{split}_images")
        annotation_file = getattr(args, f"{split}_json")
        if image_dir or annotation_file:
            manual[split] = {}
            if image_dir:
                manual[split]["image_dir"] = image_dir
            if annotation_file:
                manual[split]["annotation_file"] = annotation_file
    return manual


def _prepare_one(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config) if args.config else default_experiment_config(dataset, args.model_mode)
    config = load_config(config_path)
    manual = _manual_paths(args)
    if manual:
        config.setdefault("dataset", {})["manual_paths"] = manual
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
    parser.add_argument("--train-images", default=None, help="TinyPerson manual train image directory.")
    parser.add_argument("--train-json", default=None, help="TinyPerson manual train COCO json.")
    parser.add_argument("--val-images", default=None, help="TinyPerson manual val image directory.")
    parser.add_argument("--val-json", default=None, help="TinyPerson manual val COCO json.")
    parser.add_argument("--test-images", default=None, help="TinyPerson manual test image directory.")
    parser.add_argument("--test-json", default=None, help="TinyPerson manual test COCO json.")
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
