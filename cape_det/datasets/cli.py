from __future__ import annotations

import argparse
from typing import Any


def add_tinyperson_manual_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-images",
        "--tinyperson-train-images",
        dest="train_images",
        default=None,
        help="TinyPerson manual train image directory.",
    )
    parser.add_argument(
        "--train-json",
        "--tinyperson-train-json",
        dest="train_json",
        default=None,
        help="TinyPerson manual train COCO annotation json.",
    )
    parser.add_argument(
        "--val-images",
        "--tinyperson-val-images",
        dest="val_images",
        default=None,
        help="TinyPerson manual val image directory.",
    )
    parser.add_argument(
        "--val-json",
        "--tinyperson-val-json",
        dest="val_json",
        default=None,
        help="TinyPerson manual val COCO annotation json.",
    )
    parser.add_argument(
        "--test-images",
        "--tinyperson-test-images",
        dest="test_images",
        default=None,
        help="TinyPerson manual test image directory.",
    )
    parser.add_argument(
        "--test-json",
        "--tinyperson-test-json",
        dest="test_json",
        default=None,
        help="TinyPerson manual test COCO annotation json.",
    )


def collect_tinyperson_manual_paths(args: argparse.Namespace) -> dict[str, Any]:
    manual: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        image_dir = getattr(args, f"{split}_images", None)
        annotation_file = getattr(args, f"{split}_json", None)
        if image_dir or annotation_file:
            manual[split] = {}
            if image_dir:
                manual[split]["image_dir"] = image_dir
            if annotation_file:
                manual[split]["annotation_file"] = annotation_file
    return manual


def apply_tinyperson_manual_paths(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    manual = collect_tinyperson_manual_paths(args)
    if manual:
        config.setdefault("dataset", {})["manual_paths"] = manual
    return config
