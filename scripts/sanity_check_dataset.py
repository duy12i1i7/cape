#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cape_det.datasets import build_dataset
from cape_det.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    split = args.split or config.get("dataset", {}).get("train_split", "train")
    dataset = build_dataset(config, split=split)
    print(dataset.summarize())


if __name__ == "__main__":
    main()
