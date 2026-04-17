#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from cape_det.utils.io import read_json
from cape_det.utils.visualizer import save_boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="outputs/visualizations/prediction.png")
    args = parser.parse_args()
    pred = read_json(args.prediction_json)[args.index]
    image = Image.open(args.image).convert("RGB")
    save_boxes(args.output, image, pred.get("boxes", []), scores=pred.get("scores", []))
    print(args.output)


if __name__ == "__main__":
    main()
