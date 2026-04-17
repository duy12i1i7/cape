#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cape_det.metrics.reporting import FIGURE_FILES, OPTIONAL_CURVE_FILES, plot_required_figures, write_optional_curve_exports
from cape_det.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--export-optional-curves", action="store_true")
    args = parser.parse_args()
    rows = []
    for path in args.metrics:
        payload = read_json(path)
        if isinstance(payload, list):
            rows.extend(payload)
        else:
            rows.append(payload)
    paths = plot_required_figures(rows, args.output_dir)
    if args.export_optional_curves:
        paths.update(write_optional_curve_exports(rows, args.output_dir))
    for name, path in paths.items():
        print(f"{name}: {path}")
    for _, csv_name in FIGURE_FILES.values():
        print(f"csv: {Path(args.output_dir) / csv_name}")
    if args.export_optional_curves:
        for csv_name in OPTIONAL_CURVE_FILES.values():
            print(f"optional_csv: {Path(args.output_dir) / csv_name}")


if __name__ == "__main__":
    main()
