from __future__ import annotations

from pathlib import Path

from cape_det.utils.io import write_csv


def flatten_curve_rows(curves: dict[str, list[dict]], dataset: str, eval_mode: str) -> list[dict]:
    rows = []
    for curve_name, points in curves.items():
        for point in points:
            row = {"Dataset": dataset, "EvalMode": eval_mode, "Curve": curve_name}
            row.update(point)
            rows.append(row)
    return rows


def export_curve_csv(rows: list[dict], path: str | Path) -> None:
    columns = sorted({key for row in rows for key in row.keys()})
    for preferred in ["Dataset", "EvalMode", "Curve", "threshold", "score", "precision", "recall", "f1", "fp_per_image"]:
        if preferred in columns:
            columns.insert(0, columns.pop(columns.index(preferred)))
    write_csv(rows, columns, path)
