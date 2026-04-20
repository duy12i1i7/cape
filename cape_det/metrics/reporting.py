from __future__ import annotations

from pathlib import Path
from typing import Any

from cape_det.utils.io import ensure_dir, write_csv, write_markdown_table


TABLE1_COLUMNS = [
    "Dataset",
    "EvalMode",
    "AP50",
    "AP50_95",
    "AP75",
    "Precision",
    "Recall",
    "F1",
    "AR1",
    "AR10",
    "AR100",
    "AP_tiny",
    "AP_small",
    "AP_medium_plus",
    "Recall_tiny",
    "Recall_small",
    "Params",
    "FLOPs",
    "Latency_ms",
    "FPS",
]

TABLE2_COLUMNS = [
    "Dataset",
    "EvalMode",
    "Pd",
    "MissRate",
    "FP_per_image",
    "Recall_IoU_0_3",
    "Recall_IoU_0_5",
    "Latency_ms",
    "FPS",
    "Energy_per_image",
    "AvgActiveHypotheses",
    "AvgRefinementBudgetUsed",
]

TABLE3_COLUMNS = [
    "Dataset",
    "EvalMode",
    "Threshold_BestF1",
    "Precision_BestF1",
    "Recall_BestF1",
    "F1_BestF1",
    "Threshold_HighRecall",
    "Precision_HighRecall",
    "Recall_HighRecall",
    "FP_per_image_HighRecall",
    "Threshold_LowFP",
    "Precision_LowFP",
    "Recall_LowFP",
    "FP_per_image_LowFP",
]

TABLE4_COLUMNS = [
    "Dataset",
    "BudgetMode",
    "MaxActiveHypotheses",
    "MaxRefinementSteps",
    "AvgActiveHypotheses",
    "AvgRefinementBudgetUsed",
    "AP_tiny",
    "Recall_tiny",
    "Pd",
    "FP_per_image",
    "Latency_ms",
    "FPS",
]

TABLE_SPECS = {
    "table1_unified_detection": TABLE1_COLUMNS,
    "table2_search_and_rescue": TABLE2_COLUMNS,
    "table3_operating_points": TABLE3_COLUMNS,
    "table4_budget_cape_ablation": TABLE4_COLUMNS,
}

FIGURE_FILES = {
    "fig1": ("fig1_precision_recall.png", "fig1_precision_recall.csv"),
    "fig2": ("fig2_recall_vs_fp_per_image.png", "fig2_recall_vs_fp_per_image.csv"),
    "fig3": ("fig3_confidence_threshold.png", "fig3_confidence_threshold.csv"),
}

OPTIONAL_CURVE_FILES = {
    "pr_by_size": "pr_by_size.csv",
    "miss_rate_vs_fp_per_image": "miss_rate_vs_fp_per_image.csv",
    "pr_under_budget": "pr_under_budget.csv",
}


def _round_row(row: dict) -> dict:
    rounded = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = f"{value:.6g}"
        else:
            rounded[key] = value
    return rounded


def _format_row(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return _round_row({col: row.get(col, "") for col in columns})


def table1_rows(metrics_rows: list[dict]) -> list[dict]:
    return [_format_row(row, TABLE1_COLUMNS) for row in metrics_rows]


def table2_rows(metrics_rows: list[dict]) -> list[dict]:
    return [_format_row(row, TABLE2_COLUMNS) for row in metrics_rows]


def table3_rows(metrics_rows: list[dict]) -> list[dict]:
    rows = []
    for row in metrics_rows:
        op = row.get("operating_points", {})
        merged = {"Dataset": row.get("Dataset", ""), "EvalMode": row.get("EvalMode", ""), **op}
        rows.append(_format_row(merged, TABLE3_COLUMNS))
    return rows


def table4_rows(metrics_rows: list[dict]) -> list[dict]:
    rows = []
    for row in metrics_rows:
        merged = {
            "Dataset": row.get("Dataset", ""),
            "BudgetMode": row.get("BudgetMode", row.get("EvalMode", "")),
            "MaxActiveHypotheses": row.get("MaxActiveHypotheses", ""),
            "MaxRefinementSteps": row.get("MaxRefinementSteps", ""),
            "AvgActiveHypotheses": row.get("AvgActiveHypotheses", ""),
            "AvgRefinementBudgetUsed": row.get("AvgRefinementBudgetUsed", ""),
            "AP_tiny": row.get("AP_tiny", ""),
            "Recall_tiny": row.get("Recall_tiny", ""),
            "Pd": row.get("Pd", ""),
            "FP_per_image": row.get("FP_per_image", ""),
            "Latency_ms": row.get("Latency_ms", ""),
            "FPS": row.get("FPS", ""),
        }
        rows.append(_format_row(merged, TABLE4_COLUMNS))
    return rows


def build_all_table_rows(metrics_rows: list[dict]) -> dict[str, tuple[list[dict], list[str]]]:
    return {
        "table1_unified_detection": (table1_rows(metrics_rows), TABLE1_COLUMNS),
        "table2_search_and_rescue": (table2_rows(metrics_rows), TABLE2_COLUMNS),
        "table3_operating_points": (table3_rows(metrics_rows), TABLE3_COLUMNS),
        "table4_budget_cape_ablation": (table4_rows(metrics_rows), TABLE4_COLUMNS),
    }


def validate_table_rows(table_rows: dict[str, tuple[list[dict], list[str]]]) -> None:
    for table_name, (rows, columns) in table_rows.items():
        expected = TABLE_SPECS[table_name]
        if columns != expected:
            raise ValueError(f"{table_name} columns do not match the protocol")
        for row_idx, row in enumerate(rows):
            missing = [col for col in expected if col not in row]
            if missing:
                raise ValueError(f"{table_name} row {row_idx} missing columns: {missing}")


def validate_metrics_rows(metrics_rows: list[dict]) -> None:
    table_rows = build_all_table_rows(metrics_rows)
    validate_table_rows(table_rows)
    for row_idx, row in enumerate(metrics_rows):
        for col in sorted(set(TABLE1_COLUMNS + TABLE2_COLUMNS + TABLE4_COLUMNS)):
            if col not in row:
                raise ValueError(f"metrics row {row_idx} missing required table column source: {col}")
        if "threshold_sweep" not in row:
            raise ValueError(f"metrics row {row_idx} missing threshold_sweep for Figure 2/Figure 3 CSV export")
        if "pr_curve" not in row:
            raise ValueError(f"metrics row {row_idx} missing pr_curve for Figure 1 CSV export")
        if "operating_points" not in row:
            raise ValueError(f"metrics row {row_idx} missing operating_points for Table 3 and Figure 3 markers")
        operating_points = row.get("operating_points", {})
        for col in TABLE3_COLUMNS:
            if col in {"Dataset", "EvalMode"}:
                continue
            if col not in operating_points:
                raise ValueError(f"metrics row {row_idx} missing operating point source: {col}")


def write_all_tables(metrics_rows: list[dict], output_dir: str | Path = "outputs/reports") -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    validate_metrics_rows(metrics_rows)
    specs = build_all_table_rows(metrics_rows)
    validate_table_rows(specs)
    paths: dict[str, Path] = {}
    for name, (rows, columns) in specs.items():
        csv_path = output_dir / f"{name}.csv"
        md_path = output_dir / f"{name}.md"
        write_csv(rows, columns, csv_path)
        write_markdown_table(rows, columns, md_path)
        paths[f"{name}_csv"] = csv_path
        paths[f"{name}_md"] = md_path
    return paths


def _operating_point_name(row: dict, threshold: float) -> str:
    op = row.get("operating_points", {})
    names = []
    for name, key in [
        ("best_f1", "Threshold_BestF1"),
        ("high_recall", "Threshold_HighRecall"),
        ("low_fp", "Threshold_LowFP"),
    ]:
        value = op.get(key)
        if value is not None and abs(float(value) - float(threshold)) <= 1e-9:
            names.append(name)
    return "+".join(names)


def plot_required_figures(metrics_rows: list[dict], output_dir: str | Path = "outputs/figures") -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validate_metrics_rows(metrics_rows)
    output_dir = ensure_dir(output_dir)
    paths: dict[str, Path] = {}

    pr_rows = []
    plt.figure(figsize=(8, 5))
    for row in metrics_rows:
        label = f"{row.get('Dataset')} {row.get('EvalMode')}"
        curve = row.get("pr_curve", [])
        if not curve:
            continue
        recalls = [p["recall"] for p in curve]
        precisions = [p["precision"] for p in curve]
        plt.plot(recalls, precisions, label=label)
        for p in curve:
            pr_rows.append(
                {
                    "Dataset": row.get("Dataset"),
                    "EvalMode": row.get("EvalMode"),
                    "precision": p.get("precision", ""),
                    "recall": p.get("recall", ""),
                    "score": p.get("score", ""),
                }
            )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.grid(True, alpha=0.3)
    if pr_rows:
        plt.legend()
    fig1 = output_dir / FIGURE_FILES["fig1"][0]
    plt.tight_layout()
    plt.savefig(fig1, dpi=160)
    plt.close()
    write_csv(pr_rows, ["Dataset", "EvalMode", "precision", "recall", "score"], output_dir / FIGURE_FILES["fig1"][1])
    paths["fig1"] = fig1

    fp_rows = []
    plt.figure(figsize=(8, 5))
    for row in metrics_rows:
        sweep = row.get("threshold_sweep", [])
        if not sweep:
            continue
        label = f"{row.get('Dataset')} {row.get('EvalMode')}"
        fp = [p["fp_per_image"] for p in sweep]
        recall = [p["recall"] for p in sweep]
        plt.plot(fp, recall, label=label)
        for p in sweep:
            fp_rows.append(
                {
                    "Dataset": row.get("Dataset"),
                    "EvalMode": row.get("EvalMode"),
                    "threshold": p.get("threshold", ""),
                    "recall": p.get("recall", ""),
                    "fp_per_image": p.get("fp_per_image", ""),
                    "precision": p.get("precision", ""),
                    "f1": p.get("f1", ""),
                    "miss_rate": p.get("miss_rate", ""),
                    "pd": p.get("pd", ""),
                }
            )
    plt.xlabel("FP/image")
    plt.ylabel("Recall")
    plt.title("Recall vs FP/image")
    plt.grid(True, alpha=0.3)
    if fp_rows:
        plt.legend()
    fig2 = output_dir / FIGURE_FILES["fig2"][0]
    plt.tight_layout()
    plt.savefig(fig2, dpi=160)
    plt.close()
    write_csv(
        fp_rows,
        ["Dataset", "EvalMode", "threshold", "recall", "fp_per_image", "precision", "f1", "miss_rate", "pd"],
        output_dir / FIGURE_FILES["fig2"][1],
    )
    paths["fig2"] = fig2

    conf_rows = []
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for row in metrics_rows:
        sweep = row.get("threshold_sweep", [])
        if not sweep:
            continue
        label = f"{row.get('Dataset')} {row.get('EvalMode')}"
        thr = [p["threshold"] for p in sweep]
        axes[0].plot(thr, [p["precision"] for p in sweep], label=label)
        axes[1].plot(thr, [p["recall"] for p in sweep], label=label)
        axes[2].plot(thr, [p["f1"] for p in sweep], label=label)
        op = row.get("operating_points", {})
        for ax in axes:
            for key in ["Threshold_BestF1", "Threshold_HighRecall", "Threshold_LowFP"]:
                if key in op:
                    ax.axvline(op[key], color="black", alpha=0.12, linewidth=1)
        for p in sweep:
            threshold = p.get("threshold", "")
            conf_rows.append(
                {
                    "Dataset": row.get("Dataset"),
                    "EvalMode": row.get("EvalMode"),
                    "threshold": threshold,
                    "precision": p.get("precision", ""),
                    "recall": p.get("recall", ""),
                    "f1": p.get("f1", ""),
                    "fp_per_image": p.get("fp_per_image", ""),
                    "miss_rate": p.get("miss_rate", ""),
                    "pd": p.get("pd", ""),
                    "operating_point": _operating_point_name(row, float(threshold)) if threshold != "" else "",
                }
            )
    axes[0].set_ylabel("Precision")
    axes[1].set_ylabel("Recall")
    axes[2].set_ylabel("F1")
    axes[2].set_xlabel("Confidence threshold")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        if conf_rows:
            ax.legend()
    fig3 = output_dir / FIGURE_FILES["fig3"][0]
    fig.tight_layout()
    fig.savefig(fig3, dpi=160)
    plt.close(fig)
    write_csv(
        conf_rows,
        ["Dataset", "EvalMode", "threshold", "precision", "recall", "f1", "fp_per_image", "miss_rate", "pd", "operating_point"],
        output_dir / FIGURE_FILES["fig3"][1],
    )
    paths["fig3"] = fig3
    return paths


def _budget_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "BudgetMode": row.get("BudgetMode", row.get("EvalMode", "")),
        "MaxActiveHypotheses": row.get("MaxActiveHypotheses", ""),
        "MaxRefinementSteps": row.get("MaxRefinementSteps", ""),
        "AvgActiveHypotheses": row.get("AvgActiveHypotheses", ""),
        "AvgRefinementBudgetUsed": row.get("AvgRefinementBudgetUsed", ""),
    }


def pr_by_size_rows(metrics_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in metrics_rows:
        for size_bin, payload in row.get("pr_by_size", {}).items():
            points = payload.get("points", []) if isinstance(payload, dict) else []
            total_gt = payload.get("total_gt", "") if isinstance(payload, dict) else ""
            for point in points:
                rows.append(
                    {
                        "Dataset": row.get("Dataset"),
                        "EvalMode": row.get("EvalMode"),
                        "SizeBin": size_bin,
                        "TotalGT": total_gt,
                        "precision": point.get("precision", ""),
                        "recall": point.get("recall", ""),
                        "score": point.get("score", ""),
                    }
                )
    return rows


def miss_rate_vs_fp_rows(metrics_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in metrics_rows:
        for point in row.get("threshold_sweep", []):
            rows.append(
                {
                    "Dataset": row.get("Dataset"),
                    "EvalMode": row.get("EvalMode"),
                    "threshold": point.get("threshold", ""),
                    "miss_rate": point.get("miss_rate", ""),
                    "fp_per_image": point.get("fp_per_image", ""),
                    "pd": point.get("pd", ""),
                    "recall": point.get("recall", ""),
                    "precision": point.get("precision", ""),
                    "f1": point.get("f1", ""),
                }
            )
    return rows


def pr_under_budget_rows(metrics_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in metrics_rows:
        budget = _budget_fields(row)
        for point in row.get("pr_curve", []):
            rows.append(
                {
                    "Dataset": row.get("Dataset"),
                    "EvalMode": row.get("EvalMode"),
                    **budget,
                    "precision": point.get("precision", ""),
                    "recall": point.get("recall", ""),
                    "score": point.get("score", ""),
                }
            )
    return rows


def write_optional_curve_exports(metrics_rows: list[dict], output_dir: str | Path = "outputs/figures") -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    paths: dict[str, Path] = {}
    exports = [
        (
            "pr_by_size",
            pr_by_size_rows(metrics_rows),
            ["Dataset", "EvalMode", "SizeBin", "TotalGT", "precision", "recall", "score"],
        ),
        (
            "miss_rate_vs_fp_per_image",
            miss_rate_vs_fp_rows(metrics_rows),
            ["Dataset", "EvalMode", "threshold", "miss_rate", "fp_per_image", "pd", "recall", "precision", "f1"],
        ),
        (
            "pr_under_budget",
            pr_under_budget_rows(metrics_rows),
            [
                "Dataset",
                "EvalMode",
                "BudgetMode",
                "MaxActiveHypotheses",
                "MaxRefinementSteps",
                "AvgActiveHypotheses",
                "AvgRefinementBudgetUsed",
                "precision",
                "recall",
                "score",
            ],
        ),
    ]
    for name, rows, columns in exports:
        path = output_dir / OPTIONAL_CURVE_FILES[name]
        write_csv(rows, columns, path)
        paths[name] = path
    return paths


def write_all_reports(
    metrics_rows: list[dict],
    reports_dir: str | Path = "outputs/reports",
    figures_dir: str | Path = "outputs/figures",
    optional_curves: bool = False,
) -> dict[str, Path]:
    validate_metrics_rows(metrics_rows)
    paths = write_all_tables(metrics_rows, reports_dir)
    paths.update(plot_required_figures(metrics_rows, figures_dir))
    if optional_curves:
        paths.update(write_optional_curve_exports(metrics_rows, figures_dir))
    return paths


def expected_report_files(reports_dir: str | Path = "outputs/reports", figures_dir: str | Path = "outputs/figures") -> list[Path]:
    reports_dir = Path(reports_dir)
    figures_dir = Path(figures_dir)
    files: list[Path] = []
    for table_name in TABLE_SPECS:
        files.append(reports_dir / f"{table_name}.csv")
        files.append(reports_dir / f"{table_name}.md")
    for png_name, csv_name in FIGURE_FILES.values():
        files.append(figures_dir / png_name)
        files.append(figures_dir / csv_name)
    return files


def verify_report_files(reports_dir: str | Path = "outputs/reports", figures_dir: str | Path = "outputs/figures") -> list[Path]:
    missing = [path for path in expected_report_files(reports_dir, figures_dir) if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing or empty report files: " + ", ".join(str(path) for path in missing))
    return expected_report_files(reports_dir, figures_dir)
