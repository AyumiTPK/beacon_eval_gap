from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .deployment_metrics import evaluate_deploy_metrics


def _read_many(paths):
    paths = list(paths)
    if not paths:
        return pd.DataFrame()
    return pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )


def _macro(df: pd.DataFrame, group_cols: list[str], aggregation: str) -> pd.DataFrame:
    """Macro-average batch-level metric columns, excluding bookkeeping numbers."""
    excluded = {
        "batch_order",
        "fit_seconds",
        "score_seconds",
        "threshold",
        "threshold_quantile",
        "is_labeled_normal",
    }
    numeric_columns = df.select_dtypes(include="number").columns
    metric_columns = [
        column
        for column in numeric_columns
        if column not in excluded
    ]
    grouped_metrics = df.groupby(group_cols, as_index=False)[metric_columns].mean()
    batch_counts = (
        df.groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "n_batches"})
    )
    result = grouped_metrics.merge(batch_counts, on=group_cols, how="left")
    result["aggregation"] = aggregation
    return result


def _threshold_sensitivity(root: Path) -> pd.DataFrame:
    rows = []
    for run_path in root.glob("*/run.json"):
        model_dir = run_path.parent
        scores_path = model_dir / "sequential_scores.csv"
        if not scores_path.exists():
            continue

        run = json.loads(run_path.read_text())
        thresholds = run.get("sensitivity_thresholds", {})
        if not thresholds:
            continue

        scores = pd.read_csv(scores_path)
        for quantile_text, threshold in thresholds.items():
            quantile = float(quantile_text)
            for batch_id, batch_scores in scores.groupby("batch_id", sort=False):
                batch_scores = batch_scores.sort_values("timestamp_index")
                time = None
                if "time" in batch_scores and batch_scores["time"].notna().all():
                    time = batch_scores["time"].to_numpy()
                metrics = evaluate_deploy_metrics(
                    batch_scores["y_true"].to_numpy(),
                    batch_scores["score"].to_numpy(),
                    threshold=float(threshold),
                    time=time,
                )
                first = batch_scores.iloc[0]
                metrics.update({
                    "model": first["model"],
                    "family": first["family"],
                    "protocol": "sequential",
                    "aggregation": "batch",
                    "batch_id": batch_id,
                    "batch_order": first["batch_order"],
                    "fault_type": first["fault_type"],
                    "is_labeled_normal": bool(first["is_labeled_normal"]),
                    "threshold_quantile": quantile,
                    "threshold": float(threshold),
                })
                rows.append(metrics)
    return pd.DataFrame(rows)


def _clean_batch_fpr(sensitivity: pd.DataFrame) -> pd.DataFrame:
    clean = sensitivity[sensitivity["is_labeled_normal"]].copy()
    if clean.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["model", "family", "threshold_quantile", "threshold"]
    for keys, group in clean.groupby(group_cols, sort=False):
        model, family, quantile, threshold = keys
        false_positives = int(group["false_positives"].sum())
        normal_scorable_points = int(group["normal_scorable_points"].sum())
        rows.append({
            "model": model,
            "family": family,
            "threshold_quantile": quantile,
            "threshold": threshold,
            "nominal_reference_exceedance": 1.0 - quantile,
            "false_positives": false_positives,
            "normal_scorable_points": normal_scorable_points,
            "false_positive_rate": (
                false_positives / normal_scorable_points
                if normal_scorable_points else np.nan
            ),
            "macro_false_positive_rate": group["false_positive_rate"].mean(),
            "n_batches": len(group),
            "aggregation": "pooled_clean_batches",
        })
    return pd.DataFrame(rows)


def analyze(results_dir: str | Path) -> None:
    """Create batch, fault, overall, threshold-sensitivity, and clean-batch summaries."""
    root = Path(results_dir)
    retro_batch = _read_many(root.glob("*/retrospective_metrics_by_batch.csv"))
    seq_batch = _read_many(root.glob("*/sequential_metrics_by_batch.csv"))

    if not retro_batch.empty:
        retro_batch.to_csv(root / "retrospective_by_batch.csv", index=False)
        _macro(retro_batch, ["model", "family", "fault_type"], "macro_batches_by_fault").to_csv(
            root / "retrospective_by_fault.csv", index=False
        )
        _macro(retro_batch, ["model", "family"], "macro_batches_overall").to_csv(
            root / "retrospective_overall.csv", index=False
        )

    if not seq_batch.empty:
        seq_batch.to_csv(root / "sequential_by_batch.csv", index=False)
        _macro(seq_batch, ["model", "family", "fault_type"], "macro_batches_by_fault").to_csv(
            root / "sequential_by_fault.csv", index=False
        )
        _macro(seq_batch, ["model", "family"], "macro_batches_overall").to_csv(
            root / "sequential_overall.csv", index=False
        )

    sensitivity = _threshold_sensitivity(root)
    if not sensitivity.empty:
        sensitivity.to_csv(root / "sequential_threshold_sensitivity_by_batch.csv", index=False)
        _macro(
            sensitivity,
            ["model", "family", "fault_type", "threshold_quantile"],
            "macro_batches_by_fault",
        ).to_csv(root / "sequential_threshold_sensitivity_by_fault.csv", index=False)
        _macro(
            sensitivity,
            ["model", "family", "threshold_quantile"],
            "macro_batches_overall",
        ).to_csv(root / "sequential_threshold_sensitivity_overall.csv", index=False)

        clean_fpr = _clean_batch_fpr(sensitivity)
        if not clean_fpr.empty:
            clean_fpr.to_csv(root / "sequential_clean_batch_fpr.csv", index=False)
