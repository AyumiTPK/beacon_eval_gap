from __future__ import annotations
from pathlib import Path
import pandas as pd


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
    }

    numeric_columns = df.select_dtypes(include="number").columns

    metric_columns = [
        column
        for column in numeric_columns
        if column not in excluded
    ]

    grouped_metrics = (
        df.groupby(group_cols, as_index=False)[metric_columns]
        .mean()
    )

    batch_counts = (
        df.groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "n_batches"})
    )

    result = grouped_metrics.merge(
        batch_counts,
        on=group_cols,
        how="left",
    )

    result["aggregation"] = aggregation
    return result


def analyze(results_dir: str | Path) -> None:
    """Create batch, fault, and overall summaries for both evaluation protocols."""
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
