from __future__ import annotations
from pathlib import Path
import pandas as pd


def _read_many(paths):
    paths = list(paths)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def _macro(df: pd.DataFrame, group_cols: list[str], aggregation: str) -> pd.DataFrame:
    """Macro-average batch-level metric columns, excluding bookkeeping numbers."""
    excluded = {"batch_order", "fit_seconds", "score_seconds", "threshold"}
    numeric = [c for c in df.select_dtypes(include="number").columns if c not in excluded]
    out = df.groupby(group_cols, as_index=False)[numeric].mean()
    counts = df.groupby(group_cols, as_index=False).size().rename(columns={"size": "n_batches"})
    out = out.merge(counts, on=group_cols, how="left")
    out["aggregation"] = aggregation
    return out


def analyze(results_dir: str | Path) -> None:
    """Create batch, fault, and overall summaries for both evaluation protocols.

    Independent process/simulation runs are the aggregation unit. Retrospective
    AUC/VUS metrics are first computed per batch and then macro-averaged by fault
    and overall. Sequential deployment metrics are aggregated in the same way.
    The metric families themselves remain protocol-specific and are not treated
    as interchangeable measurements.
    """
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
