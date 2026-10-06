from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .deployment_metrics import binary_predictions, contiguous_events, evaluate_deploy_metrics
from .models import DetectorPipeline, ModelSpec
from .retrospective_metrics import evaluate_retrospective_metrics
from .types import Batch


SENSITIVITY_QUANTILES = (0.990, 0.995, 0.999)


def _threshold(scores: np.ndarray, quantile: float) -> float:
    finite_scores = np.asarray(scores, dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if not len(finite_scores):
        raise ValueError("No finite reference scores")
    return float(np.quantile(finite_scores, quantile))


def resolve_vus_window(evaluation: list[Batch], configured_window: int | None = None) -> int:
    """Return the configured VUS window or the median anomaly-event duration."""
    if configured_window is not None:
        if configured_window < 1:
            raise ValueError("VUS window must be at least 1")
        return int(configured_window)
    durations = [
        end - start + 1
        for batch in evaluation
        for start, end in contiguous_events(batch.y)
    ]
    if not durations:
        raise ValueError("Cannot resolve VUS window: no anomaly events in evaluation batches")
    return max(1, int(np.median(durations)))


def run_model(
    spec: ModelSpec,
    reference: Batch,
    evaluation: list[Batch],
    out_dir: str | Path,
    threshold_quantile: float = 0.995,
    sliding_window: int = 100,
):
    """Fit on the reference batch and evaluate each test batch independently."""
    out = Path(out_dir) / spec.name
    out.mkdir(parents=True, exist_ok=True)

    model = DetectorPipeline(spec, reference.X.shape[1])
    start = time.perf_counter()
    model.fit_reference(reference.X)
    fit_seconds = time.perf_counter() - start

    reference_scores = model.score(reference.X)
    sensitivity_thresholds = {
        quantile: _threshold(reference_scores, quantile)
        for quantile in SENSITIVITY_QUANTILES
    }
    threshold = _threshold(reference_scores, threshold_quantile)

    reference_scores_df = pd.DataFrame({
        "model": spec.name,
        "family": spec.family,
        "batch_id": reference.batch_id,
        "timestamp_index": np.arange(len(reference.X)),
        "y_true": reference.y,
        "score": reference_scores,
    })
    reference_scores_df.to_csv(out / "reference_scores.csv", index=False)

    retrospective_rows = []
    retrospective_score_frames = []
    sequential_rows = []
    sequential_score_frames = []

    for batch_order, batch in enumerate(evaluation):
        fault_type = batch.metadata.get("fault_type", "unknown")

        start = time.perf_counter()
        score = model.score(batch.X)
        score_seconds = time.perf_counter() - start

        print(
            f"batch={batch.batch_id}, "
            f"fault_type={fault_type}, "
            f"n={len(batch.y)}, "
            f"anomalies={int(np.sum(batch.y == 1))}, "
            f"finite_scores={int(np.isfinite(score).sum())}, "
            f"scorable_anomalies={int(np.sum(batch.y[np.isfinite(score)] == 1))}"
        )

        start = time.perf_counter()
        retrospective = evaluate_retrospective_metrics(
            batch.y,
            score,
            sliding_window,
        )
        retrospective_seconds = time.perf_counter() - start

        retrospective.update({
            "model": spec.name,
            "family": spec.family,
            "protocol": "retrospective",
            "aggregation": "batch",
            "batch_id": batch.batch_id,
            "batch_order": batch_order,
            "fault_type": fault_type,
            "is_labeled_normal": batch.is_labeled_normal,
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "retrospective_seconds": retrospective_seconds,
            "threshold": np.nan,
        })
        retrospective_rows.append(retrospective)

        retrospective_score_frames.append(pd.DataFrame({
            "model": spec.name,
            "family": spec.family,
            "protocol": "retrospective",
            "batch_id": batch.batch_id,
            "batch_order": batch_order,
            "fault_type": fault_type,
            "is_labeled_normal": batch.is_labeled_normal,
            "timestamp_index": np.arange(len(batch.X)),
            "y_true": batch.y,
            "score": score,
        }))

        prediction = binary_predictions(score, threshold)

        start = time.perf_counter()
        sequential = evaluate_deploy_metrics(
            batch.y,
            score,
            threshold=threshold,
            time=batch.time,
        )
        deployment_seconds = time.perf_counter() - start

        print(
            f"[TIMING] {spec.name} | {batch.batch_id} | "
            f"score={score_seconds:.1f}s | "
            f"retrospective={retrospective_seconds:.1f}s | "
            f"deployment={deployment_seconds:.1f}s"
        )

        sequential.update({
            "model": spec.name,
            "family": spec.family,
            "protocol": "sequential",
            "aggregation": "batch",
            "batch_id": batch.batch_id,
            "batch_order": batch_order,
            "fault_type": fault_type,
            "is_labeled_normal": batch.is_labeled_normal,
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "deployment_seconds": deployment_seconds,
            "threshold": threshold,
        })
        sequential_rows.append(sequential)
        sequential_score_frames.append(pd.DataFrame({
            "model": spec.name,
            "family": spec.family,
            "protocol": "sequential",
            "batch_id": batch.batch_id,
            "batch_order": batch_order,
            "fault_type": fault_type,
            "is_labeled_normal": batch.is_labeled_normal,
            "timestamp_index": np.arange(len(batch.X)),
            "time": batch.time if batch.time is not None else np.full(len(batch.X), np.nan),
            "y_true": batch.y,
            "score": score,
            "prediction": prediction,
            "threshold": threshold,
        }))

    retrospective_df = pd.DataFrame(retrospective_rows)
    sequential_df = pd.DataFrame(sequential_rows)
    retrospective_scores_df = pd.concat(retrospective_score_frames, ignore_index=True)
    sequential_scores_df = pd.concat(sequential_score_frames, ignore_index=True)

    excluded = {
        "batch_order",
        "fit_seconds",
        "score_seconds",
        "retrospective_seconds",
        "deployment_seconds",
        "threshold",
        "is_labeled_normal",
        "retrospective_evaluable",
    }
    metric_cols = [
        column
        for column in retrospective_df.select_dtypes(include="number").columns
        if column not in excluded
    ]
    retrospective_by_fault = retrospective_df.groupby(
        ["model", "family", "fault_type"],
        as_index=False,
    )[metric_cols].mean()
    retrospective_by_fault["protocol"] = "retrospective"
    retrospective_by_fault["aggregation"] = "macro_batches_by_fault"
    batch_counts = (
        retrospective_df.groupby(["model", "family", "fault_type"], as_index=False)
        .agg(
            n_batches=("batch_id", "size"),
            n_retrospective_evaluable=("retrospective_evaluable", "sum"),
        )
    )
    retrospective_by_fault = retrospective_by_fault.merge(
        batch_counts,
        on=["model", "family", "fault_type"],
        how="left",
    )

    retrospective_overall = retrospective_df.groupby(
        ["model", "family"],
        as_index=False,
    )[metric_cols].mean()
    retrospective_overall["protocol"] = "retrospective"
    retrospective_overall["aggregation"] = "macro_batches_overall"
    retrospective_overall["n_batches"] = len(retrospective_df)

    retrospective_scores_df.to_csv(out / "retrospective_scores.csv", index=False)
    retrospective_df.to_csv(out / "retrospective_metrics_by_batch.csv", index=False)
    retrospective_by_fault.to_csv(out / "retrospective_metrics_by_fault.csv", index=False)
    retrospective_overall.to_csv(out / "retrospective_metrics_overall.csv", index=False)
    sequential_scores_df.to_csv(out / "sequential_scores.csv", index=False)
    sequential_df.to_csv(out / "sequential_metrics_by_batch.csv", index=False)

    (out / "run.json").write_text(json.dumps({
        "model": asdict(spec),
        "threshold_quantile": threshold_quantile,
        "threshold": threshold,
        "sensitivity_quantiles": list(SENSITIVITY_QUANTILES),
        "sensitivity_thresholds": {
            str(quantile): value
            for quantile, value in sensitivity_thresholds.items()
        },
        "evaluation_unit": "batch",
        "retrospective_aggregation": "macro_from_batch_metrics",
        "sequential_aggregation": "macro_from_batch_metrics",
    }, indent=2))

    return retrospective_overall.iloc[0].to_dict(), sequential_rows
