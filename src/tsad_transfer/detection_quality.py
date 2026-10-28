"""Evaluate operational detection quality using saved benchmark scores.
The ``time`` column in sequential_scores.csv is measured in hours.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


SCORE_COLUMNS = [
    "batch_id",
    "fault_type",
    "timestamp_index",
    "time",
    "y_true",
    "score",
    "prediction",
    "threshold",
]

RETROSPECTIVE_METRICS = ("AUC-PR", "VUS-PR")


def event_starts(mask):
    """Return the first index of each contiguous True region."""
    mask = np.asarray(mask, dtype=bool)
    previous = np.concatenate(([False], mask[:-1]))
    return np.flatnonzero(mask & ~previous)


def event_ends(mask):
    """Return the last index of each contiguous True region."""
    mask = np.asarray(mask, dtype=bool)
    following = np.concatenate((mask[1:], [False]))
    return np.flatnonzero(mask & ~following)


def validate_and_select_predictions(frame, model, override_threshold):
    """Check saved alarms and optionally apply a different calibrated threshold."""
    batch_id = frame["batch_id"].iloc[0]
    scores = frame["score"].to_numpy(dtype=float)
    saved_predictions = frame["prediction"].to_numpy(dtype=int)
    scorable = np.isfinite(scores)

    saved_thresholds = frame["threshold"].dropna().unique()
    if len(saved_thresholds) != 1:
        raise ValueError(f"{model}/{batch_id}: expected one threshold")

    saved_threshold = saved_thresholds[0]
    expected_predictions = scorable & (scores >= saved_threshold)

    if not np.array_equal(saved_predictions.astype(bool), expected_predictions):
        raise ValueError(
            f"{model}/{batch_id}: predictions differ from thresholded scores"
        )

    if override_threshold is None:
        return expected_predictions.astype(int), scorable

    predictions = scorable & (scores >= override_threshold)
    return predictions.astype(int), scorable


def calculate_normal_exposure(times, normal_mask):
    """Estimate hours of scorable normal operation using forward time intervals.    """
    if not np.all(np.isfinite(times)):
        raise ValueError("Time must be finite and nondecreasing (in hours)")

    intervals = np.diff(times)
    if np.any(intervals < 0):
        raise ValueError("Time must be finite and nondecreasing (in hours)")

    if len(times) <= 1:
        return np.nan

    positive_intervals = intervals[intervals > 0]
    if len(positive_intervals) == 0:
        final_interval = np.nan
    else:
        final_interval = float(np.median(positive_intervals))

    weights = np.concatenate((intervals, [final_interval]))
    if not np.all(np.isfinite(weights)):
        return np.nan

    return float(np.sum(weights[normal_mask]))


def calculate_event_metrics(
    labels,
    predictions,
    scorable,
    times,
    model,
    batch_id,
    fault_type,
    deadlines,
):
    """Measure first-alarm delay and timely detection for each fault event."""
    fault_mask = labels == 1
    starts = event_starts(fault_mask)
    ends = event_ends(fault_mask)
    rows = []

    for event_number, (start, end) in enumerate(zip(starts, ends), start=1):
        event_predictions = predictions[start : end + 1]
        event_scorable = scorable[start : end + 1]
        alarm_indices = np.flatnonzero(
            (event_predictions == 1) & event_scorable
        )

        detected = len(alarm_indices) > 0
        if detected:
            first_alarm_index = start + int(alarm_indices[0])
            delay_h = float(times[first_alarm_index] - times[start])
        else:
            delay_h = np.nan

        if np.any(event_scorable):
            fault_alarm_fraction = float(
                np.mean(event_predictions[event_scorable])
            )
        else:
            fault_alarm_fraction = np.nan

        row = {
            "model": model,
            "batch_id": batch_id,
            "fault_type": fault_type,
            "event_number": event_number,
            "onset_time_h": float(times[start]),
            "end_time_h": float(times[end]),
            "detected": detected,
            "delay_h": delay_h,
            "fault_alarm_fraction": fault_alarm_fraction,
        }

        for deadline_min in deadlines:
            column = f"detected_within_{deadline_min:g}min"
            row[column] = bool(detected and delay_h <= deadline_min / 60)

        rows.append(row)

    return rows


def calculate_batch_metrics(
    labels,
    predictions,
    scorable,
    times,
    model,
    batch_id,
    fault_type,
    normal_exposure_h,
):
    """Count normal alarms, fault alarms and false-alarm episodes in one batch."""
    normal_mask = (labels == 0) & scorable
    fault_mask = (labels == 1) & scorable
    false_alarm_mask = normal_mask & (predictions == 1)

    normal_points = int(normal_mask.sum())
    fault_points = int(fault_mask.sum())
    normal_alarm_points = int(false_alarm_mask.sum())
    fault_alarm_points = int(np.sum((predictions == 1) & fault_mask))

    if normal_points > 0:
        normal_alarm_fraction = normal_alarm_points / normal_points
    else:
        normal_alarm_fraction = np.nan

    if fault_points > 0:
        fault_alarm_fraction = float(np.mean(predictions[fault_mask]))
    else:
        fault_alarm_fraction = np.nan

    if len(times) > 1:
        duration_h = float(times[-1] - times[0])
    else:
        duration_h = 0.0

    return {
        "model": model,
        "batch_id": batch_id,
        "fault_type": fault_type,
        "is_clean": not np.any(labels == 1),
        "n_events": len(event_starts(labels == 1)),
        "duration_h": duration_h,
        "normal_exposure_h": normal_exposure_h,
        "normal_scorable_points": normal_points,
        "normal_alarm_points": normal_alarm_points,
        "false_alarm_events": int(len(event_starts(false_alarm_mask))),
        "clean_fpr": normal_alarm_fraction,
        "normal_alarm_fraction": normal_alarm_fraction,
        "fault_alarm_points": fault_alarm_points,
        "fault_scorable_points": fault_points,
        "fault_alarm_fraction": fault_alarm_fraction,
    }


def process_batch(frame, model, deadlines, override_threshold=None):
    """Evaluate a complete batch at one threshold.
    Returns one batch-statistics dictionary and a list of fault-event rows.
    """
    frame = frame.sort_values("timestamp_index", kind="stable")
    batch_id = frame["batch_id"].iloc[0]
    fault_type = frame["fault_type"].iloc[0]

    labels = frame["y_true"].to_numpy(dtype=int)
    times = frame["time"].to_numpy(dtype=float)

    predictions, scorable = validate_and_select_predictions(
        frame,
        model,
        override_threshold,
    )

    normal_mask = (labels == 0) & scorable
    normal_exposure_h = calculate_normal_exposure(times, normal_mask)

    event_rows = calculate_event_metrics(
        labels,
        predictions,
        scorable,
        times,
        model,
        batch_id,
        fault_type,
        deadlines,
    )

    batch_row = calculate_batch_metrics(
        labels,
        predictions,
        scorable,
        times,
        model,
        batch_id,
        fault_type,
        normal_exposure_h,
    )

    return batch_row, event_rows


def evaluate_complete_batch(frame, model, deadlines, quantile_thresholds):
    """Evaluate a batch at the saved threshold and at each sensitivity quantile."""
    default_batch, default_events = process_batch(frame, model, deadlines)
    sensitivity_batches = []
    sensitivity_events = []

    if quantile_thresholds is None:
        quantile_thresholds = {}

    for quantile, threshold in quantile_thresholds.items():
        batch_row, event_rows = process_batch(
            frame,
            model,
            deadlines,
            override_threshold=threshold,
        )
        batch_row["threshold_quantile"] = quantile
        sensitivity_batches.append(batch_row)

        for event_row in event_rows:
            event_row["threshold_quantile"] = quantile
        sensitivity_events.extend(event_rows)

    return default_batch, default_events, sensitivity_batches, sensitivity_events


def load_model(path, deadlines, chunksize, quantile_thresholds=None):
    """Read one model's score CSV without splitting batches at chunk boundaries.

    The benchmark stores each batch contiguously in sequential_scores.csv.
    """
    model = path.parent.name
    batches = []
    events = []
    sensitivity_batches = []
    sensitivity_events = []

    current_batch_id = None
    batch_pieces = []

    def finish_current_batch():
        if not batch_pieces:
            return

        complete_batch = pd.concat(batch_pieces, ignore_index=True)
        results = evaluate_complete_batch(
            complete_batch,
            model,
            deadlines,
            quantile_thresholds,
        )
        default_batch, default_events, quantile_batches, quantile_events = results

        batches.append(default_batch)
        events.extend(default_events)
        sensitivity_batches.extend(quantile_batches)
        sensitivity_events.extend(quantile_events)

    for chunk in pd.read_csv(path, usecols=SCORE_COLUMNS, chunksize=chunksize):
        batch_ids = chunk["batch_id"].to_numpy()
        batch_changed = batch_ids[1:] != batch_ids[:-1]
        change_indices = np.flatnonzero(batch_changed) + 1
        boundaries = np.concatenate(([0], change_indices, [len(chunk)]))

        for left, right in zip(boundaries[:-1], boundaries[1:]):
            piece = chunk.iloc[left:right]
            piece_batch_id = str(piece["batch_id"].iloc[0])

            if current_batch_id is not None and piece_batch_id != current_batch_id:
                finish_current_batch()
                batch_pieces = []

            current_batch_id = piece_batch_id
            batch_pieces.append(piece)

    finish_current_batch()
    return batches, events, sensitivity_batches, sensitivity_events


def load_calibrated_thresholds(run_path, quantiles):
    """Read reference-calibrated thresholds from a model's run.json."""
    if not run_path.is_file():
        raise FileNotFoundError(f"Missing calibrated thresholds: {run_path}")

    with run_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    available = {}
    for quantile, threshold in metadata["sensitivity_thresholds"].items():
        available[float(quantile)] = float(threshold)

    selected = {}
    missing = []

    for requested_quantile in quantiles:
        matched_threshold = None

        for available_quantile, threshold in available.items():
            if np.isclose(requested_quantile, available_quantile, atol=1e-10):
                matched_threshold = threshold
                break

        if matched_threshold is None:
            missing.append(requested_quantile)
        else:
            selected[requested_quantile] = matched_threshold

    if missing:
        raise ValueError(
            f"{run_path.parent.name}: missing calibration quantiles "
            f"{missing} in {run_path}"
        )

    default_quantile = float(metadata["threshold_quantile"])
    return selected, default_quantile


def add_timely_detection_rates(row, model_events, deadlines):
    """Add event-level timely detection rates, counting missed events as failures."""
    for deadline_min in deadlines:
        event_column = f"detected_within_{deadline_min:g}min"
        summary_column = f"timely_detection_{deadline_min:g}min"

        if len(model_events) > 0:
            row[summary_column] = float(model_events[event_column].mean())
        else:
            row[summary_column] = np.nan


def summarize_default(batches, events, deadlines):
    """Aggregate default-threshold statistics by model."""
    rows = []

    for model, model_batches in batches.groupby("model", sort=True):
        clean_batches = model_batches[model_batches["is_clean"]]
        fault_batches = model_batches[~model_batches["is_clean"]]
        model_events = events[events["model"] == model]

        clean_exposure_h = clean_batches["normal_exposure_h"].sum()
        false_alarm_events = int(clean_batches["false_alarm_events"].sum())
        normal_points = int(clean_batches["normal_scorable_points"].sum())
        false_alarm_points = int(clean_batches["normal_alarm_points"].sum())
        fault_points = int(fault_batches["fault_scorable_points"].sum())
        fault_alarm_points = int(fault_batches["fault_alarm_points"].sum())
        fault_batch_normal_points = int(fault_batches["normal_scorable_points"].sum())
        fault_batch_normal_alarms = int(fault_batches["normal_alarm_points"].sum())

        if len(model_events) > 0:
            event_detection_rate = float(model_events["detected"].mean())
            detected_events = model_events[model_events["detected"]]
            median_delay_h = float(detected_events["delay_h"].median())
        else:
            event_detection_rate = np.nan
            median_delay_h = np.nan

        if normal_points > 0:
            clean_fpr = false_alarm_points / normal_points
        else:
            clean_fpr = np.nan

        if clean_exposure_h > 0:
            false_alarms_per_hour = false_alarm_events / clean_exposure_h
        else:
            false_alarms_per_hour = np.nan

        if fault_points > 0:
            fault_alarm_fraction = fault_alarm_points / fault_points
        else:
            fault_alarm_fraction = np.nan

        if fault_batch_normal_points > 0:
            fault_batch_normal_fraction = (
                fault_batch_normal_alarms / fault_batch_normal_points
            )
        else:
            fault_batch_normal_fraction = np.nan

        row = {
            "model": model,
            "n_clean_batches": len(clean_batches),
            "n_fault_batches": len(fault_batches),
            "n_fault_events": len(model_events),
            "event_detection_rate": event_detection_rate,
            "median_delay_h_detected_only": median_delay_h,
            "clean_fpr": clean_fpr,
            "clean_false_alarm_events": false_alarm_events,
            "clean_exposure_h": clean_exposure_h,
            "clean_false_alarm_events_per_hour": false_alarms_per_hour,
            "fault_alarm_fraction": fault_alarm_fraction,
            "normal_alarm_fraction_fault_batches": fault_batch_normal_fraction,
        }

        add_timely_detection_rates(row, model_events, deadlines)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_threshold_sensitivity(batches, events, deadlines):
    """Aggregate detection and clean-operation false alarms at each quantile."""
    rows = []
    grouped_batches = batches.groupby(["model", "threshold_quantile"], sort=True)

    for (model, quantile), model_batches in grouped_batches:
        same_model = events["model"] == model
        same_quantile = events["threshold_quantile"] == quantile
        model_events = events[same_model & same_quantile]

        clean_batches = model_batches[model_batches["is_clean"]]
        fault_batches = model_batches[~model_batches["is_clean"]]

        normal_points = int(clean_batches["normal_scorable_points"].sum())
        false_alarm_points = int(clean_batches["normal_alarm_points"].sum())
        false_alarm_events = int(clean_batches["false_alarm_events"].sum())
        clean_exposure_h = float(clean_batches["normal_exposure_h"].sum())

        if len(model_events) > 0:
            event_detection_rate = float(model_events["detected"].mean())
        else:
            event_detection_rate = np.nan

        if normal_points > 0:
            clean_fpr = float(false_alarm_points / normal_points)
        else:
            clean_fpr = np.nan

        if clean_exposure_h > 0:
            false_alarms_per_hour = float(false_alarm_events / clean_exposure_h)
        else:
            false_alarms_per_hour = np.nan

        row = {
            "model": model,
            "threshold_quantile": quantile,
            "n_clean_batches": len(clean_batches),
            "n_fault_batches": len(fault_batches),
            "n_fault_events": len(model_events),
            "event_detection_rate": event_detection_rate,
            "clean_fpr": clean_fpr,
            "clean_false_alarm_events": false_alarm_events,
            "clean_exposure_h": clean_exposure_h,
            "clean_false_alarm_events_per_hour": false_alarms_per_hour,
        }

        add_timely_detection_rates(row, model_events, deadlines)
        rows.append(row)

    return pd.DataFrame(rows)


def load_retrospective_metrics(results, operational_models):
    """Load model-level retrospective AUC-PR and VUS-PR for ranking transfer."""
    candidate_names = [
        "retrospective_metrics_overall.csv",
        "retrospective_overall.csv",
    ]
    retrospective_path = None

    for filename in candidate_names:
        candidate = results / filename
        if candidate.is_file():
            retrospective_path = candidate
            break

    if retrospective_path is None:
        raise FileNotFoundError(
            "Missing retrospective_metrics_overall.csv or "
            f"retrospective_overall.csv under {results}"
        )

    retrospective = pd.read_csv(retrospective_path)
    required_columns = {"model", *RETROSPECTIVE_METRICS}
    missing_columns = required_columns - set(retrospective.columns)

    if missing_columns:
        raise ValueError(f"{retrospective_path} lacks columns {missing_columns}")

    if retrospective["model"].duplicated().any():
        raise ValueError(
            f"Duplicate models in {retrospective_path}; "
            "choose one overall aggregation"
        )

    if set(retrospective["model"]) != set(operational_models):
        raise ValueError(
            "Retrospective and operational model sets differ; "
            "ranking comparison would be invalid"
        )

    return retrospective


def calculate_ranking_transfer(sensitivity, retrospective, deadlines):
    """Calculate Spearman correlations across models at each calibration quantile."""
    metric_columns = ["model", *RETROSPECTIVE_METRICS]
    retrospective_subset = retrospective[metric_columns]

    merged = sensitivity.merge(
        retrospective_subset,
        on="model",
        validate="many_to_one",
    )
    rows = []

    for quantile, group in merged.groupby("threshold_quantile", sort=True):
        for retrospective_metric in RETROSPECTIVE_METRICS:
            for deadline_min in deadlines:
                operational_metric = f"timely_detection_{deadline_min:g}min"
                metric_pair = group[[retrospective_metric, operational_metric]]
                valid_models = metric_pair.dropna()

                enough_models = len(valid_models) >= 3
                retrospective_varies = (
                    valid_models[retrospective_metric].nunique() > 1
                )
                operational_varies = (
                    valid_models[operational_metric].nunique() > 1
                )

                if enough_models and retrospective_varies and operational_varies:
                    correlation = valid_models[retrospective_metric].corr(
                        valid_models[operational_metric],
                        method="spearman",
                    )
                else:
                    correlation = np.nan

                row = {
                    "threshold_quantile": quantile,
                    "retrospective_metric": retrospective_metric,
                    "operational_metric": operational_metric,
                    "n_models": len(valid_models),
                    "spearman_rho": correlation,
                }
                rows.append(row)

    return pd.DataFrame(rows)


def calculate_ranking_stability(sensitivity, deadlines, default_quantile):
    """Calculate model ranks and changes from the default calibration quantile."""
    rows = []

    for deadline_min in deadlines:
        metric = f"timely_detection_{deadline_min:g}min"
        ranks = sensitivity[["model", "threshold_quantile", metric]].copy()

        ranks["rank"] = ranks.groupby("threshold_quantile")[metric].rank(
            method="average",
            ascending=False,
        )

        at_default = np.isclose(ranks["threshold_quantile"], default_quantile)
        baseline = ranks.loc[at_default, ["model", "rank"]].copy()
        baseline = baseline.rename(columns={"rank": "default_rank"})

        ranks = ranks.merge(baseline, on="model", validate="many_to_one")
        ranks["rank_change_from_default"] = (
            ranks["rank"] - ranks["default_rank"]
        )

        for record in ranks.to_dict("records"):
            row = {
                "model": record["model"],
                "threshold_quantile": record["threshold_quantile"],
                "operational_metric": metric,
                "value": record[metric],
                "rank": record["rank"],
                "default_threshold_quantile": default_quantile,
                "default_rank": record["default_rank"],
                "rank_change_from_default": record["rank_change_from_default"],
            }
            rows.append(row)

    return pd.DataFrame(rows)


def analyze_detection_quality(
    results: Path,
    output: Path,
    deadlines=(10, 30, 60),
    chunksize=150000,
    quantiles=(0.990, 0.995, 0.999),
):
    """Run the post-analysis and write the six operational-quality CSV files."""
    results = Path(results)
    output = Path(output)

    score_paths = sorted(results.glob("*/sequential_scores.csv"))
    if not score_paths:
        raise FileNotFoundError(f"No */sequential_scores.csv under {results}")

    output.mkdir(parents=True, exist_ok=True)

    default_batches = []
    default_events = []
    sensitivity_batches = []
    sensitivity_events = []
    default_quantiles = set()

    for score_path in score_paths:
        print(f"Reading {score_path}", flush=True)

        run_path = score_path.parent / "run.json"
        thresholds, default_quantile = load_calibrated_thresholds(
            run_path,
            quantiles,
        )
        default_quantiles.add(default_quantile)

        model_results = load_model(
            score_path,
            deadlines,
            chunksize,
            thresholds,
        )
        model_batches, model_events, model_sensitivity_batches, model_sensitivity_events = model_results

        default_batches.extend(model_batches)
        default_events.extend(model_events)
        sensitivity_batches.extend(model_sensitivity_batches)
        sensitivity_events.extend(model_sensitivity_events)

    batch_df = pd.DataFrame(default_batches)
    event_df = pd.DataFrame(default_events)
    batch_df.to_csv(output / "operational_by_batch.csv", index=False)
    event_df.to_csv(output / "operational_by_event.csv", index=False)

    summary = summarize_default(batch_df, event_df, deadlines)
    summary.to_csv(output / "operational_model_summary.csv", index=False)

    sensitivity_batch_df = pd.DataFrame(sensitivity_batches)
    sensitivity_event_df = pd.DataFrame(sensitivity_events)
    sensitivity = summarize_threshold_sensitivity(
        sensitivity_batch_df,
        sensitivity_event_df,
        deadlines,
    )
    sensitivity.to_csv(output / "operational_threshold_sensitivity.csv", index=False)

    retrospective = load_retrospective_metrics(results, sensitivity["model"])
    ranking_transfer = calculate_ranking_transfer(
        sensitivity,
        retrospective,
        deadlines,
    )
    ranking_transfer.to_csv(output / "operational_ranking_transfer.csv", index=False)

    if len(default_quantiles) != 1:
        raise ValueError(
            f"Inconsistent default calibration quantiles: {default_quantiles}"
        )

    default_quantile = next(iter(default_quantiles))
    default_is_requested = any(
        np.isclose(default_quantile, quantile) for quantile in quantiles
    )
    if not default_is_requested:
        raise ValueError(
            f"Default quantile {default_quantile} not among requested {quantiles}"
        )

    ranking_stability = calculate_ranking_stability(
        sensitivity,
        deadlines,
        default_quantile,
    )
    ranking_stability.to_csv(output / "operational_ranking_stability.csv", index=False)

    return summary
