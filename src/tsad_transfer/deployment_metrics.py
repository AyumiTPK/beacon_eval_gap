from __future__ import annotations
import numpy as np


def binary_predictions(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    p = np.asarray(probability, dtype=float).reshape(-1)
    pred = np.zeros(len(p), dtype=int)
    pred[np.isfinite(p) & (p >= threshold)] = 1
    return pred


def contiguous_events(y: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive (start, end) intervals for contiguous positive labels."""
    y = np.asarray(y, dtype=int).reshape(-1)
    idx = np.flatnonzero(y == 1)
    if len(idx) == 0:
        return []
    events = []
    start = prev = int(idx[0])
    for cur in idx[1:]:
        cur = int(cur)
        if cur != prev + 1:
            events.append((start, prev))
            start = cur
        prev = cur
    events.append((start, prev))
    return events


def evaluate_deploy_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float = 0.5,
    time: np.ndarray | None = None,
) -> dict:
    """Event detection rates, first-detection delays, and point FPR.
    """
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("y_true and probability lengths differ")
    pred = binary_predictions(p, threshold=threshold)
    scorable = np.isfinite(p)

    if time is not None:
        t = np.asarray(time)
        if len(t) != len(y):
            raise ValueError("time and y_true lengths differ")
    else:
        t = None

    events = contiguous_events(y)
    delays_samples: list[float] = []
    delays_time: list[float] = []
    detected = 0
    process_delays_samples: list[float] = []
    process_delays_time: list[float] = []
    detected_by_process_end = 0

    for start, end in events:
        hits = np.flatnonzero(
            (pred[start:end + 1] == 1) & scorable[start:end + 1]
        )
        if len(hits):
            first = start + int(hits[0])
            detected += 1
            delays_samples.append(float(first - start))
            if t is not None:
                delays_time.append(float(t[first] - t[start]))

        process_hits = np.flatnonzero((pred[start:] == 1) & scorable[start:])
        if len(process_hits):
            first_process = start + int(process_hits[0])
            detected_by_process_end += 1
            process_delays_samples.append(float(first_process - start))
            if t is not None:
                process_delays_time.append(float(t[first_process] - t[start]))

    normal_mask = (y == 0) & scorable
    false_positive_mask = (pred == 1) & normal_mask
    fp = int(np.sum(false_positive_mask))
    false_positive_events = len(contiguous_events(false_positive_mask.astype(int)))
    normal_count = int(np.sum(normal_mask))
    fpr = fp / normal_count if normal_count else np.nan

    result = {
        "n_events": len(events),
        "detected_events": detected,
        "detection_rate": detected / len(events) if events else np.nan,
        "delay_samples_mean": float(np.mean(delays_samples)) if delays_samples else np.nan,
        "delay_samples_median": float(np.median(delays_samples)) if delays_samples else np.nan,
        "delay_samples": delays_samples,
        "detected_events_by_process_end": detected_by_process_end,
        "detection_rate_by_process_end": detected_by_process_end / len(events) if events else np.nan,
        "delay_samples_by_process_end_mean": float(np.mean(process_delays_samples)) if process_delays_samples else np.nan,
        "delay_samples_by_process_end_median": float(np.median(process_delays_samples)) if process_delays_samples else np.nan,
        "delay_samples_by_process_end": process_delays_samples,
        "false_positives": fp,
        "false_positive_events": false_positive_events,
        "normal_scorable_points": normal_count,
        "false_positive_rate": float(fpr) if np.isfinite(fpr) else np.nan,
        "scorable_fraction": float(np.mean(scorable)) if len(scorable) else np.nan,
    }
    if t is not None:
        result.update({
            "delay_time_mean": float(np.mean(delays_time)) if delays_time else np.nan,
            "delay_time_median": float(np.median(delays_time)) if delays_time else np.nan,
            "delay_time": delays_time,
            "delay_time_by_process_end_mean": float(np.mean(process_delays_time)) if process_delays_time else np.nan,
            "delay_time_by_process_end_median": float(np.median(process_delays_time)) if process_delays_time else np.nan,
            "delay_time_by_process_end": process_delays_time,
        })
    return result
