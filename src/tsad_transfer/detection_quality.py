"""Post-hoc operational detection quality from saved sequential scores.

No model fitting, score recomputation or threshold changes. `time` must be hours.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

COLS = ['batch_id', 'fault_type', 'timestamp_index', 'time', 'y_true', 'score', 'prediction', 'threshold']


def event_starts(mask):
    mask = np.asarray(mask, dtype=bool)
    return np.flatnonzero(mask & ~np.r_[False, mask[:-1]])


def process_batch(frame, model, deadlines):
    frame = frame.sort_values('timestamp_index', kind='stable')
    y = frame['y_true'].to_numpy(dtype=int)
    scores = frame['score'].to_numpy(dtype=float)
    pred = frame['prediction'].to_numpy(dtype=int)
    t = frame['time'].to_numpy(dtype=float)
    finite = np.isfinite(scores)
    # Validate the saved predictions against the benchmark's >= threshold rule.
    thresholds = frame['threshold'].dropna().unique()
    if len(thresholds) != 1:
        raise ValueError(f'{model}/{frame.batch_id.iloc[0]}: expected one threshold')
    expected = finite & (scores >= thresholds[0])
    if not np.array_equal(pred.astype(bool), expected):
        raise ValueError(f'{model}/{frame.batch_id.iloc[0]}: predictions differ from thresholded scores')
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) < 0):
        raise ValueError('Time must be finite and nondecreasing (in hours)')
    normal = (y == 0) & finite
    fault = (y == 1) & finite
    false = normal & (pred == 1)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    # Scorable normal exposure: duration represented by normal, finite samples.
    # Forward time differences, last interval estimated from median positive step.
    dt = np.diff(t)
    step = float(np.median(dt[dt > 0])) if np.any(dt > 0) else np.nan
    weights = np.r_[dt, step] if len(t) > 1 else np.array([np.nan])
    normal_hours = float(np.sum(weights[normal])) if np.all(np.isfinite(weights)) else np.nan
    starts = event_starts(y == 1)
    # Compute actual event ends, not the next event start (normal gaps may exist).
    ends = np.flatnonzero((y == 1) & ~np.r_[y[1:] == 1, False])
    event_rows = []
    for j, (s, e) in enumerate(zip(starts, ends), 1):
        hits = np.flatnonzero((pred[s:e+1] == 1) & finite[s:e+1])
        first = int(s + hits[0]) if len(hits) else None
        delay_h = float(t[first] - t[s]) if first is not None else np.nan
        row = {'model': model, 'batch_id': frame.batch_id.iloc[0], 'fault_type': frame.fault_type.iloc[0],
               'event_number': j, 'onset_time_h': float(t[s]), 'end_time_h': float(t[e]),
               'detected': first is not None, 'delay_h': delay_h,
               'fault_alarm_fraction': float(np.mean(pred[s:e+1][finite[s:e+1]])) if np.any(finite[s:e+1]) else np.nan}
        for d in deadlines:
            row[f'detected_within_{d:g}min'] = bool(first is not None and delay_h <= d/60)
        event_rows.append(row)
    row = {'model': model, 'batch_id': frame.batch_id.iloc[0], 'fault_type': frame.fault_type.iloc[0],
           'is_clean': not np.any(y == 1), 'n_events': len(starts), 'duration_h': duration,
           'normal_exposure_h': normal_hours, 'normal_scorable_points': int(normal.sum()),
           'normal_alarm_points': int(false.sum()), 'false_alarm_events': int(len(event_starts(false))),
           'clean_fpr': float(false.sum()/normal.sum()) if normal.any() else np.nan,
           'normal_alarm_fraction': float(false.sum()/normal.sum()) if normal.any() else np.nan,
           'fault_alarm_points': int(np.sum((pred == 1) & fault)),
           'fault_scorable_points': int(fault.sum()),
           'fault_alarm_fraction': float(np.mean(pred[fault])) if fault.any() else np.nan}
    return row, event_rows


def load_model(path, deadlines, chunksize):
    model = path.parent.name
    batches, events = [], []
    current_id, pieces = None, []
    def flush():
        if pieces:
            b, e = process_batch(pd.concat(pieces, ignore_index=True), model, deadlines)
            batches.append(b)
            events.extend(e)
    for chunk in pd.read_csv(path, usecols=COLS, chunksize=chunksize):
        # Input produced by protocols.py is batch-ordered; split at boundaries.
        ids = chunk['batch_id'].to_numpy()
        boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1])+1, len(chunk)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            part = chunk.iloc[left:right]
            bid = str(part['batch_id'].iloc[0])
            if current_id is not None and bid != current_id:
                flush()
                pieces = []
            current_id = bid
            pieces.append(part)
    flush()
    return batches, events


def analyze_detection_quality(results: Path, output: Path, deadlines=(10, 30, 60), chunksize=150000):
    """Read per-model sequential_scores.csv and write three reproducible CSV tables."""
    results, output = Path(results), Path(output)
    paths = sorted(results.glob('*/sequential_scores.csv'))
    if not paths:
        raise FileNotFoundError(f'No */sequential_scores.csv under {results}')
    output.mkdir(parents=True, exist_ok=True)
    batches, events = [], []
    for path in paths:
        print(f'Reading {path}', flush=True)
        b, e = load_model(path, deadlines, chunksize)
        batches.extend(b)
        events.extend(e)
    bdf, edf = pd.DataFrame(batches), pd.DataFrame(events)
    bdf.to_csv(output / 'operational_by_batch.csv', index=False)
    edf.to_csv(output / 'operational_by_event.csv', index=False)
    summary = []
    for model, b in bdf.groupby('model', sort=True):
        clean = b[b.is_clean]
        e = edf[edf.model == model]
        clean_exposure = clean.normal_exposure_h.sum()
        clean_fp_events = int(clean.false_alarm_events.sum())
        clean_normal_points = int(clean.normal_scorable_points.sum())
        clean_fp_points = int(clean.normal_alarm_points.sum())
        fault_b = b[~b.is_clean]
        row = {'model': model, 'n_clean_batches': len(clean), 'n_fault_batches': len(fault_b),
               'n_fault_events': len(e), 'event_detection_rate': float(e.detected.mean()) if len(e) else np.nan,
               'median_delay_h_detected_only': float(e.loc[e.detected, 'delay_h'].median()) if len(e) else np.nan,
               'clean_fpr': clean_fp_points/clean_normal_points if clean_normal_points else np.nan,
               'clean_false_alarm_events': clean_fp_events,
               'clean_exposure_h': clean_exposure,
               'clean_false_alarm_events_per_hour': clean_fp_events/clean_exposure if clean_exposure > 0 else np.nan,
               'fault_alarm_fraction': fault_b.fault_alarm_points.sum()/fault_b.fault_scorable_points.sum() if fault_b.fault_scorable_points.sum() else np.nan,
               'normal_alarm_fraction_fault_batches': fault_b.normal_alarm_points.sum()/fault_b.normal_scorable_points.sum() if fault_b.normal_scorable_points.sum() else np.nan}
        for d in deadlines:
            row[f'timely_detection_{d:g}min'] = float(e[f'detected_within_{d:g}min'].mean()) if len(e) else np.nan
        summary.append(row)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(output / 'operational_model_summary.csv', index=False)
    return sdf
