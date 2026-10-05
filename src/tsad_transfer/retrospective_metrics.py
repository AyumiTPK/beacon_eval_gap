from __future__ import annotations
import numpy as np
from .basic_metrics import basic_metricor, generate_curve


def evaluate_retrospective_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    sliding_window: int = 100,
) -> dict:
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    scores = np.asarray(score, dtype=float).reshape(-1)
    if len(labels) != len(scores):
        raise ValueError("y_true and score lengths differ")

    scorable = np.isfinite(scores)
    labels = labels[scorable]
    scores = scores[scorable]
    if not len(scores):
        raise ValueError("No finite anomaly scores")

    scorable_fraction = float(np.mean(scorable))
    if len(np.unique(labels)) < 2:
        return {
            "AUC-ROC": np.nan,
            "AUC-PR": np.nan,
            "Point-F1": np.nan,
            "Range-F1": np.nan,
            "Point-F1-PA": np.nan,
            "Event-F1-PA": np.nan,
            "VUS-ROC": np.nan,
            "VUS-PR": np.nan,
            "scorable_fraction": scorable_fraction,
            "retrospective_evaluable": False,
        }

    metric_calculator = basic_metricor()
    curve_results = generate_curve(labels, scores, sliding_window, "opt", 250)
    vus_roc = curve_results[-2]
    vus_pr = curve_results[-1]

    return {
        "AUC-ROC": metric_calculator.metric_ROC(labels, scores),
        "AUC-PR": metric_calculator.metric_PR(labels, scores),
        "Point-F1": metric_calculator.metric_PointF1(labels, scores),
        "Range-F1": metric_calculator.metric_RF1(labels, scores),
        "Point-F1-PA": metric_calculator.metric_PointF1PA(labels, scores),
        "Event-F1-PA": metric_calculator.metric_EventF1PA(labels, scores),
        "VUS-ROC": vus_roc,
        "VUS-PR": vus_pr,
        "scorable_fraction": scorable_fraction,
        "retrospective_evaluable": True,
    }