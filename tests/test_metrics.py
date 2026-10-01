import numpy as np

from tsad_transfer.deployment_metrics import evaluate_deploy_metrics


def test_deploy_metrics_detect_active_event_and_false_positive():
    y = np.array([0, 0, 1, 1, 0])
    score = np.array([0.1, 0.8, 0.2, 0.9, 0.1])
    result = evaluate_deploy_metrics(y, score, threshold=0.5)

    assert result["detection_rate"] == 1.0
    assert result["delay_samples_mean"] == 1.0
    assert result["false_positives"] == 1
    assert result["false_positive_events"] == 1
    assert result["normal_scorable_points"] == 3
    assert result["false_positive_rate"] == 1 / 3


def test_deploy_metrics_excludes_nan_scores_from_fpr_denominator():
    y = np.array([0, 0, 1, 1])
    score = np.array([np.nan, 0.8, np.nan, 0.9])
    result = evaluate_deploy_metrics(y, score, threshold=0.5)

    assert result["normal_scorable_points"] == 1
    assert result["false_positives"] == 1
    assert result["scorable_fraction"] == 0.5


def test_deploy_metrics_counts_contiguous_false_alarms_as_events():
    y = np.zeros(8, dtype=int)
    score = np.array([0.1, 0.8, 0.9, 0.1, 0.7, 0.8, 0.1, 0.1])
    result = evaluate_deploy_metrics(y, score, threshold=0.5)

    assert result["false_positives"] == 4
    assert result["false_positive_events"] == 2

from tsad_transfer.retrospective_metrics import evaluate_retrospective_metrics


def test_retrospective_metrics_returns_tsb_ad_metric_set():
    y = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    score = np.array([0.1, 0.2, np.nan, 0.9, 0.8, 0.3, 0.2, 0.85, 0.75, 0.1])
    result = evaluate_retrospective_metrics(y, score, sliding_window=2)

    expected_metrics = {
        "AUC-ROC", "AUC-PR", "Point-F1", "Range-F1", "Point-F1-PA",
        "Event-F1-PA", "VUS-ROC", "VUS-PR", "scorable_fraction",
    }
    assert set(result) == expected_metrics
    assert all(np.isfinite(value) for value in result.values())
    assert result["scorable_fraction"] == 0.9
