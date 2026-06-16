from __future__ import annotations

import numpy as np

from cpg_vuln.training.metrics import (
    classification_metrics,
    select_f1_threshold,
    select_mcc_threshold,
)


def test_select_threshold_uses_validation_f1() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])

    threshold = select_f1_threshold(labels, probabilities)
    metrics = classification_metrics(labels, probabilities, threshold=threshold)

    assert threshold == 0.6
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_classification_metrics_include_false_positive_diagnostics() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.2, 0.8, 0.7, 0.9])

    metrics = classification_metrics(labels, probabilities, threshold=0.5)

    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["specificity"] == 0.5
    assert metrics["predicted_positive_rate"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["mcc"] > 0.0


def test_select_mcc_threshold_prefers_balanced_decision_boundary() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.2, 0.4, 0.6, 0.8])

    threshold = select_mcc_threshold(labels, probabilities)

    assert threshold == 0.6
