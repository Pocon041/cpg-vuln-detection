from __future__ import annotations

import numpy as np

from cpg_vuln.training.metrics import classification_metrics, select_f1_threshold


def test_select_threshold_uses_validation_f1() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])

    threshold = select_f1_threshold(labels, probabilities)
    metrics = classification_metrics(labels, probabilities, threshold=threshold)

    assert threshold == 0.6
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0

