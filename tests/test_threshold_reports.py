from __future__ import annotations

import numpy as np
import pytest

from cpg_vuln.training.thresholds import (
    metrics_at_validation_thresholds,
    select_validation_thresholds,
    selected_threshold_key,
)


def test_select_validation_thresholds_uses_only_validation_labels() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float32)

    thresholds = select_validation_thresholds(labels, probabilities)

    assert thresholds["fixed_0_5"] == 0.5
    assert thresholds["val_f1"] == pytest.approx(0.6)
    assert thresholds["val_mcc"] == pytest.approx(0.6)


def test_metrics_at_validation_thresholds_reuses_frozen_thresholds() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.2, 0.7, 0.8, 0.9], dtype=np.float32)
    thresholds = {"fixed_0_5": 0.5, "val_f1": 0.8, "val_mcc": 0.8}

    metrics = metrics_at_validation_thresholds(labels, probabilities, thresholds)

    assert set(metrics) == {"fixed_0_5", "val_f1", "val_mcc"}
    assert metrics["fixed_0_5"]["threshold"] == 0.5
    assert metrics["val_f1"]["threshold"] == 0.8
    assert metrics["val_mcc"]["threshold"] == 0.8
    assert metrics["fixed_0_5"]["predicted_positive_rate"] == 0.75
    assert metrics["val_f1"]["predicted_positive_rate"] == 0.5


def test_selected_threshold_key_rejects_unknown_strategy() -> None:
    assert selected_threshold_key("fixed_0_5") == "fixed_0_5"
    assert selected_threshold_key("val_f1") == "val_f1"
    assert selected_threshold_key("val_mcc") == "val_mcc"

    try:
        selected_threshold_key("test_f1")
    except ValueError as error:
        assert "unsupported threshold strategy" in str(error)
    else:
        raise AssertionError("expected ValueError")
