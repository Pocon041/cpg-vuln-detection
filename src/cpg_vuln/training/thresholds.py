from __future__ import annotations

import numpy as np

from cpg_vuln.training.metrics import (
    classification_metrics,
    select_f1_threshold,
    select_mcc_threshold,
)


THRESHOLD_KEYS = ("fixed_0_5", "val_f1", "val_mcc")


def select_validation_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "fixed_0_5": 0.5,
        "val_f1": select_f1_threshold(labels, probabilities),
        "val_mcc": select_mcc_threshold(labels, probabilities),
    }


def metrics_at_validation_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, dict[str, object]]:
    missing = set(THRESHOLD_KEYS) - set(thresholds)
    if missing:
        raise ValueError(f"missing threshold(s): {sorted(missing)}")
    return {
        name: classification_metrics(
            labels,
            probabilities,
            threshold=float(thresholds[name]),
        )
        for name in THRESHOLD_KEYS
    }


def selected_threshold_key(strategy: str) -> str:
    if strategy == "fixed_0_5":
        return "fixed_0_5"
    if strategy == "val_f1":
        return "val_f1"
    if strategy == "val_mcc":
        return "val_mcc"
    raise ValueError(f"unsupported threshold strategy: {strategy}")
