from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = sorted(set(float(value) for value in probabilities))
    if not candidates:
        return 0.5
    return max(
        candidates,
        key=lambda threshold: (
            f1_score(labels, probabilities >= threshold, zero_division=0),
            threshold,
        ),
    )


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "samples": int(labels.size),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": _safe_score(roc_auc_score, labels, probabilities),
        "pr_auc": _safe_score(average_precision_score, labels, probabilities),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def _safe_score(function, labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    try:
        value = float(function(labels, probabilities))
    except ValueError:
        return None
    return None if math.isnan(value) else value

