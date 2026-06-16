from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
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


def select_mcc_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = sorted(set(float(value) for value in probabilities))
    if not candidates:
        return 0.5
    return max(
        candidates,
        key=lambda threshold: (
            matthews_corrcoef(labels, probabilities >= threshold),
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
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, _fn, _tp = matrix.ravel()
    specificity = 0.0 if tn + fp == 0 else tn / (tn + fp)
    return {
        "samples": int(labels.size),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "predicted_positive_rate": float(predictions.mean()) if predictions.size else 0.0,
        "roc_auc": _safe_score(roc_auc_score, labels, probabilities),
        "pr_auc": _safe_score(average_precision_score, labels, probabilities),
        "confusion_matrix": matrix.tolist(),
    }


def _safe_score(function, labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    try:
        value = float(function(labels, probabilities))
    except ValueError:
        return None
    return None if math.isnan(value) else value
