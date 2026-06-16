from __future__ import annotations

import numpy as np

from cpg_vuln.mining.hard_negative_bank import HardNegativePair


def pair_ranking_accuracy(pairs: list[HardNegativePair], risk_logits: dict[str, float]) -> float:
    if not pairs:
        return 0.0
    correct = sum(
        float(risk_logits[pair.positive_id]) > float(risk_logits[pair.negative_id])
        for pair in pairs
    )
    return correct / len(pairs)


def matched_negative_fpr(
    pairs: list[HardNegativePair],
    probabilities: dict[str, float],
    *,
    threshold: float,
) -> float:
    negative_ids = sorted({pair.negative_id for pair in pairs})
    if not negative_ids:
        return 0.0
    false_positives = sum(float(probabilities[sample_id]) >= threshold for sample_id in negative_ids)
    return false_positives / len(negative_ids)


def average_risk_logit_gap(pairs: list[HardNegativePair], risk_logits: dict[str, float]) -> float:
    if not pairs:
        return 0.0
    gaps = [
        float(risk_logits[pair.positive_id]) - float(risk_logits[pair.negative_id])
        for pair in pairs
    ]
    return float(np.mean(gaps))


def pair_coverage(pairs: list[HardNegativePair], *, positive_ids: list[str]) -> float:
    if not positive_ids:
        return 0.0
    covered = {pair.positive_id for pair in pairs}
    return len(covered & set(positive_ids)) / len(set(positive_ids))


def average_pairs_per_positive(pairs: list[HardNegativePair]) -> float:
    if not pairs:
        return 0.0
    positives = {pair.positive_id for pair in pairs}
    return len(pairs) / len(positives)
