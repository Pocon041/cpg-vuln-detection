from __future__ import annotations

import numpy as np

from cpg_vuln.evaluation.ramp_metrics import (
    average_pairs_per_positive,
    average_risk_logit_gap,
    matched_negative_fpr,
    pair_coverage,
    pair_ranking_accuracy,
)
from cpg_vuln.mining.hard_negative_bank import HardNegativePair


def _pairs() -> list[HardNegativePair]:
    return [
        HardNegativePair("p1", "n1", 0.9, 0.9, 0.8, 0.7, 0.6),
        HardNegativePair("p1", "n2", 0.8, 0.8, 0.7, 0.6, 0.5),
        HardNegativePair("p2", "n3", 0.7, 0.7, 0.6, 0.5, 0.4),
    ]


def test_pair_level_ramp_metrics() -> None:
    risk_logits = {"p1": 2.0, "p2": 0.5, "n1": 1.0, "n2": 3.0, "n3": 0.1}
    probabilities = {"n1": 0.6, "n2": 0.4, "n3": 0.8}

    assert pair_ranking_accuracy(_pairs(), risk_logits) == 2 / 3
    assert matched_negative_fpr(_pairs(), probabilities, threshold=0.5) == 2 / 3
    assert average_risk_logit_gap(_pairs(), risk_logits) == np.mean([1.0, -1.0, 0.4])
    assert pair_coverage(_pairs(), positive_ids=["p1", "p2", "p3"]) == 2 / 3
    assert average_pairs_per_positive(_pairs()) == 1.5
