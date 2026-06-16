from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cpg_vuln.mining.motif import motif_feature_groups, motif_feature_names


class MiningStrategy:
    RANDOM = "random"
    FALSE_POSITIVE_ONLY = "false_positive_only"
    MOTIF_MATCHED = "motif_matched"


@dataclass(frozen=True)
class HardNegativeConfig:
    strategy: str = MiningStrategy.MOTIF_MATCHED
    max_pairs_per_positive: int = 3
    minimum_pair_score: float = 0.2
    lower_percentile: float = 0.05
    upper_percentile: float = 0.30
    motif_weight: float = 0.40
    structure_weight: float = 0.30
    scale_weight: float = 0.15
    false_positive_weight: float = 0.15
    fp_top_fraction: float = 0.10
    review_percentile: float = 0.05
    seed: int = 42


@dataclass(frozen=True)
class HardNegativePair:
    positive_id: str
    negative_id: str
    pair_score: float
    motif_similarity: float
    structure_similarity: float
    scale_similarity: float
    false_positive_probability: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HardNegativePair":
        return cls(
            positive_id=str(payload["positive_id"]),
            negative_id=str(payload["negative_id"]),
            pair_score=float(payload["pair_score"]),
            motif_similarity=float(payload["motif_similarity"]),
            structure_similarity=float(payload["structure_similarity"]),
            scale_similarity=float(payload["scale_similarity"]),
            false_positive_probability=float(payload["false_positive_probability"]),
        )


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def scale_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(1.0 / (1.0 + np.mean(np.abs(left - right))))


def select_feature_group(vector: np.ndarray, group: str) -> np.ndarray:
    names = motif_feature_names()
    groups = motif_feature_groups()
    indices = [names.index(name) for name in groups[group]]
    return vector[indices]


def build_hard_negative_bank(
    *,
    sample_ids: list[str],
    labels: dict[str, int],
    retrieval_vectors: dict[str, np.ndarray],
    false_positive_probabilities: dict[str, float],
    config: HardNegativeConfig,
) -> tuple[list[HardNegativePair], list[HardNegativePair]]:
    if config.strategy == MiningStrategy.RANDOM:
        return _build_random_negative_bank(
            sample_ids=sample_ids,
            labels=labels,
            false_positive_probabilities=false_positive_probabilities,
            config=config,
        )
    if config.strategy == MiningStrategy.FALSE_POSITIVE_ONLY:
        return _build_false_positive_negative_bank(
            sample_ids=sample_ids,
            labels=labels,
            false_positive_probabilities=false_positive_probabilities,
            config=config,
        )
    positive_ids = [sample_id for sample_id in sample_ids if labels[sample_id] == 1]
    negative_ids = [sample_id for sample_id in sample_ids if labels[sample_id] == 0]
    pairs: list[HardNegativePair] = []
    review: list[HardNegativePair] = []
    for positive_id in tqdm(
        positive_ids,
        desc=f"mine {config.strategy} hard negatives",
        unit="positive",
        ascii=True,
    ):
        candidates = [
            _score_pair(
                positive_id,
                negative_id,
                retrieval_vectors,
                false_positive_probabilities,
                config,
            )
            for negative_id in negative_ids
        ]
        candidates = [
            candidate for candidate in candidates if candidate.pair_score >= config.minimum_pair_score
        ]
        candidates.sort(key=lambda item: item.pair_score, reverse=True)
        if not candidates:
            continue
        review_cutoff = max(1, int(len(candidates) * config.review_percentile))
        review.extend(candidates[:review_cutoff])
        start = int(len(candidates) * config.lower_percentile)
        end = max(start + 1, int(len(candidates) * config.upper_percentile))
        semi_hard = candidates[start:end]
        pairs.extend(semi_hard[: config.max_pairs_per_positive])
    return pairs, review


def write_pair_audit_sample(
    bank_path: Path,
    output_path: Path,
    *,
    limit: int = 100,
    seed: int = 42,
) -> None:
    rows = [
        json.loads(line)
        for line in tqdm(
            bank_path.read_text(encoding="utf-8").splitlines(),
            desc="read hard pairs",
            unit="pair",
            ascii=True,
        )
        if line.strip()
    ]
    random.Random(seed).shuffle(rows)
    selected = list(
        tqdm(
            rows[:limit],
            desc="write pair audit sample",
            unit="pair",
            ascii=True,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")


def _score_pair(
    positive_id: str,
    negative_id: str,
    retrieval_vectors: dict[str, np.ndarray],
    false_positive_probabilities: dict[str, float],
    config: HardNegativeConfig,
) -> HardNegativePair:
    positive = retrieval_vectors[positive_id]
    negative = retrieval_vectors[negative_id]
    motif_similarity = cosine_similarity(
        select_feature_group(positive, "motif"),
        select_feature_group(negative, "motif"),
    )
    structure_similarity = cosine_similarity(
        select_feature_group(positive, "structure"),
        select_feature_group(negative, "structure"),
    )
    scale_similarity_value = scale_similarity(
        select_feature_group(positive, "scale"),
        select_feature_group(negative, "scale"),
    )
    false_positive_probability = float(false_positive_probabilities.get(negative_id, 0.0))
    pair_score = (
        config.motif_weight * motif_similarity
        + config.structure_weight * structure_similarity
        + config.scale_weight * scale_similarity_value
        + config.false_positive_weight * false_positive_probability
    )
    return HardNegativePair(
        positive_id=positive_id,
        negative_id=negative_id,
        pair_score=float(pair_score),
        motif_similarity=float(motif_similarity),
        structure_similarity=float(structure_similarity),
        scale_similarity=float(scale_similarity_value),
        false_positive_probability=false_positive_probability,
    )


def _build_random_negative_bank(
    *,
    sample_ids: list[str],
    labels: dict[str, int],
    false_positive_probabilities: dict[str, float],
    config: HardNegativeConfig,
) -> tuple[list[HardNegativePair], list[HardNegativePair]]:
    import random

    rng = random.Random(config.seed)
    positive_ids = [sample_id for sample_id in sample_ids if labels[sample_id] == 1]
    negative_ids = [sample_id for sample_id in sample_ids if labels[sample_id] == 0]
    pairs: list[HardNegativePair] = []
    for positive_id in tqdm(
        positive_ids,
        desc="mine random hard negatives",
        unit="positive",
        ascii=True,
    ):
        selected = rng.sample(negative_ids, k=min(config.max_pairs_per_positive, len(negative_ids)))
        for negative_id in selected:
            pairs.append(
                HardNegativePair(
                    positive_id=positive_id,
                    negative_id=negative_id,
                    pair_score=0.0,
                    motif_similarity=0.0,
                    structure_similarity=0.0,
                    scale_similarity=0.0,
                    false_positive_probability=float(false_positive_probabilities.get(negative_id, 0.0)),
                )
            )
    return pairs, []


def _build_false_positive_negative_bank(
    *,
    sample_ids: list[str],
    labels: dict[str, int],
    false_positive_probabilities: dict[str, float],
    config: HardNegativeConfig,
) -> tuple[list[HardNegativePair], list[HardNegativePair]]:
    import random

    rng = random.Random(config.seed)
    positive_ids = [sample_id for sample_id in sample_ids if labels[sample_id] == 1]
    negative_ids = [sample_id for sample_id in sample_ids if labels[sample_id] == 0]
    ranked_negatives = sorted(
        negative_ids,
        key=lambda sample_id: (
            float(false_positive_probabilities.get(sample_id, 0.0)),
            sample_id,
        ),
        reverse=True,
    )
    pool_size = max(1, int(len(ranked_negatives) * config.fp_top_fraction))
    pool = ranked_negatives[:pool_size]
    pairs: list[HardNegativePair] = []
    for positive_id in tqdm(
        positive_ids,
        desc="mine false-positive hard negatives",
        unit="positive",
        ascii=True,
    ):
        selected = rng.sample(pool, k=min(config.max_pairs_per_positive, len(pool)))
        for negative_id in selected:
            probability = float(false_positive_probabilities.get(negative_id, 0.0))
            pairs.append(
                HardNegativePair(
                    positive_id=positive_id,
                    negative_id=negative_id,
                    pair_score=probability,
                    motif_similarity=0.0,
                    structure_similarity=0.0,
                    scale_similarity=0.0,
                    false_positive_probability=probability,
                )
            )
    return pairs, []
