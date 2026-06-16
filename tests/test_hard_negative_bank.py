from __future__ import annotations

import numpy as np

from cpg_vuln.mining.hard_negative_bank import (
    HardNegativeConfig,
    MiningStrategy,
    build_hard_negative_bank,
    cosine_similarity,
    scale_similarity,
)


def test_cosine_similarity_handles_zero_vectors() -> None:
    assert cosine_similarity(np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0])) == 0.0
    assert cosine_similarity(np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])) == 1.0


def test_scale_similarity_uses_distance_not_cosine_angle() -> None:
    assert scale_similarity(np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0])) == 1.0
    assert scale_similarity(np.asarray([1.0, 1.0]), np.asarray([4.0, 4.0])) < 1.0


def test_build_bank_uses_only_dataset_labeled_negatives() -> None:
    from cpg_vuln.mining.motif import motif_feature_names

    names = motif_feature_names()

    def vector(**values: float) -> np.ndarray:
        result = np.zeros(len(names), dtype=np.float32)
        for name, value in values.items():
            result[names.index(name)] = value
        return result

    sample_ids = ["p1", "n1", "n2", "p2"]
    labels = {"p1": 1, "n1": 0, "n2": 0, "p2": 1}
    vectors = {
        "p1": vector(known_alloc_api=1.0, num_ast_edges=1.0, num_nodes=1.0),
        "n1": vector(known_alloc_api=1.0, num_ast_edges=0.9, num_nodes=1.0),
        "n2": vector(known_copy_api=1.0, num_cfg_edges=1.0, num_nodes=1.0),
        "p2": vector(known_copy_api=1.0, num_cfg_edges=0.9, num_nodes=1.0),
    }
    fp_probs = {"n1": 0.8, "n2": 0.1}

    pairs, review = build_hard_negative_bank(
        sample_ids=sample_ids,
        labels=labels,
        retrieval_vectors=vectors,
        false_positive_probabilities=fp_probs,
        config=HardNegativeConfig(
            strategy=MiningStrategy.MOTIF_MATCHED,
            max_pairs_per_positive=1,
            minimum_pair_score=0.1,
            lower_percentile=0.0,
            upper_percentile=1.0,
        ),
    )

    assert [(pair.positive_id, pair.negative_id) for pair in pairs] == [("p1", "n1"), ("p2", "n2")]
    assert all(labels[pair.negative_id] == 0 for pair in pairs)
    assert isinstance(review, list)


def test_random_strategy_ignores_similarity_scores() -> None:
    sample_ids = ["p1", "n1", "n2"]
    labels = {"p1": 1, "n1": 0, "n2": 0}
    vectors = {
        "p1": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        "n1": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        "n2": np.asarray([0.0, 1.0, 4.0], dtype=np.float32),
    }

    pairs, _ = build_hard_negative_bank(
        sample_ids=sample_ids,
        labels=labels,
        retrieval_vectors=vectors,
        false_positive_probabilities={"n1": 1.0, "n2": 0.0},
        config=HardNegativeConfig(
            strategy=MiningStrategy.RANDOM,
            max_pairs_per_positive=2,
            minimum_pair_score=0.0,
            seed=7,
        ),
    )

    assert len(pairs) == 2
    assert {pair.negative_id for pair in pairs} == {"n1", "n2"}


def test_false_positive_strategy_keeps_high_fp_negatives() -> None:
    sample_ids = ["p1", "p2", "n-low", "n-mid", "n-high"]
    labels = {"p1": 1, "p2": 1, "n-low": 0, "n-mid": 0, "n-high": 0}
    vectors = {sample_id: np.zeros(1, dtype=np.float32) for sample_id in sample_ids}

    pairs, _ = build_hard_negative_bank(
        sample_ids=sample_ids,
        labels=labels,
        retrieval_vectors=vectors,
        false_positive_probabilities={"n-low": 0.1, "n-mid": 0.4, "n-high": 0.9},
        config=HardNegativeConfig(
            strategy=MiningStrategy.FALSE_POSITIVE_ONLY,
            max_pairs_per_positive=1,
            fp_top_fraction=0.34,
            seed=3,
        ),
    )

    assert {pair.negative_id for pair in pairs} == {"n-high"}
    assert all(pair.false_positive_probability == 0.9 for pair in pairs)


def test_pair_score_uses_distinct_feature_groups() -> None:
    from cpg_vuln.mining.motif import motif_feature_names

    names = motif_feature_names()

    def vector(**values: float) -> np.ndarray:
        result = np.zeros(len(names), dtype=np.float32)
        for name, value in values.items():
            result[names.index(name)] = value
        return result

    pairs, _ = build_hard_negative_bank(
        sample_ids=["p", "n"],
        labels={"p": 1, "n": 0},
        retrieval_vectors={
            "p": vector(known_alloc_api=1.0, num_ast_edges=1.0, num_nodes=1.0),
            "n": vector(known_alloc_api=1.0, num_cfg_edges=1.0, num_nodes=4.0),
        },
        false_positive_probabilities={"n": 0.0},
        config=HardNegativeConfig(
            strategy=MiningStrategy.MOTIF_MATCHED,
            motif_weight=1.0,
            structure_weight=1.0,
            scale_weight=1.0,
            false_positive_weight=0.0,
            max_pairs_per_positive=1,
            minimum_pair_score=0.0,
            lower_percentile=0.0,
            upper_percentile=1.0,
        ),
    )

    pair = pairs[0]
    assert pair.motif_similarity == 1.0
    assert pair.structure_similarity == 0.0
    assert pair.scale_similarity < 1.0


def test_hard_negative_pair_serializes_all_audit_fields() -> None:
    from cpg_vuln.mining.hard_negative_bank import HardNegativePair

    payload = HardNegativePair("p", "n", 0.7, 0.8, 0.6, 0.9, 0.4).to_dict()

    assert payload == {
        "positive_id": "p",
        "negative_id": "n",
        "pair_score": 0.7,
        "motif_similarity": 0.8,
        "structure_similarity": 0.6,
        "scale_similarity": 0.9,
        "false_positive_probability": 0.4,
    }


def test_pair_audit_sample_reports_read_and_write_progress(tmp_path, monkeypatch) -> None:
    import json

    import cpg_vuln.mining.hard_negative_bank as bank

    calls: list[dict[str, object]] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    bank_path = tmp_path / "bank.jsonl"
    output_path = tmp_path / "audit.json"
    bank_path.write_text(
        "\n".join(
            json.dumps({"positive_id": f"p{index}", "negative_id": f"n{index}"})
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bank, "tqdm", fake_tqdm)

    bank.write_pair_audit_sample(bank_path, output_path, limit=2, seed=1)

    assert [call["desc"] for call in calls] == ["read hard pairs", "write pair audit sample"]
