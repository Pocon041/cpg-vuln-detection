from __future__ import annotations

from torch_geometric.data import Data

from cpg_vuln.mining.hard_negative_bank import HardNegativePair
from cpg_vuln.mining.pair_dataset import PairBatchSampler, PairIndex


def test_pair_index_maps_sample_ids_to_dataset_indices() -> None:
    pairs = [
        HardNegativePair("p1", "n1", 0.8, 0.8, 0.8, 0.8, 0.2),
        HardNegativePair("p2", "n2", 0.7, 0.7, 0.7, 0.7, 0.3),
    ]
    index = PairIndex.from_pairs(pairs, {"p1": 0, "n1": 1, "p2": 2, "n2": 3})

    assert index.positive_indices == [0, 2]
    assert index.negative_indices == [1, 3]


def test_pair_batch_sampler_respects_pair_count() -> None:
    sampler = PairBatchSampler(pair_count=5, batch_size=2, shuffle=False, seed=42)

    assert list(iter(sampler)) == [[0, 1], [2, 3], [4]]


def test_pair_batch_sampler_respects_node_and_edge_budget() -> None:
    sampler = PairBatchSampler(
        pair_count=3,
        batch_size=3,
        shuffle=False,
        seed=42,
        pair_sizes=[(6, 10), (5, 10), (2, 4)],
        max_pair_nodes=8,
        max_pair_edges=20,
    )

    assert list(iter(sampler)) == [[0], [1, 2]]


def test_pair_batch_sampler_limits_replay_steps_per_epoch() -> None:
    sampler = PairBatchSampler(
        pair_count=5,
        batch_size=1,
        shuffle=False,
        seed=42,
        replay_steps_per_epoch=2,
    )

    assert list(iter(sampler)) == [[0], [1]]


def test_pair_index_returns_positive_and_negative_graph_lists() -> None:
    dataset = [Data(sample_id=f"s{i}") for i in range(4)]
    pairs = [HardNegativePair("s0", "s1", 0.8, 0.8, 0.8, 0.8, 0.2)]
    index = PairIndex.from_pairs(pairs, {"s0": 0, "s1": 1})

    positives, negatives = index.graphs_for(dataset, [0])

    assert positives[0].sample_id == "s0"
    assert negatives[0].sample_id == "s1"
