from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Sequence

from torch_geometric.data import Data

from cpg_vuln.mining.hard_negative_bank import HardNegativePair


@dataclass(frozen=True)
class PairIndex:
    positive_indices: list[int]
    negative_indices: list[int]

    @classmethod
    def from_pairs(cls, pairs: list[HardNegativePair], sample_to_index: dict[str, int]) -> "PairIndex":
        return cls(
            positive_indices=[sample_to_index[pair.positive_id] for pair in pairs],
            negative_indices=[sample_to_index[pair.negative_id] for pair in pairs],
        )

    def __len__(self) -> int:
        return len(self.positive_indices)

    def graphs_for(self, dataset: Sequence[Data], pair_indices: list[int]) -> tuple[list[Data], list[Data]]:
        positives = [dataset[self.positive_indices[index]] for index in pair_indices]
        negatives = [dataset[self.negative_indices[index]] for index in pair_indices]
        return positives, negatives


class PairBatchSampler:
    def __init__(
        self,
        *,
        pair_count: int,
        batch_size: int,
        shuffle: bool,
        seed: int,
        pair_sizes: list[tuple[int, int]] | None = None,
        max_pair_nodes: int | None = None,
        max_pair_edges: int | None = None,
        replay_steps_per_epoch: int | None = None,
    ) -> None:
        self.pair_count = pair_count
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.pair_sizes = pair_sizes
        self.max_pair_nodes = max_pair_nodes
        self.max_pair_edges = max_pair_edges
        self.replay_steps_per_epoch = replay_steps_per_epoch
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(self.pair_count))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        emitted = 0
        batch: list[int] = []
        batch_nodes = 0
        batch_edges = 0
        for index in indices:
            pair_nodes, pair_edges = self._pair_size(index)
            would_exceed_count = len(batch) >= self.batch_size
            would_exceed_nodes = (
                self.max_pair_nodes is not None
                and batch
                and batch_nodes + pair_nodes > self.max_pair_nodes
            )
            would_exceed_edges = (
                self.max_pair_edges is not None
                and batch
                and batch_edges + pair_edges > self.max_pair_edges
            )
            if would_exceed_count or would_exceed_nodes or would_exceed_edges:
                yield batch
                emitted += 1
                if self.replay_steps_per_epoch is not None and emitted >= self.replay_steps_per_epoch:
                    return
                batch = []
                batch_nodes = 0
                batch_edges = 0
            batch.append(index)
            batch_nodes += pair_nodes
            batch_edges += pair_edges
        if batch and (
            self.replay_steps_per_epoch is None or emitted < self.replay_steps_per_epoch
        ):
            yield batch

    def _pair_size(self, index: int) -> tuple[int, int]:
        if self.pair_sizes is None:
            return 0, 0
        return self.pair_sizes[index]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
