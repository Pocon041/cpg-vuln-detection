from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from torch.utils.data import Sampler


@dataclass(frozen=True)
class GraphSize:
    sample_id: str
    nodes: int
    edges: int


class DynamicBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        sizes: list[GraphSize],
        *,
        max_nodes: int,
        max_edges: int,
        shuffle: bool,
        seed: int = 42,
    ) -> None:
        self.sizes = sizes
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self.sizes)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        batch: list[int] = []
        nodes = 0
        edges = 0
        for index in indices:
            size = self.sizes[index]
            oversized = size.nodes > self.max_nodes or size.edges > self.max_edges
            exceeds_batch = nodes + size.nodes > self.max_nodes or edges + size.edges > self.max_edges
            if batch and (oversized or exceeds_batch):
                yield batch
                batch, nodes, edges = [], 0, 0
            if oversized:
                yield [index]
                continue
            batch.append(index)
            nodes += size.nodes
            edges += size.edges
        if batch:
            yield batch

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

