from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from cpg_vuln.data.batch import GraphSize
from cpg_vuln.data.store import load_topology
from cpg_vuln.features.cache import MemmapFeatureCache

if TYPE_CHECKING:
    from torch_geometric.data import Data


class TopologyDataset:
    def __init__(
        self,
        topology_paths: list[Path],
        *,
        node_features: MemmapFeatureCache,
        function_features: MemmapFeatureCache | None = None,
        function_indices: dict[str, int] | None = None,
        graph_sizes: list[GraphSize] | None = None,
    ) -> None:
        self.topology_paths = topology_paths
        self.node_features = node_features
        self.function_features = function_features
        self.function_indices = function_indices or {}
        self.graph_sizes = graph_sizes

    def __len__(self) -> int:
        return len(self.topology_paths)

    def __getitem__(self, index: int) -> "Data":
        from torch_geometric.data import Data

        payload = load_topology(self.topology_paths[index])
        text_ids = payload["text_id"]
        data = Data(
            x=torch.from_numpy(self.node_features.read(text_ids.numpy())).float(),
            edge_index=payload["edge_index"],
            y=payload["y"],
            node_type_id=payload["node_type_id"],
            line_numbers=payload["line_numbers"],
            sample_id=payload["sample_id"],
        )
        if "edge_type" in payload:
            data.edge_type = payload["edge_type"]
        if self.function_features is not None:
            function_index = self.function_indices[payload["sample_id"]]
            data.function_x = torch.from_numpy(
                self.function_features.read([function_index])
            ).float()
        return data
