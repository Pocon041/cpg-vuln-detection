from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from cpg_vuln.data.batch import GraphSize
from cpg_vuln.data.store import load_topology
from cpg_vuln.features.cache import MemmapFeatureCache

if TYPE_CHECKING:
    from torch_geometric.data import Data


SLICE_SEED_NONE = 0
SLICE_SEED_RISKY_CALL = 1
SLICE_SEED_MEMORY_ACCESS = 2
SLICE_SEED_GUARD = 3
SLICE_SEED_DATAFLOW = 4

_RISKY_CALL_TOKENS = {
    "alloc",
    "calloc",
    "free",
    "gets",
    "malloc",
    "memcpy",
    "memmove",
    "read",
    "recv",
    "realloc",
    "scanf",
    "sprintf",
    "strcat",
    "strcpy",
    "strlen",
}
_GUARD_CONTROL_TYPES = {"IF", "SWITCH", "FOR", "WHILE", "DO"}
_DATAFLOW_LABELS = {"IDENTIFIER", "METHOD_PARAMETER_IN", "LOCAL", "RETURN"}
_MEMORY_ACCESS_LABELS = {"CALL", "IDENTIFIER", "FIELD_IDENTIFIER", "LOCAL"}


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
        seed_type_id = _slice_seed_type_ids(payload)
        data.slice_seed_type_id = seed_type_id
        data.slice_seed_mask = seed_type_id.ne(SLICE_SEED_NONE)
        if self.function_features is not None:
            function_index = self.function_indices[payload["sample_id"]]
            data.function_x = torch.from_numpy(
                self.function_features.read([function_index])
            ).float()
        return data


def _slice_seed_type_ids(payload: dict[str, object]) -> torch.Tensor:
    labels = [str(value) for value in payload.get("node_labels", [])]
    names = [str(value).lower() for value in payload.get("node_names", [])]
    method_names = [
        str(value).rsplit(".", 1)[-1].lower()
        for value in payload.get("method_full_names", [])
    ]
    codes = [str(value).lower() for value in payload.get("code_summaries", [])]
    control_types = [
        str(value).upper() for value in payload.get("control_structure_types", [])
    ]
    count = len(labels)
    result = torch.zeros(count, dtype=torch.long)
    for index in range(count):
        label = labels[index]
        code = codes[index] if index < len(codes) else ""
        name = names[index] if index < len(names) else ""
        method_name = method_names[index] if index < len(method_names) else ""
        control_type = control_types[index] if index < len(control_types) else ""
        combined = f"{name} {method_name} {code}"
        if label == "CALL" and _is_risky_call(combined):
            result[index] = SLICE_SEED_RISKY_CALL
        elif label == "CONTROL_STRUCTURE" or control_type in _GUARD_CONTROL_TYPES:
            result[index] = SLICE_SEED_GUARD
        elif label in _MEMORY_ACCESS_LABELS and _is_memory_access(combined):
            result[index] = SLICE_SEED_MEMORY_ACCESS
        elif label in _DATAFLOW_LABELS and _is_dataflow_anchor(combined):
            result[index] = SLICE_SEED_DATAFLOW
    return result


def _is_risky_call(text: str) -> bool:
    return any(token in text for token in _RISKY_CALL_TOKENS)


def _is_memory_access(text: str) -> bool:
    return any(token in text for token in ("[", "->", "*", "<operator>.indexaccess", "<operator>.fieldaccess"))


def _is_dataflow_anchor(text: str) -> bool:
    return any(token in text for token in ("len", "size", "count", "src", "dst", "buf", "ptr"))
