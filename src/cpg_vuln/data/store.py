from __future__ import annotations

import json
import uuid
from pathlib import Path

import torch

from cpg_vuln.data.topology import GraphTopology
from cpg_vuln.features.text import NodeTextRegistry, normalize_node_text
from cpg_vuln.utils.fingerprint import write_json_atomic


TOPOLOGY_CACHE_SCHEMA_VERSION = 1


class NodeTypeRegistry:
    def __init__(self, values: list[str] | None = None) -> None:
        self.values = list(values or [])
        self._ids = {value: index for index, value in enumerate(self.values)}

    def __len__(self) -> int:
        return len(self.values)

    def add(self, value: str) -> int:
        existing = self._ids.get(value)
        if existing is not None:
            return existing
        index = len(self.values)
        self.values.append(value)
        self._ids[value] = index
        return index

    def write(self, path: Path) -> None:
        write_json_atomic(path, self.values)

    @classmethod
    def read(cls, path: Path) -> "NodeTypeRegistry":
        if not path.is_file():
            return cls()
        return cls(json.loads(path.read_text(encoding="utf-8")))


def save_topology(
    path: Path,
    topology: GraphTopology,
    sample_id: str,
    label: int,
    texts: NodeTextRegistry,
    node_types: NodeTypeRegistry,
    *,
    commit_id: str | None = None,
) -> dict[str, object]:
    commit_id = commit_id if commit_id is not None else uuid.uuid4().hex
    payload = build_topology_payload(
        topology,
        sample_id,
        label,
        texts,
        node_types,
        commit_id=commit_id,
    )
    save_topology_payload(path, payload)
    return topology_index_record(path, payload)


def build_topology_payload(
    topology: GraphTopology,
    sample_id: str,
    label: int,
    texts: NodeTextRegistry,
    node_types: NodeTypeRegistry,
    *,
    commit_id: str,
) -> dict[str, object]:
    edge_index = torch.tensor(topology.edges, dtype=torch.long)
    if edge_index.numel():
        edge_index = edge_index.t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    payload: dict[str, object] = {
        "cache_schema_version": TOPOLOGY_CACHE_SCHEMA_VERSION,
        "commit_id": commit_id,
        "sample_id": sample_id,
        "y": torch.tensor([label], dtype=torch.long),
        "view": topology.view,
        "edge_index": edge_index,
        "text_id": torch.tensor(
            [texts.add(normalize_node_text(node)) for node in topology.nodes],
            dtype=torch.long,
        ),
        "node_type_id": torch.tensor(
            [node_types.add(node.label) for node in topology.nodes],
            dtype=torch.long,
        ),
        "original_node_ids": topology.original_node_ids,
        "line_numbers": torch.tensor(
            [_line_number(node.attrs.get("LINE_NUMBER")) for node in topology.nodes],
            dtype=torch.long,
        ),
        "code_summaries": [
            node.attrs.get("CODE", node.attrs.get("NAME", node.label))[:240]
            for node in topology.nodes
        ],
        "relation_names": sorted(topology.relation_names),
        "edge_type_names": topology.edge_type_names,
    }
    if topology.edge_types:
        payload["edge_type"] = torch.tensor(topology.edge_types, dtype=torch.long)
    return payload


def save_topology_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def topology_index_record(
    path: Path,
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        "commit_id": str(payload["commit_id"]),
        "sample_id": str(payload["sample_id"]),
        "view": str(payload["view"]),
        "path": str(path.resolve()),
        "nodes": int(payload["text_id"].numel()),
        "edges": int(payload["edge_index"].shape[1]),
    }


def load_topology(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _line_number(value: str | None) -> int:
    try:
        return int(value) if value else -1
    except ValueError:
        return -1
