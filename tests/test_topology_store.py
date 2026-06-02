from __future__ import annotations

from pathlib import Path

import numpy as np

from cpg_vuln.data.dataset import TopologyDataset
from cpg_vuln.data.graphml import GraphMLParser, choose_primary_method
from cpg_vuln.data.store import (
    NodeTypeRegistry,
    build_topology_payload,
    load_topology,
    save_topology,
    topology_index_record,
)
from cpg_vuln.data.topology import build_view
from cpg_vuln.features.cache import MemmapFeatureCache
from cpg_vuln.features.text import NodeTextRegistry

from .helpers import write_graphml


def test_topology_payload_records_cache_schema_and_commit_id(tmp_path: Path) -> None:
    graph_path = tmp_path / "sample.graphml"
    write_graphml(graph_path)
    graph = GraphMLParser().parse(graph_path)
    topology = build_view(graph, choose_primary_method(graph), "core-cpg")

    payload = build_topology_payload(
        topology,
        "sample_1",
        1,
        NodeTextRegistry(),
        NodeTypeRegistry(),
        commit_id="commit-123",
    )

    assert payload["cache_schema_version"] == 1
    assert payload["commit_id"] == "commit-123"
    assert topology_index_record(tmp_path / "sample.pt", payload) == {
        "commit_id": "commit-123",
        "sample_id": "sample_1",
        "view": "core-cpg",
        "path": str((tmp_path / "sample.pt").resolve()),
        "nodes": len(topology.nodes),
        "edges": len(topology.edges),
    }


def test_save_topology_returns_index_record_with_commit_id(tmp_path: Path) -> None:
    graph_path = tmp_path / "sample.graphml"
    write_graphml(graph_path)
    graph = GraphMLParser().parse(graph_path)
    topology = build_view(graph, choose_primary_method(graph), "core-cpg")
    topology_path = tmp_path / "sample.pt"

    index_record = save_topology(
        topology_path,
        topology,
        "sample_1",
        1,
        NodeTextRegistry(),
        NodeTypeRegistry(),
        commit_id="commit-123",
    )

    assert index_record["commit_id"] == "commit-123"
    assert load_topology(topology_path)["commit_id"] == "commit-123"


def test_saved_topology_loads_as_pyg_data_with_external_feature_cache(tmp_path: Path) -> None:
    graph_path = tmp_path / "sample.graphml"
    write_graphml(graph_path)
    graph = GraphMLParser().parse(graph_path)
    topology = build_view(graph, choose_primary_method(graph), "core-cpg")
    texts = NodeTextRegistry()
    node_types = NodeTypeRegistry()
    topology_path = tmp_path / "sample.pt"
    save_topology(topology_path, topology, "sample_1", 1, texts, node_types)
    features = MemmapFeatureCache.create(tmp_path / "features", rows=len(texts), dim=4)
    for index in range(len(texts)):
        features.write([index], np.full((1, 4), index, dtype=np.float32))

    dataset = TopologyDataset([topology_path], node_features=features)
    data = dataset[0]

    assert data.sample_id == "sample_1"
    assert data.y.tolist() == [1]
    assert data.x.shape == (4, 4)
    assert data.edge_index.shape[0] == 2
    assert data.edge_type.shape[0] == data.edge_index.shape[1]
