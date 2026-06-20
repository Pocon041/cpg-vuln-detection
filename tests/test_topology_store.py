from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cpg_vuln.data.graphml import GraphEdge, GraphMLParser, GraphNode, ParsedGraph, choose_primary_method
from cpg_vuln.data.store import (
    NodeTypeRegistry,
    TOPOLOGY_CACHE_SCHEMA_VERSION,
    build_topology_payload,
    load_topology,
    save_topology,
    save_topology_payload,
    topology_index_record,
)
from cpg_vuln.data.topology import GraphTopology, build_view
from cpg_vuln.features.cache import MemmapFeatureCache
from cpg_vuln.features.normalization import (
    IdentifierSemanticNormalizer,
    NormalizationSpec,
    build_scope_context,
)
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

    assert payload["cache_schema_version"] == TOPOLOGY_CACHE_SCHEMA_VERSION
    assert payload["commit_id"] == "commit-123"
    assert topology_index_record(tmp_path / "sample.pt", payload) == {
        "commit_id": "commit-123",
        "sample_id": "sample_1",
        "view": "core-cpg",
        "path": str((tmp_path / "sample.pt").resolve()),
        "nodes": len(topology.nodes),
        "edges": len(topology.edges),
    }


def test_topology_payload_records_slice_node_mask() -> None:
    topology = GraphTopology(
        view="core-cpg",
        original_node_ids=["1", "2", "3"],
        nodes=[
            GraphNode("1", "METHOD", {"NAME": "f"}),
            GraphNode("2", "CALL", {"CODE": "strcpy(dst, src)"}),
            GraphNode("3", "CALL", {"CODE": "log_event()"}),
        ],
        edges=[],
        relation_names=set(),
        edge_types=[],
        edge_type_names={},
    )

    payload = build_topology_payload(
        topology,
        "sample_1",
        1,
        NodeTextRegistry(),
        NodeTypeRegistry(),
        commit_id="commit-123",
        slice_node_ids={"1", "2"},
    )

    assert payload["slice_node_mask"].tolist() == [True, True, False]


def test_topology_payload_keeps_structured_node_fields() -> None:
    topology = GraphTopology(
        view="core-cpg",
        original_node_ids=["1", "2", "3"],
        nodes=[
            GraphNode("1", "METHOD", {"NAME": "f", "CODE": "int f()", "FULL_NAME": "f"}),
            GraphNode("2", "CALL", {"NAME": "av_malloc", "METHOD_FULL_NAME": "av_malloc"}),
            GraphNode("3", "CONTROL_STRUCTURE", {"CONTROL_STRUCTURE_TYPE": "FOR"}),
        ],
        edges=[(0, 1), (1, 2)],
        relation_names={"AST", "CFG"},
        edge_types=[0, 1],
        edge_type_names={"AST": 0, "CFG": 1},
    )
    texts = NodeTextRegistry()
    node_types = NodeTypeRegistry()

    payload = build_topology_payload(topology, "sample-1", 1, texts, node_types, commit_id="c1")

    assert payload["node_labels"] == ["METHOD", "CALL", "CONTROL_STRUCTURE"]
    assert payload["node_names"] == ["f", "av_malloc", ""]
    assert payload["method_full_names"] == ["", "av_malloc", ""]
    assert payload["control_structure_types"] == ["", "", "FOR"]
    assert payload["node_type_histogram"]["CALL"] == 1
    assert payload["edge_type_histogram"]["AST"] == 1
    assert payload["edge_type_histogram"]["CFG"] == 1


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
    pytest.importorskip("torch_geometric")
    from cpg_vuln.data.dataset import TopologyDataset

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


def test_topology_dataset_exposes_slice_seed_metadata(tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    from cpg_vuln.data.dataset import (
        SLICE_SEED_MEMORY_ACCESS,
        SLICE_SEED_RISKY_CALL,
        TopologyDataset,
    )

    topology = GraphTopology(
        view="core-cpg",
        original_node_ids=["1", "2", "3", "4"],
        nodes=[
            GraphNode("1", "METHOD", {"NAME": "f", "CODE": "int f(char *src)"}),
            GraphNode("2", "CONTROL_STRUCTURE", {"CONTROL_STRUCTURE_TYPE": "IF", "CODE": "if (len < sizeof(buf))"}),
            GraphNode("3", "CALL", {"NAME": "memcpy", "CODE": "memcpy(buf, src, len)"}),
            GraphNode("4", "IDENTIFIER", {"CODE": "buf[i]"}),
        ],
        edges=[(0, 1), (1, 2), (2, 3)],
        relation_names={"AST", "CFG", "CDG"},
        edge_types=[0, 1, 2],
        edge_type_names={"AST": 0, "CFG": 1, "CDG": 2},
    )
    texts = NodeTextRegistry()
    node_types = NodeTypeRegistry()
    topology_path = tmp_path / "sample.pt"
    save_topology(topology_path, topology, "sample_1", 0, texts, node_types)
    memory_only_topology = GraphTopology(
        view="core-cpg",
        original_node_ids=["1", "2", "3"],
        nodes=[
            GraphNode("1", "METHOD", {"NAME": "f", "CODE": "int f(char *src)"}),
            GraphNode("2", "CONTROL_STRUCTURE", {"CONTROL_STRUCTURE_TYPE": "IF", "CODE": "if (len < sizeof(buf))"}),
            GraphNode("3", "IDENTIFIER", {"CODE": "buf[i]"}),
        ],
        edges=[(0, 1), (1, 2)],
        relation_names={"AST", "CFG"},
        edge_types=[0, 1],
        edge_type_names={"AST": 0, "CFG": 1},
    )
    memory_path = tmp_path / "memory_only.pt"
    save_topology(memory_path, memory_only_topology, "sample_2", 0, texts, node_types)
    features = MemmapFeatureCache.create(tmp_path / "features", rows=len(texts), dim=4)
    features.write(list(range(len(texts))), np.ones((len(texts), 4), dtype=np.float32))

    data = TopologyDataset([topology_path], node_features=features)[0]

    assert data.slice_seed_mask.tolist() == [False, False, True, False]
    assert data.slice_seed_type_id.tolist()[2] == SLICE_SEED_RISKY_CALL

    memory_data = TopologyDataset([memory_path], node_features=features)[0]

    assert memory_data.slice_seed_mask.tolist() == [False, False, True]
    assert memory_data.slice_seed_type_id.tolist()[2] == SLICE_SEED_MEMORY_ACCESS


def test_topology_dataset_exposes_slice_node_mask(tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    from cpg_vuln.data.dataset import TopologyDataset

    topology = GraphTopology(
        view="core-cpg",
        original_node_ids=["1", "2", "3"],
        nodes=[
            GraphNode("1", "METHOD", {"NAME": "f"}),
            GraphNode("2", "CALL", {"CODE": "strcpy(dst, src)"}),
            GraphNode("3", "CALL", {"CODE": "log_event()"}),
        ],
        edges=[],
        relation_names=set(),
        edge_types=[],
        edge_type_names={},
    )
    texts = NodeTextRegistry()
    node_types = NodeTypeRegistry()
    payload = build_topology_payload(
        topology,
        "sample_1",
        1,
        texts,
        node_types,
        commit_id="commit-123",
        slice_node_ids={"1", "2"},
    )
    topology_path = tmp_path / "sample.pt"
    save_topology_payload(topology_path, payload)
    features = MemmapFeatureCache.create(tmp_path / "features", rows=len(texts), dim=4)
    features.write(list(range(len(texts))), np.ones((len(texts), 4), dtype=np.float32))

    data = TopologyDataset([topology_path], node_features=features)[0]

    assert data.slice_node_mask.tolist() == [True, True, False]


def test_topology_payload_rejects_non_raw_without_scope() -> None:
    topology = GraphTopology(
        view="ast",
        original_node_ids=["1"],
        nodes=[GraphNode("1", "IDENTIFIER", {"CODE": "src_len"})],
        edges=[],
        relation_names=set(),
        edge_types=[],
        edge_type_names={},
    )
    spec = NormalizationSpec(mode="semantic-anon")

    with pytest.raises(ValueError, match="requires ScopeContext"):
        build_topology_payload(
            topology,
            "sample_1",
            1,
            NodeTextRegistry(),
            NodeTypeRegistry(),
            commit_id="abc",
            normalizer=IdentifierSemanticNormalizer(spec),
            scope=None,
            spec=spec,
        )


def test_topology_payload_records_normalization_spec() -> None:
    root = GraphNode("1", "METHOD", {"NAME": "copy", "SIGNATURE": "int(char*,char*,int)"})
    graph = ParsedGraph(
        nodes={
            "1": root,
            "2": GraphNode("2", "METHOD_PARAMETER_IN", {"NAME": "dst", "CODE": "char *dst", "TYPE_FULL_NAME": "char *"}),
            "3": GraphNode("3", "METHOD_PARAMETER_IN", {"NAME": "src_len", "CODE": "int src_len", "TYPE_FULL_NAME": "int"}),
            "4": GraphNode("4", "CALL", {"NAME": "memcpy", "CODE": "memcpy(dst, dst, src_len)"}),
        },
        edges=[
            GraphEdge("1", "2", "AST", {}),
            GraphEdge("1", "3", "AST", {}),
            GraphEdge("1", "4", "AST", {}),
        ],
    )
    spec = NormalizationSpec(mode="semantic-anon")
    scope = build_scope_context(graph, root, spec)
    payload = build_topology_payload(
        build_view(graph, root, "ast"),
        "sample_1",
        1,
        NodeTextRegistry(),
        NodeTypeRegistry(),
        commit_id="abc",
        normalizer=IdentifierSemanticNormalizer(spec),
        scope=scope,
        spec=spec,
    )

    assert payload["normalization_mode"] == "semantic-anon"
    assert payload["normalization_key"] == "semantic-anon-v1"
    assert payload["normalization_fingerprint"] == spec.fingerprint
