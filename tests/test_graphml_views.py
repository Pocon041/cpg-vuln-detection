from __future__ import annotations

from pathlib import Path

from cpg_vuln.data.graphml import GraphMLParser, choose_primary_method
from cpg_vuln.data.topology import build_view

from .helpers import write_graphml


def test_choose_primary_method_prefers_largest_ast_subtree(tmp_path: Path) -> None:
    graph_path = tmp_path / "sample.graphml"
    write_graphml(graph_path, include_macro_helper=True)
    graph = GraphMLParser().parse(graph_path)

    selected = choose_primary_method(graph)

    assert selected.node_id == "1"
    assert selected.attrs["NAME"] == "target"


def test_build_views_keep_only_allowed_internal_relations(tmp_path: Path) -> None:
    graph_path = tmp_path / "sample.graphml"
    write_graphml(graph_path)
    graph = GraphMLParser().parse(graph_path)
    root = choose_primary_method(graph)

    ast = build_view(graph, root, "ast")
    cfg = build_view(graph, root, "cfg")
    pdg = build_view(graph, root, "pdg")
    core = build_view(graph, root, "core-cpg")
    dataflow = build_view(graph, root, "dataflow-cpg")

    assert ast.relation_names == {"AST"}
    assert cfg.relation_names == {"CFG"}
    assert pdg.relation_names == {"CDG", "REACHING_DEF"}
    assert core.relation_names == {"AST", "CFG", "CDG", "REACHING_DEF"}
    assert dataflow.relation_names == {"CFG", "CDG", "REACHING_DEF"}
    assert "5" not in core.original_node_ids
    assert len(core.edge_types) == len(core.edges)
    assert "SELF_LOOP" in core.edge_type_names
    assert len(dataflow.edge_types) == len(dataflow.edges)
