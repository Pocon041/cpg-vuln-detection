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
    slice_cpg = build_view(graph, root, "slice-cpg")

    assert ast.relation_names == {"AST"}
    assert cfg.relation_names == {"CFG"}
    assert pdg.relation_names == {"CDG", "REACHING_DEF"}
    assert core.relation_names == {"AST", "CFG", "CDG", "REACHING_DEF"}
    assert dataflow.relation_names == {"CFG", "CDG", "REACHING_DEF"}
    assert slice_cpg.relation_names == {"AST", "CFG", "CDG", "REACHING_DEF"}
    assert "5" not in core.original_node_ids
    assert len(core.edge_types) == len(core.edges)
    assert "SELF_LOOP" in core.edge_type_names
    assert len(dataflow.edge_types) == len(dataflow.edges)
    assert len(slice_cpg.edge_types) == len(slice_cpg.edges)


def test_slice_cpg_keeps_risky_flow_context_and_drops_unrelated_ast(tmp_path: Path) -> None:
    graph_path = tmp_path / "sample.graphml"
    write_graphml(graph_path, include_unrelated_ast=True)
    graph = GraphMLParser().parse(graph_path)
    root = choose_primary_method(graph)

    sliced = build_view(graph, root, "slice-cpg")

    assert {"1", "2", "3", "4"}.issubset(set(sliced.original_node_ids))
    assert "7" not in sliced.original_node_ids
    assert "SELF_LOOP" in sliced.edge_type_names
    assert any(
        sliced.edge_type_names[name] in sliced.edge_types
        for name in ("CFG", "CDG", "REACHING_DEF")
    )


def test_slice_cpg_without_risky_seed_keeps_only_method_shell(tmp_path: Path) -> None:
    graph_path = tmp_path / "no_risk.graphml"
    _write_linear_graphml(graph_path, risky=False)
    graph = GraphMLParser().parse(graph_path)
    root = choose_primary_method(graph)

    sliced = build_view(graph, root, "slice-cpg")

    assert set(sliced.original_node_ids) == {"1", "2"}
    assert "CFG" not in sliced.relation_names
    assert "REACHING_DEF" not in sliced.relation_names


def test_slice_cpg_keeps_local_cfg_context_but_drops_distant_chain(tmp_path: Path) -> None:
    graph_path = tmp_path / "risky.graphml"
    _write_linear_graphml(graph_path, risky=True)
    graph = GraphMLParser().parse(graph_path)
    root = choose_primary_method(graph)

    sliced = build_view(graph, root, "slice-cpg")

    assert {"1", "2", "3", "4", "7"}.issubset(set(sliced.original_node_ids))
    assert "5" not in sliced.original_node_ids
    assert "6" not in sliced.original_node_ids


def test_slice_cpg_prioritizes_risky_api_over_unrelated_memory_access(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "risky_with_memory_access.graphml"
    _write_linear_graphml(graph_path, risky=True, include_memory_access=True)
    graph = GraphMLParser().parse(graph_path)
    root = choose_primary_method(graph)

    sliced = build_view(graph, root, "slice-cpg")

    assert "3" in sliced.original_node_ids
    assert "8" not in sliced.original_node_ids


def test_slice_cpg_uses_memory_access_seed_when_no_risky_api(tmp_path: Path) -> None:
    graph_path = tmp_path / "memory_access.graphml"
    _write_linear_graphml(graph_path, risky=False, include_memory_access=True)
    graph = GraphMLParser().parse(graph_path)
    root = choose_primary_method(graph)

    sliced = build_view(graph, root, "slice-cpg")

    assert {"1", "2", "8"}.issubset(set(sliced.original_node_ids))
    assert "5" not in sliced.original_node_ids
    assert "6" not in sliced.original_node_ids


def _write_linear_graphml(
    path: Path,
    *,
    risky: bool,
    include_memory_access: bool = False,
) -> None:
    first_call = "strcpy(dst, src)" if risky else "helper(dst, src)"
    memory_node = """
    <node id="8"><data key="labelV">CALL</data><data key="node__CALL__CODE">items[i]</data></node>
    """ if include_memory_access else ""
    memory_edge = """
    <edge source="2" target="8"><data key="labelE">AST</data></edge>
    """ if include_memory_access else ""
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="labelV" for="node" attr.name="labelV" attr.type="string"/>
  <key id="labelE" for="edge" attr.name="labelE" attr.type="string"/>
  <key id="node__METHOD__NAME" for="node" attr.name="NAME" attr.type="string"/>
  <key id="node__METHOD__SIGNATURE" for="node" attr.name="SIGNATURE" attr.type="string"/>
  <key id="node__METHOD__IS_EXTERNAL" for="node" attr.name="IS_EXTERNAL" attr.type="boolean"/>
  <key id="node__METHOD__CODE" for="node" attr.name="CODE" attr.type="string"/>
  <key id="node__BLOCK__CODE" for="node" attr.name="CODE" attr.type="string"/>
  <key id="node__CALL__CODE" for="node" attr.name="CODE" attr.type="string"/>
  <key id="node__IDENTIFIER__CODE" for="node" attr.name="CODE" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="1">
      <data key="labelV">METHOD</data>
      <data key="node__METHOD__NAME">target</data>
      <data key="node__METHOD__SIGNATURE">int(char*)</data>
      <data key="node__METHOD__IS_EXTERNAL">false</data>
      <data key="node__METHOD__CODE">int target(char *src)</data>
    </node>
    <node id="2"><data key="labelV">BLOCK</data><data key="node__BLOCK__CODE">{{...}}</data></node>
    <node id="3"><data key="labelV">CALL</data><data key="node__CALL__CODE">{first_call}</data></node>
    <node id="4"><data key="labelV">CALL</data><data key="node__CALL__CODE">sanitize(src)</data></node>
    <node id="5"><data key="labelV">CALL</data><data key="node__CALL__CODE">log_event()</data></node>
    <node id="6"><data key="labelV">CALL</data><data key="node__CALL__CODE">cleanup()</data></node>
    <node id="7"><data key="labelV">IDENTIFIER</data><data key="node__IDENTIFIER__CODE">src</data></node>
    {memory_node}
    <edge source="1" target="2"><data key="labelE">AST</data></edge>
    <edge source="2" target="3"><data key="labelE">AST</data></edge>
    <edge source="2" target="4"><data key="labelE">AST</data></edge>
    <edge source="2" target="5"><data key="labelE">AST</data></edge>
    <edge source="2" target="6"><data key="labelE">AST</data></edge>
    <edge source="3" target="7"><data key="labelE">AST</data></edge>
    {memory_edge}
    <edge source="3" target="4"><data key="labelE">CFG</data></edge>
    <edge source="4" target="5"><data key="labelE">CFG</data></edge>
    <edge source="5" target="6"><data key="labelE">CFG</data></edge>
    <edge source="7" target="3"><data key="labelE">REACHING_DEF</data></edge>
  </graph>
</graphml>
""",
        encoding="utf-8",
    )
