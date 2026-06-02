from __future__ import annotations

from dataclasses import dataclass

from cpg_vuln.data.graphml import GraphNode, ParsedGraph, ast_closure


VIEW_RELATIONS = {
    "ast": ("AST",),
    "cfg": ("CFG",),
    "pdg": ("CDG", "REACHING_DEF"),
    "core-cpg": ("AST", "CFG", "CDG", "REACHING_DEF"),
    "dataflow-cpg": ("CFG", "CDG", "REACHING_DEF"),
}


@dataclass(frozen=True)
class GraphTopology:
    view: str
    original_node_ids: list[str]
    nodes: list[GraphNode]
    edges: list[tuple[int, int]]
    relation_names: set[str]
    edge_types: list[int]
    edge_type_names: dict[str, int]


def build_view(graph: ParsedGraph, root: GraphNode, view: str) -> GraphTopology:
    try:
        allowed_relations = VIEW_RELATIONS[view]
    except KeyError as error:
        raise ValueError(f"unsupported graph view: {view}") from error
    closure = ast_closure(graph, root.node_id)
    base_edges = [
        edge
        for edge in graph.edges
        if edge.label in allowed_relations
        and edge.source in closure
        and edge.target in closure
    ]
    if view in {"cfg", "pdg"}:
        selected_ids = {root.node_id}
        for edge in base_edges:
            selected_ids.add(edge.source)
            selected_ids.add(edge.target)
    else:
        selected_ids = closure
    original_node_ids = sorted(selected_ids, key=_node_sort_key)
    nodes = [graph.nodes[node_id] for node_id in original_node_ids]
    local_ids = {node_id: index for index, node_id in enumerate(original_node_ids)}
    if not view.endswith("-cpg"):
        edges = _bidirectional_edges(base_edges, local_ids)
        return GraphTopology(
            view=view,
            original_node_ids=original_node_ids,
            nodes=nodes,
            edges=edges,
            relation_names={edge.label for edge in base_edges},
            edge_types=[],
            edge_type_names={},
        )
    relation_ids = {name: index for index, name in enumerate(allowed_relations)}
    relation_ids.update(
        {f"{name}_REVERSE": len(relation_ids) + index for index, name in enumerate(allowed_relations)}
    )
    relation_ids["SELF_LOOP"] = len(relation_ids)
    edges: list[tuple[int, int]] = []
    edge_types: list[int] = []
    for edge in base_edges:
        source = local_ids[edge.source]
        target = local_ids[edge.target]
        edges.extend(((source, target), (target, source)))
        edge_types.extend((relation_ids[edge.label], relation_ids[f"{edge.label}_REVERSE"]))
    for node_id in range(len(nodes)):
        edges.append((node_id, node_id))
        edge_types.append(relation_ids["SELF_LOOP"])
    return GraphTopology(
        view=view,
        original_node_ids=original_node_ids,
        nodes=nodes,
        edges=edges,
        relation_names={edge.label for edge in base_edges},
        edge_types=edge_types,
        edge_type_names=relation_ids,
    )


def _bidirectional_edges(edges: list, local_ids: dict[str, int]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for edge in edges:
        source = local_ids[edge.source]
        target = local_ids[edge.target]
        result.extend(((source, target), (target, source)))
    return result


def _node_sort_key(node_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(node_id))
    except ValueError:
        return (1, node_id)
