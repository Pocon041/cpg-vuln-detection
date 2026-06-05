from __future__ import annotations

from dataclasses import dataclass

from cpg_vuln.data.graphml import GraphNode, ParsedGraph, ast_closure


VIEW_RELATIONS = {
    "ast": ("AST",),
    "cfg": ("CFG",),
    "pdg": ("CDG", "REACHING_DEF"),
    "core-cpg": ("AST", "CFG", "CDG", "REACHING_DEF"),
    "dataflow-cpg": ("CFG", "CDG", "REACHING_DEF"),
    "slice-cpg": ("AST", "CFG", "CDG", "REACHING_DEF"),
}

FLOW_RELATIONS = {"CFG", "CDG", "REACHING_DEF"}
RISKY_CALL_TOKENS = {
    "alloc",
    "free",
    "gets",
    "memcpy",
    "memmove",
    "read",
    "recv",
    "scanf",
    "sprintf",
    "strcat",
    "strcpy",
    "strlen",
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
    elif view == "slice-cpg":
        selected_ids = _slice_node_ids(graph, root, closure, base_edges)
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
    relation_edges = [
        edge for edge in base_edges if edge.source in local_ids and edge.target in local_ids
    ]
    for edge in relation_edges:
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
        relation_names={edge.label for edge in relation_edges},
        edge_types=edge_types,
        edge_type_names=relation_ids,
    )


def _slice_node_ids(
    graph: ParsedGraph,
    root: GraphNode,
    closure: set[str],
    base_edges: list,
) -> set[str]:
    ast_parents: dict[str, str] = {}
    ast_children: dict[str, list[str]] = {}
    flow_neighbors: dict[str, set[str]] = {}
    for edge in base_edges:
        if edge.label == "AST":
            ast_parents[edge.target] = edge.source
            ast_children.setdefault(edge.source, []).append(edge.target)
        elif edge.label in FLOW_RELATIONS:
            flow_neighbors.setdefault(edge.source, set()).add(edge.target)
            flow_neighbors.setdefault(edge.target, set()).add(edge.source)

    seeds = {
        node.node_id
        for node in graph.nodes.values()
        if node.node_id in closure and _is_risky_seed(node)
    }
    if not seeds:
        seeds = {
            node_id
            for edge in base_edges
            if edge.label in FLOW_RELATIONS
            for node_id in (edge.source, edge.target)
        }
    if not seeds:
        return {root.node_id}

    selected = {root.node_id, *seeds}
    frontier = set(seeds)
    for _ in range(2):
        next_frontier: set[str] = set()
        for node_id in frontier:
            next_frontier.update(flow_neighbors.get(node_id, set()))
        next_frontier &= closure
        next_frontier -= selected
        selected.update(next_frontier)
        frontier = next_frontier

    context_nodes = set(selected)
    for node_id in list(selected):
        context_nodes.update(ast_children.get(node_id, ()))
        parent = ast_parents.get(node_id)
        while parent is not None and parent in closure:
            context_nodes.add(parent)
            if parent == root.node_id:
                break
            parent = ast_parents.get(parent)
    return context_nodes & closure


def _is_risky_seed(node: GraphNode) -> bool:
    code = node.attrs.get("CODE", node.attrs.get("NAME", "")).lower()
    if node.label == "CALL" and any(token in code for token in RISKY_CALL_TOKENS):
        return True
    return any(symbol in code for symbol in ("[", "->"))


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
