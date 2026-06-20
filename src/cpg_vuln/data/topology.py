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
AST_CHILD_CONTEXT_LABELS = {
    "CALL",
    "FIELD_IDENTIFIER",
    "IDENTIFIER",
    "LITERAL",
    "LOCAL",
    "METHOD_PARAMETER_IN",
    "RETURN",
}
RISKY_CALL_TOKENS = {
    "alloc",
    "calloc",
    "free",
    "gets",
    "malloc",
    "memcpy",
    "memmove",
    "read",
    "realloc",
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
    relation_neighbors: dict[str, dict[str, set[str]]] = {
        "CFG": {},
        "CDG": {},
        "REACHING_DEF": {},
    }
    for edge in base_edges:
        if edge.label == "AST":
            ast_parents[edge.target] = edge.source
            ast_children.setdefault(edge.source, []).append(edge.target)
        elif edge.label in FLOW_RELATIONS:
            neighbors = relation_neighbors[edge.label]
            neighbors.setdefault(edge.source, set()).add(edge.target)
            neighbors.setdefault(edge.target, set()).add(edge.source)

    seeds = _slice_seed_node_ids(graph, closure)
    if not seeds:
        return _method_shell_node_ids(root.node_id, ast_children, closure)

    selected = {root.node_id, *seeds}
    selected.update(
        _bounded_relation_context(
            seeds, relation_neighbors["REACHING_DEF"], closure, radius=2
        )
    )
    selected.update(
        _bounded_relation_context(seeds, relation_neighbors["CDG"], closure, radius=1)
    )
    selected.update(
        _bounded_relation_context(seeds, relation_neighbors["CFG"], closure, radius=1)
    )

    context_nodes = set(selected)
    for node_id in list(selected):
        if _keeps_ast_children(graph.nodes[node_id]):
            context_nodes.update(ast_children.get(node_id, ()))
        parent = ast_parents.get(node_id)
        while parent is not None and parent in closure:
            context_nodes.add(parent)
            if parent == root.node_id:
                break
            parent = ast_parents.get(parent)
    return context_nodes & closure


def _is_risky_seed(node: GraphNode) -> bool:
    return _is_risky_call_seed(node) or _is_memory_access_seed(node)


def _slice_seed_node_ids(graph: ParsedGraph, closure: set[str]) -> set[str]:
    risky_calls = {
        node.node_id
        for node in graph.nodes.values()
        if node.node_id in closure and _is_risky_call_seed(node)
    }
    if risky_calls:
        return risky_calls
    return {
        node.node_id
        for node in graph.nodes.values()
        if node.node_id in closure and _is_memory_access_seed(node)
    }


def _is_risky_call_seed(node: GraphNode) -> bool:
    code = node.attrs.get("CODE", node.attrs.get("NAME", "")).lower()
    return node.label == "CALL" and any(token in code for token in RISKY_CALL_TOKENS)


def _is_memory_access_seed(node: GraphNode) -> bool:
    code = node.attrs.get("CODE", node.attrs.get("NAME", "")).lower()
    return (
        node.label in {"CALL", "FIELD_IDENTIFIER", "IDENTIFIER", "LOCAL"}
        and _has_memory_access_syntax(code)
    )


def _bounded_relation_context(
    seeds: set[str],
    neighbors: dict[str, set[str]],
    closure: set[str],
    *,
    radius: int,
) -> set[str]:
    selected: set[str] = set()
    frontier = set(seeds)
    for _ in range(radius):
        next_frontier: set[str] = set()
        for node_id in frontier:
            next_frontier.update(neighbors.get(node_id, set()))
        next_frontier &= closure
        next_frontier -= selected
        next_frontier -= seeds
        selected.update(next_frontier)
        frontier = next_frontier
    return selected


def _method_shell_node_ids(
    root_id: str,
    ast_children: dict[str, list[str]],
    closure: set[str],
) -> set[str]:
    shell = {root_id}
    shell.update(node_id for node_id in ast_children.get(root_id, ()) if node_id in closure)
    return shell


def _keeps_ast_children(node: GraphNode) -> bool:
    return node.label in AST_CHILD_CONTEXT_LABELS


def _has_memory_access_syntax(code: str) -> bool:
    return any(
        token in code
        for token in (
            "[",
            "->",
            "<operator>.fieldaccess",
            "<operator>.indirectfieldaccess",
            "<operator>.indirection",
            "<operator>.indexaccess",
        )
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
