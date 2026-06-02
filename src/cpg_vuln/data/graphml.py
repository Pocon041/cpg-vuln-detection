from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    label: str
    attrs: dict[str, str]


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    label: str
    attrs: dict[str, str]


@dataclass(frozen=True)
class ParsedGraph:
    nodes: dict[str, GraphNode]
    edges: list[GraphEdge]


class GraphMLParser:
    def parse(self, path: Path) -> ParsedGraph:
        root = ElementTree.parse(path).getroot()
        key_names = {
            element.attrib["id"]: element.attrib.get("attr.name", element.attrib["id"])
            for element in root
            if _local_name(element.tag) == "key"
        }
        graph = next(element for element in root if _local_name(element.tag) == "graph")
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        for element in graph:
            tag = _local_name(element.tag)
            values = {
                key_names.get(data.attrib["key"], data.attrib["key"]): data.text or ""
                for data in element
                if _local_name(data.tag) == "data"
            }
            if tag == "node":
                node_id = element.attrib["id"]
                nodes[node_id] = GraphNode(
                    node_id=node_id,
                    label=values.pop("labelV", "UNKNOWN"),
                    attrs=values,
                )
            elif tag == "edge":
                edges.append(
                    GraphEdge(
                        source=element.attrib["source"],
                        target=element.attrib["target"],
                        label=values.pop("labelE", "UNKNOWN"),
                        attrs=values,
                    )
                )
        return ParsedGraph(nodes=nodes, edges=edges)


def choose_primary_method(graph: ParsedGraph) -> GraphNode:
    candidates = [
        node
        for node in graph.nodes.values()
        if node.label == "METHOD"
        and node.attrs.get("IS_EXTERNAL", "false").lower() != "true"
        and node.attrs.get("NAME") != "<global>"
    ]
    if not candidates:
        raise ValueError("graph does not contain an internal non-global method")
    return sorted(
        candidates,
        key=lambda node: (
            -len(ast_closure(graph, node.node_id)),
            -len(node.attrs.get("CODE", "")),
            _node_sort_key(node.node_id),
        ),
    )[0]


def ast_closure(graph: ParsedGraph, root_id: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.label == "AST":
            children.setdefault(edge.source, []).append(edge.target)
    seen: set[str] = set()
    pending = [root_id]
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        pending.extend(children.get(node_id, ()))
    return seen


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _node_sort_key(node_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(node_id))
    except ValueError:
        return (1, node_id)

