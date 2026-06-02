from __future__ import annotations

from cpg_vuln.data.graphml import GraphNode
from cpg_vuln.features.text import NodeTextRegistry, normalize_node_text, tokenize_c


def test_normalize_node_text_uses_stable_type_specific_fallbacks() -> None:
    method = GraphNode("1", "METHOD", {"NAME": "copy", "SIGNATURE": "int(char*)"})
    block = GraphNode("2", "BLOCK", {"CODE": "{ return 0; }"})
    identifier = GraphNode("3", "IDENTIFIER", {"CODE": "source_buffer"})

    assert normalize_node_text(method) == "copy int(char*)"
    assert normalize_node_text(block) == "<BLOCK>"
    assert normalize_node_text(identifier) == "source_buffer"
    assert tokenize_c("source_buffer[i] = 42;") == ["source_buffer", "[", "i", "]", "=", "42", ";"]


def test_text_registry_round_trips(tmp_path) -> None:
    registry = NodeTextRegistry()
    first = registry.add("alpha")
    second = registry.add("beta")
    assert registry.add("alpha") == first
    registry.write(tmp_path / "registry.json")

    restored = NodeTextRegistry.read(tmp_path / "registry.json")

    assert restored.values == ["alpha", "beta"]
    assert second == 1

