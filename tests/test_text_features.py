from __future__ import annotations

from cpg_vuln.data.graphml import GraphNode
from cpg_vuln.features.lexer import TokenKind, lex_c
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


def test_lexer_keeps_string_literal_as_single_token() -> None:
    tokens = lex_c('printf("packet_size=%d", size)')

    assert [token.kind for token in tokens] == [
        TokenKind.IDENTIFIER,
        TokenKind.OPERATOR,
        TokenKind.STRING_LITERAL,
        TokenKind.OPERATOR,
        TokenKind.IDENTIFIER,
        TokenKind.OPERATOR,
    ]
    assert tokens[2].text == '"packet_size=%d"'
    assert "packet_size" not in tokenize_c('printf("packet_size=%d", size)')


def test_lexer_keeps_joern_operator_as_single_token() -> None:
    assert tokenize_c("<operator>.assignment") == ["<operator>.assignment"]


def test_lexer_recognizes_numeric_suffixes_and_keeps_minus_separate() -> None:
    assert tokenize_c("return -1") == ["return", "-", "1"]
    assert tokenize_c("mask = 32U + 1LL + 1.5e-3") == [
        "mask",
        "=",
        "32U",
        "+",
        "1LL",
        "+",
        "1.5e-3",
    ]


def test_lexer_recognizes_escaped_char_literal() -> None:
    tokens = lex_c(r"c = '\n'")

    assert tokens[-1].kind is TokenKind.CHAR_LITERAL
    assert tokens[-1].text == r"'\n'"
