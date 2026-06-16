from __future__ import annotations

import pytest

from cpg_vuln.data.graphml import GraphEdge, GraphNode, ParsedGraph
from cpg_vuln.features.normalization import (
    ApiTaxonomy,
    IdentifierSemanticNormalizer,
    NormalizationSpec,
    ScopeContext,
    build_scope_context,
    normalize_source_text,
    stable_function_placeholder,
    stable_unknown_placeholder,
    type_identifiers,
)
from scripts.audit_normalized_texts import audit_normalized_values
from scripts.rename_robustness import summarize_prediction_shift


def _graph_for_scope_test() -> tuple[ParsedGraph, GraphNode]:
    root = GraphNode("1", "METHOD", {"NAME": "copy_items", "SIGNATURE": "int(char*, char*, int)"})
    nodes = {
        "1": root,
        "2": GraphNode("2", "METHOD_PARAMETER_IN", {"NAME": "dst", "CODE": "char *dst", "TYPE_FULL_NAME": "char *", "LINE_NUMBER": "10", "COLUMN_NUMBER": "20"}),
        "3": GraphNode("3", "METHOD_PARAMETER_IN", {"NAME": "src", "CODE": "char *src", "TYPE_FULL_NAME": "char *", "LINE_NUMBER": "10", "COLUMN_NUMBER": "31"}),
        "4": GraphNode("4", "METHOD_PARAMETER_IN", {"NAME": "src_len", "CODE": "int src_len", "TYPE_FULL_NAME": "int", "LINE_NUMBER": "10", "COLUMN_NUMBER": "42"}),
        "5": GraphNode("5", "LOCAL", {"NAME": "dst_len", "CODE": "int dst_len", "TYPE_FULL_NAME": "int", "LINE_NUMBER": "12", "COLUMN_NUMBER": "5"}),
        "6": GraphNode("6", "CALL", {"NAME": "memcpy", "CODE": "memcpy(dst, src, src_len)", "LINE_NUMBER": "13", "COLUMN_NUMBER": "5"}),
        "7": GraphNode("7", "CALL", {"NAME": "av_malloc", "CODE": "av_malloc(dst_len)", "LINE_NUMBER": "14", "COLUMN_NUMBER": "5"}),
        "8": GraphNode("8", "CALL", {"NAME": "decode_packet", "CODE": "decode_packet(dst)", "LINE_NUMBER": "15", "COLUMN_NUMBER": "5"}),
        "9": GraphNode("9", "IDENTIFIER", {"NAME": "FFSIGN", "CODE": "FFSIGN", "LINE_NUMBER": "16", "COLUMN_NUMBER": "5"}),
        "10": GraphNode("10", "FIELD_IDENTIFIER", {"CANONICAL_NAME": "packet_size", "CODE": "packet_size", "LINE_NUMBER": "17", "COLUMN_NUMBER": "15"}),
        "11": GraphNode("11", "LOCAL", {"NAME": "gb", "CODE": "GetBitContext *gb", "TYPE_FULL_NAME": "GetBitContext *", "LINE_NUMBER": "18", "COLUMN_NUMBER": "5"}),
    }
    edges = [
        GraphEdge("1", "2", "AST", {}),
        GraphEdge("1", "3", "AST", {}),
        GraphEdge("1", "4", "AST", {}),
        GraphEdge("1", "5", "AST", {}),
        GraphEdge("1", "6", "AST", {}),
        GraphEdge("1", "7", "AST", {}),
        GraphEdge("1", "8", "AST", {}),
        GraphEdge("1", "9", "AST", {}),
        GraphEdge("1", "10", "AST", {}),
        GraphEdge("1", "11", "AST", {}),
    ]
    return ParsedGraph(nodes=nodes, edges=edges), root


def test_normalization_spec_accepts_only_v1_modes() -> None:
    raw = NormalizationSpec(mode="raw", version=1)
    semantic = NormalizationSpec(mode="semantic-anon", version=1)
    full = NormalizationSpec(mode="full-anon", version=1)

    assert raw.normalization_key == "raw-v1"
    assert semantic.normalization_key == "semantic-anon-v1"
    assert full.normalization_key == "full-anon-v1"
    with pytest.raises(ValueError, match="unsupported normalization mode"):
        NormalizationSpec(mode="structure-only", version=1)


def test_normalization_spec_fingerprint_changes_with_versions() -> None:
    base = NormalizationSpec(mode="semantic-anon", version=1)
    changed = NormalizationSpec(mode="semantic-anon", version=2)

    assert base.fingerprint != changed.fingerprint


def test_api_taxonomy_separates_standard_api_from_third_party_wrapper() -> None:
    taxonomy = ApiTaxonomy.default()

    assert taxonomy.classify("memcpy").normalized_tokens("semantic-anon") == ("memcpy", "API_COPY")
    assert taxonomy.classify("malloc").normalized_tokens("semantic-anon") == ("malloc", "API_ALLOC")
    assert taxonomy.classify("av_malloc").normalized_tokens("semantic-anon") == ("API_ALLOC",)
    assert taxonomy.classify("OPENSSL_malloc").normalized_tokens("semantic-anon") == ("API_ALLOC",)
    assert taxonomy.classify("memcpy").normalized_tokens("full-anon") == ("API_COPY",)
    assert taxonomy.classify("dirac_get_se_golomb") is None


def test_scope_context_preserves_identity_and_precomputes_unknowns() -> None:
    graph, root = _graph_for_scope_test()
    scope = build_scope_context(graph, root, NormalizationSpec(mode="semantic-anon"))

    assert scope.parameters["src_len"].placeholder == "PARAM_3"
    assert scope.parameters["src_len"].tags == ("SEM_SRC", "SEM_LEN")
    assert scope.locals["dst_len"].placeholder == "VAR_1"
    assert scope.locals["dst_len"].tags == ("SEM_DST", "SEM_LEN")
    assert scope.fields["packet_size"].placeholder == "FIELD_1"
    assert scope.fields["packet_size"].tags == ("SEM_BUF", "SEM_SIZE")
    assert scope.functions["decode_packet"].placeholder == "USER_FUNC_1"
    assert "memcpy" not in scope.functions
    assert "av_malloc" not in scope.functions
    assert scope.unknowns["FFSIGN"].placeholder.startswith("UNKNOWN_ID_")


def test_user_type_identity_uses_complete_identifier() -> None:
    assert type_identifiers("GetBitContext *") == ["GetBitContext"]
    assert type_identifiers("const AVCodecContext *") == ["AVCodecContext"]


def test_stable_fallback_placeholders_do_not_depend_on_call_order() -> None:
    assert stable_unknown_placeholder("FFSIGN") == stable_unknown_placeholder("FFSIGN")
    assert stable_function_placeholder("decode_packet", semantic=True) == stable_function_placeholder("decode_packet", semantic=True)
    assert stable_function_placeholder("decode_packet", semantic=True).startswith("USER_FUNC_")
    assert stable_function_placeholder("decode_packet", semantic=False).startswith("FUNC_")


def test_normalizer_does_not_merge_two_length_variables() -> None:
    graph, root = _graph_for_scope_test()
    spec = NormalizationSpec(mode="semantic-anon")
    scope = build_scope_context(graph, root, spec)
    normalizer = IdentifierSemanticNormalizer(spec)

    node = GraphNode("20", "CONTROL_STRUCTURE", {"CODE": "if (src_len < dst_len)"})

    assert normalizer.normalize_node(node, scope) == (
        "if ( PARAM_3 SEM_SRC SEM_LEN < VAR_1 SEM_DST SEM_LEN )"
    )


def test_method_return_anonymizes_user_defined_type() -> None:
    graph, root = _graph_for_scope_test()
    spec = NormalizationSpec(mode="semantic-anon")
    scope = build_scope_context(graph, root, spec)
    normalizer = IdentifierSemanticNormalizer(spec)

    node = GraphNode("21", "METHOD_RETURN", {"TYPE_FULL_NAME": "GetBitContext *"})

    assert normalizer.normalize_node(node, scope) == "<RET USER_TYPE_1 *>"


def test_standard_api_does_not_shift_user_function_numbering() -> None:
    graph, root = _graph_for_scope_test()
    spec = NormalizationSpec(mode="semantic-anon")
    scope = build_scope_context(graph, root, spec)
    normalizer = IdentifierSemanticNormalizer(spec)

    memcpy_node = GraphNode("6", "CALL", {"NAME": "memcpy", "CODE": "memcpy(dst, src, src_len)"})
    av_node = GraphNode("7", "CALL", {"NAME": "av_malloc", "CODE": "av_malloc(dst_len)"})
    user_node = GraphNode("8", "CALL", {"NAME": "decode_packet", "CODE": "decode_packet(dst)"})

    assert normalizer.normalize_node(memcpy_node, scope) == (
        "memcpy API_COPY ( PARAM_1 SEM_DST , PARAM_2 SEM_SRC , PARAM_3 SEM_SRC SEM_LEN )"
    )
    assert normalizer.normalize_node(av_node, scope) == "API_ALLOC ( VAR_1 SEM_DST SEM_LEN )"
    assert normalizer.normalize_node(user_node, scope) == "USER_FUNC_1 ( PARAM_1 SEM_DST )"


def test_unknown_identifier_fallback_is_stable() -> None:
    spec = NormalizationSpec(mode="semantic-anon")
    normalizer = IdentifierSemanticNormalizer(spec)
    empty_scope = ScopeContext()
    node = GraphNode("30", "CALL", {"CODE": "FFSIGN(coeff)"})

    first = normalizer.normalize_node(node, empty_scope)
    second = normalizer.normalize_node(node, empty_scope)

    assert first == second
    assert first.startswith("UNKNOWN_ID_")
    assert "USER_FUNC_" not in first


def test_normalize_source_text_reuses_scope_placeholders() -> None:
    graph, root = _graph_for_scope_test()
    spec = NormalizationSpec(mode="full-anon")
    scope = build_scope_context(graph, root, spec)

    normalized = normalize_source_text(
        "int f(char *dst, char *src, int src_len) { memcpy(dst, src, src_len); }",
        scope,
        spec,
    )

    assert "dst" not in normalized
    assert "src_len" not in normalized
    assert "PARAM_1" in normalized
    assert "PARAM_2" in normalized
    assert "PARAM_3" in normalized
    assert "API_COPY" in normalized


def test_non_raw_normalization_requires_scope_context() -> None:
    normalizer = IdentifierSemanticNormalizer(NormalizationSpec(mode="semantic-anon"))
    node = GraphNode("1", "IDENTIFIER", {"CODE": "src_len"})

    with pytest.raises(ValueError, match="requires a frozen ScopeContext"):
        normalizer.normalize_node(node, None)


def test_audit_normalized_values_counts_string_and_path_leaks() -> None:
    stats = audit_normalized_values([
        "printf ( STR , VAR_1 )",
        "VAR_1 = \"packet_size=%d\"",
        "load ( C:\\\\tmp\\\\project\\\\file.c )",
        "UNKNOWN_ID_abcd1234 = NUM_1",
    ])

    assert stats["total_texts"] == 4
    assert stats["string_literal_leaks"] == 1
    assert stats["path_like_leaks"] == 1
    assert stats["unknown_identifier_texts"] == 1


def test_summarize_prediction_shift_reports_probability_change() -> None:
    summary = summarize_prediction_shift(
        before={"a": 0.10, "b": 0.80},
        after={"a": 0.15, "b": 0.70},
        threshold=0.5,
    )

    assert summary["mean_abs_probability_delta"] == pytest.approx(0.075)
    assert summary["prediction_label_agreement"] == pytest.approx(1.0)
