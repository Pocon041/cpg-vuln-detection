from __future__ import annotations

import torch

from cpg_vuln.mining.motif import RiskMotif, extract_risk_motif, motif_feature_names


def _payload() -> dict[str, object]:
    return {
        "sample_id": "sample-1",
        "y": torch.tensor([1], dtype=torch.long),
        "code_summaries": [
            "buf = av_malloc(atom->size)",
            "memcpy(buf, src, len)",
            "value = table[index]",
            "if (ret < 0) return AVERROR_INVALIDDATA",
        ],
        "node_labels": ["CALL", "CALL", "CALL", "CALL", "CONTROL_STRUCTURE"],
        "node_names": [
            "av_malloc",
            "memcpy",
            "<operator>.indexAccess",
            "<operator>.fieldAccess",
            "",
        ],
        "method_full_names": [
            "av_malloc",
            "memcpy",
            "<operator>.indexAccess",
            "<operator>.fieldAccess",
            "",
        ],
        "control_structure_types": ["", "", "", "", "IF"],
        "node_type_histogram": {"CALL": 4, "CONTROL_STRUCTURE": 1},
        "edge_type_histogram": {"AST": 1, "CFG": 1, "CDG": 1, "REACHING_DEF": 1},
        "node_type_id": torch.tensor([1, 2, 2, 2, 3], dtype=torch.long),
        "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        "edge_type": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "edge_type_names": {"AST": 0, "CFG": 1, "CDG": 2, "REACHING_DEF": 3},
    }


def test_extract_risk_motif_counts_lexical_and_graph_features() -> None:
    motif = extract_risk_motif(_payload())

    assert motif.sample_id == "sample-1"
    assert motif.label == 1
    assert motif.features["known_alloc_api"] == 1.0
    assert motif.features["known_copy_api"] == 1.0
    assert motif.features["field_access"] == 1.0
    assert motif.features["array_index"] == 1.0
    assert motif.features["num_nodes"] > 0.0
    assert motif.features["num_edges"] > 0.0
    assert motif.features["num_reaching_def_edges"] > 0.0


def test_motif_vector_uses_stable_feature_order() -> None:
    motif = RiskMotif(sample_id="x", label=0, features={"known_alloc_api": 1.0})

    vector = motif.to_vector()

    assert vector.shape == (len(motif_feature_names()),)
    assert vector[motif_feature_names().index("known_alloc_api")] == 1.0


def test_motif_feature_groups_are_disjoint_and_cover_scored_dimensions() -> None:
    from cpg_vuln.mining.motif import motif_feature_groups

    groups = motif_feature_groups()
    scored = groups["motif"] + groups["structure"] + groups["scale"]

    assert len(scored) == len(set(scored))
    assert "known_alloc_api" in groups["motif"]
    assert "num_reaching_def_edges" in groups["structure"]
    assert "num_nodes" in groups["scale"]


def test_structured_motif_extractor_avoids_string_level_false_hits() -> None:
    payload = {
        "sample_id": "sample-2",
        "label": 0,
        "code_summaries": ["ctx->field = size * count; helper(value)"],
        "node_labels": ["IDENTIFIER", "CALL"],
        "node_names": ["ctx", "helper"],
        "method_full_names": ["", "helper"],
        "control_structure_types": ["", ""],
        "node_type_histogram": {"IDENTIFIER": 1, "CALL": 1},
        "edge_type_histogram": {},
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "node_type_id": torch.tensor([1, 2], dtype=torch.long),
    }

    motif = extract_risk_motif(payload)

    assert motif.features["known_alloc_api"] == 0.0
    assert motif.features["integer_mul"] == 0.0
    assert motif.features["integer_sub"] == 0.0
    assert motif.features["field_access"] == 0.0


def test_api_motif_uses_only_call_nodes() -> None:
    payload = {
        "sample_id": "sample-3",
        "label": 0,
        "node_labels": ["IDENTIFIER", "FIELD_IDENTIFIER"],
        "node_names": ["read", "free"],
        "method_full_names": ["", ""],
        "control_structure_types": ["", ""],
        "node_type_histogram": {"IDENTIFIER": 1, "FIELD_IDENTIFIER": 1},
        "edge_type_histogram": {},
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "node_type_id": torch.tensor([1, 2], dtype=torch.long),
    }

    motif = extract_risk_motif(payload)

    assert motif.features["known_io_api"] == 0.0
    assert motif.features["known_release_api"] == 0.0
