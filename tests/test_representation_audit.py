from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_audit_module():
    path = Path("scripts/audit_intermediate_representations.py")
    spec = importlib.util.spec_from_file_location("audit_intermediate_representations", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_threshold_metrics_report_ppr_and_ranking_scores() -> None:
    audit = _load_audit_module()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.8, 0.4, 0.9], dtype=np.float32)

    metrics = audit.fixed_threshold_metrics(labels, probabilities, threshold=0.5)

    assert metrics["predicted_positive_rate"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["roc_auc"] == 0.75


def test_histogram_feature_matrix_uses_shape_and_histograms() -> None:
    audit = _load_audit_module()
    records = [
        {
            "nodes": 9,
            "edges": 15,
            "node_histogram": {"CALL": 3, "METHOD": 1},
            "edge_histogram": {"AST": 8, "CFG": 2},
        },
        {
            "nodes": 4,
            "edges": 4,
            "node_histogram": {"IDENTIFIER": 2},
            "edge_histogram": {"AST": 1, "REACHING_DEF": 3},
        },
    ]

    matrix, feature_names = audit.histogram_feature_matrix(records)

    assert matrix.shape == (2, len(feature_names))
    assert feature_names[:2] == ["log_nodes", "log_edges"]
    assert "node:CALL_fraction" in feature_names
    assert "edge:REACHING_DEF_log_count" in feature_names
    call_index = feature_names.index("node:CALL_fraction")
    assert matrix[0, call_index] == 3 / 9
    assert matrix[1, call_index] == 0.0


def test_pool_node_features_returns_mean_and_max_per_graph() -> None:
    audit = _load_audit_module()
    graph_vectors = [
        np.asarray([[1.0, 2.0], [3.0, 1.0]], dtype=np.float32),
        np.asarray([[0.0, 4.0]], dtype=np.float32),
    ]

    pooled = audit.pool_node_feature_arrays(graph_vectors)

    np.testing.assert_allclose(
        pooled,
        np.asarray(
            [
                [2.0, 1.5, 3.0, 2.0],
                [0.0, 4.0, 0.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )
