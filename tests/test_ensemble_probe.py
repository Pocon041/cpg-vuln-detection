from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_probe_module():
    path = Path("scripts/probe_full_anon_ensemble.py")
    spec = importlib.util.spec_from_file_location("probe_full_anon_ensemble", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_multi_probe_module():
    path = Path("scripts/probe_full_anon_multi_ensemble.py")
    spec = importlib.util.spec_from_file_location("probe_full_anon_multi_ensemble", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fixed05_probe_module():
    path = Path("scripts/probe_fixed05_prediction_ensemble.py")
    spec = importlib.util.spec_from_file_location("probe_fixed05_prediction_ensemble", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cross_norm_probe_module():
    path = Path("scripts/probe_cross_norm_fixed05_ensemble.py")
    spec = importlib.util.spec_from_file_location("probe_cross_norm_fixed05_ensemble", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guarded_threshold_scan_rejects_all_positive_f1_shortcut() -> None:
    probe = _load_probe_module()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)

    unconstrained = probe.scan_threshold(labels, probabilities, objective="f1")
    guarded = probe.scan_threshold(
        labels,
        probabilities,
        objective="f1",
        recall_max=0.75,
        positive_rate_max=0.75,
    )

    assert unconstrained["recall"] == 1.0
    assert unconstrained["predicted_positive_rate"] == 1.0
    assert guarded["recall"] <= 0.75
    assert guarded["predicted_positive_rate"] <= 0.75
    assert guarded["threshold"] > unconstrained["threshold"]


def test_ramp_model_name_is_read_from_run_metrics(tmp_path: Path) -> None:
    probe = _load_probe_module()
    run_dir = tmp_path / "ramp-E4-strict"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        '{"run_metadata": {"model_name": "selective-fusion"}}\n',
        encoding="utf-8",
    )

    assert probe.ramp_model_name_for_run(run_dir, requested="auto") == "selective-fusion"


def test_explicit_ramp_model_name_overrides_run_metrics(tmp_path: Path) -> None:
    probe = _load_probe_module()
    run_dir = tmp_path / "ramp-E4-strict"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        '{"run_metadata": {"model_name": "selective-fusion"}}\n',
        encoding="utf-8",
    )

    assert probe.ramp_model_name_for_run(run_dir, requested="ramp-v2-rgcn") == "ramp-v2-rgcn"


def test_simplex_weights_cover_non_negative_sum_to_one_grid() -> None:
    probe = _load_multi_probe_module()

    weights = list(probe.simplex_weights(model_count=3, grid_steps=2))

    assert len(weights) == 6
    assert (1.0, 0.0, 0.0) in weights
    assert (0.0, 1.0, 0.0) in weights
    assert (0.0, 0.0, 1.0) in weights
    assert (0.5, 0.5, 0.0) in weights
    for row in weights:
        assert all(value >= 0.0 for value in row)
        assert sum(row) == 1.0


def test_multi_probe_ranking_rows_use_weight_mapping() -> None:
    probe = _load_multi_probe_module()
    report = {
        "best_by_strategy": {
            "guarded": {
                "weights": {"a": 0.25, "b": 0.75},
                "threshold": 0.4,
                "test": {
                    "f1": 0.6,
                    "mcc": 0.2,
                    "precision": 0.7,
                    "recall": 0.5,
                    "predicted_positive_rate": 0.4,
                },
            }
        }
    }

    rows = list(probe.ranking_rows(report))

    assert rows == [
        {
            "strategy": "guarded",
            "f1": 0.6,
            "mcc": 0.2,
            "precision": 0.7,
            "recall": 0.5,
            "predicted_positive_rate": 0.4,
            "threshold": 0.4,
            "weights": {"a": 0.25, "b": 0.75},
        }
    ]


def test_multi_probe_fixed05_strategy_keeps_threshold_fixed() -> None:
    probe = _load_multi_probe_module()
    labels = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    probabilities = np.asarray(
        [
            [0.9, 0.8, 0.7, 0.6, 0.9],
            [0.1, 0.2, 0.8, 0.9, 0.4],
        ],
        dtype=np.float32,
    )

    _rows, best = probe._scan_multi(
        model_names=["bad", "good"],
        weights=[(1.0, 0.0), (0.0, 1.0)],
        val_labels=labels,
        test_labels=labels,
        val_probabilities=probabilities,
        test_probabilities=probabilities,
        average_precision_score=lambda y, p: 0.5,
        roc_auc_score=lambda y, p: 0.5,
        tqdm=lambda items, **kwargs: items,
    )

    selected = best["fixed05_guarded_r90_p80_val_f1"]
    assert selected["threshold"] == 0.5
    assert selected["weights"] == {"bad": 0.0, "good": 1.0}
    assert selected["validation"]["f1"] > 0.7


def test_fixed05_probe_uses_fixed_threshold_and_filters_uncontrolled_rows() -> None:
    probe = _load_fixed05_probe_module()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = {
        "all_positive_like": np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        "controlled": np.asarray([0.1, 0.4, 0.6, 0.8], dtype=np.float32),
    }

    rows = probe.scan_fixed05_combinations(
        labels=labels,
        probabilities_by_model=probabilities,
        grid_steps=2,
        max_models=1,
        max_ppr=0.75,
        max_recall=1.0,
    )

    assert [row["model_weights"] for row in rows] == [{"controlled": 1.0}]
    assert rows[0]["metrics"]["threshold"] == 0.5
    assert rows[0]["metrics"]["f1"] == 1.0


def test_cross_norm_model_spec_requires_name_config_and_run() -> None:
    probe = _load_cross_norm_probe_module()

    spec = probe.parse_model_spec("raw_E4=configs/default.yaml=outputs/raw-run")

    assert spec.name == "raw_E4"
    assert spec.config_path == Path("configs/default.yaml")
    assert spec.run_dir == Path("outputs/raw-run")


def test_cross_norm_probe_reads_dual_head_ramp_model_name(tmp_path: Path) -> None:
    probe = _load_cross_norm_probe_module()
    run_dir = tmp_path / "dual-run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        '{"run_metadata": {"model_name": "ramp-v2-dual"}}\n',
        encoding="utf-8",
    )

    assert probe._model_name_for_run(run_dir, {"model_name": "ramp-v2-dual"}) == "ramp-v2-dual"


def test_cross_norm_probe_selects_weights_on_validation_at_fixed05() -> None:
    probe = _load_cross_norm_probe_module()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    val_probabilities = np.asarray(
        [
            [0.1, 0.6, 0.6, 0.4],
            [0.1, 0.4, 0.7, 0.8],
        ],
        dtype=np.float32,
    )
    test_probabilities = np.asarray(
        [
            [0.1, 0.2, 0.8, 0.9],
            [0.9, 0.8, 0.7, 0.6],
        ],
        dtype=np.float32,
    )

    selected = probe.select_fixed05_weights(
        model_names=["test_better", "val_better"],
        weights=[(1.0, 0.0), (0.0, 1.0)],
        val_labels=labels,
        test_labels=labels,
        val_probabilities=val_probabilities,
        test_probabilities=test_probabilities,
        objective="f1",
        max_ppr=0.8,
        max_recall=1.0,
    )

    assert selected["threshold"] == 0.5
    assert selected["weights"] == {"test_better": 0.0, "val_better": 1.0}
    assert selected["validation"]["f1"] == 1.0
    assert selected["test"]["predicted_positive_rate"] == 1.0


def test_cross_norm_probability_features_preserve_model_columns() -> None:
    probe = _load_cross_norm_probe_module()
    probabilities = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float32,
    )

    features = probe.probability_feature_matrix(probabilities, feature_space="probability")

    assert features.shape == (3, 2)
    np.testing.assert_allclose(features[:, 0], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(features[:, 1], [0.7, 0.8, 0.9], atol=1e-6)


def test_cross_norm_logistic_stack_keeps_fixed05_and_filters_uncontrolled_candidates() -> None:
    probe = _load_cross_norm_probe_module()
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    probabilities = np.asarray(
        [
            [0.2, 0.4, 0.8, 0.7, 0.8, 0.9],
            [0.1, 0.2, 0.3, 0.6, 0.7, 0.8],
        ],
        dtype=np.float32,
    )

    selected = probe.select_fixed05_logistic_stack(
        model_names=["noisy", "clean"],
        val_labels=labels,
        test_labels=labels,
        val_probabilities=probabilities,
        test_probabilities=probabilities,
        objective="f1",
        max_ppr=0.75,
        max_recall=1.0,
        c_values=[0.1, 1.0],
        class_weights=["none"],
        feature_spaces=["probability"],
        cv_splits=2,
    )

    assert selected is not None
    assert selected["threshold"] == 0.5
    assert selected["validation"]["predicted_positive_rate"] <= 0.75
    assert selected["validation"]["recall"] <= 1.0
    assert selected["stacker"]["feature_space"] == "probability"
