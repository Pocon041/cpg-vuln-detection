from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cpg_vuln.data.graphml import GraphMLParser, choose_primary_method
from cpg_vuln.data.layout import ArtifactLayout
from cpg_vuln.data.store import NodeTypeRegistry, save_topology
from cpg_vuln.data.topology import build_view
from cpg_vuln.features.cache import MemmapFeatureCache
from cpg_vuln.features.normalization import NormalizationSpec
from cpg_vuln.features.text import NodeTextRegistry
from cpg_vuln.training.runner import train_baselines, train_devign, train_enhanced

from .helpers import write_graphml


def test_baseline_and_enhanced_runners_consume_cached_topologies(tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")

    artifacts = tmp_path / "artifacts"
    layout = ArtifactLayout(
        artifacts_root=artifacts,
        outputs_root=tmp_path / "outputs",
        spec=NormalizationSpec(mode="raw"),
    )
    topologies = layout.topology_dir
    graph_path = tmp_path / "sample.graphml"
    write_graphml(graph_path)
    graph = GraphMLParser().parse(graph_path)
    root = choose_primary_method(graph)
    texts = NodeTextRegistry()
    node_types = NodeTypeRegistry()
    index = []
    sample_ids = [f"sample-{number}" for number in range(8)]
    for number, sample_id in enumerate(sample_ids):
        for view in ("ast", "core-cpg", "dataflow-cpg", "slice-cpg"):
            index.append(
                save_topology(
                    topologies / view / f"{sample_id}.pt",
                    build_view(graph, root, view),
                    sample_id,
                    number % 2,
                    texts,
                    node_types,
                )
            )
    topologies.mkdir(parents=True, exist_ok=True)
    (topologies / "index.json").write_text(json.dumps(index), encoding="utf-8")
    node_types.write(topologies / "node_type_registry.json")
    split_dir = artifacts / "data" / "splits"
    split_dir.mkdir(parents=True)
    split = {"train": sample_ids[:4], "val": sample_ids[4:6], "test": sample_ids[6:]}
    (split_dir / "course.json").write_text(json.dumps(split), encoding="utf-8")
    node_cache = MemmapFeatureCache.create(
        layout.word2vec_dir / "features", rows=len(texts), dim=8
    )
    codebert_cache = MemmapFeatureCache.create(
        layout.node_codebert_dir, rows=len(texts), dim=8
    )
    values = np.ones((len(texts), 8), dtype=np.float32)
    node_cache.write(list(range(len(texts))), values)
    codebert_cache.write(list(range(len(texts))), values)
    functions = MemmapFeatureCache.create(
        layout.function_codebert_dir / "features",
        rows=len(sample_ids),
        dim=8,
    )
    functions.write(list(range(len(sample_ids))), np.ones((len(sample_ids), 8), dtype=np.float32))
    (layout.function_codebert_dir / "function_indices.json").write_text(
        json.dumps({sample_id: index for index, sample_id in enumerate(sample_ids)}),
        encoding="utf-8",
    )
    config = {
        "paths": {"artifacts_dir": str(artifacts), "outputs_dir": str(tmp_path / "outputs")},
        "model": {"hidden_dim": 8, "node_type_dim": 4, "dropout": 0.0},
        "devign": {"hidden_dim": 16, "dropout": 0.2, "steps": 3, "weight_decay": 0.0000013},
        "training": {
            "epochs": 1,
            "patience": 1,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "max_nodes": 100,
            "max_edges": 100,
            "seed": 42,
            "device": "cpu",
        },
    }

    train_baselines(config, views=("ast",), embeddings=("word2vec",), splits=("course",), epochs=1)
    train_enhanced(config, splits=("course",), variants=("selective-fusion",), epochs=1)
    train_enhanced(config, splits=("course",), variants=("slice-fusion",), epochs=1)

    run_root = tmp_path / "outputs" / "runs" / "raw-v1"
    assert (run_root / "baseline-ast-word2vec-course" / "best.pt").is_file()
    assert (run_root / "enhanced-selective-fusion-course" / "best.pt").is_file()
    assert (run_root / "enhanced-slice-fusion-course" / "best.pt").is_file()
    enhanced_metrics = json.loads(
        (run_root / "enhanced-selective-fusion-course" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert enhanced_metrics["config"]["learning_rate"] == 1e-4
    slice_metrics = json.loads(
        (run_root / "enhanced-slice-fusion-course" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert slice_metrics["run_metadata"]["view"] == "slice-cpg"
    baseline_metrics = run_root / "baseline-ast-word2vec-course" / "metrics.json"
    baseline_metrics.write_text("keep existing completed run", encoding="utf-8")

    train_baselines(config, views=("ast",), embeddings=("word2vec",), splits=("course",), epochs=1)

    assert baseline_metrics.read_text(encoding="utf-8") == "keep existing completed run"


def test_function_source_normalization_metadata_comes_from_function_cache(tmp_path: Path) -> None:
    import cpg_vuln.training.runner as runner

    layout = ArtifactLayout(
        artifacts_root=tmp_path / "artifacts",
        outputs_root=tmp_path / "outputs",
        spec=NormalizationSpec(mode="full-anon"),
    )
    layout.function_codebert_dir.mkdir(parents=True)
    (layout.function_codebert_dir / "source_normalization.json").write_text(
        json.dumps({"function_source_normalization": "full-anon-v1"}),
        encoding="utf-8",
    )

    assert runner._function_source_normalization(layout) == "full-anon-v1"


def test_ramp_warmup_is_stale_when_function_source_normalization_changes(tmp_path: Path) -> None:
    import cpg_vuln.training.runner as runner

    layout = ArtifactLayout(
        artifacts_root=tmp_path / "artifacts",
        outputs_root=tmp_path / "outputs",
        spec=NormalizationSpec(mode="full-anon"),
    )
    layout.function_codebert_dir.mkdir(parents=True)
    (layout.function_codebert_dir / "source_normalization.json").write_text(
        json.dumps({"function_source_normalization": "full-anon-v1"}),
        encoding="utf-8",
    )
    warmup_dir = tmp_path / "outputs" / "runs" / "full-anon-v1" / "warmup"
    warmup_dir.mkdir(parents=True)
    (warmup_dir / "best.pt").write_bytes(b"checkpoint")
    (warmup_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_metadata": {
                    "normalization_fingerprint": layout.spec.fingerprint,
                    "function_source_normalization": "raw",
                }
            }
        ),
        encoding="utf-8",
    )

    assert runner._ramp_warmup_matches_current_inputs(warmup_dir, layout) is False


def test_ramp_warmup_is_stale_when_model_fingerprint_changes(tmp_path: Path) -> None:
    import cpg_vuln.training.runner as runner

    layout = ArtifactLayout(
        artifacts_root=tmp_path / "artifacts",
        outputs_root=tmp_path / "outputs",
        spec=NormalizationSpec(mode="raw"),
    )
    warmup_dir = tmp_path / "outputs" / "runs" / "raw-v1" / "warmup"
    warmup_dir.mkdir(parents=True)
    (warmup_dir / "best.pt").write_bytes(b"checkpoint")
    (warmup_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_metadata": {
                    "normalization_fingerprint": layout.spec.fingerprint,
                    "function_source_normalization": "raw",
                    "model_fingerprint": "old",
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        runner._ramp_warmup_matches_current_inputs(
            warmup_dir,
            layout,
            expected_model_fingerprint="new",
        )
        is False
    )


def test_ramp_model_fingerprint_changes_with_v3_slice_settings() -> None:
    import cpg_vuln.training.runner as runner

    base = {
        "model": {"hidden_dim": 8},
        "ramp_v3": {"slice_top_k": 3},
    }
    changed = {
        "model": {"hidden_dim": 8},
        "ramp_v3": {"slice_top_k": 4},
    }

    assert runner._ramp_model_fingerprint("ramp-v3-slice-mil", "core-cpg", base) != (
        runner._ramp_model_fingerprint("ramp-v3-slice-mil", "core-cpg", changed)
    )


def test_ramp_model_constructs_gated_rgcn() -> None:
    import cpg_vuln.training.runner as runner
    from cpg_vuln.models.ramp_v2 import RampV2GatedRGCNCPG

    model = runner._ramp_model(
        model_name="ramp-v2-gated-rgcn",
        input_dim=8,
        function_dim=8,
        num_node_types=3,
        num_relations=4,
        config={
            "model": {"node_type_dim": 4},
            "ramp_v2": {
                "hidden_dim": 16,
                "layers": 2,
                "dropout": 0.0,
                "gated_rgcn": {
                    "gate_bias_init": -0.5,
                    "ffn_multiplier": 2,
                },
            },
        },
    )

    assert isinstance(model, RampV2GatedRGCNCPG)


def test_ramp_model_constructs_ramp_v3_slice_mil() -> None:
    import cpg_vuln.training.runner as runner
    from cpg_vuln.models.ramp_v2 import RampV3SliceMILCPG

    model = runner._ramp_model(
        model_name="ramp-v3-slice-mil",
        input_dim=8,
        function_dim=8,
        num_node_types=3,
        num_relations=4,
        config={
            "model": {"node_type_dim": 4},
            "ramp_v2": {
                "hidden_dim": 16,
                "layers": 2,
                "dropout": 0.0,
            },
            "ramp_v3": {
                "slice_top_k": 4,
                "slice_temperature": 0.7,
                "fusion_logit_init": [2.0, 0.0, -1.0, -2.0],
            },
        },
    )

    assert isinstance(model, RampV3SliceMILCPG)
    assert model.slice_top_k == 4
    assert model.slice_temperature == pytest.approx(0.7)
    assert torch.allclose(
        model.logit_weights.detach(),
        torch.tensor([2.0, 0.0, -1.0, -2.0]),
    )


def test_baseline_runner_reports_matrix_progress(tmp_path: Path, monkeypatch) -> None:
    import cpg_vuln.training.runner as runner

    calls: list[dict[str, object]] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    monkeypatch.setattr(runner, "tqdm", fake_tqdm)
    monkeypatch.setattr(runner, "_topology_index", lambda artifacts: [])
    monkeypatch.setattr(runner.NodeTypeRegistry, "read", lambda path: object())
    monkeypatch.setattr(runner, "_read_json", lambda path: {"train": [], "val": [], "test": []})
    monkeypatch.setattr(runner, "_completed", lambda run_dir, *, force: True)

    runner.train_baselines(
        {
            "paths": {
                "artifacts_dir": str(tmp_path / "artifacts"),
                "outputs_dir": str(tmp_path / "outputs"),
            },
        },
        views=("ast", "cfg"),
        embeddings=("codebert",),
        splits=("strict",),
    )

    assert calls[0]["desc"] == "baseline runs"
    assert calls[0]["unit"] == "run"
    assert calls[0]["total"] == 2


def test_train_devign_uses_core_cpg_codebert_training_path(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("torch_geometric")

    artifacts = tmp_path / "artifacts"
    layout = ArtifactLayout(
        artifacts_root=artifacts,
        outputs_root=tmp_path / "outputs",
        spec=NormalizationSpec(mode="raw"),
    )
    topologies = layout.topology_dir
    graph_path = tmp_path / "sample.graphml"
    write_graphml(graph_path)
    graph = GraphMLParser().parse(graph_path)
    root = choose_primary_method(graph)
    texts = NodeTextRegistry()
    node_types = NodeTypeRegistry()
    index = []
    sample_ids = [f"sample-{number}" for number in range(6)]
    for number, sample_id in enumerate(sample_ids):
        index.append(
            save_topology(
                topologies / "core-cpg" / f"{sample_id}.pt",
                build_view(graph, root, "core-cpg"),
                sample_id,
                number % 2,
                texts,
                node_types,
            )
        )
    topologies.mkdir(parents=True, exist_ok=True)
    (topologies / "index.json").write_text(json.dumps(index), encoding="utf-8")
    node_types.write(topologies / "node_type_registry.json")
    split_dir = artifacts / "data" / "splits"
    split_dir.mkdir(parents=True)
    split = {"train": sample_ids[:4], "val": sample_ids[4:5], "test": sample_ids[5:]}
    (split_dir / "strict.json").write_text(json.dumps(split), encoding="utf-8")
    codebert_cache = MemmapFeatureCache.create(
        layout.node_codebert_dir,
        rows=len(texts),
        dim=8,
    )
    codebert_cache.write(list(range(len(texts))), np.ones((len(texts), 8), dtype=np.float32))
    config = {
        "paths": {"artifacts_dir": str(artifacts), "outputs_dir": str(tmp_path / "outputs")},
        "model": {"hidden_dim": 8, "node_type_dim": 4, "dropout": 0.0},
        "devign": {"hidden_dim": 16, "dropout": 0.2, "steps": 3, "weight_decay": 0.0000013},
        "training": {
            "epochs": 3,
            "patience": 1,
            "loss": "focal",
            "focal_gamma": 2.0,
            "class_weight": [1.2, 1.0],
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "max_nodes": 100,
            "max_edges": 100,
            "seed": 42,
            "device": "cpu",
        },
    }
    received: dict[str, object] = {}

    def fake_run_training(model, **kwargs):
        received["model"] = model
        received.update(kwargs)
        return {}

    import cpg_vuln.training.engine as engine

    monkeypatch.setattr(engine, "run_training", fake_run_training)

    train_devign(
        config,
        split="strict",
        view="core-cpg",
        embedding="codebert",
        run_name="devign-raw-v1-strict",
        checkpoint_metric="pr_auc",
        threshold_strategy="fixed_0_5",
        learning_rate=0.0001,
        epochs=1,
        force=True,
    )

    assert received["output_dir"] == layout.run_root / "devign-raw-v1-strict"
    assert received["config"].epochs == 1
    assert received["config"].checkpoint_metric == "pr_auc"
    assert received["config"].threshold_strategy == "fixed_0_5"
    assert received["config"].loss == "cross_entropy"
    assert received["config"].class_weight is None
    assert received["config"].weight_decay == 0.0000013
    assert received["model"].dropout.p == 0.2
    assert received["model"].ggnn.num_layers == 3
    assert len(received["train_dataset"]) == 4
    metadata = received["run_metadata"]
    assert metadata["kind"] == "devign"
    assert metadata["model_name"] == "devign"
    assert metadata["view"] == "core-cpg"
    assert metadata["embedding"] == "codebert"
    assert metadata["split"] == "strict"


def test_train_devign_can_apply_positive_class_weight(tmp_path: Path, monkeypatch) -> None:
    import cpg_vuln.training.runner as runner

    received: dict[str, object] = {}

    class FakeCache:
        class Metadata:
            dim = 8
            text_registry_sha256 = "sha"

        metadata = Metadata()

    class FakeNodeTypes:
        def __len__(self):
            return 3

    def fake_run_training(model, **kwargs):
        received.update(kwargs)
        return {}

    import cpg_vuln.training.engine as engine

    monkeypatch.setattr(runner, "_completed", lambda run_dir, *, force: False)
    monkeypatch.setattr(runner, "_topology_index", lambda layout: [{"sample_id": "s", "view": "core-cpg", "path": "s.pt", "nodes": 1, "edges": 0}])
    monkeypatch.setattr(runner, "_read_json", lambda path: {"train": ["s"], "val": ["s"], "test": ["s"]})
    monkeypatch.setattr(runner.NodeTypeRegistry, "read", lambda path: FakeNodeTypes())
    monkeypatch.setattr(runner, "_node_feature_cache", lambda layout, embedding: FakeCache())
    monkeypatch.setattr(runner, "_datasets", lambda *args, **kwargs: {"train": [object()], "val": [object()], "test": [object()]})
    monkeypatch.setattr(engine, "run_training", fake_run_training)

    runner.train_devign(
        {
            "paths": {
                "artifacts_dir": str(tmp_path / "artifacts"),
                "outputs_dir": str(tmp_path / "outputs"),
            },
            "model": {"hidden_dim": 8, "node_type_dim": 4, "dropout": 0.0},
            "training": {
                "epochs": 1,
                "patience": 1,
                "weight_decay": 0.0001,
                "gradient_clip": 1.0,
                "max_nodes": 100,
                "max_edges": 100,
                "seed": 42,
                "device": "cpu",
            },
        },
        positive_class_weight=1.3,
        force=True,
    )

    assert received["config"].class_weight == [1.0, 1.3]


def test_train_ramp_rejects_unsupported_experiment() -> None:
    from cpg_vuln.training.runner import train_ramp

    try:
        train_ramp({}, experiment="EX", split="strict")
    except ValueError as error:
        assert "unsupported RAMP experiment" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_ramp_run_dir_uses_explicit_run_name(tmp_path: Path) -> None:
    from cpg_vuln.training.runner import ramp_run_directory

    outputs = tmp_path / "outputs"

    assert ramp_run_directory(
        outputs,
        experiment="E4",
        split="strict",
        run_name=None,
    ).name == "ramp-E4-strict"
    assert ramp_run_directory(
        outputs,
        experiment="E4",
        split="strict",
        run_name="ramp-E4-v2A-fast-strict",
    ).name == "ramp-E4-v2A-fast-strict"


def test_ramp_config_overrides_do_not_mutate_base_config() -> None:
    from cpg_vuln.training.runner import ramp_settings_for_experiment

    config = {
        "training": {"seed": 42},
        "ramp": {
            "lambda_replay": 0.5,
            "lambda_rank": 0.25,
            "margin": 0.5,
            "max_pairs_per_positive": 3,
        },
    }

    _mining, ramp_config = ramp_settings_for_experiment(
        "E4",
        config,
        lambda_replay=0.15,
        lambda_rank=0.20,
        margin=0.25,
        max_pairs_per_positive=2,
    )

    assert ramp_config.lambda_replay == 0.15
    assert ramp_config.lambda_rank == 0.20
    assert ramp_config.margin == 0.25
    assert ramp_config.max_pairs_per_positive == 2
    assert config["ramp"]["lambda_replay"] == 0.5


def test_ramp_settings_keep_immediate_ranking_by_default() -> None:
    from cpg_vuln.training.runner import ramp_settings_for_experiment

    _mining, ramp_config = ramp_settings_for_experiment("E4", {"ramp": {"lambda_rank": 0.25}})

    assert ramp_config.lambda_auxiliary == 0.0
    assert ramp_config.rank_warmup_epochs == 0
    assert ramp_config.rank_ramp_epochs == 0


def test_v3_ramp_overrides_are_model_scoped() -> None:
    import cpg_vuln.training.runner as runner

    base = runner.RampConfig(lambda_rank=0.25)
    config = {
        "ramp_v3": {
            "lambda_auxiliary": 0.2,
            "rank_warmup_epochs": 5,
            "rank_ramp_epochs": 4,
        }
    }

    v2_config = runner._apply_model_ramp_overrides(
        "ramp-v2-rgcn",
        base,
        config,
    )
    v3_config = runner._apply_model_ramp_overrides(
        "ramp-v3-slice-mil",
        base,
        config,
    )

    assert v2_config == base
    assert v3_config.lambda_auxiliary == pytest.approx(0.2)
    assert v3_config.rank_warmup_epochs == 5
    assert v3_config.rank_ramp_epochs == 4


def test_ramp_settings_encode_e1_to_e4_negative_strategies() -> None:
    from cpg_vuln.mining.hard_negative_bank import MiningStrategy
    from cpg_vuln.training.runner import ramp_settings_for_experiment

    base = {
        "ramp": {
            "margin": 0.5,
            "lambda_replay": 0.5,
            "lambda_rank": 0.25,
            "pair_batch_size": 2,
            "warmup_epochs": 3,
            "max_pairs_per_positive": 3,
            "minimum_pair_score": 0.2,
            "semi_hard_lower_percentile": 0.05,
            "semi_hard_upper_percentile": 0.30,
        }
    }

    e1_mining, e1_ramp = ramp_settings_for_experiment("E1", base)
    assert e1_mining.strategy == MiningStrategy.RANDOM
    assert e1_ramp.lambda_replay == 0.5
    assert e1_ramp.lambda_rank == 0.0

    e2_mining, e2_ramp = ramp_settings_for_experiment("E2", base)
    assert e2_mining.strategy == MiningStrategy.FALSE_POSITIVE_ONLY
    assert e2_mining.motif_weight == 0.0
    assert e2_mining.structure_weight == 0.0
    assert e2_mining.scale_weight == 0.0
    assert e2_mining.false_positive_weight == 1.0
    assert e2_ramp.lambda_replay == 0.5
    assert e2_ramp.lambda_rank == 0.0

    e3_mining, e3_ramp = ramp_settings_for_experiment("E3", base)
    assert e3_mining.strategy == MiningStrategy.MOTIF_MATCHED
    assert e3_mining.false_positive_weight > 0.0
    assert e3_ramp.lambda_replay == 0.5
    assert e3_ramp.lambda_rank == 0.0

    e4_mining, e4_ramp = ramp_settings_for_experiment("E4", base)
    assert e4_mining.strategy == MiningStrategy.MOTIF_MATCHED
    assert e4_mining == e3_mining
    assert e4_ramp.lambda_replay == 0.5
    assert e4_mining.false_positive_weight > 0.0
    assert e4_ramp.lambda_rank == 0.25
    assert not hasattr(e4_ramp, "refresh_interval")


def test_ramp_strategy_manifest_contains_experiment_controls(tmp_path: Path) -> None:
    from cpg_vuln.training.runner import write_ramp_strategy_manifest

    write_ramp_strategy_manifest(
        tmp_path / "strategy.json",
        experiment="E2",
        mining_strategy="false_positive_only",
        mining_weights={
            "motif_weight": 0.0,
            "structure_weight": 0.0,
            "scale_weight": 0.0,
            "false_positive_weight": 1.0,
        },
        lambda_rank=0.0,
        split="strict",
        bank_mode="static",
        false_positive_source="train_warmup_snapshot",
        warmup_epochs=3,
        warmup_checkpoint_sha256="abc123",
        pair_count=10,
        bank_path="artifacts/retrieval/strict/E2/bank.jsonl",
        review_path="artifacts/retrieval/strict/E2/review_queue.jsonl",
    )

    payload = json.loads((tmp_path / "strategy.json").read_text(encoding="utf-8"))
    assert payload["experiment"] == "E2"
    assert payload["mining_strategy"] == "false_positive_only"
    assert payload["mining_weights"]["false_positive_weight"] == 1.0
    assert payload["lambda_rank"] == 0.0
    assert payload["bank_mode"] == "static"
    assert payload["false_positive_source"] == "train_warmup_snapshot"
    assert payload["warmup_epochs"] == 3
    assert payload["warmup_checkpoint_sha256"] == "abc123"
    assert payload["pair_count"] == 10


def test_false_positive_snapshot_keeps_only_train_labeled_negatives() -> None:
    import pytest

    from cpg_vuln.training.runner import false_positive_probabilities_from_prediction

    prediction = {
        "sample_ids": ["p1", "n1", "n2"],
        "labels": np.asarray([1, 0, 0], dtype=np.int64),
        "probabilities": np.asarray([0.9, 0.7, 0.2], dtype=np.float32),
    }

    probabilities = false_positive_probabilities_from_prediction(prediction)

    assert set(probabilities) == {"n1", "n2"}
    assert probabilities["n1"] == pytest.approx(0.7)
    assert probabilities["n2"] == pytest.approx(0.2)
