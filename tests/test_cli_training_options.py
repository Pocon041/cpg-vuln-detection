from __future__ import annotations

from pathlib import Path

import yaml

import cpg_vuln.cli as cli
from cpg_vuln.data.layout import ArtifactLayout
from cpg_vuln.features.normalization import NormalizationSpec
from cpg_vuln.features.text import NodeTextRegistry


def test_cli_passes_requested_baseline_matrix_filters(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"paths": {"artifacts_dir": str(tmp_path / "artifacts")}}))
    received: dict[str, object] = {}

    def fake_train_baselines(config, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(cli, "train_baselines", fake_train_baselines)

    cli.main(
        [
            "--config",
            str(config),
            "train-baselines",
            "--embeddings",
            "word2vec",
            "--splits",
            "course",
            "--epochs",
            "1",
        ]
    )

    assert received == {
        "views": ("ast", "cfg", "pdg"),
        "embeddings": ("word2vec",),
        "splits": ("course",),
        "epochs": 1,
        "force": False,
    }


def test_cli_accepts_slice_fusion_enhanced_variant(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "outputs_dir": str(tmp_path / "outputs"),
                }
            }
        )
    )
    received: dict[str, object] = {}

    def fake_train_enhanced(config, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(cli, "train_enhanced", fake_train_enhanced)

    cli.main(
        [
            "--config",
            str(config),
            "train-enhanced",
            "--variants",
            "slice-fusion",
            "--splits",
            "strict",
            "--epochs",
            "1",
            "--force",
        ]
    )

    assert received == {
        "splits": ("strict",),
        "variants": ("slice-fusion",),
        "epochs": 1,
        "force": True,
    }


def test_cli_passes_word2vec_force_and_batch_size(tmp_path: Path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    NodeTextRegistry(["return 0"]).write(
        artifacts / "normalization" / "raw-v1" / "topologies" / "text_registry.json"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {"artifacts_dir": str(artifacts)},
                "features": {"word2vec_batch_size": 7},
            }
        )
    )
    received: dict[str, object] = {}

    def fake_build_word2vec_cache(registry, output_dir, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(cli, "build_word2vec_cache", fake_build_word2vec_cache)

    cli.main(["--config", str(config), "build-word2vec", "--force"])

    assert received["batch_size"] == 7
    assert received["force"] is True


def test_artifact_layout_uses_normalization_key(tmp_path: Path) -> None:
    layout = ArtifactLayout(
        artifacts_root=tmp_path / "artifacts",
        outputs_root=tmp_path / "outputs",
        spec=NormalizationSpec(mode="semantic-anon"),
    )

    assert layout.topology_dir == tmp_path / "artifacts" / "normalization" / "semantic-anon-v1" / "topologies"
    assert layout.word2vec_dir == tmp_path / "artifacts" / "normalization" / "semantic-anon-v1" / "features" / "word2vec"
    assert layout.node_codebert_dir == tmp_path / "artifacts" / "normalization" / "semantic-anon-v1" / "features" / "codebert" / "nodes"
    assert layout.function_codebert_dir == tmp_path / "artifacts" / "normalization" / "semantic-anon-v1" / "features" / "codebert" / "functions"
    assert layout.retrieval_dir == tmp_path / "artifacts" / "normalization" / "semantic-anon-v1" / "retrieval"
    assert layout.run_root == tmp_path / "outputs" / "runs" / "semantic-anon-v1"


def test_raw_artifact_layout_keeps_existing_function_cache_path(tmp_path: Path) -> None:
    layout = ArtifactLayout(
        artifacts_root=tmp_path / "artifacts",
        outputs_root=tmp_path / "outputs",
        spec=NormalizationSpec(mode="raw"),
    )

    assert layout.function_codebert_dir == tmp_path / "artifacts" / "features" / "codebert" / "functions-raw"


def test_cli_accepts_weak_baselines_command(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "outputs_dir": str(tmp_path / "outputs"),
                }
            }
        )
    )
    called: dict[str, object] = {}

    def fake_run_weak_baselines(config, *, split, view):
        called.update({"split": split, "view": view})

    monkeypatch.setattr(cli, "run_weak_baselines", fake_run_weak_baselines)

    cli.main(["--config", str(config), "weak-baselines", "--split", "strict", "--view", "core-cpg"])

    assert called == {"split": "strict", "view": "core-cpg"}


def test_cli_passes_train_ramp_options(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "outputs_dir": str(tmp_path / "outputs"),
                }
            }
        )
    )
    received: dict[str, object] = {}

    def fake_train_ramp(config, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(cli, "train_ramp", fake_train_ramp)

    cli.main(
        [
            "--config",
            str(config),
            "train-ramp",
            "--experiment",
            "E4",
            "--split",
            "strict",
            "--epochs",
            "1",
            "--force",
        ]
    )

    assert received == {
        "experiment": "E4",
        "split": "strict",
        "view": "core-cpg",
        "model_name": "selective-fusion",
        "run_name": None,
        "lambda_replay": None,
        "lambda_rank": None,
        "margin": None,
        "max_pairs_per_positive": None,
        "checkpoint_metric": None,
        "threshold_strategy": None,
        "learning_rate": None,
        "positive_class_weight": None,
        "checkpoint_min_ppr": None,
        "checkpoint_max_ppr": None,
        "checkpoint_max_recall": None,
        "evaluate_test": True,
        "epochs": 1,
        "force": True,
    }


def test_cli_passes_fast_proof_ramp_options(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "outputs_dir": str(tmp_path / "outputs"),
                }
            }
        )
    )
    received: dict[str, object] = {}

    def fake_train_ramp(config, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(cli, "train_ramp", fake_train_ramp)

    cli.main(
        [
            "--config",
            str(config),
            "train-ramp",
            "--experiment",
            "E4",
            "--split",
            "strict",
            "--view",
            "core-cpg",
            "--model",
            "ramp-v2-rgcn",
            "--run-name",
            "ramp-E4-v2A-fast-strict",
            "--lambda-replay",
            "0.25",
            "--lambda-rank",
            "0.20",
            "--margin",
            "0.35",
            "--max-pairs-per-positive",
            "2",
            "--checkpoint-metric",
            "pr_auc",
            "--threshold-strategy",
            "fixed_0_5",
            "--learning-rate",
            "0.0003",
            "--positive-class-weight",
            "2.0",
            "--checkpoint-min-ppr",
            "0.25",
            "--checkpoint-max-ppr",
            "0.75",
            "--checkpoint-max-recall",
            "0.90",
            "--defer-test",
            "--epochs",
            "1",
            "--force",
        ]
    )

    assert received["experiment"] == "E4"
    assert received["split"] == "strict"
    assert received["view"] == "core-cpg"
    assert received["model_name"] == "ramp-v2-rgcn"
    assert received["run_name"] == "ramp-E4-v2A-fast-strict"
    assert received["lambda_replay"] == 0.25
    assert received["lambda_rank"] == 0.20
    assert received["margin"] == 0.35
    assert received["max_pairs_per_positive"] == 2
    assert received["checkpoint_metric"] == "pr_auc"
    assert received["threshold_strategy"] == "fixed_0_5"
    assert received["learning_rate"] == 0.0003
    assert received["positive_class_weight"] == 2.0
    assert received["checkpoint_min_ppr"] == 0.25
    assert received["checkpoint_max_ppr"] == 0.75
    assert received["checkpoint_max_recall"] == 0.90
    assert received["evaluate_test"] is False
    assert received["epochs"] == 1
    assert received["force"] is True


def test_cli_accepts_dual_head_ramp_model(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "outputs_dir": str(tmp_path / "outputs"),
                }
            }
        )
    )
    received: dict[str, object] = {}

    def fake_train_ramp(config, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(cli, "train_ramp", fake_train_ramp)

    cli.main(
        [
            "--config",
            str(config),
            "train-ramp",
            "--model",
            "ramp-v2-dual",
        ]
    )

    assert received["model_name"] == "ramp-v2-dual"


def test_cli_passes_train_devign_options(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "outputs_dir": str(tmp_path / "outputs"),
                }
            }
        )
    )
    received: dict[str, object] = {}

    def fake_train_devign(config, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(cli, "train_devign", fake_train_devign)

    cli.main(
        [
            "--config",
            str(config),
            "train-devign",
            "--split",
            "strict",
            "--view",
            "core-cpg",
            "--embedding",
            "codebert",
            "--run-name",
            "devign-raw-v1-strict",
            "--checkpoint-metric",
            "pr_auc",
            "--threshold-strategy",
            "fixed_0_5",
            "--learning-rate",
            "0.0001",
            "--positive-class-weight",
            "1.2",
            "--epochs",
            "1",
            "--force",
        ]
    )

    assert received == {
        "split": "strict",
        "view": "core-cpg",
        "embedding": "codebert",
        "run_name": "devign-raw-v1-strict",
        "checkpoint_metric": "pr_auc",
        "threshold_strategy": "fixed_0_5",
        "learning_rate": 0.0001,
        "positive_class_weight": 1.2,
        "evaluate_test": True,
        "epochs": 1,
        "force": True,
    }


def test_cli_passes_evaluate_ramp_options(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "outputs_dir": str(tmp_path / "outputs"),
                }
            }
        )
    )
    received: dict[str, object] = {}

    def fake_evaluate_ramp(config, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(cli, "evaluate_ramp", fake_evaluate_ramp)

    cli.main(
        [
            "--config",
            str(config),
            "evaluate-ramp",
            "--run",
            "ramp-E4-v2A-fast-strict",
            "--split",
            "strict",
            "--export-attention",
        ]
    )

    assert received == {
        "run": "ramp-E4-v2A-fast-strict",
        "split": "strict",
        "export_attention": True,
    }


def test_cli_accepts_audit_hard_pairs_command(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "outputs_dir": str(tmp_path / "outputs"),
                }
            }
        )
    )
    called: dict[str, object] = {}

    def fake_write_pair_audit_sample(bank_path, output_path, *, limit=100, seed=42):
        called.update(
            {
                "bank_path": bank_path,
                "output_path": output_path,
                "limit": limit,
                "seed": seed,
            }
        )

    monkeypatch.setattr(cli, "write_pair_audit_sample", fake_write_pair_audit_sample)

    cli.main(
        [
            "--config",
            str(config),
            "audit-hard-pairs",
            "--split",
            "strict",
            "--experiment",
            "E3",
            "--limit",
            "5",
            "--seed",
            "7",
        ]
    )

    assert called["bank_path"] == tmp_path / "artifacts" / "normalization" / "raw-v1" / "retrieval" / "strict" / "E3" / "bank.jsonl"
    assert called["output_path"] == tmp_path / "outputs" / "reports" / "raw-v1" / "pair_audit_strict_E3.json"
    assert called["limit"] == 5
    assert called["seed"] == 7


def test_cli_accepts_visualize_attention_command(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "outputs_dir": str(tmp_path / "outputs"),
                },
                "source_mapping": {
                    "source_map_path": str(tmp_path / "source_map.csv"),
                },
            }
        )
    )
    called: dict[str, object] = {}

    def fake_export_attention_dashboard(*args, **kwargs):
        called["args"] = args
        called.update(kwargs)

    monkeypatch.setattr(cli, "export_attention_dashboard", fake_export_attention_dashboard)

    cli.main(
        [
            "--config",
            str(config),
            "visualize-attention",
            "--run",
            "ramp-E4-strict",
            "--top-samples",
            "3",
            "--top-lines",
            "5",
            "--context",
            "2",
        ]
    )

    assert called["args"] == (
        tmp_path / "outputs" / "runs" / "raw-v1" / "ramp-E4-strict" / "node_attention.json",
        tmp_path / "outputs" / "runs" / "raw-v1" / "ramp-E4-strict" / "predictions.csv",
        tmp_path / "source_map.csv",
        tmp_path / "outputs" / "reports" / "raw-v1" / "attention" / "ramp-E4-strict",
    )
    assert called["run_name"] == "ramp-E4-strict"
    assert called["top_samples"] == 3
    assert called["top_lines"] == 5
    assert called["context_radius"] == 2
