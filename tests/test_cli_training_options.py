from __future__ import annotations

from pathlib import Path

import yaml

import cpg_vuln.cli as cli
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
    NodeTextRegistry(["return 0"]).write(artifacts / "topologies" / "text_registry.json")
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
