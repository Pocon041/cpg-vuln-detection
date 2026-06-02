from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

import cpg_vuln.training.engine as engine
from cpg_vuln.models.gcn import GCNClassifier
from cpg_vuln.training.engine import TrainConfig, run_training


def _graphs() -> list[Data]:
    graphs = []
    for index in range(8):
        graphs.append(
            Data(
                x=torch.randn(3, 8),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
                node_type_id=torch.tensor([0, 1, 2], dtype=torch.long),
                y=torch.tensor([index % 2], dtype=torch.long),
                sample_id=f"sample-{index}",
                line_numbers=torch.tensor([1, 2, 3], dtype=torch.long),
            )
        )
    return graphs


def test_training_smoke_writes_metrics_predictions_and_checkpoint(tmp_path: Path) -> None:
    graphs = _graphs()
    model = GCNClassifier(input_dim=8, num_node_types=3, hidden_dim=8)

    result = run_training(
        model,
        train_dataset=graphs[:4],
        val_dataset=graphs[4:6],
        test_dataset=graphs[6:],
        output_dir=tmp_path,
        config=TrainConfig(epochs=1, patience=1, device="cpu", max_nodes=100, max_edges=100),
        run_metadata={"feature_dim": 8},
    )

    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "predictions.csv").is_file()
    assert result["test"]["samples"] == 2
    assert result["run_metadata"]["feature_dim"] == 8


def test_training_reports_epoch_progress_and_final_metrics(tmp_path: Path, capsys) -> None:
    graphs = _graphs()
    model = GCNClassifier(input_dim=8, num_node_types=3, hidden_dim=8)

    run_training(
        model,
        train_dataset=graphs[:4],
        val_dataset=graphs[4:6],
        test_dataset=graphs[6:],
        output_dir=tmp_path,
        config=TrainConfig(epochs=1, patience=1, device="cpu", max_nodes=100, max_edges=100),
    )

    captured = capsys.readouterr()
    assert "train epochs" in captured.err
    assert "val_f1" in captured.err
    assert "validation metrics:" in captured.out
    assert "test metrics:" in captured.out


def test_epoch_loss_is_weighted_by_batch_sample_count() -> None:
    assert engine._sample_weighted_mean_loss([(1.0, 1), (3.0, 3)]) == 2.5


def test_training_selects_checkpoint_by_validation_roc_auc(tmp_path: Path, monkeypatch) -> None:
    metric_values = iter(
        [
            _metrics(f1=0.9, roc_auc=0.6),
            _metrics(f1=0.7, roc_auc=0.8),
            _metrics(f1=0.7, roc_auc=0.8),
            _metrics(f1=0.6, roc_auc=0.7),
        ]
    )
    prediction = {
        "labels": np.asarray([0, 1], dtype=np.int64),
        "probabilities": np.asarray([0.1, 0.9], dtype=np.float32),
        "sample_ids": ["sample-0", "sample-1"],
        "explanations": {},
    }
    monkeypatch.setattr(engine, "_train_epoch", lambda *args: 0.5)
    monkeypatch.setattr(engine, "_predict", lambda *args: prediction)
    monkeypatch.setattr(engine, "classification_metrics", lambda *args, **kwargs: next(metric_values))

    result = run_training(
        nn.Linear(1, 2),
        train_dataset=[],
        val_dataset=[],
        test_dataset=[],
        output_dir=tmp_path,
        config=TrainConfig(epochs=2, patience=2, device="cpu"),
    )

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert result["best_epoch"] == 2
    assert history[1]["val_roc_auc"] == 0.8


def _metrics(*, f1: float, roc_auc: float) -> dict[str, object]:
    return {
        "samples": 2,
        "threshold": 0.5,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": 0.5,
        "confusion_matrix": [[1, 0], [1, 0]],
    }
