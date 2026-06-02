from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

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


def test_train_epoch_passes_actual_batch_sizes_to_weighted_loss(monkeypatch) -> None:
    class BatchSizeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = nn.Parameter(torch.zeros(2))

        def forward(self, batch: Data) -> SimpleNamespace:
            return SimpleNamespace(logits=self.logits.expand(batch.y.numel(), -1))

    captured: list[tuple[float, int]] = []

    def capture_weighted_loss(losses_and_samples) -> float:
        captured.extend(losses_and_samples)
        return 123.0

    model = BatchSizeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    monkeypatch.setattr(engine, "_sample_weighted_mean_loss", capture_weighted_loss)

    loss = engine._train_epoch(
        model,
        [Data(y=torch.tensor([0])), Data(y=torch.tensor([0, 1, 0]))],
        optimizer,
        scaler,
        torch.device("cpu"),
        gradient_clip=1.0,
    )

    assert [samples for _, samples in captured] == [1, 3]
    assert loss == 123.0


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


def test_training_saves_fallback_checkpoint_and_stops_when_validation_roc_auc_is_none(
    tmp_path: Path, monkeypatch
) -> None:
    prediction = {
        "labels": np.asarray([0, 0], dtype=np.int64),
        "probabilities": np.asarray([0.1, 0.2], dtype=np.float32),
        "sample_ids": ["sample-0", "sample-1"],
        "explanations": {},
    }
    monkeypatch.setattr(engine, "_train_epoch", lambda *args: 0.5)
    monkeypatch.setattr(engine, "_predict", lambda *args: prediction)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_training(
            nn.Linear(1, 2),
            train_dataset=[],
            val_dataset=[],
            test_dataset=[],
            output_dir=tmp_path,
            config=TrainConfig(epochs=10, patience=2, device="cpu"),
        )

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert (tmp_path / "best.pt").is_file()
    assert result["best_epoch"] == 1
    assert result["validation"]["roc_auc"] is None
    assert [epoch["val_roc_auc"] for epoch in history] == [None, None, None]


def _metrics(*, f1: float, roc_auc: float | None) -> dict[str, object]:
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
