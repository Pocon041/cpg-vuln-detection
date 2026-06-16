from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data

import cpg_vuln.training.engine as engine
from cpg_vuln.data.batch import GraphSize
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


def test_training_can_defer_strict_test_metrics(tmp_path: Path) -> None:
    graphs = _graphs()
    model = GCNClassifier(input_dim=8, num_node_types=3, hidden_dim=8)

    result = run_training(
        model,
        train_dataset=graphs[:4],
        val_dataset=graphs[4:6],
        test_dataset=graphs[6:],
        output_dir=tmp_path,
        config=TrainConfig(
            epochs=1,
            patience=1,
            device="cpu",
            max_nodes=100,
            max_edges=100,
            evaluate_test=False,
        ),
    )

    assert result["evaluation_mode"] == "validation_only"
    assert result["selection_source"] == "validation"
    assert result["validation_val_f1"]["samples"] == 2
    assert result["validation_val_mcc"]["samples"] == 2
    assert result["test"] is None
    assert result["test_fixed_0_5"] is None
    assert result["test_val_f1"] is None
    assert result["test_val_mcc"] is None
    assert not (tmp_path / "predictions.csv").is_file()


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


def test_epoch_summary_reports_predicted_positive_rate() -> None:
    summary = engine._epoch_summary(
        epoch=1,
        loss=0.5,
        metrics=_metrics(f1=0.6, roc_auc=0.7),
        checkpoint_metric="pr_auc",
        best_checkpoint_score=0.8,
        remaining_patience=3,
    )

    assert "val_ppr=0.5000" in summary


def test_epoch_loss_is_weighted_by_batch_sample_count() -> None:
    assert engine._sample_weighted_mean_loss([(1.0, 1), (3.0, 3)]) == 2.5


def test_checkpoint_score_can_guard_predicted_positive_rate() -> None:
    valid_metrics = _metrics(f1=0.7, roc_auc=0.8)
    invalid_metrics = {**valid_metrics, "predicted_positive_rate": 0.95}

    assert (
        engine._checkpoint_score(
            "f1",
            loss=0.5,
            metrics=valid_metrics,
            checkpoint_min_ppr=0.25,
            checkpoint_max_ppr=0.75,
            checkpoint_max_recall=0.9,
        )
        == 0.7
    )
    assert (
        engine._checkpoint_score(
            "f1",
            loss=0.5,
            metrics=invalid_metrics,
            checkpoint_min_ppr=0.25,
            checkpoint_max_ppr=0.75,
            checkpoint_max_recall=0.9,
        )
        is None
    )


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


def test_loader_skips_oversized_graphs_before_fetching_dataset_item() -> None:
    class SizedDataset:
        graph_sizes = [
            GraphSize("small", nodes=2, edges=2),
            GraphSize("huge", nodes=9000, edges=70000),
        ]

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> Data:
            if index == 1:
                raise AssertionError("oversized graph should not be fetched")
            return Data(
                x=torch.randn(2, 1),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                y=torch.tensor([0]),
                sample_id="small",
            )

    loader, _ = engine._loader(
        SizedDataset(),
        config=TrainConfig(device="cpu", max_nodes=8000, max_edges=60000),
        shuffle=False,
    )

    batches = list(loader)

    assert len(batches) == 1
    assert batches[0].sample_id == ["small"]


def test_focal_loss_downweights_easy_examples() -> None:
    logits = torch.tensor([[4.0, -4.0], [-4.0, 4.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    cross_entropy = engine._classification_loss(
        logits,
        targets,
        loss_name="cross_entropy",
        focal_gamma=2.0,
        class_weight=None,
    )
    focal = engine._classification_loss(
        logits,
        targets,
        loss_name="focal",
        focal_gamma=2.0,
        class_weight=None,
    )

    assert focal < cross_entropy


def test_ramp_training_uses_pair_ranking_loss(tmp_path: Path) -> None:
    from cpg_vuln.mining.hard_negative_bank import HardNegativePair
    from cpg_vuln.training.ramp import RampConfig, run_ramp_training

    class FixedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, batch: Data) -> SimpleNamespace:
            labels = batch.y.view(-1).float()
            logits = torch.stack((1 - labels, labels), dim=-1) * self.weight
            return SimpleNamespace(logits=logits, node_attention=torch.ones(batch.x.shape[0]))

    graphs = _graphs()
    pairs = [HardNegativePair("sample-1", "sample-0", 0.9, 0.9, 0.9, 0.9, 0.8)]

    result = run_ramp_training(
        FixedModel(),
        train_dataset=graphs[:4],
        val_dataset=graphs[4:6],
        test_dataset=graphs[6:],
        output_dir=tmp_path,
        config=TrainConfig(epochs=1, patience=1, device="cpu", max_nodes=100, max_edges=100),
        ramp_config=RampConfig(lambda_rank=0.25, margin=0.5, pair_batch_size=1),
        initial_pairs=pairs,
    )

    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "predictions.csv").is_file()
    assert (tmp_path / "node_attention.json").is_file()
    assert result["run_metadata"]["training_mode"] == "ramp"
    assert result["run_metadata"]["bank_mode"] == "static"
    assert result["run_metadata"]["initial_pair_count"] == 1


def test_ramp_training_can_defer_strict_test_metrics(tmp_path: Path) -> None:
    from cpg_vuln.training.ramp import RampConfig, run_ramp_training

    class FixedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, batch: Data) -> SimpleNamespace:
            labels = batch.y.view(-1).float()
            logits = torch.stack((1 - labels, labels), dim=-1) * self.weight
            return SimpleNamespace(logits=logits, node_attention=None)

    result = run_ramp_training(
        FixedModel(),
        train_dataset=_graphs()[:4],
        val_dataset=_graphs()[4:6],
        test_dataset=_graphs()[6:],
        output_dir=tmp_path,
        config=TrainConfig(
            epochs=1,
            patience=1,
            device="cpu",
            max_nodes=100,
            max_edges=100,
            evaluate_test=False,
        ),
        ramp_config=RampConfig(lambda_rank=0.0),
        initial_pairs=[],
    )

    assert result["evaluation_mode"] == "validation_only"
    assert result["selection_source"] == "validation"
    assert result["validation_val_f1"]["samples"] == 2
    assert result["test"] is None
    assert not (tmp_path / "predictions.csv").is_file()


def test_ramp_history_logs_loss_parts_with_separate_denominators(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import cpg_vuln.training.ramp as ramp

    class FixedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, batch: Data) -> SimpleNamespace:
            logits = torch.stack((1 - batch.y.float(), batch.y.float()), dim=-1) * self.weight
            return SimpleNamespace(logits=logits, node_attention=None)

    calls = iter(
        [
            (
                torch.tensor(1.0, requires_grad=True),
                {"loss": 1.0, "main_loss": 0.4, "replay_loss": 0.2, "ranking_loss": 0.1},
            ),
            (
                torch.tensor(3.0, requires_grad=True),
                {"loss": 3.0, "main_loss": 0.8, "replay_loss": 0.6, "ranking_loss": 0.5},
            ),
        ]
    )

    def fake_compute_ramp_loss(**kwargs):
        return next(calls)

    monkeypatch.setattr(ramp, "compute_ramp_loss", fake_compute_ramp_loss)

    ramp.run_ramp_training(
        FixedModel(),
        train_dataset=_graphs()[:4],
        val_dataset=_graphs()[4:6],
        test_dataset=_graphs()[6:],
        output_dir=tmp_path,
        config=TrainConfig(
            epochs=1,
            patience=1,
            device="cpu",
            max_nodes=6,
            max_edges=10,
            evaluate_test=False,
        ),
        ramp_config=ramp.RampConfig(lambda_rank=0.0),
        initial_pairs=[],
    )

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert history[0]["main_loss"] == pytest.approx(0.6)
    assert history[0]["replay_loss"] == 0.0
    assert history[0]["ranking_loss"] == 0.0
    assert "val_f1" in history[0]
    assert "val_precision" in history[0]
    assert "val_recall" in history[0]
    assert "val_ppr" in history[0]
    assert history[0]["replay_pairs"] == 0
    assert history[0]["ranking_pairs"] == 0
    assert history[0]["objective_per_step"] == pytest.approx(2.0)


def test_ramp_training_reports_epoch_progress(tmp_path: Path, monkeypatch) -> None:
    import cpg_vuln.training.ramp as ramp

    class RecordingProgress:
        def __init__(self, iterable, **kwargs) -> None:
            self.iterable = iterable
            self.kwargs = kwargs

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs) -> None:
            pass

    calls: list[dict[str, object]] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return RecordingProgress(iterable, **kwargs)

    class FixedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, batch: Data) -> SimpleNamespace:
            labels = batch.y.view(-1).float()
            logits = torch.stack((1 - labels, labels), dim=-1) * self.weight
            return SimpleNamespace(logits=logits, node_attention=None)

    monkeypatch.setattr(ramp, "tqdm", fake_tqdm)
    graphs = _graphs()

    ramp.run_ramp_training(
        FixedModel(),
        train_dataset=graphs[:4],
        val_dataset=graphs[4:6],
        test_dataset=graphs[6:],
        output_dir=tmp_path,
        config=TrainConfig(epochs=1, patience=1, device="cpu", max_nodes=100, max_edges=100),
        ramp_config=ramp.RampConfig(lambda_rank=0.0),
        initial_pairs=[],
    )

    assert calls[0]["desc"] == f"ramp epochs ({tmp_path.name})"
    assert calls[0]["unit"] == "epoch"


def test_ramp_training_persists_progress_before_later_epoch_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import cpg_vuln.training.ramp as ramp

    class FixedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, batch: Data) -> SimpleNamespace:
            labels = batch.y.view(-1).float()
            logits = torch.stack((1 - labels, labels), dim=-1) * self.weight
            return SimpleNamespace(logits=logits, node_attention=None)

    prediction_calls = 0

    def flaky_predict(*args):
        nonlocal prediction_calls
        prediction_calls += 1
        if prediction_calls == 2:
            raise RuntimeError("validation interrupted")
        return {
            "labels": np.asarray([0, 1], dtype=np.int64),
            "probabilities": np.asarray([0.1, 0.9], dtype=np.float32),
            "sample_ids": ["sample-0", "sample-1"],
            "explanations": {},
        }

    monkeypatch.setattr(ramp, "_predict", flaky_predict)

    with pytest.raises(RuntimeError, match="validation interrupted"):
        ramp.run_ramp_training(
            FixedModel(),
            train_dataset=_graphs()[:4],
            val_dataset=_graphs()[4:6],
            test_dataset=_graphs()[6:],
            output_dir=tmp_path,
            config=TrainConfig(epochs=2, patience=2, device="cpu", max_nodes=100, max_edges=100),
            ramp_config=ramp.RampConfig(lambda_rank=0.0),
            initial_pairs=[],
        )

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert history[0]["epoch"] == 1
    assert (tmp_path / "best.pt").is_file()


def test_ramp_checkpoint_guard_does_not_consume_patience(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import cpg_vuln.training.ramp as ramp

    class FixedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, batch: Data) -> SimpleNamespace:
            labels = batch.y.view(-1).float()
            logits = torch.stack((1 - labels, labels), dim=-1) * self.weight
            return SimpleNamespace(logits=logits, node_attention=None)

    metric_values = iter(
        [
            _metrics(f1=0.5, roc_auc=0.6),
            {**_metrics(f1=0.9, roc_auc=0.6), "predicted_positive_rate": 0.95, "recall": 1.0},
            _metrics(f1=0.6, roc_auc=0.6),
            _metrics(f1=0.6, roc_auc=0.6),
        ]
    )

    def fake_threshold_metrics(*args, **kwargs):
        metrics = next(metric_values)
        return {"fixed_0_5": metrics, "val_f1": metrics, "val_mcc": metrics}

    monkeypatch.setattr(
        ramp,
        "_predict",
        lambda *args: {
            "labels": np.asarray([0, 1], dtype=np.int64),
            "probabilities": np.asarray([0.1, 0.9], dtype=np.float32),
            "sample_ids": ["sample-0", "sample-1"],
            "explanations": {},
        },
    )
    monkeypatch.setattr(
        ramp,
        "select_validation_thresholds",
        lambda *args, **kwargs: {"fixed_0_5": 0.5, "val_f1": 0.5, "val_mcc": 0.5},
    )
    monkeypatch.setattr(ramp, "metrics_at_validation_thresholds", fake_threshold_metrics)

    result = ramp.run_ramp_training(
        FixedModel(),
        train_dataset=_graphs()[:4],
        val_dataset=_graphs()[4:6],
        test_dataset=_graphs()[6:],
        output_dir=tmp_path,
        config=TrainConfig(
            epochs=3,
            patience=1,
            device="cpu",
            max_nodes=100,
            max_edges=100,
            checkpoint_metric="f1",
            threshold_strategy="fixed_0_5",
            checkpoint_max_ppr=0.75,
            checkpoint_max_recall=0.9,
            evaluate_test=False,
        ),
        ramp_config=ramp.RampConfig(lambda_rank=0.0),
        initial_pairs=[],
    )

    assert result["best_epoch"] == 3


def test_training_selects_checkpoint_by_validation_roc_auc(tmp_path: Path, monkeypatch) -> None:
    metric_values = iter(
        [
            _metrics(f1=0.9, roc_auc=0.6),
            _metrics(f1=0.7, roc_auc=0.8),
            _metrics(f1=0.7, roc_auc=0.8),
            _metrics(f1=0.6, roc_auc=0.7),
            _metrics(f1=0.7, roc_auc=0.8),
            _metrics(f1=0.7, roc_auc=0.8),
        ]
    )
    prediction = {
        "labels": np.asarray([0, 1], dtype=np.int64),
        "probabilities": np.asarray([0.1, 0.9], dtype=np.float32),
        "sample_ids": ["sample-0", "sample-1"],
        "explanations": {},
    }
    monkeypatch.setattr(engine, "_train_epoch", lambda *args, **kwargs: 0.5)
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


def test_training_can_select_checkpoint_by_validation_f1(tmp_path: Path, monkeypatch) -> None:
    metric_values = iter(
        [
            _metrics(f1=0.9, roc_auc=0.6),
            _metrics(f1=0.7, roc_auc=0.8),
            _metrics(f1=0.9, roc_auc=0.6),
            _metrics(f1=0.5, roc_auc=0.5),
            _metrics(f1=0.9, roc_auc=0.6),
            _metrics(f1=0.5, roc_auc=0.5),
        ]
    )
    prediction = {
        "labels": np.asarray([0, 1], dtype=np.int64),
        "probabilities": np.asarray([0.1, 0.9], dtype=np.float32),
        "sample_ids": ["sample-0", "sample-1"],
        "explanations": {},
    }
    monkeypatch.setattr(engine, "_train_epoch", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(engine, "_predict", lambda *args: prediction)
    monkeypatch.setattr(engine, "classification_metrics", lambda *args, **kwargs: next(metric_values))

    result = run_training(
        nn.Linear(1, 2),
        train_dataset=[],
        val_dataset=[],
        test_dataset=[],
        output_dir=tmp_path,
        config=TrainConfig(epochs=2, patience=2, device="cpu", checkpoint_metric="f1"),
    )

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert result["best_epoch"] == 1
    assert history[0]["checkpoint_score"] == 0.9
    assert result["config"]["checkpoint_metric"] == "f1"


def test_training_saves_fallback_checkpoint_and_stops_when_validation_roc_auc_is_none(
    tmp_path: Path, monkeypatch
) -> None:
    prediction = {
        "labels": np.asarray([0, 0], dtype=np.int64),
        "probabilities": np.asarray([0.1, 0.2], dtype=np.float32),
        "sample_ids": ["sample-0", "sample-1"],
        "explanations": {},
    }
    monkeypatch.setattr(engine, "_train_epoch", lambda *args, **kwargs: 0.5)
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
        "specificity": 0.5,
        "balanced_accuracy": 0.5,
        "mcc": 0.0,
        "predicted_positive_rate": 0.5,
        "roc_auc": roc_auc,
        "pr_auc": 0.5,
        "confusion_matrix": [[1, 0], [1, 0]],
    }
