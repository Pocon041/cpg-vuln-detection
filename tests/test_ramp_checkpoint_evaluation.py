from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch_geometric.data import Data

from cpg_vuln.evaluation.ramp_run import evaluate_ramp_checkpoint
from cpg_vuln.training.engine import TrainConfig


class FixedEvalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: Data) -> SimpleNamespace:
        labels = batch.y.view(-1).float()
        logits = torch.stack((1 - labels, labels), dim=-1) * self.weight
        return SimpleNamespace(logits=logits, node_attention=None)


def _graph(sample_id: str, label: int) -> Data:
    return Data(
        x=torch.randn(2, 4),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_type=torch.tensor([0], dtype=torch.long),
        node_type_id=torch.tensor([0, 1], dtype=torch.long),
        y=torch.tensor([label]),
        sample_id=sample_id,
    )


def test_evaluate_ramp_checkpoint_uses_existing_thresholds_without_training(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = FixedEvalModel()
    torch.save({"model_state": model.state_dict(), "epoch": 3}, run_dir / "best.pt")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "evaluation_mode": "validation_only",
                "selection_source": "validation",
                "validation_thresholds": {
                    "fixed_0_5": 0.5,
                    "val_f1": 0.5,
                    "val_mcc": 0.5,
                },
                "run_metadata": {"model_name": "fixed"},
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_ramp_checkpoint(
        model_factory=lambda metadata: FixedEvalModel(),
        run_dir=run_dir,
        test_dataset=[_graph("n", 0), _graph("p", 1)],
        config=TrainConfig(epochs=1, device="cpu", max_nodes=100, max_edges=100),
        export_attention=False,
    )

    assert result["selection_source"] == "validation"
    assert result["checkpoint_path"].endswith("best.pt")
    assert result["test_val_f1"]["samples"] == 2
    assert (run_dir / "final_test_metrics.json").is_file()
