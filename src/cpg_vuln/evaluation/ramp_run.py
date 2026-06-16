from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import nn

from cpg_vuln.training.engine import TrainConfig, _loader, _predict, _write_json
from cpg_vuln.training.thresholds import metrics_at_validation_thresholds


def evaluate_ramp_checkpoint(
    *,
    model_factory: Callable[[dict[str, object]], nn.Module],
    run_dir: Path,
    test_dataset: Sequence,
    config: TrainConfig,
    export_attention: bool,
) -> dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best.pt"
    development_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    thresholds = development_metrics["validation_thresholds"]
    metadata = development_metrics.get("run_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("run_metadata must be a JSON object")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = model_factory(metadata)
    model.load_state_dict(checkpoint["model_state"])

    device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_loader, _ = _loader(test_dataset, config=config, shuffle=False)
    prediction = _predict(model, test_loader, device)
    test_metrics = metrics_at_validation_thresholds(
        prediction["labels"],
        prediction["probabilities"],
        thresholds,
    )

    result: dict[str, object] = {
        "evaluation_mode": "final_test",
        "selection_source": "validation",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "validation_thresholds": thresholds,
        "test_fixed_0_5": test_metrics["fixed_0_5"],
        "test_val_f1": test_metrics["val_f1"],
        "test_val_mcc": test_metrics["val_mcc"],
    }
    _write_json(run_dir / "final_test_metrics.json", result)
    if export_attention and prediction["explanations"]:
        _write_json(run_dir / "final_node_attention.json", prediction["explanations"])
    return result
