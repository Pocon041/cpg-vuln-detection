from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from cpg_vuln.data.batch import DynamicBatchSampler, GraphSize
from cpg_vuln.training.metrics import classification_metrics, select_f1_threshold


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    max_nodes: int = 8000
    max_edges: int = 60000
    seed: int = 42
    device: str = "cuda"


def run_training(
    model: nn.Module,
    *,
    train_dataset: Sequence,
    val_dataset: Sequence,
    test_dataset: Sequence,
    output_dir: Path,
    config: TrainConfig,
    run_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    _seed_everything(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    train_loader, train_sampler = _loader(train_dataset, config=config, shuffle=True)
    val_loader, _ = _loader(val_dataset, config=config, shuffle=False)
    test_loader, _ = _loader(test_dataset, config=config, shuffle=False)
    best_roc_auc: float | None = None
    has_checkpoint = False
    remaining_patience = config.patience
    history: list[dict[str, float | int | None]] = []
    started = time.perf_counter()
    epoch_progress = tqdm(
        range(config.epochs),
        desc=f"train epochs ({output_dir.name})",
        unit="epoch",
    )
    for epoch in epoch_progress:
        train_sampler.set_epoch(epoch)
        loss = _train_epoch(model, train_loader, optimizer, scaler, device, config.gradient_clip)
        validation = _predict(model, val_loader, device)
        threshold = select_f1_threshold(validation["labels"], validation["probabilities"])
        validation_metrics = classification_metrics(
            validation["labels"], validation["probabilities"], threshold=threshold
        )
        validation_roc_auc = validation_metrics["roc_auc"]
        history.append(
            {
                "epoch": epoch + 1,
                "loss": loss,
                "val_f1": validation_metrics["f1"],
                "val_roc_auc": validation_roc_auc,
            }
        )
        should_stop = False
        has_better_roc_auc = validation_roc_auc is not None and (
            best_roc_auc is None or float(validation_roc_auc) > best_roc_auc
        )
        if has_better_roc_auc or not has_checkpoint:
            if has_better_roc_auc:
                best_roc_auc = float(validation_roc_auc)
            remaining_patience = config.patience
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch + 1, "threshold": threshold},
                output_dir / "best.pt",
            )
            has_checkpoint = True
        else:
            remaining_patience -= 1
            should_stop = remaining_patience <= 0
        epoch_progress.set_postfix(
            loss=f"{loss:.4f}",
            val_f1=_format_metric(validation_metrics["f1"]),
            best_roc_auc=_format_metric(best_roc_auc),
            patience=remaining_patience,
        )
        tqdm.write(
            _epoch_summary(
                epoch=epoch + 1,
                loss=loss,
                metrics=validation_metrics,
                best_roc_auc=best_roc_auc,
                remaining_patience=remaining_patience,
            )
        )
        if should_stop:
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation = _predict(model, val_loader, device)
    threshold = select_f1_threshold(validation["labels"], validation["probabilities"])
    test = _predict(model, test_loader, device)
    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "config": asdict(config),
        "run_metadata": run_metadata or {},
        "elapsed_seconds": elapsed,
        "best_epoch": checkpoint["epoch"],
        "validation": classification_metrics(
            validation["labels"], validation["probabilities"], threshold=threshold
        ),
        "test": classification_metrics(test["labels"], test["probabilities"], threshold=threshold),
    }
    _write_json(output_dir / "metrics.json", result)
    _write_json(output_dir / "history.json", history)
    _write_predictions(output_dir / "predictions.csv", test, threshold)
    if test["explanations"]:
        _write_json(output_dir / "node_attention.json", test["explanations"])
    print(_metrics_summary("validation", result["validation"]))
    print(_metrics_summary("test", result["test"]))
    return result


def _train_epoch(model, loader, optimizer, scaler, device, gradient_clip: float) -> float:
    model.train()
    losses_and_samples: list[tuple[float, int]] = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(batch)
                loss = nn.functional.cross_entropy(output.logits, batch.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        except torch.OutOfMemoryError as error:
            sample_ids = getattr(batch, "sample_id", "<unknown>")
            raise RuntimeError(f"CUDA OOM while processing samples: {sample_ids}") from error
        losses_and_samples.append((float(loss.detach().cpu()), int(batch.y.numel())))
    return _sample_weighted_mean_loss(losses_and_samples)


def _sample_weighted_mean_loss(losses_and_samples: Sequence[tuple[float, int]]) -> float:
    if not losses_and_samples:
        return 0.0
    total_samples = sum(samples for _, samples in losses_and_samples)
    if total_samples <= 0:
        raise ValueError("total sample count must be positive")
    return sum(loss * samples for loss, samples in losses_and_samples) / total_samples


@torch.inference_mode()
def _predict(model, loader, device) -> dict[str, object]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    sample_ids: list[str] = []
    explanations: dict[str, list[dict[str, float | int]]] = {}
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        probabilities.extend(torch.softmax(output.logits, dim=-1)[:, 1].cpu().tolist())
        labels.extend(batch.y.cpu().tolist())
        ids = list(batch.sample_id) if isinstance(batch.sample_id, (list, tuple)) else [batch.sample_id]
        sample_ids.extend(ids)
        if output.node_attention is not None:
            ptr = batch.ptr.cpu().tolist()
            attention = output.node_attention.cpu()
            lines = batch.line_numbers.cpu()
            for index, sample_id in enumerate(ids):
                start, end = ptr[index], ptr[index + 1]
                explanations[sample_id] = [
                    {"line": int(lines[node]), "score": float(attention[node])}
                    for node in range(start, end)
                ]
    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "probabilities": np.asarray(probabilities, dtype=np.float32),
        "sample_ids": sample_ids,
        "explanations": explanations,
    }


def _loader(dataset: Sequence, *, config: TrainConfig, shuffle: bool):
    sizes = getattr(dataset, "graph_sizes", None)
    if sizes is None:
        sizes = [
            GraphSize(str(data.sample_id), nodes=int(data.num_nodes), edges=int(data.num_edges))
            for data in dataset
        ]
    sampler = DynamicBatchSampler(
        sizes,
        max_nodes=config.max_nodes,
        max_edges=config.max_edges,
        shuffle=shuffle,
        seed=config.seed,
    )
    return DataLoader(dataset, batch_sampler=sampler), sampler


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_predictions(path: Path, prediction: dict[str, object], threshold: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "probability", "prediction"])
        for sample_id, label, probability in zip(
            prediction["sample_ids"],
            prediction["labels"],
            prediction["probabilities"],
            strict=True,
        ):
            writer.writerow([sample_id, int(label), float(probability), int(probability >= threshold)])


def _epoch_summary(
    *,
    epoch: int,
    loss: float,
    metrics: dict[str, object],
    best_roc_auc: float | None,
    remaining_patience: int,
) -> str:
    return (
        f"epoch={epoch} loss={loss:.4f} "
        f"val_accuracy={_format_metric(metrics['accuracy'])} "
        f"val_precision={_format_metric(metrics['precision'])} "
        f"val_recall={_format_metric(metrics['recall'])} "
        f"val_f1={_format_metric(metrics['f1'])} "
        f"val_roc_auc={_format_metric(metrics['roc_auc'])} "
        f"val_pr_auc={_format_metric(metrics['pr_auc'])} "
        f"best_roc_auc={_format_metric(best_roc_auc)} patience={remaining_patience}"
    )


def _metrics_summary(name: str, metrics: object) -> str:
    values = metrics if isinstance(metrics, dict) else {}
    return (
        f"{name} metrics: "
        f"accuracy={_format_metric(values.get('accuracy'))} "
        f"precision={_format_metric(values.get('precision'))} "
        f"recall={_format_metric(values.get('recall'))} "
        f"f1={_format_metric(values.get('f1'))} "
        f"roc_auc={_format_metric(values.get('roc_auc'))} "
        f"pr_auc={_format_metric(values.get('pr_auc'))} "
        f"threshold={_format_metric(values.get('threshold'))}"
    )


def _format_metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
