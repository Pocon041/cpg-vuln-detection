from __future__ import annotations

import csv
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from cpg_vuln.data.batch import DynamicBatchSampler, GraphSize
from cpg_vuln.training.metrics import (
    classification_metrics,
    select_f1_threshold,
    select_mcc_threshold,
)
from cpg_vuln.training.thresholds import (
    metrics_at_validation_thresholds,
    select_validation_thresholds,
    selected_threshold_key,
)


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
    checkpoint_metric: str = "roc_auc"
    loss: str = "cross_entropy"
    focal_gamma: float = 2.0
    class_weight: tuple[float, float] | list[float] | None = None
    threshold_strategy: str = "val_f1"
    evaluate_test: bool = True
    checkpoint_min_ppr: float | None = None
    checkpoint_max_ppr: float | None = None
    checkpoint_max_recall: float | None = None

    def __post_init__(self) -> None:
        allowed_metrics = {"loss", "f1", "roc_auc", "pr_auc", "mcc", "balanced_accuracy"}
        if self.checkpoint_metric not in allowed_metrics:
            raise ValueError(
                f"checkpoint_metric must be one of {sorted(allowed_metrics)}"
            )
        allowed_threshold_strategies = {"fixed_0_5", "val_f1", "val_mcc"}
        if self.threshold_strategy not in allowed_threshold_strategies:
            raise ValueError(
                f"threshold_strategy must be one of {sorted(allowed_threshold_strategies)}"
            )
        allowed_losses = {"cross_entropy", "focal"}
        if self.loss not in allowed_losses:
            raise ValueError(f"loss must be one of {sorted(allowed_losses)}")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if self.class_weight is not None and len(self.class_weight) != 2:
            raise ValueError("class_weight must contain two values for labels 0 and 1")
        for name in ("checkpoint_min_ppr", "checkpoint_max_ppr", "checkpoint_max_recall"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            self.checkpoint_min_ppr is not None
            and self.checkpoint_max_ppr is not None
            and self.checkpoint_min_ppr > self.checkpoint_max_ppr
        ):
            raise ValueError("checkpoint_min_ppr must be <= checkpoint_max_ppr")


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
    best_checkpoint_score: float | None = None
    has_checkpoint = False
    remaining_patience = config.patience
    history: list[dict[str, float | int | None]] = []
    started = time.perf_counter()
    epoch_progress = tqdm(
        range(config.epochs),
        desc=f"train epochs ({output_dir.name})",
        unit="epoch",
        ascii=True,
    )
    for epoch in epoch_progress:
        train_sampler.set_epoch(epoch)
        loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config.gradient_clip,
            loss_name=config.loss,
            focal_gamma=config.focal_gamma,
            class_weight=config.class_weight,
        )
        validation = _predict(model, val_loader, device)
        threshold = _select_threshold(
            validation["labels"],
            validation["probabilities"],
            strategy=config.threshold_strategy,
        )
        validation_metrics = classification_metrics(
            validation["labels"], validation["probabilities"], threshold=threshold
        )
        validation_roc_auc = validation_metrics["roc_auc"]
        checkpoint_guarded_out = config.checkpoint_metric != "loss" and _violates_checkpoint_guard(
            validation_metrics,
            checkpoint_min_ppr=config.checkpoint_min_ppr,
            checkpoint_max_ppr=config.checkpoint_max_ppr,
            checkpoint_max_recall=config.checkpoint_max_recall,
        )
        checkpoint_score = _checkpoint_score(
            config.checkpoint_metric,
            loss=loss,
            metrics=validation_metrics,
            checkpoint_min_ppr=config.checkpoint_min_ppr,
            checkpoint_max_ppr=config.checkpoint_max_ppr,
            checkpoint_max_recall=config.checkpoint_max_recall,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": loss,
                "val_f1": validation_metrics["f1"],
                "val_mcc": validation_metrics["mcc"],
                "val_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "val_roc_auc": validation_roc_auc,
                "val_pr_auc": validation_metrics["pr_auc"],
                "checkpoint_metric": config.checkpoint_metric,
                "checkpoint_score": checkpoint_score,
                "checkpoint_guarded_out": checkpoint_guarded_out,
            }
        )
        should_stop = False
        has_better_checkpoint = _is_better_checkpoint_score(
            config.checkpoint_metric,
            checkpoint_score,
            best_checkpoint_score,
        )
        if has_better_checkpoint or not has_checkpoint:
            if has_better_checkpoint:
                best_checkpoint_score = float(checkpoint_score)
            remaining_patience = config.patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch + 1,
                    "threshold": threshold,
                    "checkpoint_metric": config.checkpoint_metric,
                    "checkpoint_score": checkpoint_score,
                },
                output_dir / "best.pt",
            )
            has_checkpoint = True
        else:
            if not checkpoint_guarded_out:
                remaining_patience -= 1
            should_stop = remaining_patience <= 0
        epoch_progress.set_postfix(
            loss=f"{loss:.4f}",
            val_f1=_format_metric(validation_metrics["f1"]),
            best_checkpoint=_format_metric(best_checkpoint_score),
            patience=remaining_patience,
        )
        tqdm.write(
            _epoch_summary(
                epoch=epoch + 1,
                loss=loss,
                metrics=validation_metrics,
                checkpoint_metric=config.checkpoint_metric,
                best_checkpoint_score=best_checkpoint_score,
                remaining_patience=remaining_patience,
            )
        )
        if should_stop:
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation = _predict(model, val_loader, device)
    validation_thresholds = select_validation_thresholds(
        validation["labels"],
        validation["probabilities"],
    )
    validation_threshold_metrics = metrics_at_validation_thresholds(
        validation["labels"],
        validation["probabilities"],
        validation_thresholds,
    )
    selected_key = selected_threshold_key(config.threshold_strategy)
    validation_metrics = validation_threshold_metrics[selected_key]
    test_fixed = None
    test_val_f1 = None
    test_val_mcc = None
    test_metrics = None
    test_prediction = None
    test_explanations = {}
    if config.evaluate_test:
        test_loader, _ = _loader(test_dataset, config=config, shuffle=False)
        test_prediction = _predict(model, test_loader, device)
        test_threshold_metrics = metrics_at_validation_thresholds(
            test_prediction["labels"],
            test_prediction["probabilities"],
            validation_thresholds,
        )
        test_fixed = test_threshold_metrics["fixed_0_5"]
        test_val_f1 = test_threshold_metrics["val_f1"]
        test_val_mcc = test_threshold_metrics["val_mcc"]
        test_metrics = test_threshold_metrics[selected_key]
        test_explanations = test_prediction["explanations"]
    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "config": asdict(config),
        "run_metadata": run_metadata or {},
        "evaluation_mode": "final_test" if config.evaluate_test else "validation_only",
        "selection_source": "validation",
        "elapsed_seconds": elapsed,
        "best_epoch": checkpoint["epoch"],
        "checkpoint_metric": config.checkpoint_metric,
        "checkpoint_score": checkpoint.get("checkpoint_score"),
        "validation_thresholds": validation_thresholds,
        "validation_fixed_0_5": validation_threshold_metrics["fixed_0_5"],
        "validation_val_f1": validation_threshold_metrics["val_f1"],
        "validation_val_mcc": validation_threshold_metrics["val_mcc"],
        "validation": validation_metrics,
        "test_fixed_0_5": test_fixed,
        "test_val_f1": test_val_f1,
        "test_val_mcc": test_val_mcc,
        "test": test_metrics,
    }
    _write_json(output_dir / "metrics.json", result)
    _write_json(output_dir / "history.json", history)
    if test_prediction is not None:
        _write_predictions(
            output_dir / "predictions.csv",
            test_prediction,
            validation_thresholds[selected_key],
        )
    if test_explanations:
        _write_json(output_dir / "node_attention.json", test_explanations)
    print(_metrics_summary("validation", result["validation"]))
    print(_metrics_summary("test", result["test"]))
    return result


def _train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    gradient_clip: float,
    *,
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
    class_weight: Sequence[float] | None = None,
) -> float:
    model.train()
    losses_and_samples: list[tuple[float, int]] = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(batch)
                loss = _classification_loss(
                    output.logits,
                    batch.y,
                    loss_name=loss_name,
                    focal_gamma=focal_gamma,
                    class_weight=class_weight,
                )
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


def _classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_name: str,
    focal_gamma: float,
    class_weight: Sequence[float] | None,
) -> torch.Tensor:
    weight = _class_weight_tensor(class_weight, logits)
    if loss_name == "cross_entropy":
        return nn.functional.cross_entropy(logits, targets, weight=weight)
    if loss_name != "focal":
        raise ValueError(f"unsupported loss: {loss_name}")
    unweighted = nn.functional.cross_entropy(logits, targets, reduction="none")
    weighted = nn.functional.cross_entropy(logits, targets, weight=weight, reduction="none")
    true_class_probability = torch.exp(-unweighted)
    focal_factor = (1 - true_class_probability).pow(focal_gamma)
    return (focal_factor * weighted).mean()


def _class_weight_tensor(
    class_weight: Sequence[float] | None,
    logits: torch.Tensor,
) -> torch.Tensor | None:
    if class_weight is None:
        return None
    return torch.tensor(class_weight, dtype=logits.dtype, device=logits.device)


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
    oversized = [
        size
        for size in sizes
        if size.nodes > config.max_nodes or size.edges > config.max_edges
    ]
    if oversized:
        preview = ", ".join(size.sample_id for size in oversized[:5])
        suffix = "" if len(oversized) <= 5 else f", ... (+{len(oversized) - 5} more)"
        warnings.warn(
            f"Skipping {len(oversized)} graph(s) over max_nodes={config.max_nodes} "
            f"or max_edges={config.max_edges}: {preview}{suffix}",
            RuntimeWarning,
            stacklevel=2,
        )
    sampler = DynamicBatchSampler(
        sizes,
        max_nodes=config.max_nodes,
        max_edges=config.max_edges,
        shuffle=shuffle,
        seed=config.seed,
        skip_oversized=True,
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


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray, *, strategy: str) -> float:
    if strategy == "fixed_0_5":
        return 0.5
    if strategy == "val_mcc":
        return select_mcc_threshold(labels, probabilities)
    if strategy == "val_f1":
        return select_f1_threshold(labels, probabilities)
    raise ValueError(f"unsupported threshold strategy: {strategy}")


def _epoch_summary(
    *,
    epoch: int,
    loss: float,
    metrics: dict[str, object],
    checkpoint_metric: str,
    best_checkpoint_score: float | None,
    remaining_patience: int,
) -> str:
    return (
        f"epoch={epoch} loss={loss:.4f} "
        f"val_accuracy={_format_metric(metrics['accuracy'])} "
        f"val_precision={_format_metric(metrics['precision'])} "
        f"val_recall={_format_metric(metrics['recall'])} "
        f"val_f1={_format_metric(metrics['f1'])} "
        f"val_ppr={_format_metric(metrics.get('predicted_positive_rate'))} "
        f"val_roc_auc={_format_metric(metrics['roc_auc'])} "
        f"val_pr_auc={_format_metric(metrics['pr_auc'])} "
        f"best_{checkpoint_metric}={_format_metric(best_checkpoint_score)} "
        f"patience={remaining_patience}"
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


def _checkpoint_score(
    checkpoint_metric: str,
    *,
    loss: float,
    metrics: dict[str, object],
    checkpoint_min_ppr: float | None = None,
    checkpoint_max_ppr: float | None = None,
    checkpoint_max_recall: float | None = None,
) -> float | None:
    if checkpoint_metric == "loss":
        return float(loss)
    if _violates_checkpoint_guard(
        metrics,
        checkpoint_min_ppr=checkpoint_min_ppr,
        checkpoint_max_ppr=checkpoint_max_ppr,
        checkpoint_max_recall=checkpoint_max_recall,
    ):
        return None
    value = metrics.get(checkpoint_metric)
    return None if value is None else float(value)


def _violates_checkpoint_guard(
    metrics: dict[str, object],
    *,
    checkpoint_min_ppr: float | None,
    checkpoint_max_ppr: float | None,
    checkpoint_max_recall: float | None,
) -> bool:
    ppr = metrics.get("predicted_positive_rate")
    recall = metrics.get("recall")
    if checkpoint_min_ppr is not None and (ppr is None or float(ppr) < checkpoint_min_ppr):
        return True
    if checkpoint_max_ppr is not None and (ppr is None or float(ppr) > checkpoint_max_ppr):
        return True
    if checkpoint_max_recall is not None and (recall is None or float(recall) > checkpoint_max_recall):
        return True
    return False


def _is_better_checkpoint_score(
    checkpoint_metric: str,
    score: float | None,
    best_score: float | None,
) -> bool:
    if score is None:
        return False
    if best_score is None:
        return True
    if checkpoint_metric == "loss":
        return score < best_score
    return score > best_score
