from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch_geometric.data import Batch
from tqdm import tqdm

from cpg_vuln.mining.hard_negative_bank import HardNegativePair
from cpg_vuln.mining.pair_dataset import PairBatchSampler, PairIndex
from cpg_vuln.training.engine import (
    TrainConfig,
    _checkpoint_score,
    _is_better_checkpoint_score,
    _loader,
    _predict,
    _seed_everything,
    _violates_checkpoint_guard,
    _write_json,
    _write_predictions,
)
from cpg_vuln.training.losses import compute_ramp_loss
from cpg_vuln.training.thresholds import (
    metrics_at_validation_thresholds,
    select_validation_thresholds,
    selected_threshold_key,
)


@dataclass(frozen=True)
class RampConfig:
    lambda_replay: float = 0.5
    lambda_rank: float = 0.25
    lambda_auxiliary: float = 0.0
    margin: float = 0.5
    pair_batch_size: int = 2
    warmup_epochs: int = 3
    rank_warmup_epochs: int = 0
    rank_ramp_epochs: int = 0
    max_pairs_per_positive: int = 3
    minimum_pair_score: float = 0.2
    max_pair_nodes: int = 8000
    max_pair_edges: int = 60000
    replay_steps_per_epoch: int = 200


def run_ramp_training(
    model: nn.Module,
    *,
    train_dataset: Sequence,
    val_dataset: Sequence,
    test_dataset: Sequence,
    output_dir: Path,
    config: TrainConfig,
    ramp_config: RampConfig,
    initial_pairs: list[HardNegativePair],
    run_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    _seed_everything(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader, train_sampler = _loader(train_dataset, config=config, shuffle=True)
    val_loader, _ = _loader(val_dataset, config=config, shuffle=False)
    sample_to_index = {
        str(train_dataset[index].sample_id): index for index in range(len(train_dataset))
    }
    pair_index = PairIndex.from_pairs(initial_pairs, sample_to_index) if initial_pairs else PairIndex([], [])
    pair_sizes = [
        (
            int(train_dataset[pair_index.positive_indices[index]].num_nodes)
            + int(train_dataset[pair_index.negative_indices[index]].num_nodes),
            int(train_dataset[pair_index.positive_indices[index]].edge_index.shape[1])
            + int(train_dataset[pair_index.negative_indices[index]].edge_index.shape[1]),
        )
        for index in range(len(pair_index))
    ]

    best_score: float | None = None
    best_state: dict[str, object] | None = None
    remaining_patience = config.patience
    history: list[dict[str, object]] = []
    epoch_progress = tqdm(
        range(config.epochs),
        desc=f"ramp epochs ({output_dir.name})",
        unit="epoch",
        ascii=True,
    )
    for epoch in epoch_progress:
        train_sampler.set_epoch(epoch)
        pair_sampler = PairBatchSampler(
            pair_count=len(pair_index),
            batch_size=ramp_config.pair_batch_size,
            shuffle=True,
            seed=config.seed,
            pair_sizes=pair_sizes,
            max_pair_nodes=ramp_config.max_pair_nodes,
            max_pair_edges=ramp_config.max_pair_edges,
            replay_steps_per_epoch=ramp_config.replay_steps_per_epoch,
        )
        pair_sampler.set_epoch(epoch)
        pair_batches = iter(pair_sampler)
        total_loss = 0.0
        total_samples = 0
        main_loss_sum = 0.0
        main_item_count = 0
        replay_loss_sum = 0.0
        replay_pair_count = 0
        ranking_loss_sum = 0.0
        ranking_pair_count = 0
        auxiliary_loss_sum = 0.0
        auxiliary_item_count = 0
        objective_step_sum = 0.0
        pre_rank_objective_step_sum = 0.0
        weighted_auxiliary_step_sum = 0.0
        weighted_replay_step_sum = 0.0
        weighted_ranking_step_sum = 0.0
        optimizer_step_count = 0
        pair_batches_used = 0
        effective_lambda_rank = _effective_rank_lambda(ramp_config, epoch=epoch)
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            positive_logits = output.logits.new_empty((0, 2))
            negative_logits = output.logits.new_empty((0, 2))
            positive_evidence_logits = None
            negative_evidence_logits = None
            try:
                pair_batch_indices = next(pair_batches)
            except StopIteration:
                pair_batch_indices = []
            if pair_batch_indices:
                positives, negatives = pair_index.graphs_for(train_dataset, pair_batch_indices)
                positive_batch = Batch.from_data_list(positives).to(device)
                negative_batch = Batch.from_data_list(negatives).to(device)
                positive_output = model(positive_batch)
                negative_output = model(negative_batch)
                positive_logits = positive_output.logits
                negative_logits = negative_output.logits
                positive_evidence_logits = _evidence_logits(positive_output)
                negative_evidence_logits = _evidence_logits(negative_output)
            loss, parts = compute_ramp_loss(
                batch_logits=output.logits,
                batch_targets=batch.y,
                positive_logits=positive_logits,
                matched_negative_logits=negative_logits,
                auxiliary_logits=getattr(output, "auxiliary_logits", None),
                positive_evidence_logits=positive_evidence_logits,
                matched_negative_evidence_logits=negative_evidence_logits,
                margin=ramp_config.margin,
                lambda_replay=ramp_config.lambda_replay,
                lambda_rank=effective_lambda_rank,
                lambda_auxiliary=ramp_config.lambda_auxiliary,
                class_weight=config.class_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            main_items = int(batch.y.numel())
            pair_items = len(pair_batch_indices)
            total_loss += float(loss.detach().cpu()) * main_items
            total_samples += main_items
            main_loss_sum += float(parts["main_loss"]) * main_items
            main_item_count += main_items
            auxiliary_loss_sum += float(parts.get("auxiliary_loss", 0.0)) * main_items
            auxiliary_item_count += main_items
            if pair_items:
                replay_loss_sum += float(parts["replay_loss"]) * pair_items
                replay_pair_count += pair_items
                ranking_loss_sum += float(parts["ranking_loss"]) * pair_items
                ranking_pair_count += pair_items
                pair_batches_used += 1
            objective_step_sum += float(parts["loss"])
            pre_rank_objective_step_sum += float(parts.get("pre_rank_loss", parts["loss"]))
            weighted_auxiliary_step_sum += float(parts.get("weighted_auxiliary_loss", 0.0))
            weighted_replay_step_sum += float(parts.get("weighted_replay_loss", 0.0))
            weighted_ranking_step_sum += float(parts.get("weighted_ranking_loss", 0.0))
            optimizer_step_count += 1
        train_loss = total_loss / max(total_samples, 1)
        validation = _predict(model, val_loader, device)
        validation_thresholds = select_validation_thresholds(
            validation["labels"],
            validation["probabilities"],
        )
        selected_key = selected_threshold_key(config.threshold_strategy)
        validation_threshold_metrics = metrics_at_validation_thresholds(
            validation["labels"],
            validation["probabilities"],
            validation_thresholds,
        )
        threshold = validation_thresholds[selected_key]
        validation_metrics = validation_threshold_metrics[selected_key]
        checkpoint_score = _checkpoint_score(
            config.checkpoint_metric,
            loss=train_loss,
            metrics=validation_metrics,
            checkpoint_min_ppr=config.checkpoint_min_ppr,
            checkpoint_max_ppr=config.checkpoint_max_ppr,
            checkpoint_max_recall=config.checkpoint_max_recall,
        )
        checkpoint_guarded_out = config.checkpoint_metric != "loss" and _violates_checkpoint_guard(
            validation_metrics,
            checkpoint_min_ppr=config.checkpoint_min_ppr,
            checkpoint_max_ppr=config.checkpoint_max_ppr,
            checkpoint_max_recall=config.checkpoint_max_recall,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": train_loss,
                "main_loss": main_loss_sum / max(main_item_count, 1),
                "auxiliary_loss": auxiliary_loss_sum / max(auxiliary_item_count, 1),
                "replay_loss": replay_loss_sum / max(replay_pair_count, 1),
                "ranking_loss": ranking_loss_sum / max(ranking_pair_count, 1),
                "objective_per_step": objective_step_sum / max(optimizer_step_count, 1),
                "pre_rank_objective_per_step": pre_rank_objective_step_sum
                / max(optimizer_step_count, 1),
                "weighted_auxiliary_loss_per_step": weighted_auxiliary_step_sum
                / max(optimizer_step_count, 1),
                "weighted_replay_loss_per_step": weighted_replay_step_sum
                / max(optimizer_step_count, 1),
                "weighted_ranking_loss_per_step": weighted_ranking_step_sum
                / max(optimizer_step_count, 1),
                "effective_lambda_rank": effective_lambda_rank,
                "main_items": main_item_count,
                "replay_pairs": replay_pair_count,
                "ranking_pairs": ranking_pair_count,
                "optimizer_steps": optimizer_step_count,
                "pair_batches_used": pair_batches_used,
                "val_precision": validation_metrics["precision"],
                "val_recall": validation_metrics["recall"],
                "val_f1": validation_metrics["f1"],
                "val_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "val_mcc": validation_metrics["mcc"],
                "val_ppr": validation_metrics["predicted_positive_rate"],
                "val_roc_auc": validation_metrics["roc_auc"],
                "val_pr_auc": validation_metrics["pr_auc"],
                "checkpoint_metric": config.checkpoint_metric,
                "checkpoint_score": checkpoint_score,
                "checkpoint_guarded_out": checkpoint_guarded_out,
            }
        )
        _write_json(output_dir / "history.json", history)
        epoch_progress.set_postfix(
            loss=f"{train_loss:.4f}",
            val_f1=f"{float(validation_metrics['f1']):.4f}",
            val_mcc=f"{float(validation_metrics['mcc']):.4f}",
            val_ppr=f"{float(validation_metrics['predicted_positive_rate']):.4f}",
            patience=remaining_patience,
        )
        has_better_checkpoint = _is_better_checkpoint_score(
            config.checkpoint_metric,
            checkpoint_score,
            best_score,
        )
        if has_better_checkpoint or best_state is None:
            if has_better_checkpoint:
                best_score = float(checkpoint_score)
            best_state = {
                "model_state": copy.deepcopy(model.state_dict()),
                "epoch": epoch + 1,
                "threshold": threshold,
                "checkpoint_metric": config.checkpoint_metric,
                "checkpoint_score": checkpoint_score,
            }
            torch.save(best_state, output_dir / "best.pt")
            remaining_patience = config.patience
        else:
            if not checkpoint_guarded_out:
                remaining_patience -= 1
            if remaining_patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        torch.save(best_state, output_dir / "best.pt")
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
    metadata = {
        "training_mode": "ramp",
        "bank_mode": "static",
        "initial_pair_count": len(initial_pairs),
        **(run_metadata or {}),
    }
    result: dict[str, object] = {
        "config": asdict(config),
        "ramp_config": asdict(ramp_config),
        "run_metadata": metadata,
        "evaluation_mode": "final_test" if config.evaluate_test else "validation_only",
        "selection_source": "validation",
        "elapsed_seconds": time.perf_counter() - started,
        "best_epoch": None if best_state is None else best_state["epoch"],
        "checkpoint_metric": config.checkpoint_metric,
        "checkpoint_score": None if best_state is None else best_state["checkpoint_score"],
        "validation_thresholds": validation_thresholds,
        "validation_fixed_0_5": validation_threshold_metrics["fixed_0_5"],
        "validation_val_f1": validation_threshold_metrics["val_f1"],
        "validation_val_mcc": validation_threshold_metrics["val_mcc"],
        "validation": validation_metrics,
        "validation_diagnostics": validation.get("diagnostics", {}),
        "test_fixed_0_5": test_fixed,
        "test_val_f1": test_val_f1,
        "test_val_mcc": test_val_mcc,
        "test": test_metrics,
        "test_diagnostics": None
        if test_prediction is None
        else test_prediction.get("diagnostics", {}),
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
    return result


def _effective_rank_lambda(ramp_config: RampConfig, *, epoch: int) -> float:
    if ramp_config.lambda_rank <= 0:
        return 0.0
    if epoch < ramp_config.rank_warmup_epochs:
        return 0.0
    if ramp_config.rank_ramp_epochs <= 0:
        return float(ramp_config.lambda_rank)
    progress = (epoch - ramp_config.rank_warmup_epochs + 1) / ramp_config.rank_ramp_epochs
    return float(ramp_config.lambda_rank) * min(1.0, max(0.0, progress))


def _evidence_logits(output) -> torch.Tensor:
    evidence_logits = getattr(output, "evidence_logits", None)
    return output.logits if evidence_logits is None else evidence_logits
