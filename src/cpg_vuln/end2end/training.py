from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from cpg_vuln.data.audit import read_manifest
from cpg_vuln.end2end.dataset import RawCodeDataset
from cpg_vuln.end2end.model import RawCodeMILTransformer
from cpg_vuln.training.engine import TrainConfig, run_training
from cpg_vuln.training.runner import (
    _completed,
    _layout,
    _read_json,
    _seed_model,
    _train_config,
)


def train_end2end(
    config: dict,
    *,
    split: str = "strict",
    model_name: str | None = None,
    run_name: str | None = None,
    max_length: int | None = None,
    stride: int | None = None,
    max_chunks: int | None = None,
    max_batch_chunks: int | None = None,
    freeze_encoder: bool | None = None,
    checkpoint_metric: str | None = None,
    threshold_strategy: str | None = None,
    learning_rate: float | None = None,
    positive_class_weight: float | None = None,
    checkpoint_min_ppr: float | None = None,
    checkpoint_max_ppr: float | None = None,
    checkpoint_max_recall: float | None = None,
    evaluate_test: bool = True,
    epochs: int | None = None,
    force: bool = False,
) -> None:
    artifacts = Path(config["paths"]["artifacts_dir"])
    layout = _layout(config)
    end2end_config = _end2end_config(config)
    model_name = model_name or str(end2end_config["model_name"])
    max_length = int(max_length or end2end_config["max_length"])
    stride = int(stride or end2end_config["stride"])
    max_chunks = int(max_chunks or end2end_config["max_chunks"])
    max_batch_chunks = int(max_batch_chunks or end2end_config["max_batch_chunks"])
    freeze_encoder = bool(
        end2end_config["freeze_encoder"] if freeze_encoder is None else freeze_encoder
    )
    run_dir = layout.run_root / (
        run_name or f"end2end-raw-code-mil-{split}-fixed05"
    )
    if _completed(run_dir, force=force):
        return
    stage_progress = tqdm(
        total=4,
        desc=f"end2end {split} stages",
        unit="stage",
        ascii=True,
    )
    try:
        records = read_manifest(artifacts / "data" / "manifest.jsonl")
        split_payload = _read_json(artifacts / "data" / "splits" / f"{split}.json")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=bool(end2end_config["local_files_only"]),
        )
        stage_progress.update(1)
        datasets = {
            split_name: RawCodeDataset(
                records,
                sample_ids,
                tokenizer=tokenizer,
                max_length=max_length,
                stride=stride,
                max_chunks=max_chunks,
                desc=f"tokenize end2end {split_name}",
            )
            for split_name, sample_ids in split_payload.items()
        }
        stage_progress.update(1)
        _seed_model(config["training"]["seed"])
        model = RawCodeMILTransformer(
            model_name=model_name,
            dropout=float(end2end_config["dropout"]),
            freeze_encoder=freeze_encoder,
            local_files_only=bool(end2end_config["local_files_only"]),
        )
        stage_progress.update(1)
        base_config = _train_config(
            config,
            epochs=epochs,
            learning_rate=float(
                end2end_config["learning_rate"]
                if learning_rate is None
                else learning_rate
            ),
        )
        positive_class_weight = _configured_float(
            positive_class_weight,
            end2end_config["positive_class_weight"],
        )
        checkpoint_min_ppr = _configured_float(
            checkpoint_min_ppr,
            end2end_config["checkpoint_min_ppr"],
        )
        checkpoint_max_ppr = _configured_float(
            checkpoint_max_ppr,
            end2end_config["checkpoint_max_ppr"],
        )
        checkpoint_max_recall = _configured_float(
            checkpoint_max_recall,
            end2end_config["checkpoint_max_recall"],
        )
        train_config = TrainConfig(
            **{
                **asdict(base_config),
                "max_nodes": max_batch_chunks,
                "checkpoint_metric": checkpoint_metric or "f1",
                "loss": str(end2end_config["loss"]),
                "class_weight": None
                if positive_class_weight is None
                else [1.0, positive_class_weight],
                "checkpoint_min_ppr": checkpoint_min_ppr,
                "checkpoint_max_ppr": checkpoint_max_ppr,
                "checkpoint_max_recall": checkpoint_max_recall,
                "threshold_strategy": threshold_strategy or "fixed_0_5",
                "evaluate_test": evaluate_test,
            }
        )
        run_training(
            model,
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["test"],
            output_dir=run_dir,
            config=train_config,
            run_metadata={
                "kind": "end2end",
                "model_name": "raw-code-mil-transformer",
                "transformer_model": model_name,
                "split": split,
                "run_name": run_dir.name,
                "evaluate_test": evaluate_test,
                "representation": "raw-source",
                "uses_intermediate_representation": False,
                "max_length": max_length,
                "stride": stride,
                "max_chunks": max_chunks,
                "max_batch_chunks": max_batch_chunks,
                "freeze_encoder": freeze_encoder,
                "normalization_mode": "raw-source",
                "function_source_normalization": "raw",
            },
        )
        stage_progress.update(1)
    finally:
        stage_progress.close()


def _end2end_config(config: dict) -> dict[str, object]:
    values = config.get("end2end", {})
    return {
        "model_name": values.get(
            "model_name",
            config.get("features", {}).get("codebert_model", "microsoft/codebert-base"),
        ),
        "max_length": int(values.get("max_length", 256)),
        "stride": int(values.get("stride", 128)),
        "max_chunks": int(values.get("max_chunks", 8)),
        "max_batch_chunks": int(values.get("max_batch_chunks", 16)),
        "learning_rate": float(values.get("learning_rate", 2e-5)),
        "dropout": float(values.get("dropout", 0.2)),
        "freeze_encoder": bool(values.get("freeze_encoder", False)),
        "local_files_only": bool(values.get("local_files_only", True)),
        "loss": str(values.get("loss", "cross_entropy")),
        "positive_class_weight": _optional_float(values.get("positive_class_weight")),
        "checkpoint_min_ppr": _optional_float(values.get("checkpoint_min_ppr")),
        "checkpoint_max_ppr": _optional_float(values.get("checkpoint_max_ppr")),
        "checkpoint_max_recall": _optional_float(values.get("checkpoint_max_recall")),
    }


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _configured_float(explicit: float | None, configured: object) -> float | None:
    return float(configured) if explicit is None and configured is not None else explicit
