from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from tqdm import tqdm

from cpg_vuln.data.batch import GraphSize
from cpg_vuln.data.dataset import TopologyDataset
from cpg_vuln.data.layout import ArtifactLayout
from cpg_vuln.data.store import (
    TOPOLOGY_CACHE_SCHEMA_VERSION,
    NodeTypeRegistry,
    load_topology,
)
from cpg_vuln.features.cache import MemmapFeatureCache
from cpg_vuln.features.normalization import NormalizationSpec
from cpg_vuln.mining.hard_negative_bank import (
    HardNegativeConfig,
    HardNegativePair,
    MiningStrategy,
    build_hard_negative_bank,
)
from cpg_vuln.mining.motif import extract_risk_motif
from cpg_vuln.mining.retrieval_features import retrieval_vector_map
from cpg_vuln.utils.fingerprint import sha256_json


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


def train_baselines(
    config: dict,
    *,
    views: tuple[str, ...] = ("ast", "cfg", "pdg"),
    embeddings: tuple[str, ...] = ("word2vec", "codebert"),
    splits: tuple[str, ...] = ("course", "strict"),
    epochs: int | None = None,
    force: bool = False,
) -> None:
    artifacts = Path(config["paths"]["artifacts_dir"])
    layout = _layout(config)
    outputs = layout.run_root
    index = _topology_index(layout)
    node_types = NodeTypeRegistry.read(layout.topology_dir / "node_type_registry.json")
    run_specs = [
        (split_name, view, embedding)
        for split_name in splits
        for view in views
        for embedding in embeddings
    ]
    split_cache: dict[str, dict[str, list[str]]] = {}
    for split_name, view, embedding in tqdm(
        run_specs,
        desc="baseline runs",
        unit="run",
        total=len(run_specs),
        ascii=True,
    ):
        split = split_cache.setdefault(
            split_name,
            _read_json(artifacts / "data" / "splits" / f"{split_name}.json"),
        )
        run_dir = outputs / f"baseline-{view}-{embedding}-{split_name}"
        if _completed(run_dir, force=force):
            continue
        cache = _node_feature_cache(layout, embedding)
        datasets = _datasets(index, split, view, cache)
        from cpg_vuln.models.gcn import GCNClassifier
        from cpg_vuln.training.engine import run_training

        _seed_model(config["training"]["seed"])
        model = GCNClassifier(
            input_dim=cache.metadata.dim,
            num_node_types=len(node_types),
            **_model_args(config),
        )
        run_training(
            model,
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["test"],
            output_dir=run_dir,
            config=_train_config(config, epochs=epochs, learning_rate=1e-3),
            run_metadata={
                "kind": "baseline",
                "view": view,
                "embedding": embedding,
                "split": split_name,
                "feature_dim": cache.metadata.dim,
                "normalization_mode": layout.spec.mode,
                "normalization_key": layout.spec.normalization_key,
                "normalization_fingerprint": layout.spec.fingerprint,
                "text_registry_sha256": cache.metadata.text_registry_sha256,
                "function_source_normalization": "not-used",
            },
        )


def train_enhanced(
    config: dict,
    *,
    splits: tuple[str, ...] = ("course", "strict"),
    variants: tuple[str, ...] = (
        "selective-fusion",
        "no-semantics",
        "dataflow-only",
        "slice-fusion",
    ),
    epochs: int | None = None,
    force: bool = False,
) -> None:
    artifacts = Path(config["paths"]["artifacts_dir"])
    layout = _layout(config)
    outputs = layout.run_root
    index = _topology_index(layout)
    node_types = NodeTypeRegistry.read(layout.topology_dir / "node_type_registry.json")
    node_cache = _node_feature_cache(layout, "codebert")
    function_cache = MemmapFeatureCache.open(
        layout.function_codebert_dir / "features",
        read_only=True,
    )
    function_indices = _read_json(
        layout.function_codebert_dir / "function_indices.json"
    )
    run_specs = [
        (split_name, variant)
        for split_name in splits
        for variant in variants
    ]
    split_cache: dict[str, dict[str, list[str]]] = {}
    for split_name, variant in tqdm(
        run_specs,
        desc="enhanced runs",
        unit="run",
        total=len(run_specs),
        ascii=True,
    ):
        split = split_cache.setdefault(
            split_name,
            _read_json(artifacts / "data" / "splits" / f"{split_name}.json"),
        )
        view = _enhanced_view(variant)
        run_dir = outputs / f"enhanced-{variant}-{split_name}"
        if _completed(run_dir, force=force):
            continue
        datasets = _datasets(
            index,
            split,
            view,
            node_cache,
            function_cache=function_cache,
            function_indices=function_indices,
        )
        sample_payload = load_topology(datasets["train"].topology_paths[0])
        from cpg_vuln.models.selective_fusion import SelectiveFusionCPG
        from cpg_vuln.training.engine import run_training

        _seed_model(config["training"]["seed"])
        model = SelectiveFusionCPG(
            input_dim=node_cache.metadata.dim,
            function_dim=function_cache.metadata.dim,
            num_node_types=len(node_types),
            num_relations=len(sample_payload["edge_type_names"]),
            use_semantics=variant != "no-semantics",
            **_model_args(config),
        )
        run_training(
            model,
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["test"],
            output_dir=run_dir,
            config=_train_config(config, epochs=epochs, learning_rate=1e-4),
            run_metadata={
                "kind": "enhanced",
                "variant": variant,
                "view": view,
                "embedding": "codebert",
                "split": split_name,
                "feature_dim": node_cache.metadata.dim,
                "function_dim": function_cache.metadata.dim,
                "normalization_mode": layout.spec.mode,
                "normalization_key": layout.spec.normalization_key,
                "normalization_fingerprint": layout.spec.fingerprint,
                "text_registry_sha256": node_cache.metadata.text_registry_sha256,
                "function_source_normalization": _function_source_normalization(layout),
            },
        )


def train_devign(
    config: dict,
    *,
    split: str = "strict",
    view: str = "core-cpg",
    embedding: str = "codebert",
    run_name: str | None = None,
    checkpoint_metric: str | None = None,
    threshold_strategy: str | None = None,
    learning_rate: float | None = None,
    positive_class_weight: float | None = None,
    evaluate_test: bool = True,
    epochs: int | None = None,
    force: bool = False,
) -> None:
    if embedding not in {"word2vec", "codebert"}:
        raise ValueError(f"unsupported Devign embedding: {embedding}")
    artifacts = Path(config["paths"]["artifacts_dir"])
    layout = _layout(config)
    outputs = layout.run_root
    run_dir = outputs / (run_name or f"devign-{view}-{embedding}-{split}")
    if _completed(run_dir, force=force):
        return
    stage_progress = tqdm(
        total=3,
        desc=f"devign {split} stages",
        unit="stage",
        ascii=True,
    )
    try:
        index = _topology_index(layout)
        split_payload = _read_json(artifacts / "data" / "splits" / f"{split}.json")
        node_types = NodeTypeRegistry.read(layout.topology_dir / "node_type_registry.json")
        node_cache = _node_feature_cache(layout, embedding)
        stage_progress.update(1)
        datasets = _datasets(index, split_payload, view, node_cache)
        from cpg_vuln.models.devign import DevignCPG
        from cpg_vuln.training.engine import TrainConfig, run_training

        _seed_model(config["training"]["seed"])
        model = DevignCPG(
            input_dim=node_cache.metadata.dim,
            num_node_types=len(node_types),
            **_devign_model_args(config),
        )
        base_train_config = _train_config(
            config,
            epochs=epochs,
            learning_rate=1e-4 if learning_rate is None else learning_rate,
        )
        train_config = TrainConfig(
            **{
                **asdict(base_train_config),
                "checkpoint_metric": checkpoint_metric or "pr_auc",
                "loss": "cross_entropy",
                "class_weight": None
                if positive_class_weight is None
                else [1.0, positive_class_weight],
                "weight_decay": _devign_weight_decay(config, base_train_config.weight_decay),
                "threshold_strategy": threshold_strategy or "fixed_0_5",
                "evaluate_test": evaluate_test,
            }
        )
        stage_progress.update(1)
        run_training(
            model,
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["test"],
            output_dir=run_dir,
            config=train_config,
            run_metadata={
                "kind": "devign",
                "model_name": "devign",
                "view": view,
                "embedding": embedding,
                "split": split,
                "run_name": run_dir.name,
                "evaluate_test": evaluate_test,
                "feature_dim": node_cache.metadata.dim,
                "normalization_mode": layout.spec.mode,
                "normalization_key": layout.spec.normalization_key,
                "normalization_fingerprint": layout.spec.fingerprint,
                "text_registry_sha256": node_cache.metadata.text_registry_sha256,
                "function_source_normalization": "not-used",
            },
        )
        stage_progress.update(1)
    finally:
        stage_progress.close()


def ramp_run_directory(
    outputs: Path,
    *,
    experiment: str,
    split: str,
    run_name: str | None,
) -> Path:
    return outputs / (run_name or f"ramp-{experiment}-{split}")


def train_ramp(
    config: dict,
    *,
    experiment: str = "E4",
    split: str = "strict",
    view: str = "core-cpg",
    model_name: str = "selective-fusion",
    run_name: str | None = None,
    lambda_replay: float | None = None,
    lambda_rank: float | None = None,
    lambda_auxiliary: float | None = None,
    margin: float | None = None,
    max_pairs_per_positive: int | None = None,
    rank_warmup_epochs: int | None = None,
    rank_ramp_epochs: int | None = None,
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
    if experiment not in {"E0", "E1", "E2", "E3", "E4"}:
        raise ValueError(f"unsupported RAMP experiment: {experiment}")
    artifacts = Path(config["paths"]["artifacts_dir"])
    layout = _layout(config)
    outputs = layout.run_root
    run_dir = ramp_run_directory(
        outputs,
        experiment=experiment,
        split=split,
        run_name=run_name,
    )
    if _completed(run_dir, force=force):
        return
    stage_progress = tqdm(
        total=6,
        desc=f"ramp {experiment}-{split} stages",
        unit="stage",
        ascii=True,
    )
    index = _topology_index(layout)
    split_payload = _read_json(artifacts / "data" / "splits" / f"{split}.json")
    node_types = NodeTypeRegistry.read(layout.topology_dir / "node_type_registry.json")
    node_cache = _node_feature_cache(layout, "codebert")
    function_cache = MemmapFeatureCache.open(
        layout.function_codebert_dir / "features",
        read_only=True,
    )
    function_indices = _read_json(
        layout.function_codebert_dir / "function_indices.json"
    )
    datasets = _datasets(
        index,
        split_payload,
        view,
        node_cache,
        function_cache=function_cache,
        function_indices=function_indices,
    )
    sample_payload = load_topology(datasets["train"].topology_paths[0])
    model_fingerprint = _ramp_model_fingerprint(model_name, view, config)
    _seed_model(config["training"]["seed"])
    model = _ramp_model(
        model_name=model_name,
        input_dim=node_cache.metadata.dim,
        function_dim=function_cache.metadata.dim,
        num_node_types=len(node_types),
        num_relations=len(sample_payload["edge_type_names"]),
        config=config,
    )
    stage_progress.update(1)
    mining_config, ramp_config = ramp_settings_for_experiment(
        experiment,
        config,
        lambda_replay=lambda_replay,
        lambda_rank=lambda_rank,
        margin=margin,
        max_pairs_per_positive=max_pairs_per_positive,
    )
    ramp_config = _apply_model_ramp_overrides(model_name, ramp_config, config)
    ramp_config = _apply_explicit_ramp_training_overrides(
        ramp_config,
        lambda_auxiliary=lambda_auxiliary,
        rank_warmup_epochs=rank_warmup_epochs,
        rank_ramp_epochs=rank_ramp_epochs,
    )
    from cpg_vuln.training.engine import TrainConfig, _loader, _predict, run_training
    from cpg_vuln.training.ramp import run_ramp_training

    base_train_config = _train_config(
        config,
        epochs=epochs,
        learning_rate=1e-4 if learning_rate is None else learning_rate,
    )
    train_config = TrainConfig(
        **{
            **asdict(base_train_config),
            "checkpoint_metric": checkpoint_metric or "mcc",
            "loss": "cross_entropy",
            "class_weight": None
            if positive_class_weight is None
            else [1.0, positive_class_weight],
            "checkpoint_min_ppr": checkpoint_min_ppr,
            "checkpoint_max_ppr": checkpoint_max_ppr,
            "checkpoint_max_recall": checkpoint_max_recall,
            "threshold_strategy": threshold_strategy or "val_mcc",
            "evaluate_test": evaluate_test,
        }
    )
    warmup_key = "" if model_name == "selective-fusion" and view == "core-cpg" else f"-{model_name}-{view}"
    warmup_dir = outputs / f"ramp-warmup{warmup_key}-{split}"
    warmup_artifact_dir = layout.retrieval_dir / split / f"warmup{warmup_key}"
    warmup_artifact_dir.mkdir(parents=True, exist_ok=True)
    warmup_checkpoint = warmup_dir / "best.pt"
    fp_snapshot_path = warmup_artifact_dir / "train_fp_snapshot.json"

    if not _ramp_warmup_matches_current_inputs(
        warmup_dir,
        layout,
        expected_model_fingerprint=model_fingerprint,
    ):
        warmup_config = TrainConfig(
            **{
                **asdict(train_config),
                "epochs": ramp_config.warmup_epochs,
                "checkpoint_metric": "mcc",
                "loss": "cross_entropy",
                "class_weight": train_config.class_weight,
                "threshold_strategy": "val_mcc",
            }
        )
        run_training(
            model,
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["val"],
            output_dir=warmup_dir,
            config=warmup_config,
            run_metadata={
                "kind": "ramp-warmup",
                "experiment": experiment,
                "split": split,
                "view": view,
                "model_name": model_name,
                "model_fingerprint": model_fingerprint,
                "bank_mode": "static",
                "normalization_mode": layout.spec.mode,
                "normalization_key": layout.spec.normalization_key,
                "normalization_fingerprint": layout.spec.fingerprint,
                "text_registry_sha256": node_cache.metadata.text_registry_sha256,
                "function_source_normalization": _function_source_normalization(layout),
            },
        )
    stage_progress.update(1)

    checkpoint = torch.load(warmup_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(
        train_config.device
        if train_config.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    model = model.to(device)
    if fp_snapshot_path.is_file():
        false_positive_probabilities = json.loads(fp_snapshot_path.read_text(encoding="utf-8"))
    else:
        train_loader, _ = _loader(datasets["train"], config=train_config, shuffle=False)
        train_prediction = _predict(model, train_loader, device)
        false_positive_probabilities = false_positive_probabilities_from_prediction(train_prediction)
        fp_snapshot_path.write_text(
            json.dumps(false_positive_probabilities, indent=2) + "\n",
            encoding="utf-8",
        )
    stage_progress.update(1)
    warmup_sha256 = _file_sha256(warmup_checkpoint)

    if experiment == "E0":
        pairs: list[HardNegativePair] = []
        bank_path = ""
        review_path = ""
        ramp_config = RampConfig(
            **{
                **asdict(ramp_config),
                "lambda_replay": 0.0,
                "lambda_rank": 0.0,
            }
        )
    else:
        train_motifs = [
            extract_risk_motif(load_topology(path))
            for path in tqdm(
                datasets["train"].topology_paths,
                desc=f"extract {experiment} train motifs",
                unit="graph",
                ascii=True,
            )
        ]
        stage_progress.update(1)
        retrieval_vectors = retrieval_vector_map(train_motifs)
        labels = {motif.sample_id: motif.label for motif in train_motifs}
        experiment_retrieval_dir = layout.retrieval_dir / split / experiment
        bank_file = experiment_retrieval_dir / "bank.jsonl"
        review_file = experiment_retrieval_dir / "review_queue.jsonl"
        if experiment == "E4":
            e3_bank_path = layout.retrieval_dir / split / "E3" / "bank.jsonl"
            e3_review_path = layout.retrieval_dir / split / "E3" / "review_queue.jsonl"
            if not e3_bank_path.is_file():
                raise RuntimeError("E4 requires the E3 static bank to exist; run E3 first")
            pairs = load_pair_jsonl(e3_bank_path)
            bank_file.parent.mkdir(parents=True, exist_ok=True)
            bank_file.write_text(e3_bank_path.read_text(encoding="utf-8"), encoding="utf-8")
            review_file.write_text(
                e3_review_path.read_text(encoding="utf-8")
                if e3_review_path.is_file()
                else "",
                encoding="utf-8",
            )
        else:
            pairs, review_pairs = build_hard_negative_bank(
                sample_ids=[motif.sample_id for motif in train_motifs],
                labels=labels,
                retrieval_vectors=retrieval_vectors,
                false_positive_probabilities=false_positive_probabilities,
                config=mining_config,
            )
            write_pair_jsonl(bank_file, pairs)
            write_pair_jsonl(review_file, review_pairs)
        bank_path = str(bank_file)
        review_path = str(review_file)
    if experiment == "E0":
        stage_progress.update(1)
    stage_progress.update(1)

    write_ramp_strategy_manifest(
        run_dir / "ramp_strategy.json",
        experiment=experiment,
        mining_strategy=mining_config.strategy,
        mining_weights={
            "motif_weight": mining_config.motif_weight,
            "structure_weight": mining_config.structure_weight,
            "scale_weight": mining_config.scale_weight,
            "false_positive_weight": mining_config.false_positive_weight,
        },
        lambda_rank=ramp_config.lambda_rank,
        split=split,
        bank_mode="static",
        false_positive_source="train_warmup_snapshot",
        warmup_epochs=ramp_config.warmup_epochs,
        warmup_checkpoint_sha256=warmup_sha256,
        pair_count=len(pairs),
        bank_path=bank_path,
        review_path=review_path,
    )
    run_ramp_training(
        model,
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        output_dir=run_dir,
        config=train_config,
        ramp_config=ramp_config,
        initial_pairs=pairs,
        run_metadata={
            "kind": "ramp",
            "experiment": experiment,
            "split": split,
            "view": view,
            "model_name": model_name,
            "model_fingerprint": model_fingerprint,
            "run_name": run_dir.name,
            "evaluate_test": evaluate_test,
            "feature_dim": node_cache.metadata.dim,
            "function_dim": function_cache.metadata.dim,
            "warmup_checkpoint_sha256": warmup_sha256,
            "normalization_mode": layout.spec.mode,
            "normalization_key": layout.spec.normalization_key,
            "normalization_fingerprint": layout.spec.fingerprint,
            "text_registry_sha256": node_cache.metadata.text_registry_sha256,
            "function_source_normalization": _function_source_normalization(layout),
        },
    )
    stage_progress.update(1)
    stage_progress.close()


def evaluate_ramp(
    config: dict,
    *,
    run: str,
    split: str = "strict",
    export_attention: bool = False,
) -> dict[str, object]:
    artifacts = Path(config["paths"]["artifacts_dir"])
    layout = _layout(config)
    outputs = layout.run_root
    run_dir = outputs / run
    metrics = _read_json(run_dir / "metrics.json")
    metadata = metrics.get("run_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("run_metadata must be a JSON object")
    view = str(metadata.get("view", "core-cpg"))
    model_name = str(metadata.get("model_name", "selective-fusion"))

    index = _topology_index(layout)
    split_payload = _read_json(artifacts / "data" / "splits" / f"{split}.json")
    node_types = NodeTypeRegistry.read(layout.topology_dir / "node_type_registry.json")
    node_cache = _node_feature_cache(layout, "codebert")
    function_cache = MemmapFeatureCache.open(
        layout.function_codebert_dir / "features",
        read_only=True,
    )
    function_indices = _read_json(
        layout.function_codebert_dir / "function_indices.json"
    )
    datasets = _datasets(
        index,
        split_payload,
        view,
        node_cache,
        function_cache=function_cache,
        function_indices=function_indices,
    )
    sample_payload = load_topology(datasets["test"].topology_paths[0])
    from cpg_vuln.evaluation.ramp_run import evaluate_ramp_checkpoint
    from cpg_vuln.training.engine import TrainConfig

    train_config = TrainConfig(
        **{
            **config["training"],
            "learning_rate": 1e-4,
        }
    )

    def model_factory(_metadata: dict[str, object]) -> torch.nn.Module:
        return _ramp_model(
            model_name=model_name,
            input_dim=node_cache.metadata.dim,
            function_dim=function_cache.metadata.dim,
            num_node_types=len(node_types),
            num_relations=len(sample_payload["edge_type_names"]),
            config=config,
        )

    return evaluate_ramp_checkpoint(
        model_factory=model_factory,
        run_dir=run_dir,
        test_dataset=datasets["test"],
        config=train_config,
        export_attention=export_attention,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _function_source_normalization(layout: ArtifactLayout) -> str:
    metadata_path = layout.function_codebert_dir / "source_normalization.json"
    if not metadata_path.is_file():
        return "raw"
    metadata = _read_json(metadata_path)
    value = metadata.get("function_source_normalization")
    return str(value) if value else "raw"


def _ramp_warmup_matches_current_inputs(
    warmup_dir: Path,
    layout: ArtifactLayout,
    *,
    expected_model_fingerprint: str | None = None,
) -> bool:
    if not (warmup_dir / "best.pt").is_file():
        return False
    metrics_path = warmup_dir / "metrics.json"
    if not metrics_path.is_file():
        return False
    metadata = _read_json(metrics_path).get("run_metadata", {})
    if not isinstance(metadata, dict):
        return False
    matches = (
        metadata.get("normalization_fingerprint") == layout.spec.fingerprint
        and metadata.get("function_source_normalization")
        == _function_source_normalization(layout)
    )
    if expected_model_fingerprint is not None:
        matches = (
            matches
            and metadata.get("model_fingerprint") == expected_model_fingerprint
        )
    return matches


def _ramp_model_fingerprint(model_name: str, view: str, config: dict) -> str:
    return sha256_json(
        {
            "model_name": model_name,
            "view": view,
            "model": config.get("model", {}),
            "ramp_v2": config.get("ramp_v2", {}),
            "ramp_v3": config.get("ramp_v3", {}),
            "topology_cache_schema_version": TOPOLOGY_CACHE_SCHEMA_VERSION,
        }
    )


def false_positive_probabilities_from_prediction(prediction: dict[str, object]) -> dict[str, float]:
    sample_ids = list(prediction["sample_ids"])
    labels = prediction["labels"]
    probabilities = prediction["probabilities"]
    return {
        str(sample_id): float(probability)
        for sample_id, label, probability in zip(sample_ids, labels, probabilities, strict=True)
        if int(label) == 0
    }


def ramp_settings_for_experiment(
    experiment: str,
    config: dict,
    *,
    lambda_replay: float | None = None,
    lambda_rank: float | None = None,
    margin: float | None = None,
    max_pairs_per_positive: int | None = None,
) -> tuple[HardNegativeConfig, RampConfig]:
    ramp = config.get("ramp", {})
    effective_lambda_replay = (
        float(lambda_replay)
        if lambda_replay is not None
        else float(ramp.get("lambda_replay", 0.5))
    )
    effective_lambda_rank = (
        float(lambda_rank)
        if lambda_rank is not None
        else float(ramp.get("lambda_rank", 0.25))
    )
    effective_margin = float(margin) if margin is not None else float(ramp.get("margin", 0.5))
    effective_max_pairs = (
        int(max_pairs_per_positive)
        if max_pairs_per_positive is not None
        else int(ramp.get("max_pairs_per_positive", 3))
    )
    base_mining = {
        "max_pairs_per_positive": effective_max_pairs,
        "minimum_pair_score": float(ramp.get("minimum_pair_score", 0.2)),
        "lower_percentile": float(ramp.get("semi_hard_lower_percentile", 0.05)),
        "upper_percentile": float(ramp.get("semi_hard_upper_percentile", 0.30)),
        "seed": int(config.get("training", {}).get("seed", 42)),
    }
    rank_lambda = effective_lambda_rank if experiment == "E4" else 0.0
    ramp_config = RampConfig(
        lambda_replay=0.0 if experiment == "E0" else effective_lambda_replay,
        lambda_rank=rank_lambda,
        lambda_auxiliary=float(ramp.get("lambda_auxiliary", 0.0)),
        margin=effective_margin,
        pair_batch_size=int(ramp.get("pair_batch_size", 2)),
        warmup_epochs=int(ramp.get("warmup_epochs", 3)),
        rank_warmup_epochs=int(ramp.get("rank_warmup_epochs", 0)),
        rank_ramp_epochs=int(ramp.get("rank_ramp_epochs", 0)),
        max_pairs_per_positive=effective_max_pairs,
        minimum_pair_score=float(ramp.get("minimum_pair_score", 0.2)),
        max_pair_nodes=int(ramp.get("max_pair_nodes", 8000)),
        max_pair_edges=int(ramp.get("max_pair_edges", 60000)),
        replay_steps_per_epoch=int(ramp.get("replay_steps_per_epoch", 200)),
    )
    if experiment == "E0":
        return HardNegativeConfig(strategy=MiningStrategy.RANDOM, **base_mining), ramp_config
    if experiment == "E1":
        return HardNegativeConfig(strategy=MiningStrategy.RANDOM, **base_mining), ramp_config
    if experiment == "E2":
        return (
            HardNegativeConfig(
                strategy=MiningStrategy.FALSE_POSITIVE_ONLY,
                motif_weight=0.0,
                structure_weight=0.0,
                scale_weight=0.0,
                false_positive_weight=1.0,
                **base_mining,
            ),
            ramp_config,
        )
    if experiment in {"E3", "E4"}:
        return HardNegativeConfig(strategy=MiningStrategy.MOTIF_MATCHED, **base_mining), ramp_config
    raise ValueError(f"unsupported RAMP experiment: {experiment}")


def _apply_model_ramp_overrides(
    model_name: str,
    ramp_config: RampConfig,
    config: dict,
) -> RampConfig:
    if model_name != "ramp-v3-slice-mil":
        return ramp_config
    ramp_v3_config = config.get("ramp_v3", {})
    return replace(
        ramp_config,
        lambda_auxiliary=float(
            ramp_v3_config.get("lambda_auxiliary", ramp_config.lambda_auxiliary)
        ),
        rank_warmup_epochs=int(
            ramp_v3_config.get("rank_warmup_epochs", ramp_config.rank_warmup_epochs)
        ),
        rank_ramp_epochs=int(
            ramp_v3_config.get("rank_ramp_epochs", ramp_config.rank_ramp_epochs)
        ),
    )


def _apply_explicit_ramp_training_overrides(
    ramp_config: RampConfig,
    *,
    lambda_auxiliary: float | None = None,
    rank_warmup_epochs: int | None = None,
    rank_ramp_epochs: int | None = None,
) -> RampConfig:
    return replace(
        ramp_config,
        lambda_auxiliary=ramp_config.lambda_auxiliary
        if lambda_auxiliary is None
        else float(lambda_auxiliary),
        rank_warmup_epochs=ramp_config.rank_warmup_epochs
        if rank_warmup_epochs is None
        else int(rank_warmup_epochs),
        rank_ramp_epochs=ramp_config.rank_ramp_epochs
        if rank_ramp_epochs is None
        else int(rank_ramp_epochs),
    )


def write_ramp_strategy_manifest(
    path: Path,
    *,
    experiment: str,
    mining_strategy: str,
    mining_weights: dict[str, float],
    lambda_rank: float,
    split: str,
    bank_mode: str,
    false_positive_source: str,
    warmup_epochs: int,
    warmup_checkpoint_sha256: str,
    pair_count: int,
    bank_path: str,
    review_path: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment,
        "split": split,
        "mining_strategy": mining_strategy,
        "mining_weights": mining_weights,
        "lambda_rank": lambda_rank,
        "bank_mode": bank_mode,
        "false_positive_source": false_positive_source,
        "warmup_epochs": warmup_epochs,
        "warmup_checkpoint_sha256": warmup_checkpoint_sha256,
        "pair_count": pair_count,
        "bank_path": bank_path,
        "review_path": review_path,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_pair_jsonl(path: Path, pairs: list[HardNegativePair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_dict(), sort_keys=True) + "\n")


def load_pair_jsonl(path: Path) -> list[HardNegativePair]:
    return [
        HardNegativePair.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _datasets(
    index: list[dict],
    split: dict[str, list[str]],
    view: str,
    node_cache: MemmapFeatureCache,
    *,
    function_cache: MemmapFeatureCache | None = None,
    function_indices: dict[str, int] | None = None,
) -> dict[str, TopologyDataset]:
    items = {
        item["sample_id"]: item
        for item in index
        if item["view"] == view
    }
    return {
        split_name: TopologyDataset(
            [Path(items[sample_id]["path"]) for sample_id in sample_ids],
            node_features=node_cache,
            function_features=function_cache,
            function_indices=function_indices,
            graph_sizes=[
                GraphSize(
                    sample_id,
                    nodes=int(items[sample_id]["nodes"]),
                    edges=int(items[sample_id]["edges"]),
                )
                for sample_id in sample_ids
            ],
        )
        for split_name, sample_ids in split.items()
    }


def _node_feature_cache(layout: ArtifactLayout, embedding: str) -> MemmapFeatureCache:
    path = (
        layout.word2vec_dir / "features"
        if embedding == "word2vec"
        else layout.node_codebert_dir
    )
    return MemmapFeatureCache.open(path, read_only=True)


def _topology_index(layout: ArtifactLayout) -> list[dict]:
    return _read_json(layout.topology_dir / "index.json")


def _normalization_spec(config: dict) -> NormalizationSpec:
    values = config.get("features", {}).get("node_text_normalization", {})
    return NormalizationSpec(
        mode=values.get("mode", "raw"),
        version=int(values.get("version", 1)),
        api_taxonomy_version=int(values.get("api_taxonomy_version", 1)),
        tokenizer_version=int(values.get("tokenizer_version", 2)),
    )


def _layout(config: dict) -> ArtifactLayout:
    return ArtifactLayout(
        artifacts_root=Path(config["paths"]["artifacts_dir"]),
        outputs_root=Path(config["paths"]["outputs_dir"]),
        spec=_normalization_spec(config),
    )


def _model_args(config: dict) -> dict:
    return {
        "hidden_dim": config["model"]["hidden_dim"],
        "node_type_dim": config["model"]["node_type_dim"],
        "dropout": config["model"]["dropout"],
    }


def _devign_model_args(config: dict) -> dict:
    devign = config.get("devign", {})
    model = config.get("model", {})
    return {
        "hidden_dim": int(devign.get("hidden_dim", model.get("hidden_dim", 128))),
        "node_type_dim": int(devign.get("node_type_dim", model.get("node_type_dim", 32))),
        "dropout": float(devign.get("dropout", model.get("dropout", 0.3))),
        "steps": int(devign.get("steps", 6)),
        "max_nodes": int(devign.get("max_nodes", 205)),
    }


def _devign_weight_decay(config: dict, fallback: float) -> float:
    return float(config.get("devign", {}).get("weight_decay", fallback))


def _ramp_model(
    *,
    model_name: str,
    input_dim: int,
    function_dim: int,
    num_node_types: int,
    num_relations: int,
    config: dict,
    ) -> torch.nn.Module:
    model_config = config.get("model", {})
    ramp_v2_config = config.get("ramp_v2", {})
    ramp_v3_config = config.get("ramp_v3", {})
    if model_name == "selective-fusion":
        from cpg_vuln.models.selective_fusion import SelectiveFusionCPG

        return SelectiveFusionCPG(
            input_dim=input_dim,
            function_dim=function_dim,
            num_node_types=num_node_types,
            num_relations=num_relations,
            use_semantics=True,
            **_model_args(config),
        )
    if model_name in {
        "ramp-v2-rgcn",
        "ramp-v2-dual",
        "ramp-v2-gated-rgcn",
        "ramp-v3-slice-mil",
    }:
        from cpg_vuln.models.ramp_v2 import (
            RampV2CPG,
            RampV2DualHeadCPG,
            RampV2GatedRGCNCPG,
            RampV3SliceMILCPG,
        )

        model_classes = {
            "ramp-v2-rgcn": RampV2CPG,
            "ramp-v2-dual": RampV2DualHeadCPG,
            "ramp-v2-gated-rgcn": RampV2GatedRGCNCPG,
            "ramp-v3-slice-mil": RampV3SliceMILCPG,
        }
        dropout_default = (
            0.25
            if model_name in {"ramp-v2-gated-rgcn", "ramp-v3-slice-mil"}
            else 0.2
        )
        model_kwargs = {
            "input_dim": input_dim,
            "function_dim": function_dim,
            "num_node_types": num_node_types,
            "num_relations": num_relations,
            "hidden_dim": int(ramp_v2_config.get("hidden_dim", 256)),
            "node_type_dim": int(model_config.get("node_type_dim", 32)),
            "layers": int(ramp_v2_config.get("layers", 3)),
            "dropout": float(ramp_v2_config.get("dropout", dropout_default)),
            "encoder": "rgcn",
            "use_semantics": True,
        }
        if model_name == "ramp-v2-gated-rgcn":
            gated_config = ramp_v2_config.get("gated_rgcn", {})
            model_kwargs.update(
                {
                    "gate_bias_init": float(gated_config.get("gate_bias_init", -1.0)),
                    "ffn_multiplier": int(gated_config.get("ffn_multiplier", 2)),
                }
            )
        if model_name == "ramp-v3-slice-mil":
            fusion_logit_init = tuple(
                float(value)
                for value in ramp_v3_config.get(
                    "fusion_logit_init",
                    (4.0, 0.0, 0.0, -2.0),
                )
            )
            model_kwargs.update(
                {
                    "slice_top_k": int(ramp_v3_config.get("slice_top_k", 3)),
                    "slice_temperature": float(ramp_v3_config.get("slice_temperature", 1.0)),
                    "fusion_logit_init": fusion_logit_init,
                }
            )

        return model_classes[model_name](
            **model_kwargs,
        )
    raise ValueError(f"unsupported RAMP model: {model_name}")


def _train_config(config: dict, *, epochs: int | None, learning_rate: float) -> TrainConfig:
    from cpg_vuln.training.engine import TrainConfig

    values = dict(config["training"])
    values["learning_rate"] = learning_rate
    if epochs is not None:
        values["epochs"] = epochs
    return TrainConfig(**values)


def _enhanced_view(variant: str) -> str:
    if variant == "dataflow-only":
        return "dataflow-cpg"
    if variant == "slice-fusion":
        return "slice-cpg"
    return "core-cpg"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_model(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _completed(run_dir: Path, *, force: bool) -> bool:
    return not force and (run_dir / "metrics.json").is_file()
