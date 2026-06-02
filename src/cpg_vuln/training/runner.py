from __future__ import annotations

import json
from pathlib import Path

import torch

from cpg_vuln.data.batch import GraphSize
from cpg_vuln.data.dataset import TopologyDataset
from cpg_vuln.data.store import NodeTypeRegistry, load_topology
from cpg_vuln.features.cache import MemmapFeatureCache
from cpg_vuln.models.gcn import GCNClassifier
from cpg_vuln.models.selective_fusion import SelectiveFusionCPG
from cpg_vuln.training.engine import TrainConfig, run_training


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
    outputs = Path(config["paths"]["outputs_dir"]) / "runs"
    index = _topology_index(artifacts)
    node_types = NodeTypeRegistry.read(artifacts / "topologies" / "node_type_registry.json")
    for split_name in splits:
        split = _read_json(artifacts / "data" / "splits" / f"{split_name}.json")
        for view in views:
            for embedding in embeddings:
                run_dir = outputs / f"baseline-{view}-{embedding}-{split_name}"
                if _completed(run_dir, force=force):
                    continue
                cache = _node_feature_cache(artifacts, embedding)
                datasets = _datasets(index, split, view, cache)
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
                    },
                )


def train_enhanced(
    config: dict,
    *,
    splits: tuple[str, ...] = ("course", "strict"),
    variants: tuple[str, ...] = ("selective-fusion", "no-semantics", "dataflow-only"),
    epochs: int | None = None,
    force: bool = False,
) -> None:
    artifacts = Path(config["paths"]["artifacts_dir"])
    outputs = Path(config["paths"]["outputs_dir"]) / "runs"
    index = _topology_index(artifacts)
    node_types = NodeTypeRegistry.read(artifacts / "topologies" / "node_type_registry.json")
    node_cache = _node_feature_cache(artifacts, "codebert")
    function_cache = MemmapFeatureCache.open(
        artifacts / "features" / "codebert" / "functions" / "features",
        read_only=True,
    )
    function_indices = _read_json(
        artifacts / "features" / "codebert" / "functions" / "function_indices.json"
    )
    for split_name in splits:
        split = _read_json(artifacts / "data" / "splits" / f"{split_name}.json")
        for variant in variants:
            view = "dataflow-cpg" if variant == "dataflow-only" else "core-cpg"
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
                },
            )


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


def _node_feature_cache(artifacts: Path, embedding: str) -> MemmapFeatureCache:
    path = (
        artifacts / "features" / "word2vec" / "features"
        if embedding == "word2vec"
        else artifacts / "features" / "codebert" / "nodes"
    )
    return MemmapFeatureCache.open(path, read_only=True)


def _topology_index(artifacts: Path) -> list[dict]:
    return _read_json(artifacts / "topologies" / "index.json")


def _model_args(config: dict) -> dict:
    return {
        "hidden_dim": config["model"]["hidden_dim"],
        "node_type_dim": config["model"]["node_type_dim"],
        "dropout": config["model"]["dropout"],
    }


def _train_config(config: dict, *, epochs: int | None, learning_rate: float) -> TrainConfig:
    values = dict(config["training"])
    values["learning_rate"] = learning_rate
    if epochs is not None:
        values["epochs"] = epochs
    return TrainConfig(**values)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_model(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _completed(run_dir: Path, *, force: bool) -> bool:
    return not force and (run_dir / "metrics.json").is_file()
