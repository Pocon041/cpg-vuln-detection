from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cpg_vuln.data.graphml import GraphMLParser, choose_primary_method
from cpg_vuln.data.store import NodeTypeRegistry, save_topology
from cpg_vuln.data.topology import build_view
from cpg_vuln.features.cache import MemmapFeatureCache
from cpg_vuln.features.text import NodeTextRegistry
from cpg_vuln.training.runner import train_baselines, train_enhanced

from .helpers import write_graphml


def test_baseline_and_enhanced_runners_consume_cached_topologies(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    topologies = artifacts / "topologies"
    graph_path = tmp_path / "sample.graphml"
    write_graphml(graph_path)
    graph = GraphMLParser().parse(graph_path)
    root = choose_primary_method(graph)
    texts = NodeTextRegistry()
    node_types = NodeTypeRegistry()
    index = []
    sample_ids = [f"sample-{number}" for number in range(8)]
    for number, sample_id in enumerate(sample_ids):
        for view in ("ast", "core-cpg", "dataflow-cpg"):
            index.append(
                save_topology(
                    topologies / view / f"{sample_id}.pt",
                    build_view(graph, root, view),
                    sample_id,
                    number % 2,
                    texts,
                    node_types,
                )
            )
    topologies.mkdir(parents=True, exist_ok=True)
    (topologies / "index.json").write_text(json.dumps(index), encoding="utf-8")
    node_types.write(topologies / "node_type_registry.json")
    split_dir = artifacts / "data" / "splits"
    split_dir.mkdir(parents=True)
    split = {"train": sample_ids[:4], "val": sample_ids[4:6], "test": sample_ids[6:]}
    (split_dir / "course.json").write_text(json.dumps(split), encoding="utf-8")
    node_cache = MemmapFeatureCache.create(
        artifacts / "features" / "word2vec" / "features", rows=len(texts), dim=8
    )
    codebert_cache = MemmapFeatureCache.create(
        artifacts / "features" / "codebert" / "nodes", rows=len(texts), dim=8
    )
    values = np.ones((len(texts), 8), dtype=np.float32)
    node_cache.write(list(range(len(texts))), values)
    codebert_cache.write(list(range(len(texts))), values)
    functions = MemmapFeatureCache.create(
        artifacts / "features" / "codebert" / "functions" / "features",
        rows=len(sample_ids),
        dim=8,
    )
    functions.write(list(range(len(sample_ids))), np.ones((len(sample_ids), 8), dtype=np.float32))
    (artifacts / "features" / "codebert" / "functions" / "function_indices.json").write_text(
        json.dumps({sample_id: index for index, sample_id in enumerate(sample_ids)}),
        encoding="utf-8",
    )
    config = {
        "paths": {"artifacts_dir": str(artifacts), "outputs_dir": str(tmp_path / "outputs")},
        "model": {"hidden_dim": 8, "node_type_dim": 4, "dropout": 0.0},
        "training": {
            "epochs": 1,
            "patience": 1,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "max_nodes": 100,
            "max_edges": 100,
            "seed": 42,
            "device": "cpu",
        },
    }

    train_baselines(config, views=("ast",), embeddings=("word2vec",), splits=("course",), epochs=1)
    train_enhanced(config, splits=("course",), variants=("selective-fusion",), epochs=1)

    assert (tmp_path / "outputs" / "runs" / "baseline-ast-word2vec-course" / "best.pt").is_file()
    assert (tmp_path / "outputs" / "runs" / "enhanced-selective-fusion-course" / "best.pt").is_file()
    enhanced_metrics = json.loads(
        (
            tmp_path / "outputs" / "runs" / "enhanced-selective-fusion-course" / "metrics.json"
        ).read_text(encoding="utf-8")
    )
    assert enhanced_metrics["config"]["learning_rate"] == 1e-4
    baseline_metrics = tmp_path / "outputs" / "runs" / "baseline-ast-word2vec-course" / "metrics.json"
    baseline_metrics.write_text("keep existing completed run", encoding="utf-8")

    train_baselines(config, views=("ast",), embeddings=("word2vec",), splits=("course",), epochs=1)

    assert baseline_metrics.read_text(encoding="utf-8") == "keep existing completed run"
