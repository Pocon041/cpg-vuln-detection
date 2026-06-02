from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "paths": {
        "dataset_root": "data/fq_graphml_dataset",
        "metadata_csv": "data/fq_graphml_dataset/metadata/labels.csv",
        "source_root": "F&Q/F&Q",
        "excluded_csv": "data/fq_dataset/fq_dataset/metadata/excluded.csv",
        "artifacts_dir": "artifacts",
        "outputs_dir": "outputs",
    },
    "source_mapping": {
        "default_line_offset": 64,
        "source_map_path": "artifacts/manifests/source_map.csv",
        "prepared_source_root": None,
        "overrides_path": "configs/source_map_overrides.csv",
        "validate_offsets": True,
        "allow_sample_overrides": True,
        "validation": {
            "max_sampled_nodes": 32,
            "context_radius": 2,
            "minimum_token_match_ratio": 0.5,
        },
    },
    "features": {
        "word2vec_dim": 128,
        "word2vec_epochs": 10,
        "word2vec_batch_size": 1024,
        "codebert_model": "microsoft/codebert-base",
        "codebert_node_max_length": 64,
        "codebert_batch_size": 64,
        "function_max_tokens": 510,
        "function_overlap": 256,
    },
    "model": {"hidden_dim": 128, "node_type_dim": 32, "dropout": 0.3},
    "training": {
        "epochs": 50,
        "patience": 8,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "max_nodes": 8000,
        "max_edges": 60000,
        "seed": 42,
        "device": "cuda",
    },
}


def load_config(path: Path | None = None) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _deep_merge(config, loaded)
    root = Path.cwd()
    config["paths"] = {
        name: str(_resolve(root, value)) for name, value in config["paths"].items()
    }
    for name in ("source_map_path", "prepared_source_root", "overrides_path"):
        value = config["source_mapping"][name]
        config["source_mapping"][name] = str(_resolve(root, value)) if value else None
    return config


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _deep_merge(target: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
