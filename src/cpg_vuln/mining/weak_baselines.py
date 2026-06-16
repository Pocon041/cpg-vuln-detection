from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from cpg_vuln.data.store import load_topology
from cpg_vuln.mining.motif import RiskMotif, extract_risk_motif


FEATURE_GROUPS = {
    "api": [
        "known_alloc_api",
        "known_copy_api",
        "known_release_api",
        "known_io_api",
        "indirect_call",
    ],
    "scale": ["num_nodes", "num_edges", "branch_count", "loop_count", "return_count"],
    "relation": ["num_ast_edges", "num_cfg_edges", "num_cdg_edges", "num_reaching_def_edges"],
    "motif": None,
}


def feature_matrix(
    motifs: list[RiskMotif],
    *,
    group: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if group not in FEATURE_GROUPS:
        raise ValueError(f"unsupported weak baseline group: {group}")
    names = FEATURE_GROUPS[group]
    matrix = (
        np.stack([motif.to_vector() for motif in motifs]).astype(np.float32)
        if names is None
        else np.asarray(
            [[motif.features.get(name, 0.0) for name in names] for motif in motifs],
            dtype=np.float32,
        )
    )
    labels = np.asarray([motif.label for motif in motifs], dtype=np.int64)
    sample_ids = [motif.sample_id for motif in motifs]
    return matrix, labels, sample_ids


def fit_and_score_group(
    train_motifs: list[RiskMotif],
    val_motifs: list[RiskMotif],
    test_motifs: list[RiskMotif],
    *,
    group: str,
    classifier_factory: Callable[[], object],
) -> dict[str, float | int]:
    train_matrix, train_labels, _ = feature_matrix(train_motifs, group=group)
    val_matrix, val_labels, _ = feature_matrix(val_motifs, group=group)
    test_matrix, test_labels, _ = feature_matrix(test_motifs, group=group)
    model = classifier_factory()
    model.fit(train_matrix, train_labels)
    val_probabilities = model.predict_proba(val_matrix)[:, 1]
    test_probabilities = model.predict_proba(test_matrix)[:, 1]
    return {
        "train_samples": int(train_labels.size),
        "val_samples": int(val_labels.size),
        "test_samples": int(test_labels.size),
        "val_roc_auc": float(roc_auc_score(val_labels, val_probabilities)),
        "val_pr_auc": float(average_precision_score(val_labels, val_probabilities)),
        "test_roc_auc": float(roc_auc_score(test_labels, test_probabilities)),
        "test_pr_auc": float(average_precision_score(test_labels, test_probabilities)),
    }


def score_weak_baseline_groups(
    train_motifs: list[RiskMotif],
    val_motifs: list[RiskMotif],
    test_motifs: list[RiskMotif],
    *,
    classifier_factory,
) -> dict[str, dict[str, float | int]]:
    results = {}
    for group in tqdm(
        ("api", "motif", "scale", "relation"),
        desc="weak baseline groups",
        unit="group",
        ascii=True,
    ):
        results[group] = fit_and_score_group(
            train_motifs,
            val_motifs,
            test_motifs,
            group=group,
            classifier_factory=classifier_factory,
        )
    return results


def run_weak_baselines(config: dict, *, split: str, view: str) -> None:
    artifacts = Path(config["paths"]["artifacts_dir"])
    outputs = Path(config["paths"]["outputs_dir"])
    index = json.loads((artifacts / "topologies" / "index.json").read_text(encoding="utf-8"))
    split_payload = json.loads(
        (artifacts / "data" / "splits" / f"{split}.json").read_text(encoding="utf-8")
    )
    all_split_ids = split_payload["train"] + split_payload["val"] + split_payload["test"]
    paths = {
        item["sample_id"]: Path(item["path"])
        for item in index
        if item["view"] == view and item["sample_id"] in all_split_ids
    }

    def load_motifs(sample_ids: list[str], *, split_name: str) -> list[RiskMotif]:
        return [
            extract_risk_motif(load_topology(paths[sample_id]))
            for sample_id in tqdm(
                sample_ids,
                desc=f"load {split_name} weak-baseline motifs",
                unit="sample",
                ascii=True,
            )
            if sample_id in paths
        ]

    train_motifs = load_motifs(split_payload["train"], split_name="train")
    val_motifs = load_motifs(split_payload["val"], split_name="val")
    test_motifs = load_motifs(split_payload["test"], split_name="test")
    results = score_weak_baseline_groups(
        train_motifs,
        val_motifs,
        test_motifs,
        classifier_factory=lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    output = outputs / "reports" / f"weak_baselines_{split}_{view}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
