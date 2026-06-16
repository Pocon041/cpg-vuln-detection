from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from cpg_vuln.config import load_config
from cpg_vuln.data.store import load_topology
from cpg_vuln.features.cache import MemmapFeatureCache
from cpg_vuln.training.runner import _layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether extracted CPG/text/function intermediate "
            "representations contain trainable and generalizable label signal."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Config to audit. Repeatable. Defaults to raw and full-anon configs.",
    )
    parser.add_argument("--split", default="strict")
    parser.add_argument("--view", default="core-cpg")
    parser.add_argument(
        "--output",
        default="outputs/reports/intermediate_representation_audit.json",
    )
    parser.add_argument("--rf-trees", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = [
        Path(path)
        for path in (
            args.config
            or [
                "configs/default.yaml",
                "configs/full_anon.yaml",
            ]
        )
    ]
    stages = tqdm(total=len(config_paths) * 4, desc="representation audit", unit="stage", ascii=True)
    report = {
        "note": (
            "This is a shallow representation audit. High train metrics with low "
            "validation/test metrics indicate non-generalizable signal in the "
            "extracted intermediate representation."
        ),
        "split": args.split,
        "view": args.view,
        "configs": {},
    }
    for config_path in config_paths:
        config_report = audit_config(
            config_path,
            split_name=args.split,
            view=args.view,
            rf_trees=args.rf_trees,
            stages=stages,
        )
        report["configs"][str(config_path)] = config_report
        print_config_summary(str(config_path), config_report)
    stages.close()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path}", flush=True)


def audit_config(
    config_path: Path,
    *,
    split_name: str,
    view: str,
    rf_trees: int,
    stages,
) -> dict[str, object]:
    config = load_config(config_path)
    layout = _layout(config)
    split_payload = json.loads(
        (
            Path(config["paths"]["artifacts_dir"])
            / "data"
            / "splits"
            / f"{split_name}.json"
        ).read_text(encoding="utf-8")
    )
    records = load_split_records(layout.topology_dir / "index.json", split_payload, view=view)
    stages.update(1)

    node_cache = MemmapFeatureCache.open(layout.node_codebert_dir, read_only=True)
    function_cache = MemmapFeatureCache.open(
        layout.function_codebert_dir / "features",
        read_only=True,
    )
    function_indices = json.loads(
        (layout.function_codebert_dir / "function_indices.json").read_text(encoding="utf-8")
    )
    source_normalization = _read_json_optional(
        layout.function_codebert_dir / "source_normalization.json"
    )
    cache_report = {
        "node_codebert": cache_stats(node_cache),
        "function_codebert": cache_stats(function_cache),
        "function_source_normalization": source_normalization,
    }
    stages.update(1)

    summary = split_summary(records)
    labels = {
        split: np.asarray([int(record["label"]) for record in rows], dtype=np.int64)
        for split, rows in records.items()
    }
    all_records = records["train"] + records["val"] + records["test"]
    structure_matrix, structure_feature_names = histogram_feature_matrix(all_records)
    offsets = split_offsets(records)

    def split_matrix(matrix: np.ndarray, split: str) -> np.ndarray:
        start, stop = offsets[split]
        return matrix[start:stop]

    structure_results = {
        "feature_count": len(structure_feature_names),
        "feature_names": structure_feature_names,
        "logreg": fit_and_score(
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ),
            split_matrix(structure_matrix, "train"),
            labels["train"],
            {
                split: split_matrix(structure_matrix, split)
                for split in ("train", "val", "test")
            },
            labels,
        ),
        "random_forest": fit_and_score(
            RandomForestClassifier(
                n_estimators=rf_trees,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            split_matrix(structure_matrix, "train"),
            labels["train"],
            {
                split: split_matrix(structure_matrix, split)
                for split in ("train", "val", "test")
            },
            labels,
        ),
    }
    stages.update(1)

    node_features = {
        split: node_pool_features(rows, node_cache)
        for split, rows in records.items()
    }
    function_features = {
        split: function_feature_matrix(rows, function_cache, function_indices)
        for split, rows in records.items()
    }
    semantic_results = {
        "node_codebert_pool_logreg": fit_and_score(
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"),
            ),
            node_features["train"],
            labels["train"],
            node_features,
            labels,
        ),
        "function_codebert_logreg": fit_and_score(
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"),
            ),
            function_features["train"],
            labels["train"],
            function_features,
            labels,
        ),
    }
    stages.update(1)

    return {
        "normalization_key": layout.spec.normalization_key,
        "normalization_mode": layout.spec.mode,
        "topology_dir": str(layout.topology_dir),
        "node_codebert_dir": str(layout.node_codebert_dir),
        "function_codebert_dir": str(layout.function_codebert_dir),
        "split_summary": summary,
        "cache": cache_report,
        "structure": structure_results,
        "semantic": semantic_results,
    }


def fixed_threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    predictions = (probabilities >= threshold).astype(np.int64)
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    return {
        "threshold": float(threshold),
        "roc_auc": finite_metric(roc_auc_score, labels, probabilities),
        "pr_auc": finite_metric(average_precision_score, labels, probabilities),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "predicted_positive_rate": float(predictions.mean()) if predictions.size else 0.0,
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
    }


def finite_metric(function, labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    try:
        value = float(function(labels, probabilities))
    except ValueError:
        return None
    return None if math.isnan(value) else value


def histogram_feature_matrix(
    records: list[dict[str, object]],
) -> tuple[np.ndarray, list[str]]:
    node_keys = sorted(
        {
            key
            for record in records
            for key in dict(record.get("node_histogram", {}))
        }
    )
    edge_keys = sorted(
        {
            key
            for record in records
            for key in dict(record.get("edge_histogram", {}))
        }
    )
    feature_names = (
        ["log_nodes", "log_edges"]
        + [f"node:{key}_fraction" for key in node_keys]
        + [f"edge:{key}_log_count" for key in edge_keys]
    )
    rows = []
    for record in records:
        nodes = int(record.get("nodes", 0))
        edges = int(record.get("edges", 0))
        node_histogram = dict(record.get("node_histogram", {}))
        edge_histogram = dict(record.get("edge_histogram", {}))
        node_denominator = max(1, nodes)
        row = [math.log1p(nodes), math.log1p(edges)]
        row.extend(float(node_histogram.get(key, 0)) / node_denominator for key in node_keys)
        row.extend(math.log1p(float(edge_histogram.get(key, 0))) for key in edge_keys)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32), feature_names


def pool_node_feature_arrays(graph_vectors: Iterable[np.ndarray]) -> np.ndarray:
    rows = []
    for values in graph_vectors:
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            rows.append(np.empty((0,), dtype=np.float32))
            continue
        rows.append(np.concatenate([values.mean(axis=0), values.max(axis=0)], axis=0))
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    width = max(row.shape[0] for row in rows)
    return np.asarray(
        [
            row if row.shape[0] == width else np.zeros(width, dtype=np.float32)
            for row in rows
        ],
        dtype=np.float32,
    )


def load_split_records(
    index_path: Path,
    split_payload: dict[str, list[str]],
    *,
    view: str,
) -> dict[str, list[dict[str, object]]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    by_sample = {
        item["sample_id"]: item
        for item in index
        if item["view"] == view
    }
    records = {}
    for split, sample_ids in split_payload.items():
        rows = []
        for sample_id in tqdm(
            sample_ids,
            desc=f"load {view} {split} topologies",
            unit="graph",
            ascii=True,
        ):
            item = by_sample.get(sample_id)
            if item is None:
                continue
            payload = load_topology(Path(item["path"]))
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "path": str(item["path"]),
                    "label": int(payload["y"].item()),
                    "nodes": int(item["nodes"]),
                    "edges": int(item["edges"]),
                    "text_id": payload["text_id"].numpy(),
                    "node_histogram": dict(payload.get("node_type_histogram", {})),
                    "edge_histogram": dict(payload.get("edge_type_histogram", {})),
                }
            )
        records[split] = rows
    return records


def split_summary(records: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    return {
        split: summarize_split(rows)
        for split, rows in records.items()
    }


def summarize_split(records: list[dict[str, object]]) -> dict[str, object]:
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
    nodes = np.asarray([int(record["nodes"]) for record in records], dtype=np.int64)
    edges = np.asarray([int(record["edges"]) for record in records], dtype=np.int64)
    result = {
        "samples": int(labels.size),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "nodes_median": float(np.median(nodes)) if nodes.size else 0.0,
        "edges_median": float(np.median(edges)) if edges.size else 0.0,
        "by_label": {},
    }
    for label in (0, 1):
        subset = [record for record in records if int(record["label"]) == label]
        result["by_label"][str(label)] = edge_coverage_summary(subset)
    return result


def edge_coverage_summary(records: list[dict[str, object]]) -> dict[str, float | int]:
    if not records:
        return {
            "samples": 0,
            "missing_cfg_rate": 0.0,
            "missing_cdg_rate": 0.0,
            "missing_reaching_def_rate": 0.0,
            "no_non_self_edges_rate": 0.0,
        }
    return {
        "samples": len(records),
        "missing_cfg_rate": _mean_missing_edge(records, "CFG"),
        "missing_cdg_rate": _mean_missing_edge(records, "CDG"),
        "missing_reaching_def_rate": _mean_missing_edge(records, "REACHING_DEF"),
        "no_non_self_edges_rate": float(
            np.mean(
                [
                    sum(
                        int(value)
                        for key, value in dict(record.get("edge_histogram", {})).items()
                        if key != "SELF_LOOP"
                    )
                    == 0
                    for record in records
                ]
            )
        ),
    }


def _mean_missing_edge(records: list[dict[str, object]], edge_name: str) -> float:
    return float(
        np.mean(
            [
                int(dict(record.get("edge_histogram", {})).get(edge_name, 0)) == 0
                for record in records
            ]
        )
    )


def cache_stats(cache: MemmapFeatureCache) -> dict[str, object]:
    sample_count = min(2000, cache.metadata.rows)
    if sample_count:
        sample_indices = np.linspace(0, cache.metadata.rows - 1, sample_count, dtype=np.int64)
        sample = cache.read(sample_indices).astype(np.float32)
        abs_mean = float(np.abs(sample).mean())
        mean_feature_std = float(sample.std(axis=0).mean())
        zero_row_rate = float((np.linalg.norm(sample, axis=1) == 0).mean())
    else:
        abs_mean = mean_feature_std = zero_row_rate = 0.0
    return {
        "rows": int(cache.metadata.rows),
        "dim": int(cache.metadata.dim),
        "dtype": cache.metadata.dtype,
        "normalization_key": cache.metadata.normalization_key,
        "normalization_fingerprint": cache.metadata.normalization_fingerprint,
        "producer": cache.metadata.producer,
        "producer_fingerprint": cache.metadata.producer_fingerprint,
        "complete": bool(np.all(cache.completed)),
        "sample_abs_mean": abs_mean,
        "sample_mean_feature_std": mean_feature_std,
        "sample_zero_row_rate": zero_row_rate,
    }


def node_pool_features(
    records: list[dict[str, object]],
    cache: MemmapFeatureCache,
) -> np.ndarray:
    graph_vectors = [
        cache.read(record["text_id"]).astype(np.float32, copy=False)
        for record in tqdm(records, desc="pool node CodeBERT", unit="graph", ascii=True)
    ]
    return pool_node_feature_arrays(graph_vectors)


def function_feature_matrix(
    records: list[dict[str, object]],
    cache: MemmapFeatureCache,
    function_indices: dict[str, int],
) -> np.ndarray:
    indices = [int(function_indices[str(record["sample_id"])]) for record in records]
    return cache.read(indices).astype(np.float32, copy=False)


def fit_and_score(
    estimator,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    features_by_split: dict[str, np.ndarray],
    labels_by_split: dict[str, np.ndarray],
) -> dict[str, object]:
    estimator.fit(train_features, train_labels)
    results = {}
    for split, features in features_by_split.items():
        labels = labels_by_split[split]
        probabilities = estimator.predict_proba(features)[:, 1]
        results[split] = fixed_threshold_metrics(labels, probabilities, threshold=0.5)
    test_auc = results["test"]["roc_auc"]
    train_auc = results["train"]["roc_auc"]
    if train_auc is not None and test_auc is not None:
        results["train_test_auc_gap"] = float(train_auc - test_auc)
    return results


def split_offsets(records: dict[str, list[dict[str, object]]]) -> dict[str, tuple[int, int]]:
    offsets = {}
    start = 0
    for split in ("train", "val", "test"):
        stop = start + len(records[split])
        offsets[split] = (start, stop)
        start = stop
    return offsets


def _read_json_optional(path: Path) -> object | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def print_config_summary(config_name: str, report: dict[str, object]) -> None:
    print(f"\n== {config_name} ({report['normalization_key']}) ==", flush=True)
    function_source = report["cache"]["function_source_normalization"]
    print(f"function_source_normalization={function_source}", flush=True)
    for family, models in (
        ("structure", report["structure"]),
        ("semantic", report["semantic"]),
    ):
        print(f"{family}:", flush=True)
        for name, value in models.items():
            if not isinstance(value, dict) or "test" not in value:
                continue
            train = value["train"]
            test = value["test"]
            gap = value.get("train_test_auc_gap")
            print(
                f"  {name}: "
                f"train_auc={_format(train['roc_auc'])} "
                f"test_auc={_format(test['roc_auc'])} "
                f"test_prauc={_format(test['pr_auc'])} "
                f"test_f1={_format(test['f1'])} "
                f"test_mcc={_format(test['mcc'])} "
                f"gap={_format(gap)}",
                flush=True,
            )


def _format(value: object) -> str:
    return "None" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
