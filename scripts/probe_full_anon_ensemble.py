from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def finite_score(function, labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    try:
        value = float(function(labels, probabilities))
    except ValueError:
        return None
    return None if math.isnan(value) else value


def threshold_stats(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    precision = (
        0.0
        if true_positive + false_positive == 0
        else true_positive / (true_positive + false_positive)
    )
    recall = (
        0.0
        if true_positive + false_negative == 0
        else true_positive / (true_positive + false_negative)
    )
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    specificity = (
        0.0
        if true_negative + false_positive == 0
        else true_negative / (true_negative + false_positive)
    )
    accuracy = (
        (true_positive + true_negative) / labels.size
        if labels.size
        else 0.0
    )
    balanced_accuracy = (recall + specificity) / 2
    denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (
        0.0
        if denominator == 0
        else ((true_positive * true_negative) - (false_positive * false_negative))
        / denominator
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "predicted_positive_rate": (
            float(predictions.mean()) if predictions.size else 0.0
        ),
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
    }


def scan_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    objective: str,
    recall_max: float | None = None,
    positive_rate_max: float | None = None,
    min_precision: float | None = None,
) -> dict[str, object] | None:
    candidates = np.unique(probabilities.astype(np.float64))
    candidates = np.unique(
        np.concatenate([candidates, np.asarray([0.0, 0.5, 1.0], dtype=np.float64)])
    )
    best: dict[str, object] | None = None
    best_key: tuple[float, ...] | None = None
    for threshold in candidates:
        stats = threshold_stats(labels, probabilities, float(threshold))
        if recall_max is not None and float(stats["recall"]) > recall_max:
            continue
        if (
            positive_rate_max is not None
            and float(stats["predicted_positive_rate"]) > positive_rate_max
        ):
            continue
        if min_precision is not None and float(stats["precision"]) < min_precision:
            continue
        if objective == "f1":
            key = (
                float(stats["f1"]),
                float(stats["mcc"]),
                float(stats["precision"]),
                float(stats["threshold"]),
            )
        elif objective == "mcc":
            key = (
                float(stats["mcc"]),
                float(stats["f1"]),
                float(stats["precision"]),
                float(stats["threshold"]),
            )
        elif objective == "balanced_accuracy":
            key = (
                float(stats["balanced_accuracy"]),
                float(stats["mcc"]),
                float(stats["f1"]),
                float(stats["threshold"]),
            )
        else:
            raise ValueError(f"unsupported objective: {objective}")
        if best_key is None or key > best_key:
            best_key = key
            best = stats
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe a validation-selected probability ensemble between the full-anon "
            "enhanced selective-fusion model and the full-anon RAMP-v2 model."
        )
    )
    parser.add_argument("--config", default="configs/full_anon.yaml")
    parser.add_argument("--split", default="strict")
    parser.add_argument("--view", default="core-cpg")
    parser.add_argument(
        "--selective-run",
        default="outputs/runs/full-anon-v1/enhanced-selective-fusion-strict",
    )
    parser.add_argument(
        "--ramp-run",
        default="outputs/runs/full-anon-v1/ramp-E4-v2A-full-anon-strict",
    )
    parser.add_argument(
        "--ramp-model",
        choices=("auto", "selective-fusion", "ramp-v2-rgcn"),
        default="auto",
        help="Architecture for --ramp-run; auto reads run_metadata.model_name.",
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/reports/full-anon-v1/"
            "ensemble_probe_selective_ramp_v2_guarded.json"
        ),
    )
    parser.add_argument(
        "--weight-steps",
        type=int,
        default=41,
        help="Number of evenly spaced ramp weights in [0, 1].",
    )
    return parser.parse_args()


def main() -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from tqdm import tqdm

    import torch

    from cpg_vuln.config import load_config
    from cpg_vuln.data.store import NodeTypeRegistry, load_topology
    from cpg_vuln.features.cache import MemmapFeatureCache
    from cpg_vuln.models.selective_fusion import SelectiveFusionCPG
    from cpg_vuln.training.engine import _loader, _predict
    from cpg_vuln.training.runner import (
        _datasets,
        _layout,
        _model_args,
        _node_feature_cache,
        _ramp_model,
        _topology_index,
        _train_config,
    )

    args = parse_args()
    config_path = Path(args.config)
    output_path = Path(args.output)
    selective_run = Path(args.selective_run)
    ramp_run = Path(args.ramp_run)
    ramp_model_name = ramp_model_name_for_run(ramp_run, requested=args.ramp_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stages = tqdm(total=5, desc="ensemble probe stages", unit="stage", ascii=True)
    config = load_config(config_path)
    layout = _layout(config)
    index = _topology_index(layout)
    split_payload = _read_json(
        Path(config["paths"]["artifacts_dir"]) / "data" / "splits" / f"{args.split}.json"
    )
    node_types = NodeTypeRegistry.read(layout.topology_dir / "node_type_registry.json")
    node_cache = _node_feature_cache(layout, "codebert")
    function_cache = MemmapFeatureCache.open(
        layout.function_codebert_dir / "features",
        read_only=True,
    )
    function_indices = _read_json(layout.function_codebert_dir / "function_indices.json")
    datasets = _datasets(
        index,
        split_payload,
        args.view,
        node_cache,
        function_cache=function_cache,
        function_indices=function_indices,
    )
    sample_payload = load_topology(datasets["train"].topology_paths[0])
    num_relations = len(sample_payload["edge_type_names"])
    train_config = _train_config(config, epochs=None, learning_rate=1e-4)
    device = torch.device(
        train_config.device
        if train_config.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    stages.update(1)
    tqdm.write(
        "context: "
        f"device={device}, node_dim={node_cache.metadata.dim}, "
        f"function_dim={function_cache.metadata.dim}, "
        f"node_types={len(node_types)}, relations={num_relations}"
    )

    selective = SelectiveFusionCPG(
        input_dim=node_cache.metadata.dim,
        function_dim=function_cache.metadata.dim,
        num_node_types=len(node_types),
        num_relations=num_relations,
        use_semantics=True,
        **_model_args(config),
    )
    ramp = _ramp_model(
        model_name=ramp_model_name,
        input_dim=node_cache.metadata.dim,
        function_dim=function_cache.metadata.dim,
        num_node_types=len(node_types),
        num_relations=num_relations,
        config=config,
    )

    pred_selective = _predict_model(
        "selective",
        selective,
        selective_run / "best.pt",
        datasets,
        train_config,
        device,
        torch=torch,
        loader_fn=_loader,
        predict_fn=_predict,
        tqdm=tqdm,
    )
    stages.update(1)
    pred_ramp = _predict_model(
        "ramp-v2",
        ramp,
        ramp_run / "best.pt",
        datasets,
        train_config,
        device,
        torch=torch,
        loader_fn=_loader,
        predict_fn=_predict,
        tqdm=tqdm,
    )
    stages.update(1)

    ref_val_ids = list(pred_selective["val"]["sample_ids"])
    ref_test_ids = list(pred_selective["test"]["sample_ids"])
    val_labels, val_selective = _align_prediction(ref_val_ids, pred_selective["val"])
    test_labels, test_selective = _align_prediction(ref_test_ids, pred_selective["test"])
    val_labels_ramp, val_ramp = _align_prediction(ref_val_ids, pred_ramp["val"])
    test_labels_ramp, test_ramp = _align_prediction(ref_test_ids, pred_ramp["test"])
    if not np.array_equal(val_labels, val_labels_ramp):
        raise RuntimeError("validation labels differ between model predictions")
    if not np.array_equal(test_labels, test_labels_ramp):
        raise RuntimeError("test labels differ between model predictions")

    weight_count = max(2, int(args.weight_steps))
    weights = np.linspace(0.0, 1.0, weight_count)
    strategies = _strategies()
    best_by_strategy, weight_rows = _scan_ensemble(
        weights,
        strategies,
        val_labels=val_labels,
        test_labels=test_labels,
        val_selective=val_selective,
        test_selective=test_selective,
        val_ramp=val_ramp,
        test_ramp=test_ramp,
        average_precision_score=average_precision_score,
        roc_auc_score=roc_auc_score,
        tqdm=tqdm,
    )
    stages.update(1)

    _add_best_prauc_weight_strategies(
        best_by_strategy,
        weight_rows,
        val_labels=val_labels,
        test_labels=test_labels,
        val_selective=val_selective,
        test_selective=test_selective,
        val_ramp=val_ramp,
        test_ramp=test_ramp,
        average_precision_score=average_precision_score,
        roc_auc_score=roc_auc_score,
    )
    baselines = _baseline_metrics(
        strategies,
        val_labels=val_labels,
        test_labels=test_labels,
        val_selective=val_selective,
        test_selective=test_selective,
        val_ramp=val_ramp,
        test_ramp=test_ramp,
        average_precision_score=average_precision_score,
        roc_auc_score=roc_auc_score,
    )
    report = {
        "note": (
            "Ensemble probability = (1 - ramp_weight) * enhanced_selective "
            "+ ramp_weight * ramp_v2. Weights and thresholds are selected on "
            "validation only, then applied once to test."
        ),
        "config": str(config_path),
        "split": args.split,
        "view": args.view,
        "runs": {
            "enhanced_selective": str(selective_run),
            "ramp_v2": str(ramp_run),
            "ramp_model_name": ramp_model_name,
        },
        "samples": {
            "val": int(val_labels.size),
            "test": int(test_labels.size),
            "val_positive_rate": float(val_labels.mean()),
            "test_positive_rate": float(test_labels.mean()),
        },
        "weight_rows": weight_rows,
        "best_by_strategy": {
            name: None
            if value is None
            else {key: val for key, val in value.items() if key != "selection_key"}
            for name, value in best_by_strategy.items()
        },
        "baselines_from_current_predictions": baselines,
    }
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stages.update(1)
    stages.close()
    print(f"wrote {output_path}", flush=True)
    _print_ranked(report)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def ramp_model_name_for_run(run_dir: Path, *, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return "ramp-v2-rgcn"
    metrics = _read_json(metrics_path)
    if not isinstance(metrics, dict):
        return "ramp-v2-rgcn"
    metadata = metrics.get("run_metadata", {})
    if not isinstance(metadata, dict):
        return "ramp-v2-rgcn"
    model_name = metadata.get("model_name")
    if model_name in {"selective-fusion", "ramp-v2-rgcn"}:
        return str(model_name)
    return "ramp-v2-rgcn"


def _predict_model(
    name,
    model,
    checkpoint_path: Path,
    datasets,
    train_config,
    device,
    *,
    torch,
    loader_fn,
    predict_fn,
    tqdm,
) -> dict[str, object]:
    tqdm.write(f"loading {name}: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    result = {}
    for split_name in tqdm(
        ["val", "test"],
        desc=f"predict {name}",
        unit="split",
        ascii=True,
    ):
        loader, _ = loader_fn(datasets[split_name], config=train_config, shuffle=False)
        result[split_name] = predict_fn(model, loader, device)
    return result


def _align_prediction(
    reference_ids: list[str],
    prediction: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    by_id = {
        sample_id: (int(label), float(probability))
        for sample_id, label, probability in zip(
            prediction["sample_ids"],
            prediction["labels"],
            prediction["probabilities"],
        )
    }
    labels = np.asarray([by_id[sample_id][0] for sample_id in reference_ids], dtype=np.int64)
    probabilities = np.asarray(
        [by_id[sample_id][1] for sample_id in reference_ids],
        dtype=np.float32,
    )
    return labels, probabilities


def _strategies() -> list[dict[str, object]]:
    return [
        {"name": "unconstrained_val_f1", "objective": "f1"},
        {
            "name": "guarded_r90_p75_val_f1",
            "objective": "f1",
            "recall_max": 0.90,
            "positive_rate_max": 0.75,
        },
        {
            "name": "guarded_r85_p70_val_f1",
            "objective": "f1",
            "recall_max": 0.85,
            "positive_rate_max": 0.70,
        },
        {
            "name": "guarded_r80_p65_val_f1",
            "objective": "f1",
            "recall_max": 0.80,
            "positive_rate_max": 0.65,
        },
        {"name": "unconstrained_val_mcc", "objective": "mcc"},
        {
            "name": "guarded_r90_p75_val_mcc",
            "objective": "mcc",
            "recall_max": 0.90,
            "positive_rate_max": 0.75,
        },
        {
            "name": "guarded_r85_p70_val_mcc",
            "objective": "mcc",
            "recall_max": 0.85,
            "positive_rate_max": 0.70,
        },
    ]


def _scan_ensemble(
    weights: np.ndarray,
    strategies: list[dict[str, object]],
    *,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    val_selective: np.ndarray,
    test_selective: np.ndarray,
    val_ramp: np.ndarray,
    test_ramp: np.ndarray,
    average_precision_score,
    roc_auc_score,
    tqdm,
) -> tuple[dict[str, dict[str, object] | None], list[dict[str, object]]]:
    best_by_strategy: dict[str, dict[str, object] | None] = {
        str(strategy["name"]): None for strategy in strategies
    }
    weight_rows: list[dict[str, object]] = []
    for weight in tqdm(weights, desc="scan ensemble weights", unit="w", ascii=True):
        val_probabilities = ((1.0 - weight) * val_selective) + (weight * val_ramp)
        test_probabilities = ((1.0 - weight) * test_selective) + (weight * test_ramp)
        row = {
            "ramp_weight": float(weight),
            "selective_weight": float(1.0 - weight),
            "val_pr_auc": finite_score(
                average_precision_score,
                val_labels,
                val_probabilities,
            ),
            "test_pr_auc": finite_score(
                average_precision_score,
                test_labels,
                test_probabilities,
            ),
            "val_roc_auc": finite_score(roc_auc_score, val_labels, val_probabilities),
            "test_roc_auc": finite_score(roc_auc_score, test_labels, test_probabilities),
        }
        weight_rows.append(row)
        for strategy in strategies:
            name = str(strategy["name"])
            objective = str(strategy["objective"])
            constraints = _constraints(strategy)
            val_stats = scan_threshold(
                val_labels,
                val_probabilities,
                objective=objective,
                **constraints,
            )
            if val_stats is None:
                continue
            key = (
                float(val_stats[objective]),
                float(val_stats["mcc"]),
                float(val_stats["precision"]),
                float(val_stats["threshold"]),
            )
            previous = best_by_strategy[name]
            if previous is not None and key <= previous["selection_key"]:
                continue
            test_stats = threshold_stats(
                test_labels,
                test_probabilities,
                float(val_stats["threshold"]),
            )
            best_by_strategy[name] = {
                "selection_key": key,
                "ramp_weight": float(weight),
                "selective_weight": float(1.0 - weight),
                "threshold": float(val_stats["threshold"]),
                "constraints": constraints,
                "objective": objective,
                "validation": _attach_ranking_metrics(
                    val_stats,
                    val_labels,
                    val_probabilities,
                    average_precision_score=average_precision_score,
                    roc_auc_score=roc_auc_score,
                ),
                "test": _attach_ranking_metrics(
                    test_stats,
                    test_labels,
                    test_probabilities,
                    average_precision_score=average_precision_score,
                    roc_auc_score=roc_auc_score,
                ),
            }
    return best_by_strategy, weight_rows


def _add_best_prauc_weight_strategies(
    best_by_strategy: dict[str, dict[str, object] | None],
    weight_rows: list[dict[str, object]],
    *,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    val_selective: np.ndarray,
    test_selective: np.ndarray,
    val_ramp: np.ndarray,
    test_ramp: np.ndarray,
    average_precision_score,
    roc_auc_score,
) -> None:
    best_weight_by_val_prauc = max(
        weight_rows,
        key=lambda row: (
            row["val_pr_auc"] if row["val_pr_auc"] is not None else -1.0,
            row["ramp_weight"],
        ),
    )
    for recall_max, positive_rate_max in [(0.90, 0.75), (0.85, 0.70), (0.80, 0.65)]:
        weight = float(best_weight_by_val_prauc["ramp_weight"])
        val_probabilities = ((1.0 - weight) * val_selective) + (weight * val_ramp)
        test_probabilities = ((1.0 - weight) * test_selective) + (weight * test_ramp)
        val_stats = scan_threshold(
            val_labels,
            val_probabilities,
            objective="f1",
            recall_max=recall_max,
            positive_rate_max=positive_rate_max,
        )
        if val_stats is None:
            continue
        test_stats = threshold_stats(
            test_labels,
            test_probabilities,
            float(val_stats["threshold"]),
        )
        name = (
            "best_val_prauc_weight_guarded_"
            f"r{int(recall_max * 100)}_p{int(positive_rate_max * 100)}_val_f1"
        )
        best_by_strategy[name] = {
            "selection_key": (
                best_weight_by_val_prauc["val_pr_auc"],
                val_stats["f1"],
                val_stats["mcc"],
                val_stats["threshold"],
            ),
            "ramp_weight": weight,
            "selective_weight": float(1.0 - weight),
            "threshold": float(val_stats["threshold"]),
            "constraints": {
                "recall_max": recall_max,
                "positive_rate_max": positive_rate_max,
            },
            "objective": "val_pr_auc_then_guarded_f1_threshold",
            "validation": _attach_ranking_metrics(
                val_stats,
                val_labels,
                val_probabilities,
                average_precision_score=average_precision_score,
                roc_auc_score=roc_auc_score,
            ),
            "test": _attach_ranking_metrics(
                test_stats,
                test_labels,
                test_probabilities,
                average_precision_score=average_precision_score,
                roc_auc_score=roc_auc_score,
            ),
        }


def _baseline_metrics(
    strategies: list[dict[str, object]],
    *,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    val_selective: np.ndarray,
    test_selective: np.ndarray,
    val_ramp: np.ndarray,
    test_ramp: np.ndarray,
    average_precision_score,
    roc_auc_score,
) -> dict[str, dict[str, dict[str, object]]]:
    baselines: dict[str, dict[str, dict[str, object]]] = {}
    for name, val_probabilities, test_probabilities in [
        ("selective_only", val_selective, test_selective),
        ("ramp_v2_only", val_ramp, test_ramp),
    ]:
        baselines[name] = {}
        for strategy in strategies:
            objective = str(strategy["objective"])
            constraints = _constraints(strategy)
            val_stats = scan_threshold(
                val_labels,
                val_probabilities,
                objective=objective,
                **constraints,
            )
            if val_stats is None:
                continue
            test_stats = threshold_stats(
                test_labels,
                test_probabilities,
                float(val_stats["threshold"]),
            )
            baselines[name][str(strategy["name"])] = {
                "threshold": float(val_stats["threshold"]),
                "constraints": constraints,
                "objective": objective,
                "validation": _attach_ranking_metrics(
                    val_stats,
                    val_labels,
                    val_probabilities,
                    average_precision_score=average_precision_score,
                    roc_auc_score=roc_auc_score,
                ),
                "test": _attach_ranking_metrics(
                    test_stats,
                    test_labels,
                    test_probabilities,
                    average_precision_score=average_precision_score,
                    roc_auc_score=roc_auc_score,
                ),
            }
    return baselines


def _constraints(strategy: dict[str, object]) -> dict[str, float]:
    return {
        key: float(strategy[key])
        for key in ("recall_max", "positive_rate_max", "min_precision")
        if key in strategy
    }


def _attach_ranking_metrics(
    stats: dict[str, object],
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    average_precision_score,
    roc_auc_score,
) -> dict[str, object]:
    full = dict(stats)
    full["samples"] = int(labels.size)
    full["roc_auc"] = finite_score(roc_auc_score, labels, probabilities)
    full["pr_auc"] = finite_score(average_precision_score, labels, probabilities)
    return full


def _print_ranked(report: dict[str, object]) -> None:
    best_by_strategy = report["best_by_strategy"]
    print("top guarded strategies by test f1:", flush=True)
    guarded = [
        _ranking_row(name, value)
        for name, value in best_by_strategy.items()
        if value is not None and "guarded" in name
    ]
    for row in sorted(guarded, reverse=True)[:10]:
        _print_row(row)
    print("top all strategies by test f1:", flush=True)
    rows = [
        _ranking_row(name, value)
        for name, value in best_by_strategy.items()
        if value is not None
    ]
    for row in sorted(rows, reverse=True)[:10]:
        _print_row(row)


def _ranking_row(name: str, value: dict[str, object]) -> tuple:
    test = value["test"]
    return (
        test["f1"],
        test["mcc"],
        test["precision"],
        test["recall"],
        test["predicted_positive_rate"],
        name,
        value["ramp_weight"],
        value["threshold"],
    )


def _print_row(row: tuple) -> None:
    f1, mcc, precision, recall, positive_rate, name, weight, threshold = row
    print(
        f"{name}: "
        f"f1={f1:.4f} mcc={mcc:.4f} precision={precision:.4f} "
        f"recall={recall:.4f} ppr={positive_rate:.4f} "
        f"ramp_w={weight:.3f} thr={threshold:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
