from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probe_full_anon_ensemble import (  # noqa: E402
    _attach_ranking_metrics,
    finite_score,
    ramp_model_name_for_run,
    threshold_stats,
)


class ModelSpec(NamedTuple):
    name: str
    config_path: Path
    run_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe validation-selected fixed-0.5 probability ensembles across "
            "runs that may use different configs/artifact layouts."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec as NAME=CONFIG_PATH=RUN_DIR. Repeat for multiple models.",
    )
    parser.add_argument("--split", default="strict")
    parser.add_argument(
        "--view",
        default="auto",
        help="Use run_metadata.view by default; pass a view name to override all models.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/cross_norm_fixed05_ensemble.json",
    )
    parser.add_argument("--grid-steps", type=int, default=10)
    parser.add_argument("--max-ppr", type=float, default=0.80)
    parser.add_argument("--max-recall", type=float, default=0.90)
    parser.add_argument(
        "--no-logistic-stack",
        action="store_true",
        help="Disable validation-CV logistic stacking candidates.",
    )
    return parser.parse_args()


def main() -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from tqdm import tqdm

    import torch

    args = parse_args()
    specs = [parse_model_spec(value) for value in args.model]
    if len(specs) < 2:
        raise ValueError("at least two --model specs are required")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stages = tqdm(total=4, desc="cross-norm ensemble stages", unit="stage", ascii=True)
    predictions = []
    metadata_by_model: dict[str, dict[str, object]] = {}
    for spec in tqdm(specs, desc="predict models", unit="model", ascii=True):
        prediction, metadata = predict_model_for_spec(
            spec,
            split=args.split,
            view_override=args.view,
            torch=torch,
            tqdm=tqdm,
        )
        predictions.append({"name": spec.name, **prediction})
        metadata_by_model[spec.name] = metadata
    stages.update(1)

    val_ids, val_labels, val_probabilities = stack_model_predictions(predictions, "val")
    test_ids, test_labels, test_probabilities = stack_model_predictions(predictions, "test")
    stages.update(1)

    weights = list(simplex_weights(model_count=len(specs), grid_steps=args.grid_steps))
    best_by_strategy = {
        "fixed05_guarded_val_f1": select_fixed05_weights(
            model_names=[spec.name for spec in specs],
            weights=weights,
            val_labels=val_labels,
            test_labels=test_labels,
            val_probabilities=val_probabilities,
            test_probabilities=test_probabilities,
            objective="f1",
            max_ppr=args.max_ppr,
            max_recall=args.max_recall,
            average_precision_score=average_precision_score,
            roc_auc_score=roc_auc_score,
        ),
        "fixed05_guarded_val_mcc": select_fixed05_weights(
            model_names=[spec.name for spec in specs],
            weights=weights,
            val_labels=val_labels,
            test_labels=test_labels,
            val_probabilities=val_probabilities,
            test_probabilities=test_probabilities,
            objective="mcc",
            max_ppr=args.max_ppr,
            max_recall=args.max_recall,
            average_precision_score=average_precision_score,
            roc_auc_score=roc_auc_score,
        ),
    }
    if not args.no_logistic_stack:
        best_by_strategy["fixed05_stack_cv_val_f1"] = select_fixed05_logistic_stack(
            model_names=[spec.name for spec in specs],
            val_labels=val_labels,
            test_labels=test_labels,
            val_probabilities=val_probabilities,
            test_probabilities=test_probabilities,
            objective="f1",
            max_ppr=args.max_ppr,
            max_recall=args.max_recall,
            average_precision_score=average_precision_score,
            roc_auc_score=roc_auc_score,
        )
        best_by_strategy["fixed05_stack_cv_val_mcc"] = select_fixed05_logistic_stack(
            model_names=[spec.name for spec in specs],
            val_labels=val_labels,
            test_labels=test_labels,
            val_probabilities=val_probabilities,
            test_probabilities=test_probabilities,
            objective="mcc",
            max_ppr=args.max_ppr,
            max_recall=args.max_recall,
            average_precision_score=average_precision_score,
            roc_auc_score=roc_auc_score,
        )
    stages.update(1)

    report = {
        "note": (
            "All model weights are selected on validation only. Test metrics are "
            "computed once with the selected weights and a fixed threshold of 0.5."
        ),
        "split": args.split,
        "view": args.view,
        "grid_steps": args.grid_steps,
        "max_ppr": args.max_ppr,
        "max_recall": args.max_recall,
        "models": [
            {
                "name": spec.name,
                "config": str(spec.config_path),
                "run_dir": str(spec.run_dir),
                "metadata": metadata_by_model.get(spec.name, {}),
            }
            for spec in specs
        ],
        "samples": {
            "val": int(val_labels.size),
            "test": int(test_labels.size),
            "val_positive_rate": float(val_labels.mean()) if val_labels.size else 0.0,
            "test_positive_rate": float(test_labels.mean()) if test_labels.size else 0.0,
            "val_first_sample_id": val_ids[0] if val_ids else None,
            "test_first_sample_id": test_ids[0] if test_ids else None,
        },
        "best_by_strategy": best_by_strategy,
    }
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stages.update(1)
    stages.close()
    print(f"wrote {output_path}", flush=True)
    print_ranked(report)


def parse_model_spec(spec: str) -> ModelSpec:
    parts = spec.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"--model must use NAME=CONFIG_PATH=RUN_DIR format: {spec}")
    name, config_path, run_dir = parts
    return ModelSpec(name=name, config_path=Path(config_path), run_dir=Path(run_dir))


def simplex_weights(
    *,
    model_count: int,
    grid_steps: int,
) -> Iterable[tuple[float, ...]]:
    if model_count < 2:
        raise ValueError("model_count must be at least 2")
    if grid_steps < 1:
        raise ValueError("grid_steps must be positive")
    yield from _simplex_counts(model_count, grid_steps, ())


def _simplex_counts(
    model_count: int,
    remaining: int,
    prefix: tuple[int, ...],
) -> Iterable[tuple[float, ...]]:
    if len(prefix) == model_count - 1:
        counts = (*prefix, remaining)
        total = sum(counts)
        yield tuple(count / total for count in counts)
        return
    for count in range(remaining + 1):
        yield from _simplex_counts(model_count, remaining - count, (*prefix, count))


def select_fixed05_weights(
    *,
    model_names: list[str],
    weights: Iterable[tuple[float, ...]],
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    val_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    objective: str,
    max_ppr: float,
    max_recall: float,
    average_precision_score=None,
    roc_auc_score=None,
) -> dict[str, object] | None:
    if objective not in {"f1", "mcc", "balanced_accuracy"}:
        raise ValueError(f"unsupported objective: {objective}")
    best: dict[str, object] | None = None
    best_key: tuple[float, ...] | None = None
    for weights_tuple in weights:
        weights_array = np.asarray(weights_tuple, dtype=np.float32)
        val_probs = np.average(val_probabilities, axis=0, weights=weights_array)
        val_metrics = threshold_stats(val_labels, val_probs, 0.5)
        if float(val_metrics["predicted_positive_rate"]) > max_ppr:
            continue
        if float(val_metrics["recall"]) > max_recall:
            continue
        key = _selection_key(val_metrics, objective)
        if best_key is not None and key <= best_key:
            continue
        test_probs = np.average(test_probabilities, axis=0, weights=weights_array)
        test_metrics = threshold_stats(test_labels, test_probs, 0.5)
        if average_precision_score is not None and roc_auc_score is not None:
            val_metrics = _attach_ranking_metrics(
                val_metrics,
                val_labels,
                val_probs,
                average_precision_score=average_precision_score,
                roc_auc_score=roc_auc_score,
            )
            test_metrics = _attach_ranking_metrics(
                test_metrics,
                test_labels,
                test_probs,
                average_precision_score=average_precision_score,
                roc_auc_score=roc_auc_score,
            )
        best_key = key
        best = {
            "objective": objective,
            "threshold": 0.5,
            "constraints": {
                "max_ppr": float(max_ppr),
                "max_recall": float(max_recall),
            },
            "weights": {
                name: float(weight)
                for name, weight in zip(model_names, weights_tuple, strict=True)
            },
            "validation": val_metrics,
            "test": test_metrics,
        }
    return best


def select_fixed05_logistic_stack(
    *,
    model_names: list[str],
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    val_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    objective: str,
    max_ppr: float,
    max_recall: float,
    c_values: list[float] | None = None,
    class_weights: list[str] | None = None,
    feature_spaces: list[str] | None = None,
    cv_splits: int = 5,
    average_precision_score=None,
    roc_auc_score=None,
) -> dict[str, object] | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    if objective not in {"f1", "mcc", "balanced_accuracy"}:
        raise ValueError(f"unsupported objective: {objective}")
    if len(np.unique(val_labels)) < 2:
        return None
    class_counts = np.bincount(val_labels.astype(np.int64), minlength=2)
    effective_splits = min(int(cv_splits), int(class_counts[class_counts > 0].min()))
    if effective_splits < 2:
        return None
    c_values = c_values or [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
    class_weights = class_weights or ["none", "balanced"]
    feature_spaces = feature_spaces or ["probability", "logit", "probability_logit"]
    splitter = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=42,
    )
    best: dict[str, object] | None = None
    best_key: tuple[float, ...] | None = None
    for feature_space in feature_spaces:
        val_features = probability_feature_matrix(
            val_probabilities,
            feature_space=feature_space,
        )
        test_features = probability_feature_matrix(
            test_probabilities,
            feature_space=feature_space,
        )
        for c_value in c_values:
            for class_weight_name in class_weights:
                class_weight = None if class_weight_name == "none" else class_weight_name
                oof_probabilities = np.zeros(val_labels.shape[0], dtype=np.float32)
                for train_indices, holdout_indices in splitter.split(val_features, val_labels):
                    estimator = LogisticRegression(
                        C=float(c_value),
                        class_weight=class_weight,
                        max_iter=1000,
                        random_state=42,
                        solver="liblinear",
                    )
                    estimator.fit(val_features[train_indices], val_labels[train_indices])
                    oof_probabilities[holdout_indices] = estimator.predict_proba(
                        val_features[holdout_indices]
                    )[:, 1]
                val_metrics = threshold_stats(val_labels, oof_probabilities, 0.5)
                if float(val_metrics["predicted_positive_rate"]) > max_ppr:
                    continue
                if float(val_metrics["recall"]) > max_recall:
                    continue
                key = _selection_key(val_metrics, objective)
                if best_key is not None and key <= best_key:
                    continue
                final_estimator = LogisticRegression(
                    C=float(c_value),
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=42,
                    solver="liblinear",
                )
                final_estimator.fit(val_features, val_labels)
                test_stack_probabilities = final_estimator.predict_proba(test_features)[:, 1]
                test_metrics = threshold_stats(test_labels, test_stack_probabilities, 0.5)
                if average_precision_score is not None and roc_auc_score is not None:
                    val_metrics = _attach_ranking_metrics(
                        val_metrics,
                        val_labels,
                        oof_probabilities,
                        average_precision_score=average_precision_score,
                        roc_auc_score=roc_auc_score,
                    )
                    test_metrics = _attach_ranking_metrics(
                        test_metrics,
                        test_labels,
                        test_stack_probabilities,
                        average_precision_score=average_precision_score,
                        roc_auc_score=roc_auc_score,
                    )
                best_key = key
                best = {
                    "objective": objective,
                    "threshold": 0.5,
                    "constraints": {
                        "max_ppr": float(max_ppr),
                        "max_recall": float(max_recall),
                    },
                    "weights": {
                        name: 1.0 for name in model_names
                    },
                    "stacker": {
                        "type": "logistic_regression_cv",
                        "feature_space": feature_space,
                        "c": float(c_value),
                        "class_weight": class_weight_name,
                        "cv_splits": effective_splits,
                        "model_names": model_names,
                        "intercept": [
                            float(value) for value in final_estimator.intercept_.tolist()
                        ],
                        "coef": [
                            [float(value) for value in row]
                            for row in final_estimator.coef_.tolist()
                        ],
                    },
                    "validation": val_metrics,
                    "test": test_metrics,
                }
    return best


def probability_feature_matrix(
    probabilities: np.ndarray,
    *,
    feature_space: str,
) -> np.ndarray:
    probability_features = np.asarray(probabilities, dtype=np.float32).T
    if feature_space == "probability":
        return probability_features
    clipped = np.clip(probability_features, 1e-6, 1.0 - 1e-6)
    logit_features = np.log(clipped / (1.0 - clipped))
    if feature_space == "logit":
        return logit_features
    if feature_space == "probability_logit":
        return np.concatenate([probability_features, logit_features], axis=1)
    raise ValueError(f"unsupported feature_space: {feature_space}")


def _selection_key(stats: dict[str, object], objective: str) -> tuple[float, ...]:
    if objective == "f1":
        return (
            float(stats["f1"]),
            float(stats["mcc"]),
            float(stats["precision"]),
            -float(stats["predicted_positive_rate"]),
        )
    if objective == "mcc":
        return (
            float(stats["mcc"]),
            float(stats["f1"]),
            float(stats["precision"]),
            -float(stats["predicted_positive_rate"]),
        )
    return (
        float(stats["balanced_accuracy"]),
        float(stats["mcc"]),
        float(stats["f1"]),
        -float(stats["predicted_positive_rate"]),
    )


def stack_model_predictions(
    predictions: list[dict[str, object]],
    split_name: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not predictions:
        raise ValueError("no predictions provided")
    first = predictions[0]
    reference_ids = [str(value) for value in first[split_name]["sample_ids"]]
    reference_set = set(reference_ids)
    probabilities = []
    labels: np.ndarray | None = None
    for prediction in predictions:
        current_ids = [str(value) for value in prediction[split_name]["sample_ids"]]
        current_set = set(current_ids)
        if current_set != reference_set:
            name = prediction.get("name", "<unknown>")
            missing = sorted(reference_set - current_set)[:5]
            extra = sorted(current_set - reference_set)[:5]
            raise ValueError(
                f"{split_name} sample ids differ for {name}; "
                f"missing={missing}, extra={extra}"
            )
        by_id = {
            str(sample_id): (int(label), float(probability))
            for sample_id, label, probability in zip(
                prediction[split_name]["sample_ids"],
                prediction[split_name]["labels"],
                prediction[split_name]["probabilities"],
                strict=True,
            )
        }
        current_labels = np.asarray(
            [by_id[sample_id][0] for sample_id in reference_ids],
            dtype=np.int64,
        )
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            name = prediction.get("name", "<unknown>")
            raise ValueError(f"{split_name} labels differ for {name}")
        probabilities.append(
            np.asarray(
                [by_id[sample_id][1] for sample_id in reference_ids],
                dtype=np.float32,
            )
        )
    if labels is None:
        raise ValueError("no labels provided")
    return reference_ids, labels, np.stack(probabilities, axis=0)


def predict_model_for_spec(
    spec: ModelSpec,
    *,
    split: str,
    view_override: str,
    torch,
    tqdm,
) -> tuple[dict[str, object], dict[str, object]]:
    from cpg_vuln.config import load_config
    from cpg_vuln.data.store import NodeTypeRegistry, load_topology
    from cpg_vuln.features.cache import MemmapFeatureCache
    from cpg_vuln.training.engine import _loader, _predict
    from cpg_vuln.training.runner import (
        _datasets,
        _layout,
        _node_feature_cache,
        _ramp_model,
        _topology_index,
        _train_config,
    )

    config = load_config(spec.config_path)
    layout = _layout(config)
    metadata = _run_metadata(spec.run_dir)
    view = str(metadata.get("view", "core-cpg")) if view_override == "auto" else view_override
    model_name = _model_name_for_run(spec.run_dir, metadata)

    index = _topology_index(layout)
    split_payload = _read_json(
        Path(config["paths"]["artifacts_dir"]) / "data" / "splits" / f"{split}.json"
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
        view,
        node_cache,
        function_cache=function_cache,
        function_indices=function_indices,
    )
    sample_payload = load_topology(datasets["train"].topology_paths[0])
    train_config = _train_config(config, epochs=None, learning_rate=1e-4)
    device = torch.device(
        train_config.device
        if train_config.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    model = _ramp_model(
        model_name=model_name,
        input_dim=node_cache.metadata.dim,
        function_dim=function_cache.metadata.dim,
        num_node_types=len(node_types),
        num_relations=len(sample_payload["edge_type_names"]),
        config=config,
    )
    checkpoint_path = spec.run_dir / "best.pt"
    tqdm.write(
        f"loading {spec.name}: config={spec.config_path}, view={view}, "
        f"model={model_name}, checkpoint={checkpoint_path}"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    result = {}
    for split_name in tqdm(["val", "test"], desc=f"predict {spec.name}", unit="split", ascii=True):
        loader, _ = _loader(datasets[split_name], config=train_config, shuffle=False)
        result[split_name] = _predict(model, loader, device)
    return result, {
        **metadata,
        "resolved_view": view,
        "resolved_model_name": model_name,
        "normalization_key": layout.spec.normalization_key,
        "normalization_mode": layout.spec.mode,
    }


def _run_metadata(run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return {}
    metrics = _read_json(metrics_path)
    if not isinstance(metrics, dict):
        return {}
    metadata = metrics.get("run_metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _model_name_for_run(run_dir: Path, metadata: dict[str, object]) -> str:
    model_name = metadata.get("model_name")
    if model_name in {"selective-fusion", "ramp-v2-rgcn", "ramp-v2-dual"}:
        return str(model_name)
    if metadata.get("kind") == "enhanced":
        return "selective-fusion"
    return ramp_model_name_for_run(run_dir, requested="auto")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def print_ranked(report: dict[str, object]) -> None:
    print("validation-selected fixed-0.5 strategies:", flush=True)
    for name, value in report["best_by_strategy"].items():
        if value is None:
            print(f"{name}: no validation-valid weights", flush=True)
            continue
        test = value["test"]
        validation = value["validation"]
        nonzero = {
            model_name: weight
            for model_name, weight in value["weights"].items()
            if float(weight) > 0.0
        }
        weights = ", ".join(
            f"{model_name}={float(weight):.2f}"
            for model_name, weight in nonzero.items()
        )
        print(
            f"{name}: "
            f"val_f1={float(validation['f1']):.4f} val_mcc={float(validation['mcc']):.4f} "
            f"test_f1={float(test['f1']):.4f} test_mcc={float(test['mcc']):.4f} "
            f"p={float(test['precision']):.4f} r={float(test['recall']):.4f} "
            f"ppr={float(test['predicted_positive_rate']):.4f} "
            f"thr={float(value['threshold']):.1f} weights=[{weights}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
