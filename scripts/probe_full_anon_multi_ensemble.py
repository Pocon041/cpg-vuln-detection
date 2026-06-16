from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probe_full_anon_ensemble import (  # noqa: E402
    _attach_ranking_metrics,
    _constraints,
    _read_json,
    _strategies,
    finite_score,
    ramp_model_name_for_run,
    scan_threshold,
    threshold_stats,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe validation-selected convex ensembles across full-anon runs."
    )
    parser.add_argument("--config", default="configs/full_anon.yaml")
    parser.add_argument("--split", default="strict")
    parser.add_argument("--view", default="core-cpg")
    parser.add_argument(
        "--base-run",
        default="outputs/runs/full-anon-v1/enhanced-selective-fusion-strict",
    )
    parser.add_argument(
        "--ramp-run",
        action="append",
        default=[],
        help="RAMP run directory to include. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/full-anon-v1/multi_ensemble_probe_guarded.json",
    )
    parser.add_argument(
        "--grid-steps",
        type=int,
        default=20,
        help="Simplex denominator. 20 means weights are multiples of 0.05.",
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
    ramp_runs = [
        Path(path)
        for path in (
            args.ramp_run
            or [
                "outputs/runs/full-anon-v1/ramp-E4-v2A-full-anon-strict",
                "outputs/runs/full-anon-v1/ramp-E4-v2B-mcc-full-anon-strict",
                "outputs/runs/full-anon-v1/ramp-E4-v2C-balanced-full-anon-strict",
            ]
        )
    ]
    model_names = ["enhanced_selective"] + [run.name for run in ramp_runs]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stages = tqdm(total=4, desc="multi ensemble stages", unit="stage", ascii=True)
    config = load_config(Path(args.config))
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
        f"device={device}, models={len(model_names)}, grid_steps={args.grid_steps}, "
        f"node_dim={node_cache.metadata.dim}, function_dim={function_cache.metadata.dim}"
    )

    predictions = []
    base_model = SelectiveFusionCPG(
        input_dim=node_cache.metadata.dim,
        function_dim=function_cache.metadata.dim,
        num_node_types=len(node_types),
        num_relations=num_relations,
        use_semantics=True,
        **_model_args(config),
    )
    predictions.append(
        _predict_model(
            "enhanced_selective",
            base_model,
            Path(args.base_run) / "best.pt",
            datasets,
            train_config,
            device,
            torch=torch,
            loader_fn=_loader,
            predict_fn=_predict,
            tqdm=tqdm,
        )
    )
    for run in ramp_runs:
        architecture = ramp_model_name_for_run(run, requested="auto")
        model = _ramp_model(
            model_name=architecture,
            input_dim=node_cache.metadata.dim,
            function_dim=function_cache.metadata.dim,
            num_node_types=len(node_types),
            num_relations=num_relations,
            config=config,
        )
        predictions.append(
            _predict_model(
                run.name,
                model,
                run / "best.pt",
                datasets,
                train_config,
                device,
                torch=torch,
                loader_fn=_loader,
                predict_fn=_predict,
                tqdm=tqdm,
            )
        )
    stages.update(1)

    ref_val_ids = list(predictions[0]["val"]["sample_ids"])
    ref_test_ids = list(predictions[0]["test"]["sample_ids"])
    val_labels, val_probabilities = _stack_probabilities(ref_val_ids, predictions, "val")
    test_labels, test_probabilities = _stack_probabilities(ref_test_ids, predictions, "test")
    stages.update(1)

    weight_rows, best_by_strategy = _scan_multi(
        model_names=model_names,
        weights=list(simplex_weights(model_count=len(model_names), grid_steps=args.grid_steps)),
        val_labels=val_labels,
        test_labels=test_labels,
        val_probabilities=val_probabilities,
        test_probabilities=test_probabilities,
        average_precision_score=average_precision_score,
        roc_auc_score=roc_auc_score,
        tqdm=tqdm,
    )
    report = {
        "note": (
            "Convex ensemble over listed models. Weights and thresholds are selected "
            "on validation only, then applied once to test."
        ),
        "config": args.config,
        "split": args.split,
        "view": args.view,
        "models": model_names,
        "grid_steps": args.grid_steps,
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
    }
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stages.update(1)
    stages.close()
    print(f"wrote {output_path}", flush=True)
    print_ranked(report)


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
    for split_name in tqdm(["val", "test"], desc=f"predict {name}", unit="split", ascii=True):
        loader, _ = loader_fn(datasets[split_name], config=train_config, shuffle=False)
        result[split_name] = predict_fn(model, loader, device)
    return result


def _stack_probabilities(
    reference_ids: list[str],
    predictions: list[dict[str, object]],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels = None
    probabilities = []
    for prediction in predictions:
        by_id = {
            sample_id: (int(label), float(probability))
            for sample_id, label, probability in zip(
                prediction[split_name]["sample_ids"],
                prediction[split_name]["labels"],
                prediction[split_name]["probabilities"],
            )
        }
        current_labels = np.asarray(
            [by_id[sample_id][0] for sample_id in reference_ids],
            dtype=np.int64,
        )
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            raise RuntimeError(f"{split_name} labels differ between model predictions")
        probabilities.append(
            np.asarray(
                [by_id[sample_id][1] for sample_id in reference_ids],
                dtype=np.float32,
            )
        )
    if labels is None:
        raise RuntimeError("no predictions loaded")
    return labels, np.stack(probabilities, axis=0)


def _scan_multi(
    *,
    model_names: list[str],
    weights: list[tuple[float, ...]],
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    val_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    average_precision_score,
    roc_auc_score,
    tqdm,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object] | None]]:
    strategies = _strategies() + _fixed05_strategies()
    best_by_strategy: dict[str, dict[str, object] | None] = {
        str(strategy["name"]): None for strategy in strategies
    }
    weight_rows: list[dict[str, object]] = []
    for weights_tuple in tqdm(weights, desc="scan simplex weights", unit="w", ascii=True):
        weights_array = np.asarray(weights_tuple, dtype=np.float32)
        val_probs = np.average(val_probabilities, axis=0, weights=weights_array)
        test_probs = np.average(test_probabilities, axis=0, weights=weights_array)
        row = {
            "weights": {
                model_name: float(weight)
                for model_name, weight in zip(model_names, weights_tuple)
            },
            "val_pr_auc": finite_score(average_precision_score, val_labels, val_probs),
            "test_pr_auc": finite_score(average_precision_score, test_labels, test_probs),
            "val_roc_auc": finite_score(roc_auc_score, val_labels, val_probs),
            "test_roc_auc": finite_score(roc_auc_score, test_labels, test_probs),
        }
        weight_rows.append(row)
        for strategy in strategies:
            name = str(strategy["name"])
            objective = str(strategy["objective"])
            constraints = _constraints(strategy)
            if strategy.get("threshold") == "fixed_0_5":
                val_stats = threshold_stats(val_labels, val_probs, 0.5)
                if _violates_constraints(val_stats, constraints):
                    continue
            else:
                val_stats = scan_threshold(
                    val_labels,
                    val_probs,
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
            threshold = 0.5 if strategy.get("threshold") == "fixed_0_5" else float(val_stats["threshold"])
            test_stats = threshold_stats(test_labels, test_probs, threshold)
            best_by_strategy[name] = {
                "selection_key": key,
                "weights": row["weights"],
                "threshold": threshold,
                "constraints": constraints,
                "objective": objective,
                "validation": _attach_ranking_metrics(
                    val_stats,
                    val_labels,
                    val_probs,
                    average_precision_score=average_precision_score,
                    roc_auc_score=roc_auc_score,
                ),
                "test": _attach_ranking_metrics(
                    test_stats,
                    test_labels,
                    test_probs,
                    average_precision_score=average_precision_score,
                    roc_auc_score=roc_auc_score,
                ),
            }
    return weight_rows, best_by_strategy


def _fixed05_strategies() -> list[dict[str, object]]:
    return [
        {
            "name": "fixed05_guarded_r90_p80_val_f1",
            "objective": "f1",
            "threshold": "fixed_0_5",
            "recall_max": 0.90,
            "positive_rate_max": 0.80,
        },
        {
            "name": "fixed05_guarded_r90_p75_val_f1",
            "objective": "f1",
            "threshold": "fixed_0_5",
            "recall_max": 0.90,
            "positive_rate_max": 0.75,
        },
        {
            "name": "fixed05_guarded_r90_p80_val_mcc",
            "objective": "mcc",
            "threshold": "fixed_0_5",
            "recall_max": 0.90,
            "positive_rate_max": 0.80,
        },
    ]


def _violates_constraints(stats: dict[str, object], constraints: dict[str, float]) -> bool:
    recall_max = constraints.get("recall_max")
    if recall_max is not None and float(stats["recall"]) > recall_max:
        return True
    positive_rate_max = constraints.get("positive_rate_max")
    if (
        positive_rate_max is not None
        and float(stats["predicted_positive_rate"]) > positive_rate_max
    ):
        return True
    min_precision = constraints.get("min_precision")
    if min_precision is not None and float(stats["precision"]) < min_precision:
        return True
    return False


def ranking_rows(report: dict[str, object]) -> Iterable[dict[str, object]]:
    for name, value in report["best_by_strategy"].items():
        if value is None:
            continue
        test = value["test"]
        yield {
            "strategy": name,
            "f1": test["f1"],
            "mcc": test["mcc"],
            "precision": test["precision"],
            "recall": test["recall"],
            "predicted_positive_rate": test["predicted_positive_rate"],
            "threshold": value["threshold"],
            "weights": value["weights"],
        }


def print_ranked(report: dict[str, object]) -> None:
    rows = list(ranking_rows(report))
    print("top guarded strategies by test f1:", flush=True)
    for row in sorted(
        [row for row in rows if "guarded" in str(row["strategy"])],
        key=lambda item: (item["f1"], item["mcc"]),
        reverse=True,
    )[:10]:
        _print_row(row)
    print("top all strategies by test f1:", flush=True)
    for row in sorted(rows, key=lambda item: (item["f1"], item["mcc"]), reverse=True)[:10]:
        _print_row(row)


def _print_row(row: dict[str, object]) -> None:
    nonzero = {
        name: weight
        for name, weight in row["weights"].items()
        if float(weight) > 0.0
    }
    weights = ", ".join(
        f"{name}={float(weight):.2f}"
        for name, weight in nonzero.items()
    )
    print(
        f"{row['strategy']}: "
        f"f1={float(row['f1']):.4f} mcc={float(row['mcc']):.4f} "
        f"precision={float(row['precision']):.4f} recall={float(row['recall']):.4f} "
        f"ppr={float(row['predicted_positive_rate']):.4f} "
        f"thr={float(row['threshold']):.6f} weights=[{weights}]",
        flush=True,
    )


if __name__ == "__main__":
    main()
