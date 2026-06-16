from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe fixed-threshold probability ensembles from existing predictions.csv "
            "files. The decision threshold is always 0.5."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Model input as NAME=RUN_DIR_OR_PREDICTIONS_CSV. Repeat for multiple models.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/fixed05_prediction_ensemble.json",
    )
    parser.add_argument("--grid-steps", type=int, default=10)
    parser.add_argument("--max-models", type=int, default=4)
    parser.add_argument("--max-ppr", type=float, default=0.80)
    parser.add_argument("--max-recall", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels, probabilities_by_model = load_prediction_inputs(args.run)
    rows = scan_fixed05_combinations(
        labels=labels,
        probabilities_by_model=probabilities_by_model,
        grid_steps=args.grid_steps,
        max_models=args.max_models,
        max_ppr=args.max_ppr,
        max_recall=args.max_recall,
    )
    report = {
        "note": "All metrics use a fixed probability threshold of 0.5.",
        "grid_steps": args.grid_steps,
        "max_models": args.max_models,
        "max_ppr": args.max_ppr,
        "max_recall": args.max_recall,
        "sample_count": int(labels.size),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "models": list(probabilities_by_model),
        "top_by_f1": rows[: args.top_k],
        "top_by_mcc": sorted(
            rows,
            key=lambda row: (row["metrics"]["mcc"], row["metrics"]["f1"]),
            reverse=True,
        )[: args.top_k],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path}", flush=True)
    print_ranked(report["top_by_f1"][:10], title="top fixed-0.5 valid ensembles by f1")
    print_ranked(report["top_by_mcc"][:10], title="top fixed-0.5 valid ensembles by mcc")


def load_prediction_inputs(
    run_specs: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    predictions_by_model = {
        name: read_predictions(path)
        for name, path in (parse_run_spec(spec) for spec in run_specs)
    }
    if not predictions_by_model:
        raise ValueError("at least one --run is required")
    common_ids = set.intersection(*(set(rows) for rows in predictions_by_model.values()))
    if not common_ids:
        raise ValueError("prediction files have no common sample_id values")
    sample_ids = sorted(common_ids)
    first_name = next(iter(predictions_by_model))
    labels = np.asarray(
        [predictions_by_model[first_name][sample_id][0] for sample_id in sample_ids],
        dtype=np.int64,
    )
    probabilities_by_model: dict[str, np.ndarray] = {}
    for name, predictions in predictions_by_model.items():
        current_labels = np.asarray(
            [predictions[sample_id][0] for sample_id in sample_ids],
            dtype=np.int64,
        )
        if not np.array_equal(labels, current_labels):
            raise ValueError(f"labels differ for model {name}")
        probabilities_by_model[name] = np.asarray(
            [predictions[sample_id][1] for sample_id in sample_ids],
            dtype=np.float32,
        )
    return labels, probabilities_by_model


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--run must use NAME=PATH format: {spec}")
    name, raw_path = spec.split("=", 1)
    if not name:
        raise ValueError(f"--run name cannot be empty: {spec}")
    path = Path(raw_path)
    if path.is_dir():
        path = path / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path


def read_predictions(path: Path) -> dict[str, tuple[int, float]]:
    rows: dict[str, tuple[int, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[str(row["sample_id"])] = (
                int(row["label"]),
                float(row["probability"]),
            )
    return rows


def scan_fixed05_combinations(
    *,
    labels: np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
    grid_steps: int,
    max_models: int,
    max_ppr: float,
    max_recall: float,
) -> list[dict[str, object]]:
    if grid_steps < 1:
        raise ValueError("grid_steps must be positive")
    if max_models < 1:
        raise ValueError("max_models must be positive")
    names = list(probabilities_by_model)
    rows: list[dict[str, object]] = []
    for model_count in range(1, min(max_models, len(names)) + 1):
        for selected in combinations(names, model_count):
            for weights in positive_simplex_weights(model_count=model_count, grid_steps=grid_steps):
                probabilities = weighted_average(
                    [probabilities_by_model[name] for name in selected],
                    weights,
                )
                metrics = threshold_stats(labels, probabilities, threshold=0.5)
                if metrics["predicted_positive_rate"] > max_ppr:
                    continue
                if metrics["recall"] > max_recall:
                    continue
                rows.append(
                    {
                        "model_weights": {
                            name: float(weight)
                            for name, weight in zip(selected, weights, strict=True)
                        },
                        "metrics": metrics,
                    }
                )
    return sorted(
        rows,
        key=lambda row: (row["metrics"]["f1"], row["metrics"]["mcc"]),
        reverse=True,
    )


def positive_simplex_weights(
    *,
    model_count: int,
    grid_steps: int,
) -> Iterable[tuple[float, ...]]:
    if model_count == 1:
        yield (1.0,)
        return
    yield from _positive_simplex_counts(model_count, grid_steps, ())


def _positive_simplex_counts(
    model_count: int,
    remaining: int,
    prefix: tuple[int, ...],
) -> Iterable[tuple[float, ...]]:
    if len(prefix) == model_count - 1:
        if remaining > 0:
            counts = (*prefix, remaining)
            total = sum(counts)
            yield tuple(count / total for count in counts)
        return
    slots_left = model_count - len(prefix)
    for count in range(1, remaining - slots_left + 2):
        yield from _positive_simplex_counts(model_count, remaining - count, (*prefix, count))


def weighted_average(
    probability_arrays: list[np.ndarray],
    weights: tuple[float, ...],
) -> np.ndarray:
    stacked = np.stack(probability_arrays, axis=0)
    return np.average(stacked, axis=0, weights=np.asarray(weights, dtype=np.float32))


def threshold_stats(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
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
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2),
        "mcc": float(mcc),
        "predicted_positive_rate": float(predictions.mean()) if predictions.size else 0.0,
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
    }


def print_ranked(rows: list[dict[str, object]], *, title: str) -> None:
    print(title, flush=True)
    for row in rows:
        metrics = row["metrics"]
        weights = ", ".join(
            f"{name}={weight:.2f}" for name, weight in row["model_weights"].items()
        )
        print(
            f"f1={metrics['f1']:.4f} mcc={metrics['mcc']:.4f} "
            f"p={metrics['precision']:.4f} r={metrics['recall']:.4f} "
            f"ppr={metrics['predicted_positive_rate']:.4f} weights=[{weights}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
