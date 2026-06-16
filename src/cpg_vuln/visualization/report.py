from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
from tqdm import tqdm


def summarize_runs(runs_dir: Path, reports_dir: Path, *, topology_index: Path | None = None) -> pd.DataFrame:
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    run_dirs: list[Path] = []
    metrics_paths = sorted(runs_dir.glob("*/metrics.json"))
    for metrics_path in tqdm(
        metrics_paths,
        desc="read run metrics",
        unit="run",
        total=len(metrics_paths),
        ascii=True,
    ):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata = metrics.get("run_metadata", {})
        final_test = _read_optional_json(metrics_path.parent / "final_test_metrics.json")
        validation_fixed = metrics.get("validation_fixed_0_5")
        validation_val_f1 = metrics.get("validation_val_f1") or metrics.get("validation")
        validation_val_mcc = metrics.get("validation_val_mcc")
        test_fixed = (
            final_test.get("test_fixed_0_5")
            if isinstance(final_test, dict)
            else metrics.get("test_fixed_0_5")
        )
        test_val_f1 = (
            final_test.get("test_val_f1")
            if isinstance(final_test, dict)
            else metrics.get("test_val_f1") or metrics.get("test")
        )
        test_val_mcc = (
            final_test.get("test_val_mcc")
            if isinstance(final_test, dict)
            else metrics.get("test_val_mcc")
        )
        selected = test_val_f1 if isinstance(test_val_f1, dict) else validation_val_f1
        rows.append(
            {
                "run": metrics_path.parent.name,
                "feature_dim": metadata.get("feature_dim"),
                "view": metadata.get("view"),
                "model_name": metadata.get("model_name"),
                "normalization_key": metadata.get("normalization_key"),
                "normalization_mode": metadata.get("normalization_mode"),
                "function_source_normalization": metadata.get("function_source_normalization"),
                "evaluation_mode": metrics.get("evaluation_mode", "final_test"),
                "accuracy": _metric_value(selected, "accuracy"),
                "precision": _metric_value(selected, "precision"),
                "recall": _metric_value(selected, "recall"),
                "f1": _metric_value(selected, "f1"),
                "roc_auc": _metric_value(selected, "roc_auc"),
                "pr_auc": _metric_value(selected, "pr_auc"),
                "elapsed_seconds": metrics["elapsed_seconds"],
                **_flatten_threshold_metrics(
                    prefix="val",
                    fixed=validation_fixed if isinstance(validation_fixed, dict) else None,
                    val_f1=validation_val_f1 if isinstance(validation_val_f1, dict) else None,
                    val_mcc=validation_val_mcc if isinstance(validation_val_mcc, dict) else None,
                ),
                **_flatten_threshold_metrics(
                    prefix="test",
                    fixed=test_fixed if isinstance(test_fixed, dict) else None,
                    val_f1=test_val_f1 if isinstance(test_val_f1, dict) else None,
                    val_mcc=test_val_mcc if isinstance(test_val_mcc, dict) else None,
                ),
            }
        )
        run_dirs.append(metrics_path.parent)
    frame = pd.DataFrame(rows)
    frame.to_csv(reports_dir / "summary.csv", index=False)
    if frame.empty:
        return frame
    figure_tasks = [
        ("metric bars", lambda: _plot_metric_bars(frame, reports_dir / "metrics_bar.png")),
        ("radar", lambda: _plot_radar(frame, reports_dir / "radar.png")),
        ("roc", lambda: _plot_roc(run_dirs, reports_dir / "roc.png")),
        (
            "confusion matrices",
            lambda: _plot_confusion_matrices(run_dirs, reports_dir / "confusion_matrices.png"),
        ),
    ]
    if topology_index is not None and topology_index.is_file():
        figure_tasks.append(
            ("graph sizes", lambda: _plot_graph_sizes(topology_index, reports_dir / "graph_sizes.png"))
        )
    for _, write_figure in tqdm(
        figure_tasks,
        desc="write report figures",
        unit="figure",
        total=len(figure_tasks),
        ascii=True,
    ):
        write_figure()
    return frame


def _plot_metric_bars(frame: pd.DataFrame, path: Path) -> None:
    melted = frame.melt(id_vars="run", value_vars=["accuracy", "precision", "recall", "f1"])
    plt.figure(figsize=(max(8, len(frame) * 0.7), 5))
    sns.barplot(data=melted, x="run", y="value", hue="variable")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_radar(frame: pd.DataFrame, path: Path) -> None:
    metrics = ["precision", "recall", "f1"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    figure, axis = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for _, row in frame.iterrows():
        values = [float(row[name]) for name in metrics]
        axis.plot(angles, values + values[:1], label=row["run"])
    axis.set_xticks(angles[:-1], metrics)
    axis.set_ylim(0, 1)
    axis.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_roc(run_dirs: list[Path], path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plotted = False
    for run_dir in run_dirs:
        predictions_path = run_dir / "predictions.csv"
        if not predictions_path.is_file():
            continue
        labels, probabilities = _read_predictions(predictions_path)
        if len(set(labels)) < 2:
            continue
        false_positive, true_positive, _ = roc_curve(labels, probabilities)
        plt.plot(false_positive, true_positive, label=run_dir.name)
        plotted = True
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    if plotted:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_confusion_matrices(run_dirs: list[Path], path: Path) -> None:
    plot_items: list[tuple[Path, list[list[int]]]] = []
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        final_test = _read_optional_json(run_dir / "final_test_metrics.json")
        section = (
            final_test.get("test_val_f1")
            if isinstance(final_test, dict)
            else metrics.get("test_val_f1") or metrics.get("test")
        )
        if isinstance(section, dict) and isinstance(section.get("confusion_matrix"), list):
            plot_items.append((run_dir, section["confusion_matrix"]))
    if not plot_items:
        plt.figure(figsize=(4, 3))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        return
    columns = min(3, len(plot_items))
    rows = (len(plot_items) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3.5), squeeze=False)
    for axis, (run_dir, matrix) in zip(axes.flat, plot_items, strict=False):
        sns.heatmap(matrix, annot=True, fmt="d", cbar=False, ax=axis)
        axis.set_title(run_dir.name, fontsize=8)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    for axis in axes.flat[len(plot_items) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_graph_sizes(index_path: Path, path: Path) -> None:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(item for item in index if item["view"] == "core-cpg")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(frame["nodes"], bins=40, ax=axes[0])
    sns.histplot(frame["edges"], bins=40, ax=axes[1])
    axes[0].set_title("Nodes per graph")
    axes[1].set_title("Edges per graph")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _read_predictions(path: Path) -> tuple[list[int], list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [int(row["label"]) for row in rows], [float(row["probability"]) for row in rows]


def _read_optional_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(section: dict[str, object] | None, key: str) -> object:
    return None if section is None else section.get(key)


def _flatten_threshold_metrics(
    *,
    prefix: str,
    fixed: dict[str, object] | None,
    val_f1: dict[str, object] | None,
    val_mcc: dict[str, object] | None,
) -> dict[str, object]:
    return {
        f"{prefix}_f1_at_val_f1": _metric_value(val_f1, "f1"),
        f"{prefix}_precision_at_val_f1": _metric_value(val_f1, "precision"),
        f"{prefix}_recall_at_val_f1": _metric_value(val_f1, "recall"),
        f"{prefix}_positive_rate_at_val_f1": _metric_value(
            val_f1,
            "predicted_positive_rate",
        ),
        f"{prefix}_pr_auc": _metric_value(val_f1, "pr_auc"),
        f"{prefix}_roc_auc": _metric_value(val_f1, "roc_auc"),
        f"{prefix}_mcc_at_val_mcc": _metric_value(val_mcc, "mcc"),
        f"{prefix}_f1_at_fixed_0_5": _metric_value(fixed, "f1"),
    }
