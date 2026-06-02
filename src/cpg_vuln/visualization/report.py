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


def summarize_runs(runs_dir: Path, reports_dir: Path, *, topology_index: Path | None = None) -> pd.DataFrame:
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    run_dirs: list[Path] = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = metrics["test"]
        metadata = metrics.get("run_metadata", {})
        rows.append(
            {
                "run": metrics_path.parent.name,
                "feature_dim": metadata.get("feature_dim"),
                "accuracy": test["accuracy"],
                "precision": test["precision"],
                "recall": test["recall"],
                "f1": test["f1"],
                "roc_auc": test["roc_auc"],
                "pr_auc": test["pr_auc"],
                "elapsed_seconds": metrics["elapsed_seconds"],
            }
        )
        run_dirs.append(metrics_path.parent)
    frame = pd.DataFrame(rows)
    frame.to_csv(reports_dir / "summary.csv", index=False)
    if frame.empty:
        return frame
    _plot_metric_bars(frame, reports_dir / "metrics_bar.png")
    _plot_radar(frame, reports_dir / "radar.png")
    _plot_roc(run_dirs, reports_dir / "roc.png")
    _plot_confusion_matrices(run_dirs, reports_dir / "confusion_matrices.png")
    if topology_index is not None and topology_index.is_file():
        _plot_graph_sizes(topology_index, reports_dir / "graph_sizes.png")
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
    for run_dir in run_dirs:
        labels, probabilities = _read_predictions(run_dir / "predictions.csv")
        if len(set(labels)) < 2:
            continue
        false_positive, true_positive, _ = roc_curve(labels, probabilities)
        plt.plot(false_positive, true_positive, label=run_dir.name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_confusion_matrices(run_dirs: list[Path], path: Path) -> None:
    columns = min(3, len(run_dirs))
    rows = (len(run_dirs) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3.5), squeeze=False)
    for axis, run_dir in zip(axes.flat, run_dirs, strict=False):
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        sns.heatmap(metrics["test"]["confusion_matrix"], annot=True, fmt="d", cbar=False, ax=axis)
        axis.set_title(run_dir.name, fontsize=8)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    for axis in axes.flat[len(run_dirs) :]:
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
