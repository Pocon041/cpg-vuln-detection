from __future__ import annotations

import csv
import json
from pathlib import Path

from cpg_vuln.visualization.report import summarize_runs


def test_summarize_runs_writes_table_and_required_figures(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "baseline-ast-word2vec-course"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 1.2,
                "run_metadata": {"feature_dim": 128},
                "test": {
                    "accuracy": 0.75,
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": 0.74,
                    "roc_auc": 0.8,
                    "pr_auc": 0.81,
                    "confusion_matrix": [[4, 1], [2, 3]],
                },
            }
        ),
        encoding="utf-8",
    )
    with (run / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "probability", "prediction"])
        writer.writerows([["a", 0, 0.1, 0], ["b", 1, 0.8, 1]])

    summarize_runs(tmp_path / "runs", tmp_path / "reports")

    assert (tmp_path / "reports" / "summary.csv").is_file()
    assert (tmp_path / "reports" / "metrics_bar.png").is_file()
    assert (tmp_path / "reports" / "radar.png").is_file()
    assert (tmp_path / "reports" / "roc.png").is_file()
    assert (tmp_path / "reports" / "confusion_matrices.png").is_file()
    assert summarize_runs(tmp_path / "runs", tmp_path / "reports").iloc[0]["feature_dim"] == 128
