from __future__ import annotations

import csv
import json
from pathlib import Path

from cpg_vuln.visualization.explain import export_attention_dashboard
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


def test_summarize_runs_flattens_validation_and_final_test_metrics(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "ramp-E4-v2A-fast-strict"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "evaluation_mode": "validation_only",
                "elapsed_seconds": 1.0,
                "run_metadata": {"model_name": "ramp-v2-rgcn", "view": "core-cpg"},
                "validation_fixed_0_5": {"f1": 0.58},
                "validation_val_f1": {
                    "accuracy": 0.6,
                    "precision": 0.6,
                    "recall": 0.7,
                    "f1": 0.65,
                    "specificity": 0.5,
                    "mcc": 0.2,
                    "balanced_accuracy": 0.6,
                    "roc_auc": 0.61,
                    "pr_auc": 0.60,
                    "predicted_positive_rate": 0.7,
                    "threshold": 0.4,
                },
                "validation_val_mcc": {"mcc": 0.18},
                "test_fixed_0_5": None,
                "test_val_f1": None,
                "test_val_mcc": None,
                "test": None,
            }
        ),
        encoding="utf-8",
    )
    (run / "final_test_metrics.json").write_text(
        json.dumps(
            {
                "test_fixed_0_5": {"f1": 0.54},
                "test_val_f1": {
                    "precision": 0.55,
                    "recall": 0.62,
                    "f1": 0.58,
                    "roc_auc": 0.57,
                    "pr_auc": 0.56,
                    "predicted_positive_rate": 0.65,
                },
                "test_val_mcc": {"mcc": 0.12},
            }
        ),
        encoding="utf-8",
    )

    frame = summarize_runs(tmp_path / "runs", tmp_path / "reports")

    assert frame.loc[0, "evaluation_mode"] == "validation_only"
    assert frame.loc[0, "view"] == "core-cpg"
    assert frame.loc[0, "model_name"] == "ramp-v2-rgcn"
    assert frame.loc[0, "val_f1_at_val_f1"] == 0.65
    assert frame.loc[0, "val_mcc_at_val_mcc"] == 0.18
    assert frame.loc[0, "test_f1_at_val_f1"] == 0.58
    assert frame.loc[0, "test_mcc_at_val_mcc"] == 0.12


def test_summarize_runs_reports_metrics_and_figure_progress(tmp_path: Path, monkeypatch) -> None:
    import cpg_vuln.visualization.report as report

    calls: list[dict[str, object]] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    run = tmp_path / "runs" / "run-1"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 1.0,
                "run_metadata": {},
                "test": {
                    "accuracy": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "roc_auc": 1.0,
                    "pr_auc": 1.0,
                    "confusion_matrix": [[1, 0], [0, 1]],
                },
            }
        ),
        encoding="utf-8",
    )
    with (run / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "probability", "prediction"])
        writer.writerows([["a", 0, 0.1, 0], ["b", 1, 0.8, 1]])
    monkeypatch.setattr(report, "tqdm", fake_tqdm)

    report.summarize_runs(tmp_path / "runs", tmp_path / "reports")

    assert "read run metrics" in [call["desc"] for call in calls]
    assert "write report figures" in [call["desc"] for call in calls]


def test_summarize_runs_skips_roc_for_runs_without_predictions(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "ramp-E0-strict"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 1.0,
                "run_metadata": {"training_mode": "ramp"},
                "test": {
                    "accuracy": 0.5,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "roc_auc": 0.5,
                    "pr_auc": 0.5,
                    "confusion_matrix": [[1, 0], [1, 0]],
                },
            }
        ),
        encoding="utf-8",
    )

    summarize_runs(tmp_path / "runs", tmp_path / "reports")

    assert (tmp_path / "reports" / "summary.csv").is_file()
    assert (tmp_path / "reports" / "roc.png").is_file()


def test_export_attention_dashboard_writes_source_heatmap(tmp_path: Path) -> None:
    source = tmp_path / "danger.c"
    source.write_text(
        "\n".join(
            [
                "int f(char *dst, char *src) {",
                "    strcpy(dst, src);",
                "    dst[3] = 0;",
                "    return 0;",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    attention = tmp_path / "node_attention.json"
    attention.write_text(
        json.dumps(
            {
                "sample-1": [
                    {"line": 65, "score": 0.2},
                    {"line": 66, "score": 0.9},
                    {"line": 67, "score": 0.6},
                ],
                "sample-2": [{"line": 66, "score": 0.1}],
            }
        ),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "probability", "prediction"])
        writer.writerows([["sample-1", 1, 0.91, 1], ["sample-2", 0, 0.12, 0]])
    source_map = tmp_path / "source_map.csv"
    with source_map.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "raw_source_path",
                "prepared_source_path",
                "line_offset",
                "offset_source",
                "mapping_status",
                "notes",
            ]
        )
        writer.writerow(["sample-1", source, "", 64, "default", "validated_default", ""])
        writer.writerow(["sample-2", source, "", 64, "default", "validated_default", ""])

    result = export_attention_dashboard(
        attention,
        predictions,
        source_map,
        tmp_path / "dashboard",
        run_name="ramp-E4-strict",
        top_samples=1,
        top_lines=2,
        context_radius=1,
    )

    assert result["selected_samples"] == ["sample-1"]
    assert (tmp_path / "dashboard" / "index.html").is_file()
    assert (tmp_path / "dashboard" / "samples" / "sample-1.html").is_file()
    assert (tmp_path / "dashboard" / "line_attention_top.csv").is_file()
    assert (tmp_path / "dashboard" / "attention_overview.png").is_file()
    html = (tmp_path / "dashboard" / "samples" / "sample-1.html").read_text(encoding="utf-8")
    assert "strcpy(dst, src);" in html
    assert "attention-bar" in html
