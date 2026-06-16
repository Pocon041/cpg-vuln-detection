from __future__ import annotations

import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def export_top_k_explanations(attention_path: Path, output_dir: Path, *, top_k: int = 10) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    attention = json.loads(attention_path.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, float | int]]] = {}
    for sample_id, nodes in attention.items():
        by_line: dict[int, float] = {}
        for node in nodes:
            line = int(node["line"])
            if line >= 0:
                by_line[line] = max(by_line.get(line, 0.0), float(node["score"]))
        top = [
            {"line": line, "score": score}
            for line, score in sorted(by_line.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        ]
        result[sample_id] = top
        if top:
            _plot_lines(sample_id, top, output_dir / f"{sample_id}.png")
    (output_dir / "top_k_nodes.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _plot_lines(sample_id: str, top: list[dict[str, float | int]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.barh([str(item["line"]) for item in reversed(top)], [item["score"] for item in reversed(top)])
    axis.set_title(f"{sample_id}: suspicious source lines")
    axis.set_xlabel("Attention score")
    axis.set_ylabel("Source line")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


@dataclass(frozen=True)
class SourceLocation:
    path: Path
    line_offset: int


@dataclass(frozen=True)
class PredictionRow:
    sample_id: str
    label: int
    probability: float
    prediction: int


@dataclass(frozen=True)
class LineAttention:
    sample_id: str
    rank: int
    graph_line: int
    source_line: int
    score: float
    source_text: str


def export_attention_dashboard(
    attention_path: Path,
    predictions_path: Path,
    source_map_path: Path,
    output_dir: Path,
    *,
    run_name: str,
    top_samples: int = 24,
    top_lines: int = 10,
    context_radius: int = 2,
) -> dict[str, object]:
    """Write an HTML source-code heatmap for model node attention."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    attention = json.loads(attention_path.read_text(encoding="utf-8"))
    predictions = _read_prediction_rows(predictions_path)
    source_map = _read_source_map(source_map_path)

    all_top_lines = _top_line_attention_rows(attention, predictions, source_map, top_lines=top_lines)
    _write_line_attention_csv(output_dir / "line_attention_top.csv", predictions, all_top_lines)

    selected = _select_predicted_positive_samples(predictions, limit=top_samples)
    sample_links: list[dict[str, object]] = []
    for prediction in selected:
        top = all_top_lines.get(prediction.sample_id, [])
        filename = f"{_safe_filename(prediction.sample_id)}.html"
        _write_sample_heatmap_html(
            sample_dir / filename,
            prediction,
            top,
            attention.get(prediction.sample_id, []),
            source_map.get(prediction.sample_id),
            run_name=run_name,
            context_radius=context_radius,
        )
        sample_links.append(
            {
                "sample_id": prediction.sample_id,
                "label": prediction.label,
                "prediction": prediction.prediction,
                "probability": prediction.probability,
                "href": f"samples/{filename}",
                "max_attention": max((line.score for line in top), default=0.0),
            }
        )

    _plot_attention_overview(sample_links, output_dir / "attention_overview.png", run_name=run_name)
    _write_index_html(output_dir / "index.html", sample_links, run_name=run_name)
    return {
        "run": run_name,
        "selected_samples": [prediction.sample_id for prediction in selected],
        "index": str(output_dir / "index.html"),
        "overview": str(output_dir / "attention_overview.png"),
        "line_attention": str(output_dir / "line_attention_top.csv"),
    }


def _read_prediction_rows(path: Path) -> list[PredictionRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            PredictionRow(
                sample_id=row["sample_id"],
                label=int(row["label"]),
                probability=float(row["probability"]),
                prediction=int(row["prediction"]),
            )
            for row in csv.DictReader(handle)
        ]


def _read_source_map(path: Path) -> dict[str, SourceLocation]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        result: dict[str, SourceLocation] = {}
        for row in rows:
            source_path = row.get("prepared_source_path") or row["raw_source_path"]
            result[row["sample_id"]] = SourceLocation(
                Path(source_path),
                int(row["line_offset"]),
            )
        return result


def _top_line_attention_rows(
    attention: dict[str, list[dict[str, object]]],
    predictions: list[PredictionRow],
    source_map: dict[str, SourceLocation],
    *,
    top_lines: int,
) -> dict[str, list[LineAttention]]:
    result: dict[str, list[LineAttention]] = {}
    for prediction in predictions:
        by_graph_line: dict[int, float] = {}
        for node in attention.get(prediction.sample_id, []):
            line = int(node["line"])
            if line < 0:
                continue
            by_graph_line[line] = max(by_graph_line.get(line, 0.0), float(node["score"]))
        rows: list[LineAttention] = []
        for rank, (graph_line, score) in enumerate(
            sorted(by_graph_line.items(), key=lambda item: (-item[1], item[0]))[:top_lines],
            start=1,
        ):
            location = source_map.get(prediction.sample_id)
            source_line = graph_line - location.line_offset if location is not None else graph_line
            rows.append(
                LineAttention(
                    sample_id=prediction.sample_id,
                    rank=rank,
                    graph_line=graph_line,
                    source_line=source_line,
                    score=score,
                    source_text=_read_source_line(location, source_line),
                )
            )
        result[prediction.sample_id] = rows
    return result


def _write_line_attention_csv(
    path: Path,
    predictions: list[PredictionRow],
    all_top_lines: dict[str, list[LineAttention]],
) -> None:
    by_sample = {prediction.sample_id: prediction for prediction in predictions}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "label",
                "prediction",
                "probability",
                "rank",
                "graph_line",
                "source_line",
                "attention",
                "source_text",
            ]
        )
        for sample_id, rows in all_top_lines.items():
            prediction = by_sample[sample_id]
            for row in rows:
                writer.writerow(
                    [
                        sample_id,
                        prediction.label,
                        prediction.prediction,
                        prediction.probability,
                        row.rank,
                        row.graph_line,
                        row.source_line,
                        row.score,
                        row.source_text,
                    ]
                )


def _select_predicted_positive_samples(
    predictions: list[PredictionRow],
    *,
    limit: int,
) -> list[PredictionRow]:
    positives = [row for row in predictions if row.prediction == 1]
    pool = positives or predictions
    return sorted(pool, key=lambda row: (-row.probability, row.sample_id))[:limit]


def _write_index_html(
    path: Path,
    sample_links: list[dict[str, object]],
    *,
    run_name: str,
) -> None:
    cards = "\n".join(
        f"""
        <a class="card" href="{html.escape(str(item['href']))}">
          <span class="sample">{html.escape(str(item['sample_id']))}</span>
          <span>label={item['label']} pred={item['prediction']}</span>
          <span>prob={float(item['probability']):.3f}</span>
          <span>max attention={float(item['max_attention']):.3f}</span>
        </a>
        """
        for item in sample_links
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(run_name)} Attention Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    h1 {{ margin-bottom: 6px; }}
    .subtle {{ color: #4b5563; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; margin-top: 18px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 6px; padding: 12px; text-decoration: none; color: inherit; display: grid; gap: 5px; }}
    .card:hover {{ border-color: #991b1b; background: #fff7f7; }}
    .sample {{ font-weight: 700; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; margin-top: 12px; }}
  </style>
</head>
<body>
  <h1>{html.escape(run_name)} Attention Dashboard</h1>
  <p class="subtle">Predicted-positive samples sorted by vulnerability probability. Open a sample to see source lines highlighted by node attention.</p>
  <img src="attention_overview.png" alt="Attention overview">
  <div class="grid">
    {cards}
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )


def _write_sample_heatmap_html(
    path: Path,
    prediction: PredictionRow,
    top_lines: list[LineAttention],
    nodes: list[dict[str, object]],
    location: SourceLocation | None,
    *,
    run_name: str,
    context_radius: int,
) -> None:
    source_scores = {
        row.source_line: max(0.0, min(1.0, row.score))
        for row in top_lines
        if row.source_line > 0
    }
    source_lines = _read_source_lines(location)
    context_lines = _context_line_numbers(
        sorted(source_scores),
        max_line=len(source_lines),
        radius=context_radius,
    )
    top_table = "\n".join(
        f"<tr><td>{row.rank}</td><td>{row.source_line}</td><td>{row.graph_line}</td>"
        f"<td>{row.score:.3f}</td><td><code>{html.escape(row.source_text)}</code></td></tr>"
        for row in top_lines
    )
    code_rows = "\n".join(
        _render_code_line(line_number, source_lines[line_number - 1], source_scores.get(line_number, 0.0))
        if line_number > 0
        else '<div class="ellipsis">...</div>'
        for line_number in context_lines
    )
    source_path = "" if location is None else str(location.path)
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(prediction.sample_id)} Attention</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 22px; color: #111827; }}
    a {{ color: #991b1b; }}
    .meta {{ display: flex; gap: 14px; flex-wrap: wrap; color: #374151; margin: 8px 0 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; vertical-align: top; }}
    code, pre {{ font-family: Consolas, "Courier New", monospace; }}
    .code {{ border: 1px solid #d1d5db; border-radius: 6px; overflow: hidden; }}
    .line {{ display: grid; grid-template-columns: 64px 94px 1fr; align-items: stretch; min-height: 24px; }}
    .line-number {{ color: #6b7280; text-align: right; padding: 4px 10px; background: rgba(255,255,255,0.55); }}
    .attention-cell {{ padding: 4px 8px; background: rgba(255,255,255,0.4); }}
    .attention-bar {{ height: 14px; border-radius: 3px; background: #dc2626; }}
    .source {{ white-space: pre-wrap; padding: 4px 10px; }}
    .ellipsis {{ color: #6b7280; padding: 4px 10px; border-top: 1px solid #f3f4f6; border-bottom: 1px solid #f3f4f6; }}
  </style>
</head>
<body>
  <p><a href="../index.html">Back to dashboard</a></p>
  <h1>{html.escape(prediction.sample_id)}</h1>
  <div class="meta">
    <span>run={html.escape(run_name)}</span>
    <span>label={prediction.label}</span>
    <span>prediction={prediction.prediction}</span>
    <span>probability={prediction.probability:.3f}</span>
    <span>source={html.escape(source_path)}</span>
  </div>
  <h2>Top Attention Lines</h2>
  <table>
    <thead><tr><th>Rank</th><th>Source line</th><th>Graph line</th><th>Attention</th><th>Code</th></tr></thead>
    <tbody>{top_table}</tbody>
  </table>
  <h2>Highlighted Source Context</h2>
  <div class="code">{code_rows}</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def _render_code_line(line_number: int, text: str, score: float) -> str:
    alpha = 0.06 + 0.62 * max(0.0, min(1.0, score))
    width = int(max(0.0, min(1.0, score)) * 100)
    return (
        f'<div class="line" style="background: rgba(220, 38, 38, {alpha:.3f});">'
        f'<span class="line-number">{line_number}</span>'
        f'<span class="attention-cell"><span class="attention-bar" style="width: {width}%"></span></span>'
        f'<code class="source">{html.escape(text)}</code>'
        "</div>"
    )


def _plot_attention_overview(
    sample_links: list[dict[str, object]],
    path: Path,
    *,
    run_name: str,
) -> None:
    labels = [str(item["sample_id"]) for item in sample_links[:20]]
    probabilities = [float(item["probability"]) for item in sample_links[:20]]
    max_attention = [float(item["max_attention"]) for item in sample_links[:20]]
    figure, axis = plt.subplots(figsize=(max(9, len(labels) * 0.48), 4.8))
    x_positions = list(range(len(labels)))
    axis.bar(x_positions, probabilities, label="Vulnerability probability", color="#991b1b", alpha=0.82)
    axis.plot(x_positions, max_attention, label="Max line attention", color="#2563eb", marker="o", linewidth=1.5)
    axis.set_xticks(x_positions, labels, rotation=45, ha="right", fontsize=7)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title(f"{run_name}: top predicted-positive samples")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _context_line_numbers(lines: list[int], *, max_line: int, radius: int) -> list[int]:
    if not lines or max_line <= 0:
        return []
    selected: set[int] = set()
    for line in lines:
        if line <= 0:
            continue
        selected.update(range(max(1, line - radius), min(max_line, line + radius) + 1))
    ordered = sorted(selected)
    result: list[int] = []
    previous: int | None = None
    for line in ordered:
        if previous is not None and line > previous + 1:
            result.append(0)
        result.append(line)
        previous = line
    return result


def _read_source_line(location: SourceLocation | None, source_line: int) -> str:
    lines = _read_source_lines(location)
    if source_line <= 0 or source_line > len(lines):
        return ""
    return lines[source_line - 1].strip()


def _read_source_lines(location: SourceLocation | None) -> list[str]:
    if location is None or not location.path.is_file():
        return []
    return location.path.read_text(encoding="utf-8", errors="replace").splitlines()


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"
