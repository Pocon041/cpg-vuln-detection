from __future__ import annotations

import json
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

