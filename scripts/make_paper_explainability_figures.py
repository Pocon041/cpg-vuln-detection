from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


GRAPH_EXPLANATION_DIR = ROOT / "outputs" / "reports" / "explanations" / "ramp-E4-strict"
END2END_RUN_DIR = ROOT / "outputs" / "runs" / "raw-v1" / "end2end-codebert-mil-strict-fixed05-posw165"
OUT_DIR = ROOT / "figures" / "paper_explainability"
END2END_EVIDENCE_JSON = OUT_DIR / "end2end_chunk_evidence.json"

GRAPH_TP_SAMPLE = "3468_1"
GRAPH_FP_SAMPLE = "1760_0"

OKABE_BLUE = "#0072B2"
OKABE_ORANGE = "#E69F00"
OKABE_GREEN = "#009E73"
OKABE_PURPLE = "#CC79A7"
GREY = "#4B5563"
LIGHT_GREY = "#E5E7EB"
TEXT = "#111827"


@dataclass(frozen=True)
class LineAttentionRow:
    sample_id: str
    label: int
    prediction: int
    probability: float
    rank: int
    graph_line: int
    source_line: int
    attention: float
    source_text: str


@dataclass(frozen=True)
class ChunkEvidence:
    index: int
    start_token: int
    line_range: str
    weight: float
    snippet: str


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def read_line_attention(path: Path) -> list[LineAttentionRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                LineAttentionRow(
                    sample_id=row["sample_id"],
                    label=int(row["label"]),
                    prediction=int(row["prediction"]),
                    probability=float(row["probability"]),
                    rank=int(row["rank"]),
                    graph_line=int(row["graph_line"]),
                    source_line=int(row["source_line"]),
                    attention=float(row["attention"]),
                    source_text=row["source_text"],
                )
            )
    return rows


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 450},
    }.items():
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def plot_source_line_localization(rows: list[LineAttentionRow], stem: str, *, top_k: int = 6) -> None:
    selected = sorted(rows, key=lambda row: (row.rank, row.source_line))[:top_k]
    selected = sorted(selected, key=lambda row: row.source_line)
    if not selected:
        raise ValueError(f"no line attention rows for {stem}")
    values = np.array([row.attention for row in selected], dtype=float)
    vmax = max(float(values.max()), 1e-8)
    n = len(selected)
    fig, ax = plt.subplots(figsize=(7.0, 0.38 * n + 0.26))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.axis("off")

    cmap = plt.get_cmap("cividis")
    for visual_index, row in enumerate(selected):
        y = n - visual_index - 1
        norm = row.attention / vmax
        color = cmap(0.18 + 0.74 * norm)
        ax.add_patch(Rectangle((0.0, y + 0.04), 1.0, 0.90, color=color, alpha=0.18, linewidth=0))
        ax.add_patch(Rectangle((0.105, y + 0.37), 0.090 * norm, 0.20, color=OKABE_BLUE, alpha=0.95, linewidth=0))
        ax.text(
            0.076,
            y + 0.48,
            f"{row.source_line}",
            ha="right",
            va="center",
            color=GREY,
            family="DejaVu Sans Mono",
            fontsize=7.5,
        )
        ax.text(
            0.225,
            y + 0.48,
            shorten(row.source_text.strip(), width=112, placeholder=" ..."),
            ha="left",
            va="center",
            color=TEXT,
            family="DejaVu Sans Mono",
            fontsize=7.2,
        )
    save_figure(fig, stem)


def plot_node_type_evidence(path: Path, stem: str, *, case: str = "TP") -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes = payload[case]["top_nodes"]
    by_type: dict[str, float] = {}
    for node in nodes:
        label = str(node["node_label"])
        by_type[label] = max(by_type.get(label, 0.0), float(node["score"]))
    items = sorted(by_type.items(), key=lambda item: item[1])[-6:]
    labels = [item[0] for item in items]
    values = [item[1] for item in items]

    fig, ax = plt.subplots(figsize=(3.35, 1.9))
    y = np.arange(len(items))
    colors = [OKABE_BLUE if label == "IDENTIFIER" else OKABE_GREEN for label in labels]
    ax.barh(y, values, color=colors, height=0.58)
    ax.set_yticks(y, labels)
    ax.set_xlabel("max attention")
    ax.set_xlim(0, max(values) * 1.12)
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.55)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for yi, value in zip(y, values):
        ax.text(value + max(values) * 0.018, yi, f"{value:.2f}", va="center", ha="left", fontsize=7, color=GREY)
    save_figure(fig, stem)


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_end2end_tp(predictions_path: Path) -> str:
    rows = read_predictions(predictions_path)
    positives = [
        row
        for row in rows
        if int(row["label"]) == 1 and int(row["prediction"]) == 1
    ]
    if not positives:
        raise ValueError("no true-positive sample in end-to-end predictions")
    return max(positives, key=lambda row: float(row["probability"]))["sample_id"]


def source_offsets_by_token(tokenizer, source: str) -> list[tuple[int, int]]:
    try:
        encoded = tokenizer(
            source,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, NotImplementedError):
        return []
    return [(int(start), int(end)) for start, end in encoded.get("offset_mapping", [])]


def line_number_at(source: str, char_offset: int) -> int:
    return source.count("\n", 0, max(0, char_offset)) + 1


def compact_snippet(lines: list[str], start: int, end: int) -> str:
    subset = [line.strip() for line in lines[max(0, start - 1) : max(start - 1, end)] if line.strip()]
    text = " ".join(subset[:3])
    return shorten(text, width=76, placeholder=" ...")


def chunk_evidence_for_sample(sample_id: str) -> tuple[str, float, list[ChunkEvidence]]:
    os.chdir(ROOT)
    import torch
    from torch_geometric.data import Batch

    from cpg_vuln.config import load_config
    from cpg_vuln.data.audit import read_manifest
    from cpg_vuln.end2end.dataset import RawCodeDataset
    from cpg_vuln.end2end.model import RawCodeMILTransformer

    config = load_config(ROOT / "configs" / "end2end_codebert_mil.yaml")
    end2end = config["end2end"]
    records = read_manifest(Path(config["paths"]["artifacts_dir"]) / "data" / "manifest.jsonl")
    record_by_id = {record.sample_id: record for record in records}
    record = record_by_id[sample_id]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(end2end["model_name"]),
        local_files_only=bool(end2end["local_files_only"]),
    )
    dataset = RawCodeDataset(
        records,
        [sample_id],
        tokenizer=tokenizer,
        max_length=int(end2end["max_length"]),
        stride=int(end2end["stride"]),
        max_chunks=int(end2end["max_chunks"]),
        desc="tokenize explain",
    )
    batch = Batch.from_data_list([dataset[0]])

    model = RawCodeMILTransformer(
        model_name=str(end2end["model_name"]),
        dropout=float(end2end["dropout"]),
        freeze_encoder=bool(end2end["freeze_encoder"]),
        local_files_only=bool(end2end["local_files_only"]),
    )
    checkpoint = torch.load(END2END_RUN_DIR / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.inference_mode():
        output = model(batch)
        probability = float(torch.softmax(output.logits, dim=-1)[0, 1].item())
        weights = output.node_attention.detach().cpu().numpy()

    source = Path(record.source_path).read_text(encoding="utf-8", errors="replace")
    source_lines = source.splitlines()
    offsets = source_offsets_by_token(tokenizer, source)
    evidences: list[ChunkEvidence] = []
    max_content_tokens = int(end2end["max_length"]) - 2
    for index, start_token in enumerate(batch.chunk_start.cpu().tolist()):
        active_tokens = int(batch.attention_mask[index].sum().item()) - 2
        end_token = min(start_token + max(active_tokens, 0), start_token + max_content_tokens)
        if offsets and start_token < len(offsets):
            char_start = offsets[start_token][0]
            char_end = offsets[min(max(end_token - 1, start_token), len(offsets) - 1)][1]
            line_start = line_number_at(source, char_start)
            line_end = line_number_at(source, char_end)
            line_range = f"{line_start}-{line_end}" if line_start != line_end else str(line_start)
            snippet = compact_snippet(source_lines, line_start, line_end)
        else:
            ids = batch.input_ids[index].tolist()
            snippet = shorten(
                tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                width=98,
                placeholder=" ...",
            )
            line_range = str(start_token)
        evidences.append(
            ChunkEvidence(
                index=index + 1,
                start_token=int(start_token),
                line_range=line_range,
                weight=float(weights[index]),
                snippet=snippet,
            )
        )
    return sample_id, probability, evidences


def export_end2end_chunk_evidence_json(path: Path) -> None:
    sample_id = select_end2end_tp(END2END_RUN_DIR / "predictions.csv")
    raw_sample_id, probability, evidences = chunk_evidence_for_sample(sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sample_id": raw_sample_id,
                "probability": probability,
                "chunks": [evidence.__dict__ for evidence in evidences],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_or_create_end2end_chunk_evidence() -> tuple[str, float, list[ChunkEvidence]]:
    if not END2END_EVIDENCE_JSON.is_file():
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--export-end2end-data",
                str(END2END_EVIDENCE_JSON),
            ],
            cwd=ROOT,
            check=True,
        )
    payload = json.loads(END2END_EVIDENCE_JSON.read_text(encoding="utf-8"))
    return (
        str(payload["sample_id"]),
        float(payload["probability"]),
        [ChunkEvidence(**item) for item in payload["chunks"]],
    )


def plot_chunk_evidence(sample_id: str, probability: float, evidences: list[ChunkEvidence], stem: str) -> None:
    if not evidences:
        raise ValueError("no chunk evidence to plot")
    n = len(evidences)
    fig, ax = plt.subplots(figsize=(7.0, 0.48 * n + 0.40))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.axis("off")

    vmax = max(max(evidence.weight for evidence in evidences), 1e-8)
    for visual_index, evidence in enumerate(evidences):
        y = n - visual_index - 1
        norm = evidence.weight / vmax
        ax.add_patch(Rectangle((0.0, y + 0.04), 1.0, 0.90, color=OKABE_PURPLE, alpha=0.08 + 0.22 * norm, linewidth=0))
        ax.add_patch(Rectangle((0.118, y + 0.36), 0.125 * norm, 0.22, color=OKABE_PURPLE, alpha=0.95, linewidth=0))
        ax.text(
            0.080,
            y + 0.48,
            evidence.line_range,
            ha="right",
            va="center",
            color=GREY,
            family="DejaVu Sans Mono",
            fontsize=7.5,
        )
        ax.text(
            0.275,
            y + 0.48,
            evidence.snippet,
            ha="left",
            va="center",
            color=TEXT,
            family="DejaVu Sans Mono",
            fontsize=7.2,
        )
    save_figure(fig, stem)


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--export-end2end-data":
        export_end2end_chunk_evidence_json(Path(sys.argv[2]))
        return

    configure_style()
    rows = read_line_attention(GRAPH_EXPLANATION_DIR / "line_attention_top.csv")
    by_sample: dict[str, list[LineAttentionRow]] = {}
    for row in rows:
        by_sample.setdefault(row.sample_id, []).append(row)

    plot_source_line_localization(
        by_sample[GRAPH_TP_SAMPLE],
        f"graph_line_localization_tp_{GRAPH_TP_SAMPLE}",
    )
    plot_source_line_localization(
        by_sample[GRAPH_FP_SAMPLE],
        f"graph_line_localization_fp_{GRAPH_FP_SAMPLE}",
    )
    plot_node_type_evidence(
        GRAPH_EXPLANATION_DIR / "selected_top_nodes.json",
        f"graph_node_type_evidence_{GRAPH_TP_SAMPLE}",
        case="TP",
    )

    raw_sample_id, probability, evidences = load_or_create_end2end_chunk_evidence()
    plot_chunk_evidence(
        raw_sample_id,
        probability,
        evidences,
        f"end2end_chunk_evidence_{raw_sample_id}",
    )
    print(f"wrote paper explainability figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
