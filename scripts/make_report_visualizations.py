from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "report_visuals"
EXPLAIN_DIR = ROOT / "outputs" / "reports" / "explanations" / "ramp-E4-strict"
RAW_CHUNK_JSON = ROOT / "figures" / "paper_explainability" / "end2end_chunk_evidence.json"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
INK = "#111827"
MUTED = "#4B5563"
LIGHT = "#E5E7EB"
PANEL = "#F9FAFB"


@dataclass(frozen=True)
class MethodRun:
    name: str
    role: str
    path: Path
    color: str


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


METHODS = [
    MethodRun(
        "RAMP-v2A",
        "main",
        ROOT
        / "outputs"
        / "runs"
        / "raw-v1"
        / "ramp-E4-v2A-core-cpg-current-schema-fixed05-guarded-f1-rank02-replay025-20e",
        BLUE,
    ),
    MethodRun(
        "RAMP-v2Dual",
        "diagnostic",
        ROOT / "outputs" / "runs" / "raw-v1" / "ramp-E4-v2Dual-raw-v1-fixed05-mcc-lr3e4-20e",
        GREEN,
    ),
    MethodRun(
        "RAMP-v2Gated",
        "gated",
        ROOT / "outputs" / "runs" / "raw-v1" / "ramp-E4-v2Gated-raw-v1-fixed05-guarded-f1-lr3e4-25e",
        ORANGE,
    ),
    MethodRun(
        "RAMP-v3 Slice-MIL",
        "slice",
        ROOT / "outputs" / "runs" / "raw-v1" / "ramp-E4-v3SliceMIL-core-cpg-trueSliceMask-20e-noRank-aux02-fusion3",
        PURPLE,
    ),
    MethodRun(
        "RawCode-MIL",
        "raw",
        ROOT / "outputs" / "runs" / "raw-v1" / "end2end-codebert-mil-strict-fixed05-posw165",
        VERMILION,
    ),
]


CAPTIONS: list[tuple[str, str]] = []


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#9CA3AF",
            "axes.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
        }
    )


def save_figure(fig: plt.Figure, stem: str, caption: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 420},
    }.items():
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", facecolor="white", **kwargs)
    plt.close(fig)
    CAPTIONS.append((stem, caption))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_metrics() -> dict[str, dict]:
    payload: dict[str, dict] = {}
    for method in METHODS:
        metrics = load_json(method.path / "metrics.json")
        test = metrics.get("test_fixed_0_5") or metrics.get("test")
        payload[method.name] = {
            "method": method,
            "raw": metrics,
            "test": test,
            "validation": metrics.get("validation_fixed_0_5") or metrics.get("validation"),
        }
    return payload


def metric(metrics: dict[str, dict], method: str, key: str) -> float:
    return float(metrics[method]["test"][key])


def true_positive_rate(metrics: dict[str, dict], method: str) -> float:
    cm = metrics[method]["test"]["confusion_matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]
    return (tp + fn) / max(1, tn + fp + fn + tp)


def load_history(method: MethodRun) -> list[dict]:
    path = method.path / "history.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def read_line_attention(path: Path) -> list[LineAttentionRow]:
    rows: list[LineAttentionRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
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
                    source_text=row["source_text"].replace("\n", " "),
                )
            )
    return rows


def outcome(row: LineAttentionRow) -> str:
    if row.label == 1 and row.prediction == 1:
        return "TP"
    if row.label == 0 and row.prediction == 1:
        return "FP"
    if row.label == 1 and row.prediction == 0:
        return "FN"
    return "TN"


def code_category(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\b(memcpy|memmove|memset|strcpy|strncpy|snprintf|sprintf)\b", lowered):
        return "copy/API"
    if re.search(r"\b(malloc|calloc|realloc|free|g_free|alloc|av_free)\b", lowered):
        return "alloc/free"
    if re.search(r"(qtest_|qmp_|eventwait|usleep|wait|cleanup|qtest_start|qmp)", lowered):
        return "runtime/wait"
    if re.search(r"(\-\>|\[[^\]]+\]|\*|ptr|buf|dst|src|size|stride|width|height|offset|linesize)", lowered):
        return "buffer/index"
    if re.search(r"\b(if|for|while|do|switch|case)\b", lowered):
        return "control"
    if re.search(r"\b(return|goto|break|continue)\b", lowered):
        return "exit"
    return "other"


def plot_method_evolution() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.35), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.05, 0.62, 0.17, 0.18, "CPG +\nCodeBERT", BLUE),
        (0.28, 0.62, 0.17, 0.18, "Relation-aware\nRGCN", BLUE),
        (0.51, 0.78, 0.16, 0.15, "Branch\nheads", GREEN),
        (0.51, 0.56, 0.16, 0.15, "Node-update\ngate", ORANGE),
        (0.73, 0.62, 0.18, 0.18, "Slice-level\nMIL", PURPLE),
        (0.05, 0.18, 0.17, 0.18, "Raw\nsource", VERMILION),
        (0.28, 0.18, 0.17, 0.18, "Token\nwindows", VERMILION),
        (0.51, 0.18, 0.17, 0.18, "RawCode\nMIL", VERMILION),
        (0.75, 0.18, 0.16, 0.18, "Fixed 0.5\neval", INK),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=color if color == INK else "#FFFFFF",
                edgecolor=color,
                linewidth=1.7,
                joinstyle="round",
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            color="#FFFFFF" if color == INK else INK,
            fontsize=8.3,
            linespacing=1.15,
        )

    arrows = [
        ((0.22, 0.71), (0.28, 0.71), BLUE),
        ((0.45, 0.71), (0.51, 0.855), GREEN),
        ((0.45, 0.71), (0.51, 0.635), ORANGE),
        ((0.67, 0.635), (0.73, 0.71), PURPLE),
        ((0.67, 0.855), (0.73, 0.78), GREEN),
        ((0.22, 0.27), (0.28, 0.27), VERMILION),
        ((0.45, 0.27), (0.51, 0.27), VERMILION),
        ((0.68, 0.27), (0.75, 0.27), INK),
        ((0.82, 0.62), (0.82, 0.36), INK),
    ]
    for start, end, color in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.15,
                color=color,
                shrinkA=2,
                shrinkB=2,
            )
        )
    ax.plot([0.03, 0.94], [0.50, 0.50], color=LIGHT, linewidth=0.9)
    ax.text(0.955, 0.71, "graph route", ha="left", va="center", color=MUTED, fontsize=7.2)
    ax.text(0.955, 0.27, "raw-code route", ha="left", va="center", color=MUTED, fontsize=7.2)
    save_figure(
        fig,
        "01_method_evolution_paths",
        "方法路线图：图方法从 CPG+CodeBERT 逐步扩展到关系建模、分支诊断、门控更新和 Slice-MIL；RawCode-MIL 则作为不构造中间表示的端到端对照。",
    )


def plot_metric_tradeoff(metrics: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.1), constrained_layout=True)
    label_positions = {
        "RAMP-v2A": (0.6279, 0.1870, "v2A"),
        "RAMP-v2Dual": (0.6208, 0.1805, "v2Dual"),
        "RAMP-v2Gated": (0.6252, 0.1512, "v2Gated"),
        "RAMP-v3 Slice-MIL": (0.6158, 0.1789, "v3 Slice"),
        "RawCode-MIL": (0.6062, 0.1258, "RawCode"),
    }
    for name, payload in metrics.items():
        f1 = float(payload["test"]["f1"])
        mcc = float(payload["test"]["mcc"])
        ppr = float(payload["test"]["predicted_positive_rate"])
        method = payload["method"]
        ax.scatter(f1, mcc, s=2200 * (ppr - 0.50) + 180, color=method.color, alpha=0.78, edgecolor="white", linewidth=0.8)
        tx, ty, label = label_positions[name]
        ax.annotate(
            label,
            xy=(f1, mcc),
            xytext=(tx, ty),
            ha="left",
            va="center",
            fontsize=7.0,
            color=INK,
            arrowprops=dict(arrowstyle="-", color="#9CA3AF", linewidth=0.55, shrinkA=1, shrinkB=4),
        )
    ax.set_xlabel("F1 at threshold 0.5")
    ax.set_ylabel("MCC at threshold 0.5")
    ax.grid(color=LIGHT, linewidth=0.7)
    ax.set_xlim(0.600, 0.631)
    ax.set_ylim(0.115, 0.190)
    save_figure(
        fig,
        "02_metric_f1_mcc_ppr_tradeoff",
        "F1-MCC-PPR 权衡图：点越大表示 PPR 越高；RAMP-v2A 同时保持最高 F1 与最高 MCC，因此比仅追求 Recall 的方案更稳健。",
    )


def plot_precision_recall_balance(metrics: dict[str, dict]) -> None:
    names = list(metrics)
    y = np.arange(len(names))
    precision = [metric(metrics, name, "precision") for name in names]
    recall = [metric(metrics, name, "recall") for name in names]
    fig, ax = plt.subplots(figsize=(5.0, 2.8), constrained_layout=True)
    for i, name in enumerate(names):
        method = metrics[name]["method"]
        ax.plot([precision[i], recall[i]], [i, i], color=LIGHT, linewidth=5, solid_capstyle="round")
        ax.scatter(precision[i], i, s=48, color=method.color, marker="o", zorder=3)
        ax.scatter(recall[i], i, s=48, color=method.color, marker="s", zorder=3)
    ax.set_yticks(y, names)
    ax.set_xlabel("score")
    ax.set_xlim(0.50, 0.79)
    ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax.scatter([], [], color=MUTED, marker="o", label="Precision")
    ax.scatter([], [], color=MUTED, marker="s", label="Recall")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2)
    ax.invert_yaxis()
    save_figure(
        fig,
        "03_precision_recall_balance",
        "Precision-Recall 平衡图：Gated 的 Recall 最高，但 Precision 下降，说明高召回并不等价于更可靠的漏洞判别。",
    )


def plot_ppr_prevalence(metrics: dict[str, dict]) -> None:
    names = list(metrics)
    y = np.arange(len(names))
    ppr = [metric(metrics, name, "predicted_positive_rate") for name in names]
    prevalence = [true_positive_rate(metrics, name) for name in names]
    fig, ax = plt.subplots(figsize=(5.0, 2.85), constrained_layout=True)
    for i, name in enumerate(names):
        method = metrics[name]["method"]
        ax.plot([prevalence[i], ppr[i]], [i, i], color=LIGHT, linewidth=5, solid_capstyle="round")
        ax.scatter(prevalence[i], i, color="#9CA3AF", s=42, zorder=3, label="test prevalence" if i == 0 else None)
        ax.scatter(ppr[i], i, color=method.color, s=55, zorder=3, label="PPR" if i == 0 else None)
        ax.text(ppr[i] + 0.006, i, f"{ppr[i]:.3f}", va="center", ha="left", fontsize=7.2, color=MUTED)
    ax.axvline(np.mean(prevalence), color="#9CA3AF", linestyle="--", linewidth=0.9)
    ax.set_yticks(y, names)
    ax.set_xlabel("positive ratio")
    ax.set_xlim(0.44, 0.73)
    ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2)
    save_figure(
        fig,
        "04_ppr_against_test_prevalence",
        "PPR 与测试集真实正类比例对比：所有方法都偏向预测正类，Gated 偏差最大，这是固定 0.5 阈值下必须报告 PPR 的原因。",
    )


def plot_confusion_matrix(metrics: dict[str, dict]) -> None:
    cm = np.array(metrics["RAMP-v2A"]["test"]["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(2.75, 2.65), constrained_layout=True)
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            text_color = "white" if normalized[i, j] > 0.48 else INK
            ax.text(j, i - 0.08, labels[i][j], ha="center", va="center", color=text_color, fontsize=10, weight="bold")
            ax.text(j, i + 0.12, f"{int(cm[i, j])}\n{normalized[i, j] * 100:.1f}%", ha="center", va="center", color=text_color, fontsize=8)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    save_figure(
        fig,
        "05_confusion_matrix_ramp_v2a",
        "RAMP-v2A 测试集混淆矩阵：在固定 0.5 阈值下，模型仍有较多 FP，但相对其他方法保持了更好的 F1/MCC 综合平衡。",
    )


def plot_threshold_pathology(metrics: dict[str, dict]) -> None:
    raw = metrics["RAMP-v2A"]["raw"]
    variants = [
        ("fixed 0.5", raw["test_fixed_0_5"]),
        ("val-F1 thr", raw["test_val_f1"]),
        ("val-MCC thr", raw["test_val_mcc"]),
    ]
    x = np.arange(len(variants))
    ppr = [float(item["predicted_positive_rate"]) for _, item in variants]
    recall = [float(item["recall"]) for _, item in variants]
    mcc = [float(item["mcc"]) for _, item in variants]
    fig, ax = plt.subplots(figsize=(4.2, 2.65), constrained_layout=True)
    ax.plot(x, ppr, color=VERMILION, marker="o", linewidth=1.8, label="PPR")
    ax.plot(x, recall, color=BLUE, marker="s", linewidth=1.8, label="Recall")
    ax.plot(x, mcc, color=GREEN, marker="^", linewidth=1.8, label="MCC")
    ax.set_xticks(x, [name for name, _ in variants])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    for xi, value in zip(x[:2], ppr[:2]):
        ax.text(xi, value + 0.035, f"{value:.2f}", ha="center", va="bottom", color=VERMILION, fontsize=7)
    save_figure(
        fig,
        "06_threshold_choice_pathology",
        "阈值选择诊断图：验证集最优 F1 阈值会把测试集 PPR 推到 0.95 左右，因此本报告坚持固定 0.5 阈值评价。",
    )


def plot_score_distribution(method: MethodRun, stem: str, caption: str) -> None:
    rows = read_csv(method.path / "predictions.csv")
    pos = [float(row["probability"]) for row in rows if int(row["label"]) == 1]
    neg = [float(row["probability"]) for row in rows if int(row["label"]) == 0]
    bins = np.linspace(0, 1, 28)
    fig, ax = plt.subplots(figsize=(4.5, 2.75), constrained_layout=True)
    ax.hist(neg, bins=bins, density=True, histtype="stepfilled", alpha=0.32, color=SKY, label="true 0")
    ax.hist(pos, bins=bins, density=True, histtype="stepfilled", alpha=0.32, color=VERMILION, label="true 1")
    ax.hist(neg, bins=bins, density=True, histtype="step", linewidth=1.4, color=BLUE)
    ax.hist(pos, bins=bins, density=True, histtype="step", linewidth=1.4, color=VERMILION)
    ax.axvline(0.5, color=INK, linestyle="--", linewidth=1.0)
    ax.set_xlabel("predicted vulnerability probability")
    ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, stem, caption)


def plot_training_loss(metrics: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.05), constrained_layout=True)
    for method in METHODS:
        history = load_history(method)
        if not history:
            continue
        epochs = [int(item["epoch"]) for item in history]
        losses = [float(item.get("loss", item.get("objective_per_step", math.nan))) for item in history]
        if not losses or not np.isfinite(losses[0]) or losses[0] == 0:
            continue
        normalized = [loss / losses[0] for loss in losses]
        ax.plot(epochs, normalized, color=method.color, linewidth=1.6, marker="o", markersize=3, label=method.name)
    ax.axhline(1.0, color=LIGHT, linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / epoch-1 loss")
    ax.grid(color=LIGHT, linewidth=0.7)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=3)
    save_figure(
        fig,
        "08_training_loss_normalized",
        "归一化训练损失曲线：多数图方法的 loss 下降幅度有限，说明当前表示的可分性不足，而不是简单多训几个 epoch 就能解决。",
    )


def plot_checkpoint_guardrail(metrics: dict[str, dict]) -> None:
    method = metrics["RAMP-v2A"]["method"]
    history = load_history(method)
    epochs = [int(item["epoch"]) for item in history if "val_ppr" in item]
    ppr = [float(item["val_ppr"]) for item in history if "val_ppr" in item]
    guarded = [bool(item.get("checkpoint_guarded_out", False)) for item in history if "val_ppr" in item]
    fig, ax = plt.subplots(figsize=(4.5, 2.6), constrained_layout=True)
    ax.axhspan(0.25, 0.75, color=GREEN, alpha=0.08)
    for x, y, is_guarded in zip(epochs, ppr, guarded):
        ax.scatter(x, y, color=VERMILION if is_guarded else BLUE, marker="x" if is_guarded else "o", s=42, zorder=3)
    ax.plot(epochs, ppr, color="#9CA3AF", linewidth=1.0, zorder=1)
    ax.axhline(0.75, color=GREEN, linestyle="--", linewidth=0.9)
    ax.axhline(0.25, color=GREEN, linestyle="--", linewidth=0.9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation PPR")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.scatter([], [], color=BLUE, marker="o", label="eligible")
    ax.scatter([], [], color=VERMILION, marker="x", label="guarded out")
    ax.legend(frameon=False, loc="lower right")
    save_figure(
        fig,
        "09_checkpoint_ppr_guardrail",
        "checkpoint 保护带诊断：当验证集 PPR 过高或过低时，即使 F1 看起来更高，也会被排除，避免选择全正类倾向的模型。",
    )


def plot_gated_diagnostics(metrics: dict[str, dict]) -> None:
    diagnostics = metrics["RAMP-v2Gated"]["raw"]["test_diagnostics"]
    labels = ["layer 1", "layer 2", "layer 3", "fusion"]
    values = list(map(float, diagnostics["encoder_gate_mean_by_layer"])) + [float(diagnostics["fusion_gate_mean"])]
    colors = [ORANGE, ORANGE, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(3.65, 2.4), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.axhline(0.5, color=LIGHT, linewidth=0.9)
    ax.set_ylabel("mean gate value")
    ax.set_ylim(0, 0.55)
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=7.2, color=MUTED)
    save_figure(
        fig,
        "10_gated_internal_statistics",
        "Gated RGCN 内部门控统计：图消息没有完全关闭，但该门控发生在关系聚合之后，因此无法解释为关系级噪声抑制。",
    )


def plot_slice_fusion_weights(metrics: dict[str, dict]) -> None:
    diagnostics = metrics["RAMP-v3 Slice-MIL"]["raw"]["test_diagnostics"]
    labels = ["fused", "graph", "semantic", "slice"]
    values = [
        float(diagnostics["fusion_weight_fused"]),
        float(diagnostics["fusion_weight_graph"]),
        float(diagnostics["fusion_weight_semantic"]),
        float(diagnostics["fusion_weight_slice"]),
    ]
    colors = [INK, BLUE, GREEN, PURPLE]
    fig, ax = plt.subplots(figsize=(3.9, 2.45), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fusion weight")
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=7.2, color=MUTED)
    save_figure(
        fig,
        "11_slice_mil_fusion_weights",
        "Slice-MIL 融合权重图：slice 分支权重约 0.029，说明当前局部证据分支尚未真正主导最终决策。",
    )


def plot_slice_candidate_funnel(metrics: dict[str, dict]) -> None:
    diagnostics = metrics["RAMP-v3 Slice-MIL"]["raw"]["test_diagnostics"]
    labels = ["candidate\nnodes", "selected\ntop-k"]
    values = [float(diagnostics["slice_candidate_count_mean"]), float(diagnostics["slice_selected_count_mean"])]
    fig, ax = plt.subplots(figsize=(3.3, 2.45), constrained_layout=True)
    bars = ax.bar(labels, values, color=[PURPLE, BLUE], width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("mean count, log scale")
    ax.grid(axis="y", color=LIGHT, linewidth=0.7, which="both")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:.1f}", ha="center", va="bottom", fontsize=7.4, color=MUTED)
    save_figure(
        fig,
        "12_slice_candidate_funnel",
        "Slice-MIL 候选漏斗图：测试样本平均约 130 个候选节点只保留 top-3，策略非常激进，解释了局部证据学习不稳定的风险。",
    )


def plot_code_evidence(rows: list[LineAttentionRow], sample_id: str, stem: str, caption: str, top_k: int = 8) -> None:
    selected = sorted([row for row in rows if row.sample_id == sample_id], key=lambda row: row.rank)[:top_k]
    selected = sorted(selected, key=lambda row: row.source_line)
    if not selected:
        raise ValueError(f"no rows for {sample_id}")
    n = len(selected)
    values = np.array([row.attention for row in selected], dtype=float)
    vmax = max(float(values.max()), 1e-8)
    case = outcome(selected[0])
    fig, ax = plt.subplots(figsize=(7.15, 0.42 * n + 0.55), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, n + 0.35)
    ax.axis("off")
    ax.text(0.006, n + 0.10, f"{case}  sample {sample_id}  p={selected[0].probability:.3f}", ha="left", va="center", fontsize=8.2, color=MUTED)
    for visual_index, row in enumerate(selected):
        y = n - visual_index - 1
        norm = row.attention / vmax
        category = code_category(row.source_text)
        fill = plt.get_cmap("cividis")(0.22 + 0.70 * norm)
        ax.add_patch(Rectangle((0.0, y + 0.06), 1.0, 0.86, color=fill, alpha=0.17, linewidth=0))
        ax.add_patch(Rectangle((0.118, y + 0.38), 0.105 * norm, 0.20, color=BLUE, alpha=0.92, linewidth=0))
        ax.text(0.075, y + 0.48, f"{row.source_line}", ha="right", va="center", color=MUTED, family="DejaVu Sans Mono", fontsize=7.4)
        ax.text(0.245, y + 0.48, shorten(row.source_text.strip(), width=94, placeholder=" ..."), ha="left", va="center", color=INK, family="DejaVu Sans Mono", fontsize=7.15)
        ax.text(0.985, y + 0.48, category, ha="right", va="center", color=MUTED, fontsize=6.8)
    save_figure(fig, stem, caption)


def plot_raw_chunk_evidence() -> None:
    payload = load_json(RAW_CHUNK_JSON)
    chunks = payload["chunks"]
    n = len(chunks)
    weights = np.array([float(chunk["weight"]) for chunk in chunks])
    vmax = max(float(weights.max()), 1e-8)
    fig, ax = plt.subplots(figsize=(7.1, 0.62 * n + 0.65), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, n + 0.42)
    ax.axis("off")
    ax.text(0.006, n + 0.14, f"RawCode-MIL  sample {payload['sample_id']}  p={float(payload['probability']):.3f}", ha="left", va="center", fontsize=8.2, color=MUTED)
    for visual_index, chunk in enumerate(chunks):
        y = n - visual_index - 1
        weight = float(chunk["weight"])
        norm = weight / vmax
        ax.add_patch(Rectangle((0.0, y + 0.06), 1.0, 0.86, color=PURPLE, alpha=0.09 + 0.20 * norm, linewidth=0))
        ax.add_patch(Rectangle((0.135, y + 0.36), 0.120 * norm, 0.22, color=PURPLE, alpha=0.95, linewidth=0))
        ax.text(0.085, y + 0.47, str(chunk["line_range"]), ha="right", va="center", color=MUTED, family="DejaVu Sans Mono", fontsize=7.4)
        ax.text(0.282, y + 0.47, shorten(str(chunk["snippet"]), width=68, placeholder=" ..."), ha="left", va="center", color=INK, family="DejaVu Sans Mono", fontsize=7.15)
        ax.text(0.965, y + 0.47, f"w={weight:.3f}", ha="right", va="center", color=MUTED, fontsize=6.9)
    save_figure(
        fig,
        "13_rawcode_chunk_attention",
        "RawCode-MIL chunk 证据图：两个窗口权重几乎相等，说明端到端模型能给出函数级信号，但局部定位粒度明显弱于 CPG 行级证据。",
    )


def plot_attention_keyword_mix(rows: list[LineAttentionRow]) -> None:
    selected = [row for row in rows if row.rank <= 3]
    categories = ["buffer/index", "copy/API", "control", "runtime/wait", "alloc/free", "exit", "other"]
    counts = {category: 0 for category in categories}
    for row in selected:
        counts[code_category(row.source_text)] += 1
    values = np.array([counts[category] for category in categories], dtype=float)
    total = values.sum() or 1.0
    pct = values / total
    fig, ax = plt.subplots(figsize=(4.8, 2.7), constrained_layout=True)
    y = np.arange(len(categories))
    colors = [BLUE, VERMILION, GREEN, ORANGE, PURPLE, SKY, "#9CA3AF"]
    bars = ax.barh(y, pct, color=colors, height=0.62)
    ax.set_yticks(y, categories)
    ax.set_xlabel("share of top-3 evidence lines")
    ax.set_xlim(0, max(0.42, float(pct.max()) + 0.08))
    ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax.invert_yaxis()
    for bar, value, count in zip(bars, pct, values):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value * 100:.0f}% ({int(count)})", va="center", ha="left", fontsize=7.2, color=MUTED)
    save_figure(
        fig,
        "18_attention_keyword_mix",
        "Top-3 attention 行的语义类别分布：模型大量关注 buffer/index 和 copy/API 相关代码，这支持图方法确实在捕获局部程序操作信号。",
    )


def plot_outcome_category_mix(rows: list[LineAttentionRow]) -> None:
    categories = ["buffer/index", "copy/API", "control", "runtime/wait", "alloc/free", "other"]
    colors = [BLUE, VERMILION, GREEN, ORANGE, PURPLE, "#9CA3AF"]
    outcomes = ["TP", "FP", "FN", "TN"]
    matrix = np.zeros((len(outcomes), len(categories)), dtype=float)
    for row in rows:
        if row.rank > 5:
            continue
        cat = code_category(row.source_text)
        if cat == "exit":
            cat = "other"
        matrix[outcomes.index(outcome(row)), categories.index(cat)] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(5.1, 2.55), constrained_layout=True)
    left = np.zeros(len(outcomes))
    y = np.arange(len(outcomes))
    for idx, category in enumerate(categories):
        ax.barh(y, matrix[:, idx], left=left, color=colors[idx], height=0.58, label=category)
        left += matrix[:, idx]
    ax.set_yticks(y, outcomes)
    ax.set_xlim(0, 1)
    ax.set_xlabel("within-outcome share")
    ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax.invert_yaxis()
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=6.6)
    save_figure(
        fig,
        "19_attention_outcome_category_mix",
        "不同预测结果的 attention 类别组成：FP 中 runtime/wait 与资源释放类语句占比较高，可解释部分误报来自安全工程代码的风险形态相似性。",
    )


def write_index() -> None:
    lines = [
        "# Report visualization index",
        "",
        "All figures are exported as PNG, SVG, and PDF. Markdown uses PNG; paper drafts should prefer PDF or SVG.",
        "",
    ]
    for stem, caption in CAPTIONS:
        lines.append(f"- `{stem}.png` / `.svg` / `.pdf`: {caption}")
    (OUT_DIR / "figure_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    metrics = load_metrics()
    line_rows = read_line_attention(EXPLAIN_DIR / "line_attention_top.csv")

    plot_method_evolution()
    plot_metric_tradeoff(metrics)
    plot_precision_recall_balance(metrics)
    plot_ppr_prevalence(metrics)
    plot_confusion_matrix(metrics)
    plot_threshold_pathology(metrics)
    plot_score_distribution(
        METHODS[0],
        "07_score_distribution_ramp_v2a",
        "RAMP-v2A 概率分布图：正负样本分数重叠明显，解释了为什么固定阈值下 Precision、MCC 和 PPR 都需要同时观察。",
    )
    plot_training_loss(metrics)
    plot_checkpoint_guardrail(metrics)
    plot_gated_diagnostics(metrics)
    plot_slice_fusion_weights(metrics)
    plot_slice_candidate_funnel(metrics)
    plot_raw_chunk_evidence()
    plot_code_evidence(
        line_rows,
        "780_1",
        "14_attention_tp_780_1_memory_copy",
        "TP 局部证据图：模型在 780_1 中关注循环、memcpy、指针推进和 stride，符合典型内存访问风险模式。",
    )
    plot_code_evidence(
        line_rows,
        "1760_0",
        "15_attention_fp_1760_0_runtime_wait",
        "FP 局部证据图：安全样本中的等待、内存读取和资源释放语句被赋予高权重，说明模型会把工程性同步/释放逻辑误判为风险证据。",
    )
    plot_code_evidence(
        line_rows,
        "247_1",
        "16_attention_fn_247_1_state_copy",
        "FN 局部证据图：模型确实关注结构体字段复制和 memcpy，但概率未越过判别边界，说明局部证据强度与函数级标签之间仍存在弱监督落差。",
    )
    plot_code_evidence(
        line_rows,
        "6260_0",
        "17_attention_tn_6260_0_range_check",
        "TN 局部证据图：低风险样本主要关注函数签名和边界检查返回语句，模型给出低概率，有助于解释安全样本的判别模式。",
    )
    plot_attention_keyword_mix(line_rows)
    plot_outcome_category_mix(line_rows)
    write_index()
    print(f"wrote {len(CAPTIONS)} report figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
