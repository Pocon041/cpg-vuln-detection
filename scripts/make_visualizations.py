from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path("figures")

BG0 = "#070b16"
BG1 = "#0b1426"
BG2 = "#102033"
INK = "#edf6ff"
MUTED = "#95a9c6"
GRID = "#294160"
CYAN = "#22d3ee"
BLUE = "#60a5fa"
GREEN = "#34d399"
ORANGE = "#fb923c"
ROSE = "#f472b6"
VIOLET = "#a78bfa"
YELLOW = "#fbbf24"
RED = "#f87171"

METRICS = [
    {
        "name": "RAMP-v2A",
        "family": "Graph",
        "precision": 0.5399,
        "recall": 0.7468,
        "f1": 0.6267,
        "mcc": 0.1841,
        "ppr": 0.6544,
        "roc": 0.6233,
        "pr_auc": 0.6079,
        "color": CYAN,
    },
    {
        "name": "RAMP-v2Dual",
        "family": "Graph",
        "precision": 0.5400,
        "recall": 0.7240,
        "f1": 0.6186,
        "mcc": 0.1763,
        "ppr": 0.6344,
        "roc": 0.6163,
        "pr_auc": 0.5975,
        "color": BLUE,
    },
    {
        "name": "RAMP-v2Gated",
        "family": "Graph",
        "precision": 0.5218,
        "recall": 0.7760,
        "f1": 0.6240,
        "mcc": 0.1503,
        "ppr": 0.7035,
        "roc": 0.6150,
        "pr_auc": 0.5928,
        "color": ORANGE,
    },
    {
        "name": "RAMP-v3 Slice-MIL",
        "family": "Graph MIL",
        "precision": 0.5379,
        "recall": 0.7370,
        "f1": 0.6219,
        "mcc": 0.1762,
        "ppr": 0.6482,
        "roc": 0.6164,
        "pr_auc": 0.5948,
        "color": VIOLET,
    },
    {
        "name": "RawCode-MIL",
        "family": "End-to-end",
        "precision": 0.5172,
        "recall": 0.7282,
        "f1": 0.6048,
        "mcc": 0.1228,
        "ppr": 0.6672,
        "roc": 0.6165,
        "pr_auc": 0.6004,
        "color": ROSE,
    },
]


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": BG0,
        "axes.facecolor": "none",
        "savefig.facecolor": BG0,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)


def background(ax, seed: int = 1) -> None:
    x = np.linspace(0, 1, 512)
    y = np.linspace(0, 1, 512)
    xx, yy = np.meshgrid(x, y)
    z = 0.58 * xx + 0.42 * (1 - yy)
    cmap = LinearSegmentedColormap.from_list("paper_bg", [BG0, BG1, BG2])
    ax.imshow(z, extent=(0, 1, 0, 1), origin="lower", cmap=cmap, zorder=-20, aspect="auto")

    rng = np.random.default_rng(seed)
    px = rng.random(260)
    py = rng.random(260)
    sizes = rng.uniform(1.0, 6.0, 260)
    ax.scatter(px, py, s=sizes, color="#dbeafe", alpha=0.055, linewidths=0, zorder=-15)

    for yv in np.linspace(0.12, 0.88, 7):
        ax.plot([0.04, 0.96], [yv, yv], color=GRID, lw=0.45, alpha=0.14, zorder=-14)
    for xv in np.linspace(0.08, 0.92, 8):
        ax.plot([xv, xv], [0.08, 0.92], color=GRID, lw=0.45, alpha=0.11, zorder=-14)


def canvas(width: float = 14.4, height: float = 8.1, seed: int = 1):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    background(ax, seed=seed)
    return fig, ax


def glow_line(ax, x, y, color, lw=2.0, alpha=0.95, zorder=3):
    ax.plot(x, y, color=color, lw=lw + 5, alpha=0.08, solid_capstyle="round", zorder=zorder - 1)
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=zorder)


def arrow(ax, start, end, color=CYAN, rad=0.0, lw=2.2, alpha=0.92):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=7,
    )
    patch.set_path_effects([pe.Stroke(linewidth=lw + 5, foreground=color, alpha=0.12), pe.Normal()])
    ax.add_patch(patch)
    return patch


def box(
    ax,
    x,
    y,
    w,
    h,
    text,
    edge=CYAN,
    face="#0b1223",
    size=12,
    weight="semibold",
    alpha=0.92,
    radius=0.026,
    text_color=INK,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
        alpha=alpha,
        zorder=4,
    )
    patch.set_path_effects([pe.Stroke(linewidth=5.0, foreground=edge, alpha=0.12), pe.Normal()])
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=size,
        weight=weight,
        linespacing=1.18,
        zorder=8,
        path_effects=[pe.Stroke(linewidth=2.2, foreground=BG0, alpha=0.72), pe.Normal()],
    )
    return patch


def label(ax, x, y, text, color=MUTED, size=10, ha="center", va="center", weight="medium"):
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=size,
        ha=ha,
        va=va,
        weight=weight,
        zorder=9,
        path_effects=[pe.Stroke(linewidth=2.0, foreground=BG0, alpha=0.70), pe.Normal()],
    )


def metric_strip(ax, x, y, w, h, m, accent):
    items = [
        ("Precision", m["precision"]),
        ("Recall", m["recall"]),
        ("F1", m["f1"]),
        ("MCC", m["mcc"]),
    ]
    gap = 0.012
    cw = (w - gap * (len(items) - 1)) / len(items)
    for i, (name, val) in enumerate(items):
        cx = x + i * (cw + gap)
        patch = FancyBboxPatch(
            (cx, y),
            cw,
            h,
            boxstyle="round,pad=0.009,rounding_size=0.018",
            edgecolor=accent,
            facecolor="#091223",
            linewidth=1.15,
            alpha=0.86,
            zorder=5,
        )
        patch.set_path_effects([pe.Stroke(linewidth=4, foreground=accent, alpha=0.08), pe.Normal()])
        ax.add_patch(patch)
        label(ax, cx + cw * 0.5, y + h * 0.65, f"{val:.3f}", color=INK, size=13, weight="bold")
        label(ax, cx + cw * 0.5, y + h * 0.28, name, color=MUTED, size=8.5)


def draw_code_block(ax, x, y, w, h, accent=CYAN, lines=None):
    lines = lines or [
        "if (size > len) return;",
        "buf[i] = src[i];",
        "sink(buf);",
    ]
    box(ax, x, y, w, h, "", edge=accent, face="#08101f", alpha=0.86, radius=0.022)
    y0 = y + h - 0.035
    for i, line in enumerate(lines):
        color = INK if i % 2 == 0 else MUTED
        label(ax, x + 0.022, y0 - i * 0.038, line, color=color, size=8.6, ha="left", weight="regular")
    for i in range(4):
        glow_line(
            ax,
            [x + 0.020, x + w * (0.62 + 0.07 * (i % 2))],
            [y + 0.030 + i * 0.030, y + 0.030 + i * 0.030],
            accent,
            lw=1.05,
            alpha=0.22,
            zorder=6,
        )


def draw_cpg(ax, cx, cy, scale=1.0, show_legend=True):
    nodes = np.array(
        [
            [-0.22, 0.12],
            [-0.08, 0.26],
            [0.11, 0.20],
            [0.24, 0.05],
            [0.08, -0.14],
            [-0.13, -0.14],
            [-0.30, -0.04],
        ]
    )
    nodes = np.column_stack([cx + nodes[:, 0] * scale, cy + nodes[:, 1] * scale])
    edges = [
        (0, 1, CYAN, "AST"),
        (1, 2, BLUE, "CFG"),
        (2, 3, BLUE, "CFG"),
        (1, 5, VIOLET, "CDG"),
        (5, 4, GREEN, "DDG"),
        (4, 3, GREEN, "DDG"),
        (6, 0, CYAN, "AST"),
        (6, 5, VIOLET, "CDG"),
        (0, 5, ORANGE, "CALL"),
    ]
    for a, b, color, _ in edges:
        glow_line(ax, [nodes[a, 0], nodes[b, 0]], [nodes[a, 1], nodes[b, 1]], color, lw=1.7, alpha=0.66, zorder=3)
    for i, (x, y) in enumerate(nodes):
        c = Circle((x, y), 0.025 * scale, facecolor="#0e1b30", edgecolor=INK, linewidth=1.1, alpha=0.98, zorder=6)
        c.set_path_effects([pe.Stroke(linewidth=5, foreground=CYAN if i in (1, 2) else BLUE, alpha=0.12), pe.Normal()])
        ax.add_patch(c)
    if show_legend:
        label(ax, cx - 0.285 * scale, cy + 0.315 * scale, "AST", color=CYAN, size=8.0, ha="left")
        label(ax, cx - 0.185 * scale, cy + 0.315 * scale, "CFG", color=BLUE, size=8.0, ha="left")
        label(ax, cx - 0.085 * scale, cy + 0.315 * scale, "CDG", color=VIOLET, size=8.0, ha="left")
        label(ax, cx + 0.015 * scale, cy + 0.315 * scale, "DDG", color=GREEN, size=8.0, ha="left")


def draw_windows(ax, x, y, w, h, accent=ROSE, n=5):
    for i in range(n):
        ox = x + i * (w * 0.105)
        oy = y + i * (h * 0.055)
        rect = FancyBboxPatch(
            (ox, oy),
            w * 0.60,
            h * 0.46,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            edgecolor=accent if i == n - 1 else BLUE,
            facecolor="#0b1426",
            linewidth=1.1,
            alpha=0.88,
            zorder=5 + i,
        )
        rect.set_path_effects([pe.Stroke(linewidth=4, foreground=accent, alpha=0.08), pe.Normal()])
        ax.add_patch(rect)
        for j in range(4):
            glow_line(
                ax,
                [ox + 0.016, ox + w * (0.33 + 0.05 * ((i + j) % 2))],
                [oy + h * (0.10 + j * 0.07), oy + h * (0.10 + j * 0.07)],
                accent if j == 2 else MUTED,
                lw=0.9,
                alpha=0.32,
                zorder=7 + i,
            )


def draw_attention(ax, x, y, w, h, accent=ROSE):
    weights = [0.11, 0.18, 0.47, 0.29, 0.76, 0.22]
    for i, val in enumerate(weights):
        yy = y + h - (i + 1) * h / 7.3
        box_h = h * 0.085
        rect = FancyBboxPatch(
            (x, yy),
            w,
            box_h,
            boxstyle="round,pad=0.006,rounding_size=0.018",
            edgecolor=accent,
            facecolor="#0c1526",
            linewidth=0.9,
            alpha=0.64,
            zorder=4,
        )
        ax.add_patch(rect)
        glow_line(ax, [x + 0.014, x + 0.014 + (w - 0.028) * val], [yy + box_h / 2, yy + box_h / 2], accent, lw=4.0, alpha=0.82, zorder=7)
    label(ax, x + w * 0.5, y + 0.035, "gated MIL weights", color=MUTED, size=8.8)


def save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=320, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def graph_learning_story():
    fig, ax = canvas(seed=11)
    draw_code_block(
        ax,
        0.055,
        0.075,
        0.255,
        0.205,
        accent=CYAN,
        lines=["char buf[64];", "copy(buf, input);", "if (auth) sink(buf);"],
    )
    label(ax, 0.182, 0.315, "source function", color=MUTED, size=9.2)
    arrow(ax, (0.225, 0.295), (0.285, 0.420), color=CYAN, rad=-0.12)

    frame = FancyBboxPatch(
        (0.055, 0.365),
        0.395,
        0.505,
        boxstyle="round,pad=0.012,rounding_size=0.030",
        edgecolor="#24445e",
        facecolor="#08111f",
        linewidth=1.0,
        alpha=0.55,
        zorder=1,
    )
    ax.add_patch(frame)
    draw_cpg(ax, 0.265, 0.620, scale=0.72, show_legend=False)
    label(ax, 0.083, 0.830, "explicit program dependencies", color=INK, size=11, ha="left", weight="semibold")
    label(ax, 0.083, 0.790, "structure helps; noisy edges must be controlled", color=MUTED, size=9, ha="left")
    label(ax, 0.083, 0.748, "AST    CFG    CDG    DDG", color=MUTED, size=8.4, ha="left")

    box(ax, 0.525, 0.700, 0.145, 0.115, "node text\nnormalization", edge=GREEN, size=10.5)
    box(ax, 0.745, 0.700, 0.150, 0.115, "CodeBERT\nnode vectors", edge=BLUE, size=10.5)
    box(ax, 0.525, 0.485, 0.145, 0.115, "relation-aware\nRGNN", edge=CYAN, size=10.5)
    box(ax, 0.745, 0.485, 0.150, 0.115, "semantic\nfusion", edge=VIOLET, size=10.5)
    box(ax, 0.635, 0.300, 0.150, 0.100, "fixed 0.5\nclassifier", edge=YELLOW, size=10.5)

    arrow(ax, (0.450, 0.620), (0.525, 0.755), color=GREEN, rad=0.10)
    arrow(ax, (0.670, 0.757), (0.745, 0.757), color=BLUE)
    arrow(ax, (0.598, 0.700), (0.598, 0.600), color=CYAN)
    arrow(ax, (0.670, 0.542), (0.745, 0.542), color=VIOLET)
    arrow(ax, (0.820, 0.485), (0.758, 0.400), color=YELLOW, rad=-0.12)
    arrow(ax, (0.600, 0.485), (0.672, 0.400), color=YELLOW, rad=0.12)

    best = METRICS[0]
    metric_strip(ax, 0.525, 0.095, 0.370, 0.125, best, CYAN)
    label(ax, 0.710, 0.250, "best balanced graph result", color=CYAN, size=10.5, weight="semibold")
    label(ax, 0.935, 0.108, "PPR\n0.654", color=YELLOW, size=10.5, weight="bold")
    save(fig, "viz_graph_learning_story")


def end_to_end_story():
    fig, ax = canvas(seed=17)
    draw_code_block(
        ax,
        0.055,
        0.560,
        0.265,
        0.245,
        accent=ROSE,
        lines=["int parse(char *s) {", "  len = strlen(s);", "  memcpy(dst, s, len);", "  return len;"],
    )
    label(ax, 0.187, 0.512, "raw source only", color=INK, size=11, weight="semibold")
    label(ax, 0.187, 0.477, "no AST / CFG / PDG conversion", color=MUTED, size=9.0)

    draw_windows(ax, 0.390, 0.565, 0.260, 0.270, accent=ROSE, n=6)
    box(ax, 0.405, 0.255, 0.200, 0.105, "CodeBERT\nchunk encoder", edge=BLUE, size=10.5)
    box(ax, 0.698, 0.555, 0.205, 0.105, "gated attention\nMIL pooling", edge=ROSE, size=10.5)
    draw_attention(ax, 0.715, 0.250, 0.165, 0.240, accent=ROSE)
    box(ax, 0.733, 0.090, 0.130, 0.090, "fixed 0.5\nclassifier", edge=YELLOW, size=10.0)

    arrow(ax, (0.325, 0.680), (0.390, 0.690), color=ROSE)
    arrow(ax, (0.510, 0.565), (0.505, 0.360), color=BLUE, rad=0.05)
    arrow(ax, (0.605, 0.310), (0.700, 0.430), color=VIOLET, rad=-0.10)
    arrow(ax, (0.795, 0.250), (0.795, 0.180), color=YELLOW)

    raw = METRICS[-1]
    metric_strip(ax, 0.070, 0.090, 0.430, 0.130, raw, ROSE)
    label(ax, 0.286, 0.248, "end-to-end source baseline", color=ROSE, size=10.5, weight="semibold")
    label(ax, 0.642, 0.865, "long function -> overlapping windows", color=INK, size=10.6, ha="center", weight="semibold")
    label(ax, 0.642, 0.825, "MIL learns which chunks carry vulnerability evidence", color=MUTED, size=9.0, ha="center")
    save(fig, "viz_end2end_raw_mil_story")


def dual_path_motivation():
    fig, ax = canvas(seed=23)
    box(ax, 0.045, 0.415, 0.145, 0.170, "same raw\nfunction", edge=YELLOW, size=12)

    box(ax, 0.270, 0.690, 0.145, 0.105, "CPG\nextraction", edge=CYAN, size=10.5)
    box(ax, 0.485, 0.690, 0.145, 0.105, "typed relation\nmessage passing", edge=BLUE, size=10.5)
    box(ax, 0.700, 0.690, 0.145, 0.105, "graph-semantic\nfusion", edge=VIOLET, size=10.5)
    box(ax, 0.875, 0.690, 0.080, 0.105, "0.5\nscore", edge=YELLOW, size=10.5)

    box(ax, 0.270, 0.205, 0.145, 0.105, "token\nwindows", edge=ROSE, size=10.5)
    box(ax, 0.485, 0.205, 0.145, 0.105, "CodeBERT\nchunks", edge=BLUE, size=10.5)
    box(ax, 0.700, 0.205, 0.145, 0.105, "gated MIL\naggregation", edge=ROSE, size=10.5)
    box(ax, 0.875, 0.205, 0.080, 0.105, "0.5\nscore", edge=YELLOW, size=10.5)

    arrow(ax, (0.190, 0.535), (0.270, 0.742), color=CYAN, rad=0.13)
    arrow(ax, (0.190, 0.465), (0.270, 0.257), color=ROSE, rad=-0.13)
    for xs, y, color in [(0.415, 0.742, CYAN), (0.630, 0.742, BLUE), (0.845, 0.742, VIOLET)]:
        arrow(ax, (xs, y), (xs + 0.070, y), color=color)
    for xs, y, color in [(0.415, 0.257, ROSE), (0.630, 0.257, BLUE), (0.845, 0.257, ROSE)]:
        arrow(ax, (xs, y), (xs + 0.070, y), color=color)

    draw_cpg(ax, 0.345, 0.525, scale=0.28, show_legend=False)
    draw_windows(ax, 0.320, 0.400, 0.145, 0.135, accent=ROSE, n=4)

    label(ax, 0.270, 0.845, "graph learning route", color=CYAN, size=11, ha="left", weight="semibold")
    label(ax, 0.500, 0.636, "best F1 0.627 / MCC 0.184", color=INK, size=9.5, ha="left")
    label(ax, 0.500, 0.604, "models explicit dependencies; extraction noise remains visible", color=MUTED, size=8.7, ha="left")

    label(ax, 0.270, 0.365, "end-to-end route", color=ROSE, size=11, ha="left", weight="semibold")
    label(ax, 0.270, 0.160, "F1 0.605 / MCC 0.123", color=INK, size=9.5, ha="left")
    label(ax, 0.270, 0.128, "removes IR dependency; chunk evidence is learned directly", color=MUTED, size=8.7, ha="left")

    glow_line(ax, [0.225, 0.225], [0.125, 0.875], YELLOW, lw=1.4, alpha=0.38, zorder=2)
    label(ax, 0.055, 0.350, "fixed-threshold evaluation\nprevents threshold tuning gains", color=MUTED, size=8.6, ha="left")
    save(fig, "viz_dual_path_motivation")


def metric_constellation():
    fig = plt.figure(figsize=(12.8, 7.6), facecolor=BG0)
    ax = fig.add_subplot(111)
    ax.set_facecolor("none")
    xmin, xmax = 0.112, 0.193
    ymin, ymax = 0.598, 0.631
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    x = np.linspace(0, 1, 512)
    y = np.linspace(0, 1, 512)
    xx, yy = np.meshgrid(x, y)
    z = 0.54 * xx + 0.46 * (1 - yy)
    cmap = LinearSegmentedColormap.from_list("const_bg", [BG0, BG1, BG2])
    ax.imshow(z, extent=(xmin, xmax, ymin, ymax), origin="lower", cmap=cmap, zorder=-20, aspect="auto")

    for spine in ax.spines.values():
        spine.set_color("#30465f")
        spine.set_alpha(0.45)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, alpha=0.24, linewidth=0.8)
    ax.set_xlabel("MCC", color=INK, fontsize=11, labelpad=10, weight="semibold")
    ax.set_ylabel("F1", color=INK, fontsize=11, labelpad=10, weight="semibold")

    ax.axhline(METRICS[0]["f1"], color=CYAN, alpha=0.12, lw=1.2)
    ax.axvline(METRICS[0]["mcc"], color=CYAN, alpha=0.12, lw=1.2)
    ax.fill_between([xmin, xmax], 0.620, ymax, color=CYAN, alpha=0.035, zorder=-8)

    for m in METRICS:
        size = 850 + (m["ppr"] - 0.60) * 5200
        ax.scatter([m["mcc"]], [m["f1"]], s=size * 1.65, color=m["color"], alpha=0.11, linewidths=0, zorder=2)
        ax.scatter(
            [m["mcc"]],
            [m["f1"]],
            s=size,
            color=m["color"],
            alpha=0.78,
            linewidths=1.6,
            edgecolors=INK,
            zorder=5,
        )
        dx = 0.0022 if m["name"] != "RawCode-MIL" else 0.0028
        dy = 0.0010
        ax.text(
            m["mcc"] + dx,
            m["f1"] + dy,
            f"{m['name']}\nF1 {m['f1']:.3f}  PPR {m['ppr']:.3f}",
            color=INK,
            fontsize=8.8,
            weight="semibold",
            ha="left",
            va="bottom",
            linespacing=1.18,
            path_effects=[pe.Stroke(linewidth=2.5, foreground=BG0, alpha=0.85), pe.Normal()],
            zorder=7,
        )

    ax.annotate(
        "",
        xy=(0.190, 0.629),
        xytext=(0.170, 0.618),
        arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=2.0, alpha=0.85),
        zorder=8,
    )
    ax.text(
        0.171,
        0.6165,
        "higher F1 + higher MCC",
        color=YELLOW,
        fontsize=9.0,
        weight="semibold",
        path_effects=[pe.Stroke(linewidth=2.2, foreground=BG0, alpha=0.85), pe.Normal()],
    )

    ax.text(
        0.115,
        0.6286,
        "fixed threshold = 0.5",
        color=INK,
        fontsize=10.0,
        weight="semibold",
        bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.20", fc="#091223", ec=YELLOW, alpha=0.82),
        zorder=9,
    )
    ax.text(
        0.115,
        0.6259,
        "bubble area tracks predicted-positive rate",
        color=MUTED,
        fontsize=8.5,
        zorder=9,
    )
    save(fig, "viz_metric_constellation")


def fixed05_scoreboard():
    fig, ax = canvas(width=13.8, height=8.0, seed=31)
    label(ax, 0.070, 0.895, "method", color=MUTED, size=9.0, ha="left")
    label(ax, 0.360, 0.895, "F1", color=MUTED, size=9.0, ha="left")
    label(ax, 0.660, 0.895, "MCC", color=MUTED, size=9.0, ha="left")
    label(ax, 0.832, 0.895, "precision / recall / PPR", color=MUTED, size=9.0, ha="left")

    f1_min, f1_max = 0.598, 0.630
    mcc_min, mcc_max = 0.110, 0.190
    rows_y = [0.775, 0.635, 0.495, 0.355, 0.215]
    for idx, (m, y) in enumerate(zip(METRICS, rows_y)):
        h = 0.095
        face = "#0b1426" if idx == 0 else "#091223"
        edge = m["color"] if idx == 0 else "#203852"
        row = FancyBboxPatch(
            (0.050, y - h / 2),
            0.900,
            h,
            boxstyle="round,pad=0.011,rounding_size=0.024",
            edgecolor=edge,
            facecolor=face,
            linewidth=1.2,
            alpha=0.82,
            zorder=2,
        )
        row.set_path_effects([pe.Stroke(linewidth=5, foreground=m["color"], alpha=0.08 if idx else 0.16), pe.Normal()])
        ax.add_patch(row)

        family_color = CYAN if "Graph" in m["family"] else ROSE
        icon = Circle((0.083, y), 0.019, facecolor="#0d1a2d", edgecolor=family_color, linewidth=1.4, zorder=6)
        ax.add_patch(icon)
        if "End" in m["family"]:
            for k in range(3):
                glow_line(ax, [0.073, 0.093], [y - 0.009 + k * 0.009, y - 0.009 + k * 0.009], ROSE, lw=0.8, alpha=0.55, zorder=7)
        else:
            mini = [(0.078, y + 0.006), (0.088, y + 0.011), (0.091, y - 0.006), (0.074, y - 0.010)]
            for a, b in [(0, 1), (1, 2), (0, 3), (3, 2)]:
                glow_line(ax, [mini[a][0], mini[b][0]], [mini[a][1], mini[b][1]], family_color, lw=0.7, alpha=0.55, zorder=7)
            for px, py in mini:
                ax.add_patch(Circle((px, py), 0.0034, facecolor=INK, edgecolor="none", zorder=8))

        label(ax, 0.115, y + 0.017, m["name"], color=INK, size=11.0, ha="left", weight="semibold")
        label(ax, 0.115, y - 0.020, m["family"], color=MUTED, size=8.5, ha="left")

        track_x, track_w = 0.355, 0.235
        ax.add_patch(Rectangle((track_x, y - 0.012), track_w, 0.024, facecolor="#12233a", edgecolor="none", alpha=0.90, zorder=4))
        fill = np.clip((m["f1"] - f1_min) / (f1_max - f1_min), 0, 1) * track_w
        glow_line(ax, [track_x, track_x + fill], [y, y], m["color"], lw=9.0, alpha=0.92, zorder=6)
        label(ax, track_x + track_w + 0.018, y, f"{m['f1']:.3f}", color=INK, size=10.5, ha="left", weight="bold")

        mcc_x0, mcc_w = 0.645, 0.130
        glow_line(ax, [mcc_x0, mcc_x0 + mcc_w], [y, y], "#314962", lw=2.0, alpha=0.55, zorder=4)
        pos = mcc_x0 + np.clip((m["mcc"] - mcc_min) / (mcc_max - mcc_min), 0, 1) * mcc_w
        ax.scatter([pos], [y], s=115, color=m["color"], edgecolor=INK, linewidth=1.0, zorder=7)
        label(ax, pos, y - 0.034, f"{m['mcc']:.3f}", color=INK, size=8.4)

        pr_text = f"P {m['precision']:.3f}  R {m['recall']:.3f}  PPR {m['ppr']:.3f}"
        color = RED if m["ppr"] > 0.69 else (YELLOW if m["ppr"] > 0.66 else GREEN)
        label(ax, 0.825, y, pr_text, color=color, size=9.2, ha="left", weight="semibold")

    label(ax, 0.050, 0.095, "all values use threshold 0.5; high recall alone is not treated as success", color=MUTED, size=9.0, ha="left")
    save(fig, "viz_fixed05_scoreboard")


def main():
    graph_learning_story()
    end_to_end_story()
    dual_path_motivation()
    metric_constellation()
    fixed05_scoreboard()
    print(f"wrote visualizations to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
