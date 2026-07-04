"""Loss-curve and sweep-comparison charts (PNG) for model cards.

Colors are the validated categorical palette (CVD-safe order); sub-3:1 slots
carry direct labels per the relief rule.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#1baf7a", "#eda100"]  # blue, aqua, yellow
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def _style_axes(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_xlabel("iteration", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("loss", color=INK_MUTED, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)


def _label_line_end(ax, xs, ys, text, color):
    ax.annotate(
        text, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
        color=color, fontsize=9, va="center",
    )


def plot_loss_curve(history: dict, out_path: str, title: str = "Training loss"):
    """history: {"train": [[iter, loss], ...], "val": [[iter, loss], ...]}"""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax, title)
    for i, (name, points) in enumerate((("train", history["train"]), ("val", history["val"]))):
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, label=name,
                marker="o" if len(xs) < 15 else None, markersize=4)
        _label_line_end(ax, xs, ys, name, SERIES[i])
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pii_comparison(tuned: dict, baseline: dict, out_path: str):
    """Horizontal grouped bars: per-type F1, tuned vs zero-shot base model.
    tuned/baseline: {type: {"f1": float, ...}} keyed by Korean entity type."""
    plt.rcParams["font.family"] = ["AppleGothic", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    types = list(tuned.keys())
    y = range(len(types))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(types) + 1.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh([i + height / 2 for i in y], [tuned[t]["f1"] for t in types],
            height=height, color=SERIES[0], label="파인튜닝")
    ax.barh([i - height / 2 for i in y], [baseline.get(t, {}).get("f1", 0.0) for t in types],
            height=height, color=INK_MUTED, label="베이스 (zero-shot)")
    ax.set_yticks(list(y), types, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1", color=INK_MUTED, fontsize=10)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_title("유형별 F1 — 파인튜닝 vs 베이스", color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_try_progression(rows: list[dict], out_path: str):
    """Precision/recall/F1 across experiment tries (try order, not wall time).
    rows: [{"try": 1, "label": "r16-lr1e-4-b4-n3000", "precision": .., "recall": .., "f1": ..}]"""
    plt.rcParams["font.family"] = ["AppleGothic", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax, "Try별 성능 추이 (골드셋 고정)")
    ax.set_xlabel("try", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("score", color=INK_MUTED, fontsize=10)
    xs = [r["try"] for r in rows]
    for i, key in enumerate(("f1", "precision", "recall")):
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=5, label=key)
        _label_line_end(ax, xs, ys, key, SERIES[i])
    ax.set_xticks(xs, [f"#{r['try']}\n{r.get('label', '')}" for r in rows], fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sweep_comparison(runs: list[dict], out_path: str):
    """runs: [{"label": "r8 lr1e-5", "val": [[iter, loss], ...]}, ...] — val-loss
    curves of each hyperparameter combination on one axis."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax, "Validation loss by hyperparameter combination")
    for i, run in enumerate(runs):
        if not run["val"]:
            continue
        xs, ys = zip(*run["val"])
        color = SERIES[i % len(SERIES)]
        ax.plot(xs, ys, color=color, linewidth=2, label=run["label"],
                marker="o" if len(xs) < 15 else None, markersize=4)
        _label_line_end(ax, xs, ys, run["label"], color)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path
