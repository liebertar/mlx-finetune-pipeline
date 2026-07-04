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
