import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results/generated/publication_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(OUTPUT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np

models = ["LSTM", "XGBoost", "Transformer"]
y = np.array([2, 1, 0])

rmse = np.array([0.4411, 0.4239, 0.4430])
r2   = np.array([0.8591, 0.8699, 0.8579])
mape = np.array([0.7550, 0.7235, 0.7651])

panels = [
    {
        "values": rmse,
        "xlabel": "Test RMSE ↓",
        "xlim": (0.418, 0.4465),
        "xticks": [0.42, 0.43, 0.44],
        "best_idx": int(np.argmin(rmse)),
        "subtitle": "(a)"
    },
    {
        "values": r2,
        "xlabel": r"Test $R^2$ ↑",
        "xlim": (0.855, 0.8712),
        "xticks": [0.855, 0.860, 0.865, 0.870],
        "best_idx": int(np.argmax(r2)),
        "subtitle": "(b)"
    },
    {
        "values": mape,
        "xlabel": "Test MAPE (%) ↓",
        "xlim": (0.715, 0.7685),
        "xticks": [0.72, 0.74, 0.76],
        "best_idx": int(np.argmin(mape)),
        "subtitle": "(c)"
    },
]

try:
    font_manager.findfont("Arial", fallback_to_default=False)
    font_name = "Arial"
except Exception:
    try:
        font_manager.findfont("Arimo", fallback_to_default=False)
        font_name = "Arimo"
    except Exception:
        font_name = "Liberation Sans"

rcParams.update({
    "font.family": font_name,
    "font.size": 11,
    "axes.labelsize": 14,
    "xtick.labelsize": 14.0,
    "ytick.labelsize": 14.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

point_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:3]
guide_color = "#C8D0D8"
band_color = "#90AFC8"
grid_color = "#E2E7EC"
text_color = "#1F252B"

def draw_panel(ax, cfg):
    vals = cfg["values"]
    xlim = cfg["xlim"]
    best_idx = cfg["best_idx"]

    ax.set_xlim(*xlim)
    ax.set_xticks(cfg["xticks"])
    ax.set_ylim(-0.22, 2.50)
    ax.set_xlabel(cfg["xlabel"])

    best_y = y[best_idx]
    ax.axhspan(best_y - 0.34, best_y + 0.34, color=band_color, alpha=0.12, zorder=0)

    for yi, val in zip(y, vals):
        ax.hlines(yi, xlim[0], val, color=guide_color, linewidth=0.85, zorder=1)

    for i, (yi, val) in enumerate(zip(y, vals)):
        c = point_colors[i]
        ax.scatter(val, yi, s=42, color=c, edgecolor="#3B424A", linewidth=0.5, zorder=3)

        if i == best_idx:
            ax.scatter(val, yi, s=105, facecolors="none", edgecolors=c, linewidth=1.3, zorder=4)

        ax.annotate(
            f"{val:.4f}",
            xy=(val, yi),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14.0,
            fontweight="bold",
            color=text_color,
            annotation_clip=False,
            zorder=5
        )

    ax.grid(axis="x", color=grid_color, linewidth=0.8)
    ax.grid(axis="y", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3.6)

    ax.text(0.02, 0.96, cfg["subtitle"], transform=ax.transAxes,
            ha="left", va="top", fontsize=13.5, fontweight="bold")

fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), dpi=180, sharey=True)
fig.patch.set_facecolor("white")

for ax, cfg in zip(axes, panels):
    draw_panel(ax, cfg)

axes[0].set_yticks(y)
axes[0].set_yticklabels(models)
for ax in axes[1:]:
    ax.tick_params(axis="y", left=False, labelleft=False)

fig.subplots_adjust(left=0.11, right=0.99, top=0.93, bottom=0.20, wspace=0.28)
for ext in ("png", "pdf"):
    fig.savefig(OUTPUT_DIR / f"fig09_source_model_comparison.{ext}", dpi=400, facecolor="white")
plt.close(fig)
