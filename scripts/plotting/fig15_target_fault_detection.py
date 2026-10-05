from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results/generated/publication_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(OUTPUT_DIR)
DATA_DIR = ROOT / "results/reference/figure_data"

# Same as previous version, except subplot 2 (precision) Max gain box is moved lower.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

summary = pd.read_csv(DATA_DIR / "fig15_target_fault_detection_summary.csv").sort_values(
    ['transfer_learning', 'target_training_data_pct_plot_axis']
)

tl = summary[summary['transfer_learning'] == 1].sort_values('target_training_data_pct_plot_axis')
to = summary[summary['transfer_learning'] == 0].sort_values('target_training_data_pct_plot_axis')
x = tl['target_training_data_pct_plot_axis'].to_numpy()

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
    "font.size": 13.5,
    "axes.labelsize": 17.0,
    "xtick.labelsize": 13.8,
    "ytick.labelsize": 13.8,
    "legend.fontsize": 14.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

color_tl = "#2563EB"
color_to = "#F28E2B"
green_gain = "#22C55E"
red_loss = "#EF4444"
grid_color = "#DDE5EE"
border_color = "#A8B4C0"
text_color = "#1F2937"

metrics = [
    ("accuracy", "Accuracy", (0.50, 1.02)),
    ("precision", "Precision", (0.20, 1.02)),
    ("recall", "Recall", (0.50, 1.02)),
    ("f1", r"$F_1$ score", (0.20, 1.02)),
]

def smart_label_positions(y_tl, y_to, ylim):
    yr = ylim[1] - ylim[0]
    offset = 0.024 * yr
    close_extra = 0.018 * yr

    tl_pos, tl_va = [], []
    to_pos, to_va = [], []

    for yt, yo in zip(y_tl, y_to):
        gap = abs(yt - yo)
        extra = close_extra if gap < 0.07 * yr else 0.0

        if yt > ylim[1] - 0.09 * yr:
            tl_y = yt - offset - extra
            tl_align = "top"
        else:
            tl_y = yt + offset + extra
            tl_align = "bottom"

        if yo < ylim[0] + 0.09 * yr:
            to_y = yo + offset + extra
            to_align = "bottom"
        else:
            to_y = yo - offset - extra
            to_align = "top"

        tl_y = min(max(tl_y, ylim[0] + 0.035 * yr), ylim[1] - 0.035 * yr)
        to_y = min(max(to_y, ylim[0] + 0.035 * yr), ylim[1] - 0.035 * yr)

        tl_pos.append(tl_y); tl_va.append(tl_align)
        to_pos.append(to_y); to_va.append(to_align)

    return np.array(tl_pos), tl_va, np.array(to_pos), to_va

def draw_value_labels(ax, xs, y_tl, y_to, ylim):
    tl_pos, tl_va, to_pos, to_va = smart_label_positions(y_tl, y_to, ylim)

    for xx, yt, yo, p1, a1, p2, a2 in zip(xs, y_tl, y_to, tl_pos, tl_va, to_pos, to_va):
        ax.text(
            xx, p1, f"{yt:.3f}",
            ha="center", va=a1,
            fontsize=11.1, fontweight="bold", color=color_tl,
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white",
                      edgecolor="none", alpha=0.68),
            zorder=8
        )
        ax.text(
            xx, p2, f"{yo:.3f}",
            ha="center", va=a2,
            fontsize=11.1, fontweight="bold", color=color_to,
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white",
                      edgecolor="none", alpha=0.68),
            zorder=8
        )
    return tl_pos, to_pos

def pick_max_gain_box_position(metric_key, y_tl, y_to, tl_pos, to_pos, ylim):
    yr = ylim[1] - ylim[0]

    if metric_key == "precision":
        return 0.255

    if metric_key == "f1":
        left_n = min(4, len(y_tl))
        local_floor = min(
            np.min(y_tl[:left_n]),
            np.min(y_to[:left_n]),
            np.min(tl_pos[:left_n]),
            np.min(to_pos[:left_n]),
        )
        y_box = local_floor - 0.085 * yr
        y_box = max(y_box, ylim[0] + 0.14 * yr)
        return y_box

    left_n = min(3, len(y_tl))
    obstacles = np.r_[y_tl[:left_n], y_to[:left_n], tl_pos[:left_n], to_pos[:left_n]]
    candidates = np.linspace(ylim[0] + 0.30 * yr, ylim[0] + 0.56 * yr, 12)
    needed_clearance = 0.045 * yr
    chosen = candidates[0]

    for cand in candidates:
        if np.all(np.abs(cand - obstacles) > needed_clearance):
            chosen = cand
            break
    return chosen

fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.6), dpi=180)
fig.patch.set_facecolor("white")
axes = axes.ravel()

for ax, (metric_key, ylabel, ylim) in zip(axes, metrics):
    mean_tl = tl[f'{metric_key}_mean'].to_numpy()
    std_tl = tl[f'{metric_key}_std'].to_numpy()
    mean_to = to[f'{metric_key}_mean'].to_numpy()
    std_to = to[f'{metric_key}_std'].to_numpy()

    diff = mean_tl - mean_to
    best_gain_idx = int(np.argmax(diff))
    best_gain_x = x[best_gain_idx]
    best_gain = diff[best_gain_idx]

    ax.set_facecolor("white")

    ax.fill_between(x, mean_tl, mean_to, where=(mean_tl >= mean_to),
                    color=green_gain, alpha=0.10, interpolate=True, zorder=1)
    ax.fill_between(x, mean_tl, mean_to, where=(mean_tl < mean_to),
                    color=red_loss, alpha=0.10, interpolate=True, zorder=1)

    ax.fill_between(x, mean_tl - std_tl, mean_tl + std_tl,
                    color=color_tl, alpha=0.12, zorder=2)
    ax.fill_between(x, mean_to - std_to, mean_to + std_to,
                    color=color_to, alpha=0.12, zorder=2)

    ax.plot(x, mean_tl, color=color_tl, linewidth=2.0,
            marker='o', markersize=3.7, markeredgewidth=0,
            label='Transfer learning', zorder=4)
    ax.plot(x, mean_to, color=color_to, linewidth=2.0,
            marker='s', markersize=3.7, markeredgewidth=0,
            label='Target-only', zorder=4)

    ax.scatter([best_gain_x], [mean_tl[best_gain_idx]], s=52,
               color=color_tl, edgecolor='white', linewidth=0.9, zorder=6)
    ax.scatter([best_gain_x], [mean_to[best_gain_idx]], s=52,
               color=color_to, edgecolor='white', linewidth=0.9, zorder=6)
    ax.vlines(best_gain_x, mean_to[best_gain_idx], mean_tl[best_gain_idx],
              color="#697586", linewidth=0.9, linestyle=(0, (2, 2)), zorder=3)

    tl_pos, to_pos = draw_value_labels(ax, x, mean_tl, mean_to, ylim)

    y_box = pick_max_gain_box_position(metric_key, mean_tl, mean_to, tl_pos, to_pos, ylim)
    x_box = 37.5
    ax.text(
        x_box, y_box,
        f"Max gain: +{best_gain:.3f} at {best_gain_x:.1f}%",
        ha='center', va='center',
        fontsize=11.0, color=text_color,
        bbox=dict(boxstyle='round,pad=0.20',
                  facecolor='white', edgecolor='#D6DDE5',
                  linewidth=0.7, alpha=0.94),
        zorder=9
    )

    ax.set_xlim(9.5, 103.5)
    ax.set_xticks(x)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Healthy target training data (%)", fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')

    ax.set_axisbelow(True)
    ax.grid(axis='y', color=grid_color, linestyle='-', linewidth=0.75)
    ax.grid(axis='x', color=grid_color, linestyle='-', linewidth=0.4, alpha=0.30)

    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.85)
        s.set_color(border_color)
    ax.spines['left'].set_color('#5B6570')
    ax.spines['bottom'].set_color('#5B6570')
    ax.tick_params(axis='both', width=0.75, length=3.2, color='#5B6570', pad=3)

handles, labels = axes[0].get_legend_handles_labels()
leg = fig.legend(
    handles, labels, ncol=2,
    loc='upper center', bbox_to_anchor=(0.5, 0.992),
    frameon=True, fancybox=False,
    borderpad=0.42, handlelength=2.3, columnspacing=1.8
)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_edgecolor('#BAC3CC')
leg.get_frame().set_linewidth(0.8)

fig.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.08, hspace=0.29, wspace=0.18)
fig.savefig("fig15_target_fault_detection.png", dpi=400, facecolor="white")
fig.savefig("fig15_target_fault_detection.pdf", facecolor="white")
plt.close(fig)
