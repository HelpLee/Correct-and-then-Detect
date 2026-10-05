from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results/generated/publication_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(OUTPUT_DIR)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.patches import FancyBboxPatch, Patch

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
configs = ["No TL + No PROF", "No TL + PROF", "TL + No PROF", "TL + PROF"]

means = {
    "No TL + No PROF": [0.9469, 0.7640, 0.7354, 0.7493],
    "No TL + PROF":    [0.9638, 0.8531, 0.8817, 0.8546],
    "TL + No PROF":    [0.9532, 0.8240, 0.7195, 0.7682],
    "TL + PROF":       [0.9826, 0.9752, 0.8610, 0.9145],
}
sds = {
    "No TL + No PROF": [0.0057, 0.0341, 0.0196, 0.0245],
    "No TL + PROF":    [0.0339, 0.1882, 0.0377, 0.1028],
    "TL + No PROF":    [0.0009, 0.0052, 0.0043, 0.0042],
    "TL + PROF":       [0.0010, 0.0101, 0.0027, 0.0045],
}

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
    "font.size": 10.5,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "hatch.linewidth": 0.6,
})

colors = {
    "No TL + No PROF": "#7C7F88",
    "No TL + PROF":    "#1FC78F",
    "TL + No PROF":    "#FF9159",
    "TL + PROF":       "#4A8DF2",
}
label_colors = {
    "No TL + No PROF": "#3E434B",
    "No TL + PROF":    "#00A86B",
    "TL + No PROF":    "#F0511C",
    "TL + PROF":       "#125BDE",
}
hatches = {c: "////" for c in configs}

fig, ax = plt.subplots(figsize=(11.0, 5.5), dpi=180)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
fig.subplots_adjust(left=0.082, right=0.992, top=0.84, bottom=0.14)

x = np.arange(len(metrics))
bar_width = 0.18
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width

ax.set_axisbelow(True)
ax.grid(axis="y", linestyle="--", linewidth=0.75, color="#D8DDE3")
for sep in [0.5, 1.5, 2.5]:
    ax.axvline(sep, color="#E3E6EA", linestyle="--", linewidth=0.65, zorder=0)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("#20252B")
    spine.set_linewidth(0.8)

rounding = 0.018

for j, cfg in enumerate(configs):
    for i in range(len(metrics)):
        cx = x[i] + offsets[j]
        mean = means[cfg][i]
        sd = sds[cfg][i]
        left = cx - bar_width / 2
        edge = label_colors[cfg]

        bar = FancyBboxPatch(
            (left, 0), bar_width, mean,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            facecolor=colors[cfg],
            edgecolor=edge,
            linewidth=0.55,
            hatch=hatches[cfg],
            alpha=0.95,
            zorder=3,
            clip_on=True,
        )
        ax.add_patch(bar)

        ax.errorbar(
            cx, mean, yerr=sd, fmt="none",
            ecolor="#20252B",
            elinewidth=1.0,
            capsize=2.8,
            capthick=1.0,
            zorder=5,
        )

        ax.text(
            cx, mean + 0.012, f"{mean:.3f}",
            ha="center", va="bottom",
            fontsize=9.2, fontweight="bold",
            color=label_colors[cfg],
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.55)
        )

ax.set_xlim(-0.52, 3.52)
ax.set_ylim(0.5, 1.06)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight="bold")
ax.set_yticks(np.arange(0.5, 1.01, 0.1))
ax.set_ylabel("Score", fontweight="bold")
ax.tick_params(axis="both", colors="#20252B", width=0.75, length=4, pad=4)

for lab in ax.get_yticklabels():
    lab.set_fontweight("bold")

legend_handles = [
    Patch(facecolor=colors[c], edgecolor=label_colors[c], hatch=hatches[c], linewidth=0.5, label=c)
    for c in configs
]
leg = ax.legend(
    handles=legend_handles,
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    frameon=True,
    fancybox=True,
    borderpad=0.4,
    handlelength=1.9,
    columnspacing=1.2,
)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("#B8BEC5")
leg.get_frame().set_linewidth(0.8)

plt.savefig("fig16_tl_prof_ablation.png", dpi=400, facecolor="white")
plt.savefig("fig16_tl_prof_ablation.pdf", facecolor="white")
plt.savefig("fig16_tl_prof_ablation.svg", facecolor="white")
plt.close(fig)