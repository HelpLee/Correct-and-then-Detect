"""Recompute all threshold-dependent results at tau=1.8 for the first revision."""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde


HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
WORKSPACE = REVISION
MANUSCRIPT_FIGS = REVISION / "paper" / "figs"
RESPONSE_FIGS = REVISION / "response_letter" / "figs"
TAU = 1.8

sys.path.insert(0, str(WORKSPACE / "codes" / "ablation_tl_prof"))
sys.path.insert(0, str(WORKSPACE / "codes" / "model_comparison_source_611"))
import run_ablation_tl_prof as ablation
import run_source_threshold_consistent as source_threshold

NAVY, RUST, TEAL, INK, GRID = "#294C73", "#A65E2E", "#287271", "#30343B", "#D9DEE5"


def configure_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11.5, "axes.labelsize": 12.5, "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5, "legend.fontsize": 10.2, "axes.linewidth": 0.9,
        "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_all(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def source_results(device):
    validation = pd.read_csv(WORKSPACE / "codes/model_comparison_source_611/results/source_threshold_validation_consistent.csv")
    selected = validation.loc[np.isclose(validation.threshold, TAU)].iloc[0]
    test, _ = source_threshold.evaluate_thresholds(source_threshold.TEST_FAULT, np.array([TAU]), device)
    validation.to_csv(HERE / "source_threshold_validation.csv", index=False)
    test.to_csv(HERE / "source_threshold_test_tau1p8.csv", index=False)

    render_source_threshold(validation, selected)
    return selected, test.iloc[0]


def render_source_threshold(validation, selected):
    """Render the source-threshold figure from cached validation results."""

    fig, ax = plt.subplots(figsize=(7.1, 4.15))
    ax.plot(validation.threshold, validation.f1, color=NAVY, linewidth=2.0,
            marker="o", markersize=3.5, markevery=2, label="Validation F1-score")
    ax.axvline(TAU, color=RUST, linestyle="--", linewidth=1.3)
    ax.scatter([TAU], [selected.f1], color=RUST, edgecolor="white", s=48, zorder=4)
    ax.text(.97, .96, r"Selected $\tau_{\rm res}=1.8$" + f"\nF1 = {selected.f1:.3f}",
            transform=ax.transAxes, ha="right", va="top", color=INK,
            bbox={"boxstyle": "round,pad=.28", "facecolor": "white",
                  "edgecolor": GRID, "alpha": .95})
    ax.set(xlabel=r"Residual threshold, $\tau_{\rm res}$", ylabel="Validation F1-score",
           xlim=(-0.05, 6.05), ylim=(0, 0.97))
    ax.grid(axis="y", color=GRID, linewidth=0.7); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, edgecolor="#777", loc="lower center",
              bbox_to_anchor=(.5, 1.02), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 1, .91))
    save_all(fig, MANUSCRIPT_FIGS, "threshold_opt_publication")


def run_target_ratios(device):
    ablation.TAU_RES = TAU
    rows = []
    for seed in ablation.SEEDS:
        for ratio_label in range(10, 81, 10):
            ablation.CHECKPOINT_RATIO_LABEL = f"{ratio_label}%"
            # Legacy folders express the fraction relative to the complete
            # 15,600-point sequence.  The manuscript denominator is the
            # 12,480-point D_tgt^train split (80% of that sequence).
            ablation.TARGET_TRAIN_FRACTION = ratio_label / 80.0
            for use_tl in (False, True):
                summary, _ = ablation.evaluate_configuration(use_tl, True, seed, device)
                summary["ratio"] = ratio_label / 100.0
                rows.append(summary)
                print("ratio", ratio_label, "seed", seed, summary["method"], summary["f1"])
    raw = pd.DataFrame(rows)
    raw.to_csv(HERE / "target_detection_by_ratio_tau1p8.csv", index=False)
    aggregate = raw.groupby(["transfer_learning", "ratio"])[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"])
    aggregate.columns = [f"{a}_{b}" for a, b in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(HERE / "target_detection_by_ratio_summary_tau1p8.csv", index=False)

    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.15), sharex=True)
    for ax, (key, label) in zip(axes.flat, metrics):
        for tl, name, color, marker in [(1, "Transfer learning", NAVY, "o"), (0, "Target-only", RUST, "s")]:
            d = aggregate[aggregate.transfer_learning.eq(tl)].sort_values("ratio")
            x = d.ratio.to_numpy() * 125
            ax.errorbar(x, d[f"{key}_mean"], yerr=d[f"{key}_std"], color=color,
                        marker=marker, markersize=5, linewidth=1.7, elinewidth=.95,
                        capsize=2.6, label=name)
        ax.set_ylabel(label); ax.set_ylim(0.2 if key in {"precision", "f1"} else 0.5, 1.02)
        ax.grid(axis="y", color=GRID, linewidth=.7); ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Healthy target training data (%)")
        ax.set_xticks([12.5, 25, 50, 75, 100], ["12.5", "25", "50", "75", "100"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=1.5, h_pad=1.2)
    save_all(fig, MANUSCRIPT_FIGS, "test_transfer_publication")
    return raw, aggregate


def run_factorial(device):
    ablation.CHECKPOINT_RATIO_LABEL = "80%"; ablation.TARGET_TRAIN_FRACTION = 1.0; ablation.TAU_RES = TAU
    rows = []
    for use_tl, use_prof in ((False, False), (False, True), (True, False), (True, True)):
        for seed in ablation.SEEDS:
            summary, _ = ablation.evaluate_configuration(use_tl, use_prof, seed, device)
            rows.append(summary)
    raw = pd.DataFrame(rows)
    raw.to_csv(HERE / "ablation_by_seed_tau1p8.csv", index=False)
    order = ["No TL + No PROF", "No TL + PROF", "TL + No PROF", "TL + PROF"]
    summary = raw.groupby("method")[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"]).reindex(order)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(HERE / "ablation_summary_tau1p8.csv", index=False)

    render_factorial(summary)
    return raw, summary


def render_factorial(summary):
    """Render Figure 16 from the cached factorial summary."""
    order = ["No TL + No PROF", "No TL + PROF", "TL + No PROF", "TL + PROF"]

    fig, ax = plt.subplots(figsize=(9.0, 4.9)); metrics = ["accuracy", "precision", "recall", "f1"]
    colors = ["#4B5563", TEAL, RUST, NAVY]; x = np.arange(4); width = .18
    for i, method in enumerate(order):
        row = summary.loc[method]
        vals = [row[f"{m}_mean"] for m in metrics]; errs = [row[f"{m}_std"] for m in metrics]
        ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=2.5,
               color=colors[i], edgecolor="white", linewidth=.5, label=method)
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1-score"]); ax.set_ylim(.55, 1.03)
    ax.set_ylabel("Score"); ax.grid(axis="y", color=GRID, linewidth=.7); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, frameon=True, edgecolor="#777", loc="lower center",
              bbox_to_anchor=(.5, 1.02), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 1, .91))
    save_all(fig, MANUSCRIPT_FIGS, "ablation_tl_prof_publication")


def residual_figure():
    results = WORKSPACE / "codes/model_comparison_source_611/results"
    predictions = pd.read_csv(results / "residual_alignment_before_after_predictions.csv")
    metrics = pd.read_csv(results / "residual_alignment_before_after_metrics.csv").set_index("condition")
    conditions = ["Source reference", "Target before TL (direct source model)", "Target after TL"]
    arrays = [predictions.loc[predictions.condition.eq(c), "residual"].to_numpy() for c in conditions]
    all_values = np.concatenate(arrays); low, high = np.quantile(all_values, [.001, .999]); low=min(low,-2.0); high=max(high,2.0)
    x = np.linspace(low-.04*(high-low), high+.04*(high-low), 900)
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=180); max_density=0
    styles=[(NAVY,"-"),(RUST,(0,(6,2.5))),("#3568B8",(0,(5,2,1.4,2)))]
    for condition, values, (color, style) in zip(conditions, arrays, styles):
        density=gaussian_kde(values)(x); max_density=max(max_density,float(density.max())); row=metrics.loc[condition]
        ax.plot(x,density,color=color,linestyle=style,linewidth=2.4,
                label=f"{condition}  ($R^2$={row.r2:.4f}, MSE={row.mse:.4f})")
    ax.axvline(0,color=INK,lw=1,ls=(0,(2,2)),alpha=.75)
    for value in (-TAU,TAU): ax.axvline(value,color="#72777D",lw=1.15,ls=(0,(3,2)))
    ax.text(.985,.95,"Source-calibrated threshold\n"+r"$|e|=1.8\,^{\circ}$C",
            transform=ax.transAxes,ha="right",va="top",color="#555B61",
            bbox={"boxstyle":"round,pad=.32","facecolor":"white","edgecolor":GRID,"alpha":.95})
    ax.text(.015,.95,"Distance to source residuals\nWasserstein: 1.588 $\\rightarrow$ 0.157\nKS statistic: 0.820 $\\rightarrow$ 0.221",
            transform=ax.transAxes,ha="left",va="top",color=INK,
            bbox={"boxstyle":"round,pad=.35","facecolor":"white","edgecolor":GRID,"alpha":.92})
    ax.set_xlabel(r"Prediction residual, $e=y-\hat{y}$ ($^{\circ}$C)"); ax.set_ylabel("Probability density")
    ax.set_ylim(-.025*max_density,1.35*max_density); ax.grid(axis="y",color=GRID,lw=.75,alpha=.65)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.025), borderaxespad=0,
              frameon=True, edgecolor="#6E747A", handlelength=3.1)
    fig.subplots_adjust(left=.10,right=.985,bottom=.14,top=.73)
    save_all(fig, MANUSCRIPT_FIGS, "residual_alignment_publication")
    # Response letter uses the same controlled comparison.
    import shutil
    shutil.copy2(MANUSCRIPT_FIGS / "residual_alignment_publication.png", RESPONSE_FIGS / "Fig_R4.png")


def main():
    HERE.mkdir(parents=True, exist_ok=True); configure_style()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected, source_test=source_results(device)
    ratios, ratio_summary=run_target_ratios(device)
    raw, factorial=run_factorial(device)
    residual_figure()
    print("SOURCE VALIDATION", selected.to_dict()); print("SOURCE TEST", source_test.to_dict())
    print("FACTORIAL\n", factorial.to_string()); print("RATIO SUMMARY\n", ratio_summary.to_string(index=False))


if __name__ == "__main__":
    main()
