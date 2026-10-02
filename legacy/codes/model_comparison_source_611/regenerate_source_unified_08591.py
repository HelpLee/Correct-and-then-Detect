"""Regenerate root-manuscript source figures from the archived transfer checkpoint.

This trial deliberately uses the saved Option-2 validation/test predictions from
source_best_model_v3_b1.pth, the same source checkpoint that underlies the
direct-generalization and transfer results in manuscript Figures 12 and 13.
It does not touch correction_and_then_predict_1stRev.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG_DIR = ROOT / "paper" / "figs"
PREDICTIONS = (
    ROOT
    / "codes"
    / "models_reg_v3"
    / "experiments_source_v3"
    / "exp_20250411_141442_b1"
    / "predictions_results_n_past_30_n_future_10_20250411_141442.csv"
)


def metrics(truth, estimate):
    residual = truth - estimate
    mse = float(np.mean(residual**2))
    r2 = float(1.0 - np.sum(residual**2) / np.sum((truth - truth.mean()) ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
        "mape": float(np.mean(np.abs(residual / truth)) * 100.0),
    }


def point_frequency(truth, estimate):
    x_edges = np.arange(35.5, 46.6, 1.0)
    y_edges = np.linspace(35.5, 46.5, 2201)
    counts, _, _ = np.histogram2d(truth, estimate, bins=(x_edges, y_edges))
    xi = np.clip(np.digitize(truth, x_edges) - 1, 0, counts.shape[0] - 1)
    yi = np.clip(np.digitize(estimate, y_edges) - 1, 0, counts.shape[1] - 1)
    return counts[xi, yi]


def save(fig, stem):
    for suffix, kwargs in {"png": {"dpi": 600}, "pdf": {}, "svg": {}}.items():
        fig.savefig(
            FIG_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
            **kwargs,
        )


def plot_lstm_predictions(data):
    prepared = {}
    vmax = 0.0
    for split in ("Validation", "Test"):
        subset = data[(data["Dataset"] == split) & (data["Option"] == "Option 2")]
        truth = subset["True Value"].to_numpy(float)
        estimate = subset["Predicted Value"].to_numpy(float)
        frequency = point_frequency(truth, estimate)
        prepared[split] = (truth, estimate, frequency, metrics(truth, estimate))
        vmax = max(vmax, float(frequency.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.05))
    for index, (ax, split) in enumerate(zip(axes, ("Validation", "Test"))):
        truth, estimate, frequency, row = prepared[split]
        order = np.argsort(frequency)
        scatter = ax.scatter(
            truth[order], estimate[order], c=frequency[order], cmap="turbo",
            vmin=0, vmax=vmax, marker="s" if split == "Validation" else "o",
            s=8.5, linewidths=0, alpha=0.95,
            label=f"Motor 6 Temperature (°C) in {split} Set", zorder=3,
        )
        ax.plot([36, 46], [36, 46], color="#173BFF", linewidth=1.25,
                linestyle=(0, (6, 3, 1.5, 3)), label="Reference line", zorder=2)
        metric_text = (
            f"$R^2$ = {row['r2']:.4f}\n"
            f"MSE = {row['mse']:.4f}\n"
            f"MAPE = {row['mape']:.3f}%\n"
            f"RMSE = {row['rmse']:.4f}"
        )
        ax.text(0.66, 0.075, metric_text, transform=ax.transAxes, ha="left", va="bottom")
        ax.text(-0.135, 1.04, f"({chr(97 + index)})", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")
        ax.set(xlim=(36, 46), ylim=(36, 46), xlabel="Measured value", ylabel="Predicted value")
        ax.set_xticks(np.arange(36, 47, 1)); ax.set_yticks(np.arange(36, 47, 1))
        ax.xaxis.label.set_fontweight("bold"); ax.yaxis.label.set_fontweight("bold")
        ax.tick_params(direction="in", which="both", length=4, width=0.8); ax.minorticks_on()
        ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.95), frameon=False,
                  handlelength=1.8, borderaxespad=0)
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.012)
        colorbar.set_label("Frequency", rotation=90, labelpad=9)
        colorbar.ax.tick_params(labelsize=8.5, width=0.7, length=3)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.96, wspace=0.24)
    save(fig, "lstm_results_in_source")
    plt.close(fig)
    return prepared["Validation"][3], prepared["Test"][3]


def plot_model_comparison(test_metrics):
    order = ["LSTM", "XGBoost", "Transformer"]
    values = {
        "RMSE": [test_metrics["rmse"], 0.4239, 0.4430],
        "$R^2$": [test_metrics["r2"], 0.8699, 0.8579],
        "MAPE (%)": [test_metrics["mape"], 0.7235, 0.7651],
    }
    colors = ["#294D76", "#2C7A78", "#A95F2A"]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.35), dpi=180)
    for panel, (ax, (label, panel_values)) in enumerate(zip(axes, values.items())):
        panel_values = np.asarray(panel_values)
        span = max(panel_values.max() - panel_values.min(), 0.01)
        ax.hlines(y, panel_values.min() - 0.25 * span, panel_values, color="#CFD5DB", lw=1.0)
        for yi, value, color in zip(y, panel_values, colors):
            ax.scatter(value, yi, s=42, color=color, edgecolor="#20262D", linewidth=0.4, zorder=3)
            ax.text(value + 0.035 * span, yi, f"{value:.4f}", va="center", fontsize=9, fontweight="semibold")
        ax.set_xlabel(f"Test {label}")
        ax.set_yticks(y)
        if panel == 0:
            ax.set_yticklabels(order)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.invert_yaxis(); ax.grid(axis="x", color="#E1E5E9", linewidth=0.55)
        ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.21, top=0.97, wspace=0.25)
    save(fig, "source_model_accuracy_publication")
    plt.close(fig)


def main():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10.5, "axes.labelsize": 11, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5, "legend.fontsize": 9.5, "axes.linewidth": 0.9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    data = pd.read_csv(PREDICTIONS)
    validation, test = plot_lstm_predictions(data)
    plot_model_comparison(test)
    print({"validation": validation, "test": test})


if __name__ == "__main__":
    main()
