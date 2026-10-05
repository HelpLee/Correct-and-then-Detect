import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

from pathlib import Path
import argparse
ROOT = Path(__file__).resolve().parents[2]

def render(data, output_dir):
    csv_path = data
    df = pd.read_csv(csv_path).copy()

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
        "font.size": 10.0,
        "axes.labelsize": 11.5,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    x_vals = sorted(df["n_past"].unique())
    y_vals = sorted(df["n_future"].unique())

    r2_mat = (
        df.pivot(index="n_future", columns="n_past", values="r2_mean")
          .loc[y_vals, x_vals]
          .to_numpy()
    )
    mse_mat = (
        df.pivot(index="n_future", columns="n_past", values="mse_mean")
          .loc[y_vals, x_vals]
          .to_numpy()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), dpi=180)
    fig.patch.set_facecolor("white")

    def draw_heatmap(ax, data, cmap, title, cbar_label, vmin=None, vmax=None):
        im = ax.imshow(
            data, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="auto", origin="upper", interpolation="nearest"
        )

        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_xticklabels([str(v) for v in x_vals])
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_yticklabels([str(v) for v in y_vals])

        ax.set_xlabel(r"Input-window length, $W_{\mathrm{in}}$", fontweight="bold")
        ax.set_ylabel(r"Prediction horizon, $W_{\mathrm{pred}}$", fontweight="bold")
        ax.set_title(title, loc="left", fontweight="bold", pad=8)

        ax.set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        norm = plt.Normalize(vmin if vmin is not None else data.min(),
                             vmax if vmax is not None else data.max())
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                color = "white" if norm(val) > 0.58 else "#333333"
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=8.6,
                    fontweight="bold",
                    color=color
                )

        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color("#666666")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(cbar_label, fontweight="bold")
        cbar.ax.tick_params(labelsize=8.5)
        cbar.outline.set_linewidth(0.7)

    draw_heatmap(axes[0], r2_mat, plt.cm.Blues, r"(a) Validation $R^2$", r"$R^2$")
    draw_heatmap(axes[1], mse_mat, plt.cm.Oranges, "(b) Validation MSE", "MSE")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.20, wspace=0.26)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"fig10_window_sensitivity.{ext}", dpi=400, facecolor="white", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "experiments/window_sensitivity/validation_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generated/figures")
    args = parser.parse_args()
    render(args.data, args.output_dir)
