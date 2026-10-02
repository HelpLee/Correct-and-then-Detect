"""Factorial ablation of transfer learning (TL) and PROF.

This script evaluates four controlled configurations on the same target-domain
fault dataset:

1. Target-only model, without PROF
2. Target-only model, with PROF
3. Transfer-learned model, without PROF
4. Transfer-learned model, with PROF

The implementation follows Section 4.2 of Fault_detection_TL_ZZ: when the
point-wise residual at the decision index exceeds ``TAU_RES``, PROF replaces
that observation in the rolling input stream with the digital-twin prediction.
Only healthy data are used to train the checkpoints; this script performs
evaluation only and does not retrain them.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
CODES_DIR = SCRIPT_DIR.parent
DATA_DIR = CODES_DIR / "data"
MODELS_DIR = CODES_DIR / "models_reg_v3"
OUTPUT_DIR = SCRIPT_DIR / "results"
MANUSCRIPT_FIG_DIR = CODES_DIR.parent / "paper" / "figs"

SEEDS = (42, 43, 44, 45, 46)
TARGET_TRAIN_FRACTION = 1.0
# Legacy checkpoint folders encode the training block as 80% of the complete
# 15,600-point target sequence. That block is exactly 100% of the separately
# defined D_tgt^train partition (12,480 points).
CHECKPOINT_RATIO_LABEL = "80%"
TRANSFER_VARIANT = "0"  # Same transferred checkpoints used by the original PROF study.
N_PAST = 30
N_FUTURE = 10
HIDDEN_DIM = 64
NUM_LAYERS = 3
TAU_RES = 2.2
TARGET_FEATURE = "motor6_temperature"
TIMESTAMP_COLUMN = "timestamp"
LABEL_COLUMN = "label"

TARGET_HEALTHY_CSV = DATA_DIR / "target_train_data_b1.csv"
TARGET_FAULT_CSV = DATA_DIR / "target_test_faults5_T_Th1_FL30-50_TI4-8_b1.csv"
TRANSFER_FEATURE_SCALER = MODELS_DIR / "feature_scaler_v3_b1.pkl"
TRANSFER_TARGET_SCALER = MODELS_DIR / "target_scaler_v3_b1.pkl"


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, HIDDEN_DIM, NUM_LAYERS, batch_first=True
        )
        self.fc = nn.Linear(HIDDEN_DIM, N_FUTURE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def checkpoint_path(use_transfer_learning: bool, seed: int) -> Path:
    if use_transfer_learning:
        return (
            MODELS_DIR
            / f"experiments_transfer_v3_b1_{seed}"
            / f"exp_{TRANSFER_VARIANT}_{CHECKPOINT_RATIO_LABEL}"
            / "transfer_best_model.pth"
        )
    return (
        MODELS_DIR
        / f"experiments_target_v3_b1_{seed}"
        / f"exp_{CHECKPOINT_RATIO_LABEL}"
        / "target_best_model_v3.pth"
    )


def load_data_and_scalers(use_transfer_learning: bool):
    healthy = pd.read_csv(TARGET_HEALTHY_CSV).dropna().reset_index(drop=True)
    fault = pd.read_csv(TARGET_FAULT_CSV).dropna().reset_index(drop=True)

    healthy = healthy.iloc[: int(len(healthy) * TARGET_TRAIN_FRACTION)].copy()
    healthy_features = healthy.drop(columns=[TIMESTAMP_COLUMN])
    healthy_target = healthy[[TARGET_FEATURE]]
    fault_features = fault.drop(columns=[TIMESTAMP_COLUMN, LABEL_COLUMN])
    fault_target = fault[[TARGET_FEATURE]]

    if list(healthy_features.columns) != list(fault_features.columns):
        raise ValueError("Healthy and fault feature columns do not match.")

    if use_transfer_learning:
        feature_scaler = joblib.load(TRANSFER_FEATURE_SCALER)
        target_scaler = joblib.load(TRANSFER_TARGET_SCALER)
    else:
        # Target-only checkpoints were trained with scalers fitted on the same
        # target-domain healthy subset in the original experiment notebook.
        feature_scaler = MinMaxScaler().fit(healthy_features)
        target_scaler = MinMaxScaler().fit(healthy_target)

    feature_scaled = feature_scaler.transform(fault_features)
    target_scaled = target_scaler.transform(fault_target).reshape(-1)
    labels = fault[LABEL_COLUMN].astype(int).to_numpy()
    target_feature_index = list(fault_features.columns).index(TARGET_FEATURE)
    return feature_scaled, target_scaled, labels, target_feature_index, target_scaler


@torch.no_grad()
def evaluate_configuration(
    use_transfer_learning: bool,
    use_prof: bool,
    seed: int,
    device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    features, target, labels, target_index, target_scaler = load_data_and_scalers(
        use_transfer_learning
    )
    rolling_features = torch.tensor(features, dtype=torch.float32)

    model = LSTMModel(input_dim=features.shape[1]).to(device)
    ckpt = checkpoint_path(use_transfer_learning, seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    n_samples = len(features) - N_PAST - N_FUTURE + 1
    records = []
    for sample_index in range(n_samples):
        decision_index = sample_index + N_PAST + N_FUTURE - 1
        window = rolling_features[sample_index : sample_index + N_PAST]
        prediction_scaled = model(window.unsqueeze(0).to(device))[0, -1].item()

        prediction = target_scaler.inverse_transform(
            np.array([[prediction_scaled]])
        )[0, 0]
        observed = target_scaler.inverse_transform(
            np.array([[target[decision_index]]])
        )[0, 0]
        residual = abs(observed - prediction)
        predicted_label = int(residual > TAU_RES)

        # PROF: correct only the observation at the current decision index.
        # The corrected value then enters all subsequent rolling windows that
        # contain this time index, exactly as described in Section 4.2.
        if use_prof and predicted_label:
            rolling_features[decision_index, target_index] = prediction_scaled

        records.append(
            {
                "sample_index": sample_index,
                "decision_index": decision_index,
                "observed_temperature": observed,
                "predicted_temperature": prediction,
                "absolute_residual": residual,
                "true_label": int(labels[decision_index]),
                "predicted_label": predicted_label,
            }
        )

    per_sample = pd.DataFrame(records)
    y_true = per_sample["true_label"].to_numpy()
    y_pred = per_sample["predicted_label"].to_numpy()
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    method = (
        ("TL" if use_transfer_learning else "No TL")
        + " + "
        + ("PROF" if use_prof else "No PROF")
    )
    summary = {
        "method": method,
        "transfer_learning": int(use_transfer_learning),
        "prof": int(use_prof),
        "seed": seed,
        "target_train_fraction": TARGET_TRAIN_FRACTION,
        "threshold": TAU_RES,
        "total_samples": len(y_true),
        "actual_faults": int(y_true.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "checkpoint": str(ckpt.relative_to(CODES_DIR)),
    }
    per_sample.insert(0, "seed", seed)
    per_sample.insert(0, "method", method)
    return summary, per_sample


def save_summary_and_figure(seed_results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1"]
    order = ["No TL + No PROF", "No TL + PROF", "TL + No PROF", "TL + PROF"]
    aggregate = (
        seed_results.groupby("method", sort=False)[metrics]
        .agg(["mean", "std"])
        .reindex(order)
    )
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(OUTPUT_DIR / "ablation_tl_prof_summary.csv", index=False)

    # Restrained, print-friendly palette with stronger visual weight than the
    # default office colors. The distinct luminance levels also remain legible
    # when the figure is printed in grayscale.
    colors = ["#4B5563", "#287271", "#A65E2E", "#26456E"]
    x = np.arange(len(metrics))
    width = 0.18
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=180)
    for idx, (method, color) in enumerate(zip(order, colors)):
        row = aggregate.loc[aggregate["method"] == method].iloc[0]
        means = np.asarray([row[f"{metric}_mean"] for metric in metrics], dtype=float)
        stds = np.asarray([row[f"{metric}_std"] for metric in metrics], dtype=float)
        bars = ax.bar(
            x + (idx - 1.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=method,
            color=color,
            edgecolor="#20242A",
            linewidth=0.55,
            error_kw={"elinewidth": 1.05, "capthick": 1.05, "ecolor": "#20242A"},
            zorder=3,
        )
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + 0.012,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color="#20242A",
                fontweight="semibold",
                clip_on=False,
            )

    ax.set_ylabel("Detection score", fontsize=11.5, labelpad=8)
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1-score"], fontsize=10.5)
    ax.set_ylim(0.0, 1.10)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=9.5, width=0.8, length=4)
    ax.tick_params(axis="x", width=0.8, length=4)
    ax.grid(axis="y", color="#D6D9DE", linestyle="-", linewidth=0.6, alpha=0.72, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        frameon=False,
        fontsize=9.2,
        handlelength=1.4,
        columnspacing=1.35,
        handletextpad=0.55,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.14, top=0.86)
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    export_targets = [
        (OUTPUT_DIR / "ablation_tl_prof.png", {"dpi": 600}),
        (OUTPUT_DIR / "ablation_tl_prof.pdf", {}),
        (OUTPUT_DIR / "ablation_tl_prof.svg", {}),
        (MANUSCRIPT_FIG_DIR / "ablation_tl_prof_publication.png", {"dpi": 600}),
        (MANUSCRIPT_FIG_DIR / "ablation_tl_prof_publication.pdf", {}),
        (MANUSCRIPT_FIG_DIR / "ablation_tl_prof_publication.svg", {}),
    ]
    for destination, extra_options in export_targets:
        try:
            fig.savefig(
                destination,
                bbox_inches="tight",
                pad_inches=0.04,
                facecolor="white",
                **extra_options,
            )
        except PermissionError:
            # A PDF may be locked by Word/Acrobat on Windows.  The manuscript
            # uses the PNG, so keep the run reproducible and write a versioned
            # companion instead of aborting after the numerical evaluation.
            fallback = destination.with_stem(f"{destination.stem}_full_Dtgt_train")
            fig.savefig(
                fallback,
                bbox_inches="tight",
                pad_inches=0.04,
                facecolor="white",
                **extra_options,
            )
            print(f"Warning: {destination.name} was locked; saved {fallback.name}.")
    plt.close(fig)
    return aggregate


def save_component_effects(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Save paired main effects and the TL-PROF interaction for each metric."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    rows = []
    for seed in SEEDS:
        current = seed_results.loc[seed_results["seed"] == seed].set_index(
            ["transfer_learning", "prof"]
        )
        for metric in metrics:
            no_tl_no_prof = current.loc[(0, 0), metric]
            no_tl_prof = current.loc[(0, 1), metric]
            tl_no_prof = current.loc[(1, 0), metric]
            tl_prof = current.loc[(1, 1), metric]
            rows.extend(
                [
                    {
                        "seed": seed,
                        "metric": metric,
                        "effect": "PROF main effect",
                        "delta": ((no_tl_prof - no_tl_no_prof) + (tl_prof - tl_no_prof)) / 2,
                    },
                    {
                        "seed": seed,
                        "metric": metric,
                        "effect": "TL main effect",
                        "delta": ((tl_no_prof - no_tl_no_prof) + (tl_prof - no_tl_prof)) / 2,
                    },
                    {
                        "seed": seed,
                        "metric": metric,
                        "effect": "TL x PROF interaction",
                        "delta": (tl_prof - tl_no_prof) - (no_tl_prof - no_tl_no_prof),
                    },
                ]
            )
    effects_by_seed = pd.DataFrame(rows)
    effects_by_seed.to_csv(OUTPUT_DIR / "component_effects_by_seed.csv", index=False)
    effects_summary = (
        effects_by_seed.groupby(["metric", "effect"], sort=False)["delta"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "delta_mean", "std": "delta_std"})
    )
    effects_summary.to_csv(OUTPUT_DIR / "component_effects_summary.csv", index=False)
    return effects_summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configurations = ((False, False), (False, True), (True, False), (True, True))
    summaries = []
    sample_frames = []

    for use_tl, use_prof in configurations:
        for seed in SEEDS:
            summary, per_sample = evaluate_configuration(use_tl, use_prof, seed, device)
            summaries.append(summary)
            sample_frames.append(per_sample)
            print(
                f"{summary['method']:17s} seed={seed} "
                f"Acc={summary['accuracy']:.4f} Prec={summary['precision']:.4f} "
                f"Rec={summary['recall']:.4f} F1={summary['f1']:.4f}"
            )

    seed_results = pd.DataFrame(summaries)
    seed_results.to_csv(OUTPUT_DIR / "ablation_tl_prof_by_seed.csv", index=False)
    pd.concat(sample_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "ablation_tl_prof_per_sample.csv", index=False
    )
    aggregate = save_summary_and_figure(seed_results)
    effects = save_component_effects(seed_results)

    metadata = {
        "device": str(device),
        "seeds": list(SEEDS),
        "target_train_fraction": TARGET_TRAIN_FRACTION,
        "fraction_denominator": "D_tgt^train (12,480 healthy points; 12,441 windows)",
        "legacy_checkpoint_ratio_label": CHECKPOINT_RATIO_LABEL,
        "n_past": N_PAST,
        "n_future": N_FUTURE,
        "tau_res": TAU_RES,
        "target_healthy_csv": str(TARGET_HEALTHY_CSV.relative_to(CODES_DIR)),
        "target_fault_csv": str(TARGET_FAULT_CSV.relative_to(CODES_DIR)),
        "prof_definition": "Replace the target observation at the current decision index with its nominal prediction when the residual exceeds tau_res.",
    }
    (OUTPUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nMean +/- standard deviation over five seeds:")
    print(aggregate.to_string(index=False))
    print("\nPaired factorial component effects:")
    print(effects.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
