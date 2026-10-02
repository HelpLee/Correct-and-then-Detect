"""Calibrate and evaluate the source detector with one consistent PROF loop.

All residual thresholds are evaluated in parallel as the batch dimension.  A
separate rolling feature buffer is maintained for every threshold, so each
candidate receives exactly the recursive PROF corrections that it would have
received in an independent run.  This is both exact and substantially faster
than running 61 sequential passes over the validation sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "codes" / "data"
MODELS = ROOT / "codes" / "models_reg_v3"
RESULTS = HERE / "results"
FIGURES = ROOT / "paper" / "figs"

VAL_FAULT = DATA / "source_val_faults20_T_Th1_FL30-50_TI4-8_b1.csv"
TEST_FAULT = DATA / "source_test_faults20_T_Th1_FL30-50_TI4-8_b1.csv"
FEATURE_SCALER = MODELS / "feature_scaler_v3_b1.pkl"
TARGET_SCALER = MODELS / "target_scaler_v3_b1.pkl"
CHECKPOINT = MODELS / "source_best_model_v3_b1.pth"

N_PAST = 30
N_FUTURE = 10
TARGET = "motor6_temperature"
TIMESTAMP = "timestamp"
LABEL = "label"

NAVY = "#294C73"
RUST = "#B0602B"
INK = "#20242A"
GRID = "#D9DEE5"


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size=64, num_layers=3, dropout=0.2, batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, N_FUTURE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.fc(self.dropout(output[:, -1, :]))


def load_fault_sequence(path: Path):
    frame = pd.read_csv(path).dropna().reset_index(drop=True)
    features = frame.drop(columns=[TIMESTAMP, LABEL])
    feature_scaler = joblib.load(FEATURE_SCALER)
    target_scaler = joblib.load(TARGET_SCALER)
    scaled_features = feature_scaler.transform(features).astype(np.float32)
    scaled_target = target_scaler.transform(frame[[TARGET]]).astype(np.float32).ravel()
    labels = frame[LABEL].astype(int).to_numpy()
    target_index = list(features.columns).index(TARGET)
    return scaled_features, scaled_target, labels, target_index, target_scaler


@torch.inference_mode()
def evaluate_thresholds(path: Path, thresholds: np.ndarray, device: torch.device, selected_threshold: float = 2.2):
    features, target, labels, target_index, target_scaler = load_fault_sequence(path)
    threshold_tensor = torch.as_tensor(thresholds, dtype=torch.float32, device=device)
    rolling = torch.as_tensor(features, dtype=torch.float32, device=device)
    rolling = rolling.unsqueeze(0).repeat(len(thresholds), 1, 1)

    model = LSTMModel(features.shape[1]).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    scale = float(target_scaler.scale_[0])
    offset = float(target_scaler.min_[0])
    tp = torch.zeros(len(thresholds), dtype=torch.int64, device=device)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    tn = torch.zeros_like(tp)
    selected_index = int(np.argmin(np.abs(thresholds - selected_threshold)))
    selected_records = []

    n_samples = len(features) - N_PAST - N_FUTURE + 1
    for sample_index in range(n_samples):
        decision_index = sample_index + N_PAST + N_FUTURE - 1
        prediction_scaled = model(rolling[:, sample_index : sample_index + N_PAST])[:, -1]
        prediction = (prediction_scaled - offset) / scale
        observed = float((target[decision_index] - offset) / scale)
        residual = torch.abs(prediction - observed)
        predicted = residual > threshold_tensor
        truth = bool(labels[decision_index])

        if truth:
            tp += predicted
            fn += ~predicted
        else:
            fp += predicted
            tn += ~predicted

        # Each threshold has an independent rolling stream.  Only streams that
        # fire at this decision index receive the corresponding nominal value.
        if predicted.any():
            rows = torch.nonzero(predicted, as_tuple=False).squeeze(1)
            rolling[rows, decision_index, target_index] = prediction_scaled[rows]

        selected_records.append(
            {
                "sample_index": sample_index,
                "threshold": float(thresholds[selected_index]),
                "decision_index": decision_index,
                "true_temperature": observed,
                "predicted_temperature": float(prediction[selected_index].cpu()),
                "absolute_residual": float(residual[selected_index].cpu()),
                "true_label": int(truth),
                "predicted_label": int(predicted[selected_index].cpu()),
            }
        )

    tp = tp.cpu().numpy()
    fp = fp.cpu().numpy()
    fn = fn.cpu().numpy()
    tn = tn.cpu().numpy()
    total = tp + fp + fn + tn
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    metrics = pd.DataFrame(
        {
            "threshold": thresholds,
            "accuracy": (tp + tn) / total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "total_samples": total,
            "actual_faults": tp + fn,
        }
    )
    return metrics, pd.DataFrame(selected_records)


def plot_validation_curve(metrics: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11.5,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    selected = metrics.iloc[(metrics.threshold - 2.2).abs().argmin()]
    best = metrics.iloc[metrics.f1.argmax()]
    fig, ax = plt.subplots(figsize=(7.35, 4.15))
    ax.plot(metrics.threshold, metrics.f1, color=NAVY, linewidth=2.1,
            marker="o", markersize=3.8, markevery=2, label="Validation F1-score")
    ax.axvline(2.2, color=RUST, linestyle="--", linewidth=1.3)
    ax.scatter([2.2], [selected.f1], color=RUST, s=48, edgecolor="white",
               linewidth=0.7, zorder=4)
    ax.annotate(
        rf"Selected $\tau_{{\rm res}}=2.2$" + "\n" + rf"F1 = {selected.f1:.3f}",
        xy=(2.2, selected.f1), xytext=(3.05, min(0.91, selected.f1 + 0.015)),
        arrowprops={"arrowstyle": "-", "color": RUST, "lw": 1.0},
        ha="left", va="center", color=INK,
    )
    if abs(float(best.threshold) - 2.2) > 0.051:
        ax.scatter([best.threshold], [best.f1], color=INK, s=24, zorder=4)
        ax.text(best.threshold, best.f1 + 0.018, f"best {best.threshold:.1f}",
                ha="center", color=INK, fontsize=9.5)
    ax.set_xlabel(r"Residual threshold, $\tau_{\rm res}$")
    ax.set_ylabel("Validation F1-score")
    ax.set_xlim(-0.05, 6.05)
    ax.set_ylim(0, min(1.0, max(0.92, metrics.f1.max() + 0.08)))
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {"png": {"dpi": 600}, "pdf": {}, "svg": {}}.items():
        fig.savefig(FIGURES / f"threshold_opt_publication.{suffix}",
                    bbox_inches="tight", pad_inches=0.04, facecolor="white", **kwargs)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = np.round(np.arange(0.0, 6.0 + 0.1, 0.1), 1)
    validation, val_samples = evaluate_thresholds(VAL_FAULT, thresholds, device)
    test, test_samples = evaluate_thresholds(TEST_FAULT, np.array([2.2]), device)
    validation.to_csv(RESULTS / "source_threshold_validation_consistent.csv", index=False)
    test.to_csv(RESULTS / "source_threshold_test_consistent.csv", index=False)
    val_samples.to_csv(RESULTS / "source_threshold_val_tau2p2_per_sample.csv", index=False)
    test_samples.to_csv(RESULTS / "source_threshold_test_tau2p2_per_sample.csv", index=False)
    plot_validation_curve(validation)
    selected = validation.iloc[(validation.threshold - 2.2).abs().argmin()]
    best = validation.iloc[validation.f1.argmax()]
    metadata = {
        "device": str(device),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "validation_file": str(VAL_FAULT.relative_to(ROOT)),
        "test_file": str(TEST_FAULT.relative_to(ROOT)),
        "n_past": N_PAST,
        "n_future": N_FUTURE,
        "selected_threshold": 2.2,
        "selected_validation_f1": float(selected.f1),
        "best_threshold": float(best.threshold),
        "best_validation_f1": float(best.f1),
        "prof": True,
        "parallel_threshold_streams": len(thresholds),
    }
    (RESULTS / "source_threshold_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(validation.loc[validation.threshold.eq(2.2)].to_string(index=False))
    print(test.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
