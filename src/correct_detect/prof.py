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
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
CODES_DIR = Path(__file__).resolve().parents[2] / 'legacy' / 'codes'
DATA_DIR = CODES_DIR / "data"
MODELS_DIR = CODES_DIR / "models_reg_v3"
OUTPUT_DIR = SCRIPT_DIR / "results"
MANUSCRIPT_FIG_DIR = CODES_DIR.parent / "paper" / "figs"

PROFILES = json.loads((Path(__file__).resolve().parents[2] / 'configs/protocols.json').read_text(encoding='utf-8'))
SEEDS = tuple(PROFILES['detection']['seeds'])
TARGET_TRAIN_FRACTION = 1.0
# Legacy checkpoint folders encode the training block as 80% of the complete
# 15,600-point target sequence. That block is exactly 100% of the separately
# defined D_tgt^train partition (12,480 points).
CHECKPOINT_RATIO_LABEL = "80%"
TRANSFER_VARIANT = "0"  # Same transferred checkpoints used by the original PROF study.
N_PAST = PROFILES['archived_source']['past']
N_FUTURE = PROFILES['archived_source']['future']
HIDDEN_DIM = PROFILES['archived_source']['hidden']
NUM_LAYERS = PROFILES['archived_source']['layers']
TAU_RES = PROFILES['detection']['threshold']
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

