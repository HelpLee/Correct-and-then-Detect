"""Controlled residual-alignment experiment for manuscript Figure 13.

The comparison deliberately holds the source checkpoint, preprocessing,
target test set, forecast horizon, and random seed fixed.  The only difference
between the two target curves is whether the source LSTM has been fine-tuned
on the complete designated healthy target-domain training split.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
CODES = HERE.parent
RESULTS = HERE / "results"
CHECKPOINTS = RESULTS / "checkpoints"

SOURCE_CHECKPOINT = CHECKPOINTS / "lstm_seed42.pth"
FEATURE_SCALER_FILE = RESULTS / "feature_scaler.pkl"
TARGET_SCALER_FILE = RESULTS / "target_scaler.pkl"
SOURCE_PREDICTIONS = RESULTS / "source_model_test_predictions.csv"
TARGET_TRAIN = CODES / "data" / "target_train_data_b1.csv"
TARGET_VALIDATION = CODES / "data" / "target_val_data_b1.csv"
TARGET_TEST = CODES / "data" / "target_test_data_b1.csv"

OUTPUT_PREDICTIONS = RESULTS / "residual_alignment_before_after_predictions.csv"
OUTPUT_METRICS = RESULTS / "residual_alignment_before_after_metrics.csv"
OUTPUT_HISTORY = RESULTS / "residual_alignment_before_after_training_history.csv"
OUTPUT_METADATA = RESULTS / "residual_alignment_before_after_metadata.json"
OUTPUT_CHECKPOINT = CHECKPOINTS / "lstm_transfer_alignment_seed42.pth"

TARGET = "motor6_temperature"
N_PAST = 30
N_FUTURE = 10
BATCH_SIZE = 64
MAX_EPOCHS = 500
PATIENCE = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42


class LSTMForecaster(nn.Module):
    """Exact architecture of the validation-best LSTM reported in Table 6."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size=64,
            num_layers=3,
            dropout=0.2,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(64, N_FUTURE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        return self.head(self.dropout(x[:, -1, :]))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_windows(features: np.ndarray, target: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(features) - N_PAST - N_FUTURE + 1
    x = np.empty((count, N_PAST, features.shape[1]), dtype=np.float32)
    y = np.empty((count, N_FUTURE), dtype=np.float32)
    for index in range(count):
        x[index] = features[index : index + N_PAST]
        y[index] = target[index + N_PAST : index + N_PAST + N_FUTURE, 0]
    return torch.from_numpy(x), torch.from_numpy(y)


def load_target_splits():
    feature_scaler = joblib.load(FEATURE_SCALER_FILE)
    target_scaler = joblib.load(TARGET_SCALER_FILE)
    feature_columns = list(feature_scaler.feature_names_in_)
    splits = {}
    raw_rows = {}
    for name, path in (
        ("train", TARGET_TRAIN),
        ("validation", TARGET_VALIDATION),
        ("test", TARGET_TEST),
    ):
        frame = pd.read_csv(path).dropna(subset=feature_columns).reset_index(drop=True)
        raw_rows[name] = len(frame)
        features = feature_scaler.transform(frame[feature_columns]).astype(np.float32)
        target = target_scaler.transform(frame[[TARGET]]).astype(np.float32)
        splits[name] = make_windows(features, target)
    return splits, feature_scaler, target_scaler, feature_columns, raw_rows


@torch.inference_mode()
def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    batches = []
    loader = DataLoader(TensorDataset(x), batch_size=512, shuffle=False)
    for (batch,) in loader:
        batches.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(batches, axis=0)


def inverse_last_horizon(values: np.ndarray, scaler) -> np.ndarray:
    return scaler.inverse_transform(values[:, [-1]]).ravel()


def evaluate(model, tensors, target_scaler, device):
    x, y = tensors
    prediction_scaled = predict(model, x, device)
    truth = inverse_last_horizon(y.numpy(), target_scaler)
    prediction = inverse_last_horizon(prediction_scaled, target_scaler)
    return truth, prediction


def metric_row(condition: str, truth: np.ndarray, prediction: np.ndarray) -> dict:
    residual = truth - prediction
    return {
        "condition": condition,
        "n_test_windows": len(truth),
        "mse": mean_squared_error(truth, prediction),
        "rmse": math.sqrt(mean_squared_error(truth, prediction)),
        "r2": r2_score(truth, prediction),
        "mape_percent": 100.0 * mean_absolute_percentage_error(truth, prediction),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=1)),
    }


def train_transfer(model, tensors, device):
    # Paper protocol: retain the lowest-level source representation and update
    # LSTM layers 1--2 plus the forecast head.
    for name, parameter in model.lstm.named_parameters():
        if "l0" in name:
            parameter.requires_grad = False

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()
    pin_memory = device.type == "cuda"
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(*tensors["train"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory,
        generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(*tensors["validation"]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
    )

    best_loss = math.inf
    best_state = None
    stale = 0
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_sum = 0.0
        train_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_sum += loss.item()
            train_batches += 1

        model.eval()
        validation_sum = 0.0
        validation_batches = 0
        with torch.inference_mode():
            for x, y in validation_loader:
                x = x.to(device, non_blocking=pin_memory)
                y = y.to(device, non_blocking=pin_memory)
                validation_sum += criterion(model(x), y).item()
                validation_batches += 1

        train_loss = train_sum / train_batches
        validation_loss = validation_sum / validation_batches
        history.append(
            {"epoch": epoch, "train_loss_scaled": train_loss, "validation_loss_scaled": validation_loss}
        )
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} validation={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Fine-tuning did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return pd.DataFrame(history), epoch, best_loss, elapsed


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors, _, target_scaler, feature_columns, raw_rows = load_target_splits()

    source_state = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=True)
    before_model = LSTMForecaster(len(feature_columns)).to(device)
    before_model.load_state_dict(source_state)
    target_truth, prediction_before = evaluate(before_model, tensors["test"], target_scaler, device)

    transfer_model = LSTMForecaster(len(feature_columns)).to(device)
    transfer_model.load_state_dict(source_state)
    history, epochs, best_validation_loss, elapsed = train_transfer(transfer_model, tensors, device)
    truth_after, prediction_after = evaluate(transfer_model, tensors["test"], target_scaler, device)
    if not np.array_equal(target_truth, truth_after):
        raise RuntimeError("Before-TL and after-TL evaluations do not use identical test targets")

    source = pd.read_csv(SOURCE_PREDICTIONS)
    source = source[(source["model"] == "LSTM") & (source["seed"] == SEED)].copy()
    if source.empty:
        raise RuntimeError("Missing LSTM seed-42 source predictions")
    source_truth = source["truth"].to_numpy(float)
    source_prediction = source["prediction"].to_numpy(float)
    source_residual = source_truth - source_prediction
    before_residual = target_truth - prediction_before
    after_residual = target_truth - prediction_after

    rows = []
    for condition, truth, prediction in (
        ("Source reference", source_truth, source_prediction),
        ("Target before TL (direct source model)", target_truth, prediction_before),
        ("Target after TL", target_truth, prediction_after),
    ):
        rows.append(metric_row(condition, truth, prediction))
    metrics = pd.DataFrame(rows)
    metrics["wasserstein_to_source"] = [
        0.0,
        wasserstein_distance(source_residual, before_residual),
        wasserstein_distance(source_residual, after_residual),
    ]
    metrics["ks_to_source"] = [
        0.0,
        ks_2samp(source_residual, before_residual).statistic,
        ks_2samp(source_residual, after_residual).statistic,
    ]

    predictions = []
    for condition, truth, prediction in (
        ("Source reference", source_truth, source_prediction),
        ("Target before TL (direct source model)", target_truth, prediction_before),
        ("Target after TL", target_truth, prediction_after),
    ):
        predictions.append(
            pd.DataFrame(
                {
                    "condition": condition,
                    "index": np.arange(len(truth)),
                    "truth": truth,
                    "prediction": prediction,
                    "residual": truth - prediction,
                }
            )
        )
    pd.concat(predictions, ignore_index=True).to_csv(OUTPUT_PREDICTIONS, index=False)
    metrics.to_csv(OUTPUT_METRICS, index=False)
    history.to_csv(OUTPUT_HISTORY, index=False)
    torch.save(transfer_model.state_dict(), OUTPUT_CHECKPOINT)

    metadata = {
        "purpose": "Controlled before/after-TL residual alignment for manuscript Figure 13",
        "seed": SEED,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(HERE.parents[1])),
        "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
        "input_sha256": {
            str(path.relative_to(HERE.parents[1])): sha256(path)
            for path in (
                FEATURE_SCALER_FILE,
                TARGET_SCALER_FILE,
                SOURCE_PREDICTIONS,
                TARGET_TRAIN,
                TARGET_VALIDATION,
                TARGET_TEST,
            )
        },
        "raw_rows": raw_rows,
        "windows": {name: len(value[0]) for name, value in tensors.items()},
        "feature_columns": feature_columns,
        "n_past": N_PAST,
        "n_future": N_FUTURE,
        "target_train_usage": "100% of the designated healthy target-domain training split",
        "same_target_test_before_after": True,
        "fine_tuning": {
            "frozen_lstm_layer": 0,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "epochs_run": epochs,
            "best_validation_loss_scaled": best_validation_loss,
            "wall_clock_seconds": elapsed,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(metadata["fine_tuning"], indent=2), flush=True)


if __name__ == "__main__":
    main()
