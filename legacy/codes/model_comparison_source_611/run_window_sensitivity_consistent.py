"""Re-run the source-domain window sensitivity study with the deployed LSTM.

Every configuration uses the same source data, chronological split, feature
scaling, 64-unit three-layer LSTM, batch size 64, learning rate 0.005, early
stopping rule, and seed as the controlled source-model comparison.  Results
are checkpointed to CSV after every configuration so the sweep can be resumed.
"""

from __future__ import annotations

import math
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_FILE = ROOT / "codes" / "data" / "source_all_data_b1.csv"
OUT_FILE = HERE / "results" / "window_sensitivity_consistent.csv"
BASELINE_FILE = HERE / "results" / "source_model_metrics.csv"

HISTORY_LENGTHS = [20, 30, 40, 50, 60]
PREDICTION_HORIZONS = [10, 15, 20, 25, 30]
SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.005
DROPOUT = 0.2
TARGET = "motor6_temperature"
TIMESTAMP = "timestamp"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def make_windows(features: np.ndarray, target: np.ndarray, n_past: int, n_future: int):
    count = len(features) - n_past - n_future + 1
    x = np.empty((count, n_past, features.shape[1]), dtype=np.float32)
    y = np.empty((count, n_future), dtype=np.float32)
    for i in range(count):
        x[i] = features[i : i + n_past]
        y[i] = target[i + n_past : i + n_past + n_future, 0]
    return torch.from_numpy(x), torch.from_numpy(y)


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size=64,
            num_layers=3,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Linear(64, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.head(self.dropout(x[:, -1, :]))


def train_and_evaluate(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    scaler: MinMaxScaler,
    n_past: int,
    n_future: int,
    device: torch.device,
) -> dict[str, float]:
    set_seed(SEED)
    tensors = {
        name: make_windows(features, target, n_past, n_future)
        for name, (features, target) in arrays.items()
    }
    pin = device.type == "cuda"
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(*tensors["train"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(*tensors["validation"]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin,
    )
    model = LSTMForecaster(tensors["train"][0].shape[-1], n_future).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_loss = math.inf
    best_state = None
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=pin)
            y = y.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        losses = []
        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(device, non_blocking=pin)
                y = y.to(device, non_blocking=pin)
                losses.append(criterion(model(x), y).item())
        val_loss = float(np.mean(losses))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    x_test, y_test = tensors["test"]
    predictions = []
    loader = DataLoader(TensorDataset(x_test), batch_size=512, shuffle=False)
    with torch.inference_mode():
        for (x,) in loader:
            predictions.append(model(x.to(device)).cpu())
    pred_scaled = torch.cat(predictions).numpy()
    true_scaled = y_test.numpy()
    true = scaler.inverse_transform(true_scaled.reshape(-1, 1)).reshape(true_scaled.shape)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
    true_last = true[:, -1]
    pred_last = pred[:, -1]
    mse = mean_squared_error(true_last, pred_last)
    elapsed = time.perf_counter() - started
    return {
        "n_past": n_past,
        "n_future": n_future,
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(true_last, pred_last)),
        "mape_percent": float(100 * mean_absolute_percentage_error(true_last, pred_last)),
        "epochs": epoch,
        "training_seconds": elapsed,
        "seed": SEED,
        "hidden_dim": 64,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }


def load_arrays():
    data = pd.read_csv(DATA_FILE).dropna().reset_index(drop=True)
    n = len(data)
    frames = {
        "train": data.iloc[: int(0.70 * n)].copy(),
        "validation": data.iloc[int(0.70 * n) : int(0.85 * n)].copy(),
        "test": data.iloc[int(0.85 * n) :].copy(),
    }
    features = [column for column in data.columns if column != TIMESTAMP]
    feature_scaler = MinMaxScaler().fit(frames["train"][features])
    target_scaler = MinMaxScaler().fit(frames["train"][[TARGET]])
    arrays = {}
    for name, frame in frames.items():
        arrays[name] = (
            feature_scaler.transform(frame[features]).astype(np.float32),
            target_scaler.transform(frame[[TARGET]]).astype(np.float32),
        )
    return arrays, target_scaler


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(OUT_FILE) if OUT_FILE.exists() else pd.DataFrame()
    completed = set()
    if not existing.empty:
        completed = set(zip(existing["n_past"].astype(int), existing["n_future"].astype(int)))

    # Reuse the already audited controlled-comparison checkpoint for the
    # canonical 30/10 point so Fig. 10, Table 6, and Fig. 11 are identical.
    if (30, 10) not in completed:
        baseline = pd.read_csv(BASELINE_FILE)
        baseline = baseline[(baseline["model"] == "LSTM") & (baseline["split"] == "test")].iloc[0]
        row = {
            "n_past": 30,
            "n_future": 10,
            "mse": float(baseline["mse_last"]),
            "rmse": float(baseline["rmse_last"]),
            "r2": float(baseline["r2_last"]),
            "mape_percent": float(baseline["mape_last_percent"]),
            "epochs": int(baseline["epochs_run"]),
            "training_seconds": float(baseline["training_seconds"]),
            "seed": SEED,
            "hidden_dim": 64,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        }
        existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        existing.sort_values(["n_past", "n_future"]).to_csv(OUT_FILE, index=False)
        completed.add((30, 10))

    arrays, scaler = load_arrays()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = len(HISTORY_LENGTHS) * len(PREDICTION_HORIZONS)
    print(f"device={device}; completed={len(completed)}/{total}", flush=True)
    for n_past in HISTORY_LENGTHS:
        for n_future in PREDICTION_HORIZONS:
            if (n_past, n_future) in completed:
                continue
            print(f"START n_past={n_past}, n_future={n_future}", flush=True)
            row = train_and_evaluate(arrays, scaler, n_past, n_future, device)
            existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            existing = existing.drop_duplicates(["n_past", "n_future"], keep="last")
            existing.sort_values(["n_past", "n_future"]).to_csv(OUT_FILE, index=False)
            completed.add((n_past, n_future))
            print(
                f"DONE {len(completed)}/{total}: R2={row['r2']:.4f}, "
                f"MSE={row['mse']:.4f}, epochs={row['epochs']}, "
                f"seconds={row['training_seconds']:.1f}",
                flush=True,
            )
    print(OUT_FILE, flush=True)


if __name__ == "__main__":
    main()
