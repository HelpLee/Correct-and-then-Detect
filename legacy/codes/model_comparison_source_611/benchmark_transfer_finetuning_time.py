"""Wall-clock timing of the paper's LSTM transfer-learning fine-tuning stage."""

from __future__ import annotations

import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
CODES = HERE.parent
MODELS = CODES / "models_reg_v3"
RESULTS = HERE / "results"
TRAIN_DATA = CODES / "data" / "target_train_data_b1.csv"
VALIDATION_DATA = CODES / "data" / "target_val_data_b1.csv"
SOURCE_CHECKPOINT = MODELS / "source_best_model_v3_b1.pth"
FEATURE_SCALER = MODELS / "feature_scaler_v3_b1.pkl"
TARGET_SCALER = MODELS / "target_scaler_v3_b1.pkl"

SEEDS = (42, 43, 44, 45, 46)
TARGET = "motor6_temperature"
N_PAST, N_FUTURE = 30, 10
BATCH_SIZE = 64
MAX_EPOCHS = 500
PATIENCE = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 64, 3, dropout=0.2, batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, N_FUTURE)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(self.dropout(x[:, -1, :]))


def windows(features: np.ndarray, target: np.ndarray):
    count = len(features) - N_PAST - N_FUTURE + 1
    x = np.empty((count, N_PAST, features.shape[1]), dtype=np.float32)
    y = np.empty((count, N_FUTURE), dtype=np.float32)
    for i in range(count):
        x[i] = features[i : i + N_PAST]
        y[i] = target[i + N_PAST : i + N_PAST + N_FUTURE, 0]
    return torch.from_numpy(x), torch.from_numpy(y)


def load_data():
    train = pd.read_csv(TRAIN_DATA).dropna().reset_index(drop=True)
    validation = pd.read_csv(VALIDATION_DATA).dropna().reset_index(drop=True)
    columns = [column for column in train.columns if column != "timestamp"]
    feature_scaler = joblib.load(FEATURE_SCALER)
    target_scaler = joblib.load(TARGET_SCALER)

    result = {}
    for name, split in (("train", train), ("validation", validation)):
        x = feature_scaler.transform(split[columns]).astype(np.float32)
        y = target_scaler.transform(split[[TARGET]]).astype(np.float32)
        result[name] = windows(x, y)
    return result, len(train), len(validation), columns


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_seed(seed: int, tensors, input_dim: int, device: torch.device):
    set_seed(seed)
    model = LSTMModel(input_dim).to(device)
    state = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    for name, parameter in model.lstm.named_parameters():
        if "l0" in name:
            parameter.requires_grad = False

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()
    pin = device.type == "cuda"
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(*tensors["train"]), batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=pin, generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(*tensors["validation"]), batch_size=BATCH_SIZE,
        shuffle=False, pin_memory=pin,
    )

    best_loss = math.inf
    best_state = None
    stale = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
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
        total = 0.0
        batches = 0
        with torch.inference_mode():
            for x, y in validation_loader:
                x = x.to(device, non_blocking=pin)
                y = y.to(device, non_blocking=pin)
                total += criterion(model(x), y).item()
                batches += 1
        validation_loss = total / batches
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
    model.load_state_dict(best_state)
    return {
        "seed": seed,
        "fine_tuning_seconds": elapsed,
        "epochs_run": epoch,
        "seconds_per_epoch": elapsed / epoch,
        "best_validation_loss_scaled": best_loss,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    tensors, raw_train_rows, raw_validation_rows, columns = load_data()
    rows = []
    for seed in SEEDS:
        row = run_seed(seed, tensors, len(columns), device)
        rows.append(row)
        print(row, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "transfer_finetuning_time.csv", index=False)
    summary = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seeds": list(SEEDS),
        "raw_train_rows": raw_train_rows,
        "raw_validation_rows": raw_validation_rows,
        "target_train_fraction": 1.0,
        "fraction_denominator": "D_tgt^train",
        "train_windows": len(tensors["train"][0]),
        "validation_windows": len(tensors["validation"][0]),
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "frozen_lstm_layer": 0,
        "scope": "Fine-tuning train and validation loops; excludes preprocessing and checkpoint I/O.",
        "mean_seconds": float(frame.fine_tuning_seconds.mean()),
        "sample_std_seconds": float(frame.fine_tuning_seconds.std(ddof=1)),
        "mean_epochs": float(frame.epochs_run.mean()),
    }
    (RESULTS / "transfer_finetuning_time_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
