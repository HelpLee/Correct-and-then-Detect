"""Section 6.1.1 source-domain model and latency comparison.

This is an isolated benchmark and does not modify the original LSTM code or
checkpoints. LSTM, XGBoost and Transformer models share the same data split,
sliding windows, target horizon, model-selection rule and evaluation procedure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
CODES_DIR = HERE.parent
DATA_FILE = CODES_DIR / "data" / "source_all_data_b1.csv"
RESULTS_DIR = HERE / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

TIMESTAMP_COL = "timestamp"
TARGET_FEATURE = "motor6_temperature"
N_PAST = 30
N_FUTURE = 10
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.005  # Setting in the code/archive that generated Section 6.1.1.
DROPOUT = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Match the original training path and allow cuDNN to select its fastest
        # implementation for the fixed (30, 18) input shape.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def make_windows(features: np.ndarray, target: np.ndarray):
    count = len(features) - N_PAST - N_FUTURE + 1
    x = np.empty((count, N_PAST, features.shape[1]), dtype=np.float32)
    y = np.empty((count, N_FUTURE), dtype=np.float32)
    for i in range(count):
        x[i] = features[i : i + N_PAST]
        y[i] = target[i + N_PAST : i + N_PAST + N_FUTURE, 0]
    return torch.from_numpy(x), torch.from_numpy(y)


def load_data():
    data = pd.read_csv(DATA_FILE).dropna().reset_index(drop=True)
    n = len(data)
    splits = {
        "train": data.iloc[: int(0.70 * n)].copy(),
        "validation": data.iloc[int(0.70 * n) : int(0.85 * n)].copy(),
        "test": data.iloc[int(0.85 * n) :].copy(),
    }
    feature_columns = [c for c in data.columns if c != TIMESTAMP_COL]
    feature_scaler = MinMaxScaler().fit(splits["train"][feature_columns])
    target_scaler = MinMaxScaler().fit(splits["train"][[TARGET_FEATURE]])

    tensors = {}
    for name, frame in splits.items():
        features = feature_scaler.transform(frame[feature_columns]).astype(np.float32)
        target = target_scaler.transform(frame[[TARGET_FEATURE]]).astype(np.float32)
        tensors[name] = make_windows(features, target)
    return tensors, feature_scaler, target_scaler, feature_columns


class LSTMForecaster(nn.Module):
    """Architecture in the code/archive that generated Section 6.1.1."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size=64,
            num_layers=3,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Linear(64, N_FUTURE)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.head(self.dropout(x[:, -1, :]))


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        d_model = 64
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, N_PAST, d_model))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_FUTURE)

    def forward(self, x):
        x = self.input_projection(x) + self.position[:, : x.shape[1]]
        x = self.encoder(x)
        return self.head(self.norm(x[:, -1, :]))


MODEL_FACTORIES = {
    "LSTM": LSTMForecaster,
    "Transformer": TransformerForecaster,
}
MODEL_NAMES = ["LSTM", "XGBoost", "Transformer"]


def train_model(model, train_tensors, val_tensors, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = math.inf
    best_state = None
    stale = 0
    history = []
    # Follow the original source code: create CPU batches first, then transfer
    # each contiguous batch to the accelerator. Moving the full data set to the
    # GPU and gathering random indices per step was substantially slower.
    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(*train_tensors), batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(*val_tensors), batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=use_pin_memory,
    )
    if device.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_sum = 0.0
        train_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=use_pin_memory)
            y = y.to(device, non_blocking=use_pin_memory)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            train_sum += loss.item()
            train_batches += 1

        model.eval()
        val_sum = 0.0
        val_batches = 0
        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(device, non_blocking=use_pin_memory)
                y = y.to(device, non_blocking=use_pin_memory)
                loss = criterion(model(x), y)
                val_sum += loss.item()
                val_batches += 1

        train_loss = train_sum / train_batches
        val_loss = val_sum / val_batches
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"  epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    return pd.DataFrame(history), training_seconds, len(history), best_loss


def train_xgboost(train_tensors, val_tensors, seed):
    train_x, train_y = train_tensors
    val_x, val_y = val_tensors
    train_x = train_x.numpy().reshape(len(train_x), -1)
    val_x = val_x.numpy().reshape(len(val_x), -1)
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        multi_strategy="one_output_per_tree",
        random_state=seed,
        early_stopping_rounds=PATIENCE,
    )
    started = time.perf_counter()
    model.fit(
        train_x,
        train_y.numpy(),
        eval_set=[(val_x, val_y.numpy())],
        verbose=False,
    )
    training_seconds = time.perf_counter() - started
    values = model.evals_result()["validation_0"]["rmse"]
    history = pd.DataFrame(
        {"epoch": np.arange(1, len(values) + 1), "train_loss": np.nan,
         "validation_loss": np.square(values)}
    )
    best_iteration = int(model.best_iteration) + 1
    best_loss = float(values[model.best_iteration]) ** 2
    return model, history, training_seconds, best_iteration, best_loss


@torch.inference_mode()
def predict(model, x, device, batch_size=512):
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    for (batch,) in loader:
        outputs.append(model(batch.to(device)).cpu())
    return torch.cat(outputs).numpy()


def metric_row(model_name, seed, split, y_scaled, pred_scaled, scaler):
    true_all = scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(y_scaled.shape)
    pred_all = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
    true_last = true_all[:, -1]
    pred_last = pred_all[:, -1]
    return {
        "model": model_name,
        "seed": seed,
        "split": split,
        "mse_last": mean_squared_error(true_last, pred_last),
        "rmse_last": mean_squared_error(true_last, pred_last) ** 0.5,
        "r2_last": r2_score(true_last, pred_last),
        "mape_last_percent": 100 * mean_absolute_percentage_error(true_last, pred_last),
        "mse_all_horizons": mean_squared_error(true_all.reshape(-1), pred_all.reshape(-1)),
        "r2_all_horizons": r2_score(true_all.reshape(-1), pred_all.reshape(-1)),
    }, true_last, pred_last


@torch.inference_mode()
def benchmark_latency(model, input_dim, batch_size, device, warmup=100, repeats=500):
    model.eval()
    cpu_input = torch.randn(batch_size, N_PAST, input_dim, dtype=torch.float32)
    device_input = cpu_input.to(device)

    for _ in range(warmup):
        model(device_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    model_only = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        model(device_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_only.append((time.perf_counter_ns() - start) / 1e6)

    end_to_end = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        x = cpu_input.to(device)
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_to_end.append((time.perf_counter_ns() - start) / 1e6)

    def stats(values, scope):
        values = np.asarray(values)
        mean_ms = float(values.mean())
        return {
            "scope": scope,
            "latency_mean_ms": mean_ms,
            "latency_std_ms": float(values.std(ddof=1)),
            "latency_median_ms": float(np.median(values)),
            "latency_p95_ms": float(np.percentile(values, 95)),
            "throughput_samples_per_second": 1000.0 * batch_size / mean_ms,
        }

    return [stats(model_only, "model_only"), stats(end_to_end, "end_to_end")]


def benchmark_xgboost_latency(model, input_dim, batch_size, warmup=100, repeats=500):
    sample = np.random.default_rng(123).normal(
        size=(batch_size, N_PAST * input_dim)
    ).astype(np.float32)
    for _ in range(warmup):
        model.predict(sample)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        model.predict(sample)
        timings.append((time.perf_counter_ns() - started) / 1e6)
    values = np.asarray(timings)
    mean_ms = float(values.mean())
    return {
        "scope": "end_to_end",
        "latency_mean_ms": mean_ms,
        "latency_std_ms": float(values.std(ddof=1)),
        "latency_median_ms": float(np.median(values)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "throughput_samples_per_second": 1000.0 * batch_size / mean_ms,
    }


def make_plots(metrics, latency, predictions, histories):
    test = metrics.loc[metrics["split"] == "test"].copy()
    order = MODEL_NAMES
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), dpi=160)
    for ax, metric, label, lower_better in [
        (axes[0], "rmse_last", "Test RMSE", True),
        (axes[1], "r2_last", "Test R$^2$", False),
        (axes[2], "mape_last_percent", "Test MAPE (%)", True),
    ]:
        values = [test.loc[test.model == m, metric].mean() for m in order]
        ax.bar(order, values, color=["#4472C4", "#70AD47", "#ED7D31"])
        ax.set_ylabel(label)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Source-domain nominal modeling: architecture comparison")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "source_model_accuracy_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "source_model_accuracy_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    batch1 = latency[(latency.batch_size == 1) & (latency.scope == "end_to_end")]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    values = [batch1.loc[batch1.model == m, "latency_median_ms"].mean() for m in order]
    ax.bar(order, values, color=["#4472C4", "#70AD47", "#ED7D31"])
    ax.set_ylabel("Median inference latency (ms), batch=1")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_title("Digital-twin end-to-end inference latency")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "source_model_latency_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "source_model_latency_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    for model_name in order:
        frame = predictions[predictions.model == model_name].sort_values("index").head(500)
        ax.plot(frame["index"], frame["prediction"], linewidth=1.2, label=model_name)
    truth = predictions[predictions.model == order[0]].sort_values("index").head(500)
    ax.plot(truth["index"], truth["truth"], color="black", linewidth=1.4, label="True")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Motor 6 temperature")
    ax.legend(ncol=4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "source_model_test_predictions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for (model_name, seed), frame in histories.groupby(["model", "seed"]):
        ax.plot(frame.epoch, frame.validation_loss, label=f"{model_name}, seed {seed}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE loss (scaled)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "source_model_validation_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors, feature_scaler, target_scaler, feature_columns = load_data()
    joblib.dump(feature_scaler, RESULTS_DIR / "feature_scaler.pkl")
    joblib.dump(target_scaler, RESULTS_DIR / "target_scaler.pkl")

    metric_rows, latency_rows, prediction_frames, history_frames, training_rows = [], [], [], [], []
    for seed in args.seeds:
        for model_name in args.models:
            print(f"\n=== {model_name}, seed {seed}, device={device} ===")
            set_seed(seed)
            if model_name == "XGBoost":
                model, history, training_seconds, epochs_run, best_val_loss = train_xgboost(
                    tensors["train"], tensors["validation"], seed
                )
                parameter_count = np.nan
            else:
                model = MODEL_FACTORIES[model_name](len(feature_columns)).to(device)
                parameter_count = sum(p.numel() for p in model.parameters())
                history, training_seconds, epochs_run, best_val_loss = train_model(
                    model, tensors["train"], tensors["validation"], device
                )
            history.insert(0, "seed", seed)
            history.insert(0, "model", model_name)
            history_frames.append(history)
            if model_name == "XGBoost":
                checkpoint = CHECKPOINT_DIR / f"xgboost_seed{seed}.json"
                model.save_model(checkpoint)
            else:
                checkpoint = CHECKPOINT_DIR / f"{model_name.lower()}_seed{seed}.pth"
                torch.save(model.state_dict(), checkpoint)
            model_size_bytes = checkpoint.stat().st_size

            training_rows.append({
                "model": model_name, "seed": seed, "parameters": parameter_count,
                "model_size_bytes": model_size_bytes,
                "training_seconds": training_seconds, "epochs_run": epochs_run,
                "seconds_per_epoch": training_seconds / epochs_run,
                "best_validation_loss_scaled": best_val_loss,
            })

            for split in ("validation", "test"):
                x, y = tensors[split]
                if model_name == "XGBoost":
                    pred = model.predict(x.numpy().reshape(len(x), -1))
                else:
                    pred = predict(model, x, device)
                row, true_last, pred_last = metric_row(
                    model_name, seed, split, y.numpy(), pred, target_scaler
                )
                row["parameters"] = parameter_count
                row["model_size_bytes"] = model_size_bytes
                row["training_seconds"] = training_seconds
                row["epochs_run"] = epochs_run
                metric_rows.append(row)
                if split == "test":
                    prediction_frames.append(pd.DataFrame({
                        "model": model_name, "seed": seed,
                        "index": np.arange(len(true_last)), "truth": true_last,
                        "prediction": pred_last, "residual": true_last - pred_last,
                    }))

            for batch_size, repeats in ((1, 500), (64, 200)):
                if model_name == "XGBoost":
                    latency_values = [benchmark_xgboost_latency(
                        model, len(feature_columns), batch_size, repeats=repeats
                    )]
                else:
                    latency_values = benchmark_latency(
                        model, len(feature_columns), batch_size, device, repeats=repeats
                    )
                for latency in latency_values:
                    latency_rows.append({
                        "model": model_name, "seed": seed, "device": str(device),
                        "batch_size": batch_size, "repeats": repeats,
                        "parameters": parameter_count, "model_size_bytes": model_size_bytes,
                        **latency,
                    })

    metrics = pd.DataFrame(metric_rows)
    latency = pd.DataFrame(latency_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    histories = pd.concat(history_frames, ignore_index=True)
    training = pd.DataFrame(training_rows)
    metrics.to_csv(RESULTS_DIR / "source_model_metrics.csv", index=False)
    latency.to_csv(RESULTS_DIR / "source_model_latency.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "source_model_test_predictions.csv", index=False)
    histories.to_csv(RESULTS_DIR / "source_model_training_history.csv", index=False)
    training.to_csv(RESULTS_DIR / "source_model_training_time.csv", index=False)
    make_plots(metrics, latency, predictions, histories)

    metadata = {
        "data_file": str(DATA_FILE.relative_to(CODES_DIR)),
        "raw_rows": 104000,
        "split": {"train": "0-70%", "validation": "70-85%", "test": "85-100%"},
        "window": {"n_past": N_PAST, "n_future": N_FUTURE},
        "features": feature_columns,
        "target": TARGET_FEATURE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "MSELoss",
        "model_configuration": {
            "LSTM": {"hidden_size": 64, "layers": 3, "dropout": DROPOUT},
            "Transformer": {
                "d_model": 64, "heads": 4, "feedforward_size": 128,
                "layers": 3, "dropout": DROPOUT,
            },
            "XGBoost": {
                "n_estimators": 100, "learning_rate": 0.05,
                "max_depth": 8, "early_stopping_rounds": PATIENCE,
            },
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "xgboost": xgb.__version__,
        "sklearn": sklearn.__version__,
        "latency_protocol": "100 warm-up iterations; synchronized timings; batch 1: 500 repeats; batch 64: 200 repeats; model-only and CPU-to-device-plus-model scopes.",
        "xgboost_latency_note": "The reported end-to-end path accepts CPU NumPy input and includes XGBoost's DMatrix/device-mismatch fallback to the CUDA booster.",
    }
    (RESULTS_DIR / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nTest metrics:")
    print(metrics[metrics.split == "test"].to_string(index=False))
    print("\nLatency:")
    print(latency.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
