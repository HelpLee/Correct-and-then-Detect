"""Temporal windows are constructed independently inside each data partition."""
import numpy as np
import torch

def features(frame):
    return frame.drop(columns=['timestamp', 'label'], errors='ignore')

def windows(frame, feature_scaler, target_scaler, past=30, future=10):
    if len(frame) < past + future:
        raise ValueError('A partition needs at least past + future observations.')
    x = feature_scaler.transform(features(frame)).astype('float32')
    y = target_scaler.transform(frame[['motor6_temperature']]).astype('float32').ravel()
    count = len(x) - past - future + 1
    inputs = torch.from_numpy(x).unfold(0, past, 1).permute(0, 2, 1)[:count].contiguous()
    targets = np.lib.stride_tricks.sliding_window_view(y[past:], future).copy()
    return inputs, targets
