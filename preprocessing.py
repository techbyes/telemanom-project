from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
HF_DATASET_ID = "appleparan/telemanom"
LABELS_FILENAME = "labeled_anomalies.csv"
MAX_FEATURE_DIM = 55


@dataclass
class PreprocessConfig:
    sequence_length: int = 50
    stride: int = 10
    val_size: float = 0.15
    random_state: int = RANDOM_STATE
    per_feature_normalize: bool = False


@dataclass
class ModelInterface:
    input_shape: tuple[str, int, int]
    num_classes: int
    normalization: str
    sequence_length: int
    windowing: str
    feature_dim: int
    class_names: list[str] = field(default_factory=lambda: ["normal", "anomaly"])

    def as_dict(self) -> dict[str, object]:
        batch, seq_len, feat = self.input_shape
        return {
            "input_shape": f"({batch}, {seq_len}, {feat})",
            "num_classes": self.num_classes,
            "normalization": self.normalization,
            "sequence_length": self.sequence_length,
            "windowing": self.windowing,
            "feature_dim": self.feature_dim,
            "class_names": self.class_names,
        }


def feature_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c != "timestep"]


def hf_split_to_array(frame: pd.DataFrame) -> np.ndarray:
    feats = feature_columns(frame.columns)
    return clean_array(frame[feats].to_numpy(dtype=np.float64))


def clean_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D sequence data, got shape {arr.shape}")

    if np.isnan(arr).any():
        col_means = np.nanmean(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_means, inds[1])
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def parse_anomaly_sequences(value: object) -> list[list[int]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    raise TypeError(f"Unsupported anomaly_sequences type: {type(value)}")


def anomaly_mask(length: int, sequences: Iterable[Iterable[int]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in sequences:
        mask[int(start) : int(end) + 1] = True
    return mask


def make_sliding_windows(
    series: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be >= 1")
    if series.ndim != 2:
        raise ValueError(f"series must be 2D (T, F), got {series.shape}")

    n = series.shape[0]
    if n < window_size:
        return np.empty((0, window_size, series.shape[1]), dtype=series.dtype)

    starts = np.arange(0, n - window_size + 1, stride)
    return np.stack([series[i : i + window_size] for i in starts], axis=0)


def window_binary_labels(
    timestep_mask: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    n = len(timestep_mask)
    if n < window_size:
        return np.empty(0, dtype=np.int64)

    starts = np.arange(0, n - window_size + 1, stride)
    return np.array(
        [int(timestep_mask[i : i + window_size].any()) for i in starts],
        dtype=np.int64,
    )


def pad_feature_dim(array: np.ndarray, target_dim: int = MAX_FEATURE_DIM) -> np.ndarray:
    arr = clean_array(array)
    t, f = arr.shape
    if f == target_dim:
        return arr
    if f > target_dim:
        return arr[:, :target_dim]
    return np.hstack([arr, np.zeros((t, target_dim - f), dtype=arr.dtype)])


class TelemanomFeatureScaler:
    def __init__(self, per_feature_normalize: bool = False) -> None:
        self.per_feature_normalize = per_feature_normalize
        self.scaler = StandardScaler()
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._fitted = False

    def fit(self, arrays: list[np.ndarray]) -> TelemanomFeatureScaler:
        stacked = np.vstack([clean_array(a) for a in arrays if len(a)])
        self.scaler.fit(stacked)
        if self.per_feature_normalize:
            scaled = self.scaler.transform(stacked)
            self._min = scaled.min(axis=0)
            self._max = scaled.max(axis=0)
        self._fitted = True
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler is not fitted. Call fit() first.")
        flat = self.scaler.transform(clean_array(array))
        if self.per_feature_normalize and self._min is not None and self._max is not None:
            span = self._max - self._min
            span[span == 0] = 1.0
            flat = (flat - self._min) / span
        return flat

    def normalization_description(self) -> str:
        base = f"StandardScaler, SMAP padded to {MAX_FEATURE_DIM} features"
        if self.per_feature_normalize:
            return base + ", then MinMax per feature"
        return base


def build_channel_windows(
    train_array: np.ndarray,
    test_array: np.ndarray,
    anomaly_sequences: list[list[int]] | None,
    config: PreprocessConfig,
    scaler: TelemanomFeatureScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build windows from train and test streams separately (never concatenate).

    Train windows come from normal-only train CSV data. Test windows and labels
    come from the held-out test CSV (anomaly intervals when provided).
    """
    train_scaled = scaler.transform(train_array)
    test_scaled = scaler.transform(test_array)

    train_windows = make_sliding_windows(
        train_scaled, config.sequence_length, config.stride
    )
    test_windows = make_sliding_windows(
        test_scaled, config.sequence_length, config.stride
    )

    if anomaly_sequences:
        mask = anomaly_mask(len(test_scaled), anomaly_sequences)
        test_labels = window_binary_labels(mask, config.sequence_length, config.stride)
    else:
        test_labels = np.zeros(len(test_windows), dtype=np.int64)

    return train_windows, test_windows, test_labels


def chronological_train_val_split(
    x: np.ndarray,
    val_size: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Split windows in time order: first (1 - val_size) for train, tail for validation."""
    n = len(x)
    if n == 0:
        return x, x
    split = int(n * (1.0 - val_size))
    if n > 1:
        split = min(max(split, 1), n - 1)
    return x[:split], x[split:]


def build_model_interface(
    config: PreprocessConfig,
    scaler: TelemanomFeatureScaler,
    feature_dim: int,
    num_classes: int = 2,
) -> ModelInterface:
    return ModelInterface(
        input_shape=("batch_size", config.sequence_length, feature_dim),
        num_classes=num_classes,
        normalization=scaler.normalization_description(),
        sequence_length=config.sequence_length,
        windowing=f"sliding windows, len={config.sequence_length}, stride={config.stride}",
        feature_dim=feature_dim,
    )


def class_distribution(y: np.ndarray) -> pd.DataFrame:
    unique, counts = np.unique(y, return_counts=True)
    total = int(counts.sum())
    frame = pd.DataFrame(
        {
            "label": unique.astype(int),
            "class_name": np.where(unique == 1, "anomaly", "normal"),
            "count": counts.astype(int),
            "fraction": counts / total,
        }
    )
    if len(frame) == 2:
        frame.attrs["imbalance_ratio"] = float(frame["count"].max() / max(frame["count"].min(), 1))
    return frame


def sequence_length_stats(lengths: list[int]) -> pd.DataFrame:
    arr = np.array(lengths, dtype=np.int64)
    return pd.DataFrame(
        {
            "metric": ["min", "median", "mean", "max", "std"],
            "timesteps": [arr.min(), np.median(arr), arr.mean(), arr.max(), arr.std()],
        }
    )
