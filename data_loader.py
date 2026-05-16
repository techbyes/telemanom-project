from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset as TorchDataset

from preprocessing import (
    HF_DATASET_ID,
    LABELS_FILENAME,
    MAX_FEATURE_DIM,
    ModelInterface,
    PreprocessConfig,
    TelemanomFeatureScaler,
    build_channel_windows,
    build_model_interface,
    chronological_train_val_split,
    class_distribution,
    hf_split_to_array,
    pad_feature_dim,
    parse_anomaly_sequences,
)

SplitName = Literal["train", "validation", "test"]


@dataclass
class TelemanomPipelineResult:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    model_interface: ModelInterface
    config: PreprocessConfig
    channel_ids: list[str]
    label_distribution: pd.DataFrame


def load_hf_labels(dataset_id: str = HF_DATASET_ID) -> pd.DataFrame:
    path = hf_hub_download(dataset_id, LABELS_FILENAME, repo_type="dataset")
    labels = pd.read_csv(path)
    labels["anomaly_sequences"] = labels["anomaly_sequences"].apply(parse_anomaly_sequences)
    return labels


def list_hf_channels(dataset_id: str = HF_DATASET_ID) -> list[str]:
    return sorted(get_dataset_config_names(dataset_id))


def load_hf_channel(channel_id: str, dataset_id: str = HF_DATASET_ID) -> DatasetDict:
    return load_dataset(dataset_id, name=channel_id)


def channel_to_arrays(channel: DatasetDict) -> tuple[np.ndarray, np.ndarray]:
    train_arr = hf_split_to_array(channel["train"].to_pandas())
    test_arr = hf_split_to_array(channel["test"].to_pandas())
    return train_arr, test_arr


class TelemanomSequenceDataset(TorchDataset):
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have the same length")
        self.sequences = sequences
        self.labels = labels
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(self.sequences[index], dtype=self.dtype)
        y = torch.as_tensor(self.labels[index], dtype=torch.long)
        return x, y


def create_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> dict[SplitName, DataLoader]:
    loaders: dict[SplitName, DataLoader] = {}
    for name, x_split, y_split, shuffle in [
        ("train", x_train, y_train, shuffle_train),
        ("validation", x_val, y_val, False),
        ("test", x_test, y_test, False),
    ]:
        loaders[name] = DataLoader(
            TelemanomSequenceDataset(x_split, y_split),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    return loaders


def build_telemanom_pipeline(
    channel_ids: list[str] | None = None,
    config: PreprocessConfig | None = None,
    dataset_id: str = HF_DATASET_ID,
    max_channels: int | None = None,
) -> TelemanomPipelineResult:
    config = config or PreprocessConfig()
    labels_df = load_hf_labels(dataset_id)
    channels = channel_ids or list_hf_channels(dataset_id)
    if max_channels is not None:
        channels = channels[:max_channels]

    train_arrays: list[np.ndarray] = []
    channel_meta: list[tuple[np.ndarray, np.ndarray, list[list[int]] | None]] = []

    for channel_id in channels:
        train_arr, test_arr = channel_to_arrays(load_hf_channel(channel_id, dataset_id))
        train_arrays.append(train_arr)

        row = labels_df.loc[labels_df["chan_id"] == channel_id]
        sequences = row.iloc[0]["anomaly_sequences"] if not row.empty else None
        channel_meta.append((train_arr, test_arr, sequences))

    train_arrays = [pad_feature_dim(a, MAX_FEATURE_DIM) for a in train_arrays]
    channel_meta = [
        (pad_feature_dim(tr, MAX_FEATURE_DIM), pad_feature_dim(te, MAX_FEATURE_DIM), seq)
        for tr, te, seq in channel_meta
    ]

    scaler = TelemanomFeatureScaler(config.per_feature_normalize).fit(train_arrays)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    test_label_parts: list[np.ndarray] = []

    for train_arr, test_arr, sequences in channel_meta:
        train_w, test_w, y_test = build_channel_windows(
            train_arr, test_arr, sequences, config, scaler
        )
        if len(train_w):
            tr, va = chronological_train_val_split(train_w, config.val_size)
            train_parts.append(tr)
            val_parts.append(va)
        if len(test_w):
            test_parts.append(test_w)
            test_label_parts.append(y_test)

    if not train_parts:
        raise RuntimeError("No training windows created — check channels or window size.")

    x_train = np.concatenate(train_parts, axis=0)
    x_val = np.concatenate(val_parts, axis=0) if val_parts else np.empty((0, *x_train.shape[1:]))
    x_test = np.concatenate(test_parts, axis=0) if test_parts else np.empty((0, *x_train.shape[1:]))
    y_train = np.zeros(len(x_train), dtype=np.int64)
    y_val = np.zeros(len(x_val), dtype=np.int64)
    y_test = (
        np.concatenate(test_label_parts, axis=0)
        if test_label_parts
        else np.empty(0, dtype=np.int64)
    )

    return TelemanomPipelineResult(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        model_interface=build_model_interface(config, scaler, MAX_FEATURE_DIM),
        config=config,
        channel_ids=channels,
        label_distribution=class_distribution(y_test) if len(y_test) else class_distribution(y_train),
    )


def load_channel_for_analysis(
    channel_id: str,
    dataset_id: str = HF_DATASET_ID,
) -> tuple[np.ndarray, np.ndarray, pd.Series | None]:
    labels_df = load_hf_labels(dataset_id)
    train_arr, test_arr = channel_to_arrays(load_hf_channel(channel_id, dataset_id))
    row = labels_df.loc[labels_df["chan_id"] == channel_id]
    return train_arr, test_arr, row.iloc[0] if not row.empty else None
