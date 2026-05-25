"""Data ingestion pipeline for the TechStream anomaly detection task.

This module exposes :class:`ServidorDataset`, a PyTorch dataset that yields
fixed-size causal sliding windows over server telemetry features, together
with helpers to load the raw CSV, perform a temporal train/val/test split,
fit and persist a feature scaler, and build the :class:`DataLoader`
instances used by the training script.
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Final

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

FEATURE_COLS: Final[list[str]] = [
    "cpu_usage",
    "mem_usage",
    "network_traffic",
    "cpu_temp",
]
TARGET_COL: Final[str] = "failure"
WINDOW_SIZE: Final[int] = 5


@dataclass(frozen=True)
class LoaderMetadata:
    """Structured metadata returned alongside the DataLoaders."""

    pos_weight: float
    train_positives: int
    train_negatives: int
    train_windows: int
    val_windows: int
    test_windows: int
    input_dim: int
    num_features: int
    window_size: int


class ServidorDataset(Dataset):
    """Sliding-window dataset over scaled server telemetry features.

    Each sample is the concatenation of ``window_size`` consecutive feature
    vectors flattened in chronological order. The label is ``failure`` at
    the last time step of the window.

    Args:
        features: Scaled feature matrix of shape ``[T, num_features]``.
        labels: Binary failure labels of shape ``[T]``.
        window_size: Number of consecutive time steps per sample.

    Raises:
        ValueError: If ``features`` and ``labels`` lengths differ or the
            split is shorter than ``window_size``.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"features ({features.shape[0]}) and labels "
                f"({labels.shape[0]}) length mismatch"
            )
        if features.shape[0] < window_size:
            raise ValueError(
                f"Split has {features.shape[0]} rows, < window_size={window_size}"
            )

        self._features = features.astype(np.float32, copy=False)
        self._labels = labels.astype(np.float32, copy=False)
        self._window_size = window_size
        self._num_features = features.shape[1]
        self._valid_length = features.shape[0] - window_size + 1

    def __len__(self) -> int:
        return self._valid_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = index + self._window_size
        window = self._features[start:end].reshape(-1)
        label = self._labels[end - 1]
        x = torch.from_numpy(window)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y

    @property
    def window_labels(self) -> np.ndarray:
        """Return the per-sample target labels (label at end of each window)."""
        return self._labels[self._window_size - 1 :]

    @property
    def input_dim(self) -> int:
        """Flat input dimension of each sample."""
        return self._window_size * self._num_features


def load_raw_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load the telemetry CSV and validate required columns.

    Args:
        csv_path: Path to ``dataset_servidores.csv``.

    Returns:
        DataFrame ordered chronologically by ``timestamp``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        KeyError: If any required column is missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    required = {*FEATURE_COLS, TARGET_COL, "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in CSV: {sorted(missing)}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded raw dataframe: %d rows from %s", len(df), csv_path)
    return df


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered dataframe sequentially into train/val/test.

    Args:
        df: Chronologically sorted dataframe.
        train_ratio: Fraction of rows assigned to the training split.
        val_ratio: Fraction of rows assigned to validation. The remainder
            goes to the test split.

    Returns:
        Tuple ``(train_df, val_df, test_df)``.

    Raises:
        ValueError: If ratios are invalid.
    """
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("Ratios must be in (0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    logger.info(
        "Temporal split: train=%d val=%d test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def fit_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    """Fit a MinMaxScaler on the training split features only.

    Args:
        train_df: Training dataframe containing ``FEATURE_COLS``.

    Returns:
        Fitted ``MinMaxScaler``.
    """
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(train_df[FEATURE_COLS].to_numpy(dtype=np.float64))
    logger.info("Fitted MinMaxScaler on %d training rows", len(train_df))
    return scaler


def save_scaler(scaler: MinMaxScaler, path: Path) -> None:
    """Persist a fitted scaler to disk via joblib.

    Args:
        scaler: Fitted scaler instance.
        path: Destination path; parent directories are created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info("Saved scaler to %s", path)


def load_scaler(path: Path) -> MinMaxScaler:
    """Load a scaler previously saved with :func:`save_scaler`.

    Args:
        path: Path to the joblib artifact.

    Returns:
        The deserialized ``MinMaxScaler`` instance.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {path}")
    scaler: MinMaxScaler = joblib.load(path)
    logger.info("Loaded scaler from %s", path)
    return scaler


def _transform_split(
    df: pd.DataFrame, scaler: MinMaxScaler
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scaled_features, labels) arrays for a split."""
    features = scaler.transform(df[FEATURE_COLS].to_numpy(dtype=np.float64))
    labels = df[TARGET_COL].to_numpy(dtype=np.float32)
    return features, labels


def _build_loaders_from_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: MinMaxScaler,
    batch_size: int,
    window_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, LoaderMetadata]:
    """Transform splits, build datasets/loaders and compute metadata."""
    train_x, train_y = _transform_split(train_df, scaler)
    val_x, val_y = _transform_split(val_df, scaler)
    test_x, test_y = _transform_split(test_df, scaler)

    train_ds = ServidorDataset(train_x, train_y, window_size)
    val_ds = ServidorDataset(val_x, val_y, window_size)
    test_ds = ServidorDataset(test_x, test_y, window_size)

    train_window_labels = train_ds.window_labels
    positives = int(train_window_labels.sum())
    negatives = int(len(train_window_labels) - positives)
    pos_weight = float(negatives) / float(positives) if positives > 0 else 1.0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    metadata = LoaderMetadata(
        pos_weight=pos_weight,
        train_positives=positives,
        train_negatives=negatives,
        train_windows=len(train_ds),
        val_windows=len(val_ds),
        test_windows=len(test_ds),
        input_dim=train_ds.input_dim,
        num_features=len(FEATURE_COLS),
        window_size=window_size,
    )
    logger.info(
        "Built dataloaders: train_windows=%d val_windows=%d test_windows=%d "
        "pos_weight=%.4f input_dim=%d",
        metadata.train_windows,
        metadata.val_windows,
        metadata.test_windows,
        metadata.pos_weight,
        metadata.input_dim,
    )
    return train_loader, val_loader, test_loader, metadata


def build_training_dataloaders(
    csv_path: Path,
    scaler_path: Path,
    batch_size: int = 128,
    window_size: int = WINDOW_SIZE,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, LoaderMetadata]:
    """Build loaders for training and persist a scaler fitted on train only."""
    df = load_raw_dataframe(csv_path)
    train_df, val_df, test_df = temporal_split(df, train_ratio, val_ratio)
    scaler = fit_scaler(train_df)
    save_scaler(scaler, scaler_path)
    return _build_loaders_from_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        scaler=scaler,
        batch_size=batch_size,
        window_size=window_size,
        num_workers=num_workers,
    )


def build_inference_dataloaders(
    csv_path: Path,
    scaler_path: Path,
    batch_size: int = 128,
    window_size: int = WINDOW_SIZE,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, LoaderMetadata]:
    """Build loaders for evaluation/inference using an already persisted scaler."""
    df = load_raw_dataframe(csv_path)
    train_df, val_df, test_df = temporal_split(df, train_ratio, val_ratio)
    scaler = load_scaler(scaler_path)
    return _build_loaders_from_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        scaler=scaler,
        batch_size=batch_size,
        window_size=window_size,
        num_workers=num_workers,
    )
