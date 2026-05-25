"""Unit tests for the TechStream anomaly detector components.

Covers:
* :class:`AnomalyDetectorMLP` input/output tensor shapes for multiple
  batch sizes.
* :class:`ServidorDataset` causal sliding window construction and
  ``MinMaxScaler``-based feature scaling into ``[0, 1]``.
* :func:`temporal_split` sequential (non-shuffled) ordering invariant.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from src.dataset import (
    FEATURE_COLS,
    TARGET_COL,
    ServidorDataset,
    fit_scaler,
    temporal_split,
)
from src.model import AnomalyDetectorMLP


@pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
def test_model_io_shapes(batch_size: int) -> None:
    """``AnomalyDetectorMLP`` maps ``[B, 20]`` to ``[B, 1]`` for various B."""
    model = AnomalyDetectorMLP(input_dim=20, hidden_dims=(64, 32), dropout=0.3)
    model.eval()  # avoid BatchNorm issues when B=1

    x = torch.randn(batch_size, 20)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {tuple(y.shape)}"
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all(), "Model produced non-finite logits"


def test_model_rejects_wrong_input_dim() -> None:
    """The model must error if the input feature dimension is wrong."""
    model = AnomalyDetectorMLP(input_dim=20)
    model.eval()
    with pytest.raises(RuntimeError):
        model(torch.randn(4, 19))


def _synthetic_frame(num_rows: int = 50) -> pd.DataFrame:
    """Build a small synthetic dataframe matching the production schema."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="min"),
            "server_id": "SRV-TEST",
            "cpu_usage": rng.uniform(10.0, 90.0, size=num_rows),
            "mem_usage": rng.uniform(20.0, 95.0, size=num_rows),
            "network_traffic": rng.uniform(0.1, 200.0, size=num_rows),
            "cpu_temp": rng.uniform(40.0, 100.0, size=num_rows),
            "failure": rng.integers(0, 2, size=num_rows).astype(np.float32),
        }
    )


def test_dataset_scaling_and_window() -> None:
    """``ServidorDataset`` scales features to [0, 1] and uses causal windows."""
    df = _synthetic_frame(num_rows=20)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaled_features = scaler.fit_transform(
        df[FEATURE_COLS].to_numpy(dtype=np.float64)
    )
    labels = df[TARGET_COL].to_numpy(dtype=np.float32)

    window_size = 5
    num_features = len(FEATURE_COLS)
    dataset = ServidorDataset(scaled_features, labels, window_size=window_size)

    expected_len = len(df) - window_size + 1
    assert len(dataset) == expected_len

    x0, y0 = dataset[0]
    assert x0.shape == (window_size * num_features,)
    assert y0.shape == (1,)
    assert x0.dtype == torch.float32
    assert (x0 >= 0.0).all() and (x0 <= 1.0).all(), "Features must be scaled to [0, 1]"

    # Causality: the FIRST sample's flattened window must equal the first
    # `window_size` scaled rows in chronological order.
    expected_window = scaled_features[:window_size].reshape(-1).astype(np.float32)
    np.testing.assert_allclose(x0.numpy(), expected_window, rtol=0, atol=1e-6)
    # The label of the FIRST sample is the failure flag at t = window_size - 1.
    assert float(y0.item()) == float(labels[window_size - 1])

    # Causality: the LAST sample's window ends at the last row of the split.
    x_last, y_last = dataset[expected_len - 1]
    expected_last = scaled_features[-window_size:].reshape(-1).astype(np.float32)
    np.testing.assert_allclose(x_last.numpy(), expected_last, rtol=0, atol=1e-6)
    assert float(y_last.item()) == float(labels[-1])


def test_dataset_rejects_short_split() -> None:
    """``ServidorDataset`` must refuse splits smaller than ``window_size``."""
    features = np.zeros((3, len(FEATURE_COLS)), dtype=np.float32)
    labels = np.zeros((3,), dtype=np.float32)
    with pytest.raises(ValueError):
        ServidorDataset(features, labels, window_size=5)


def test_temporal_split_is_sequential() -> None:
    """``temporal_split`` preserves chronological order across splits."""
    df = _synthetic_frame(num_rows=100)
    train, val, test = temporal_split(df, train_ratio=0.70, val_ratio=0.15)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    # All train timestamps must precede every val/test timestamp.
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


def test_fit_scaler_uses_only_training_rows() -> None:
    """``fit_scaler`` must be fit exclusively on the training split."""
    df = _synthetic_frame(num_rows=100)
    train, _val, _test = temporal_split(df, train_ratio=0.70, val_ratio=0.15)
    scaler = fit_scaler(train)

    np.testing.assert_allclose(
        scaler.data_min_,
        train[FEATURE_COLS].min().to_numpy(),
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        scaler.data_max_,
        train[FEATURE_COLS].max().to_numpy(),
        rtol=0,
        atol=1e-9,
    )
