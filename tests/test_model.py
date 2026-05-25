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
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler

from src.dataset import (
    FEATURE_COLS,
    TARGET_COL,
    ServidorDataset,
    build_inference_dataloaders,
    fit_scaler,
    save_scaler,
    temporal_split,
)
from src.evaluate import load_checkpoint, main as evaluate_main
from src.model import AnomalyDetectorGRU, AnomalyDetectorMLP
from src.train import main as train_main


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


@pytest.mark.parametrize(
    ("input_dim", "hidden_dims", "dropout"),
    [
        (0, (64, 32), 0.3),
        (20, (64, 0), 0.3),
        (20, (64, 32), -0.1),
        (20, (64, 32), 1.0),
    ],
)
def test_mlp_rejects_invalid_hyperparameters(
    input_dim: int,
    hidden_dims: tuple[int, int],
    dropout: float,
) -> None:
    """Constructor must fail fast on invalid MLP hyperparameters."""
    with pytest.raises(ValueError):
        AnomalyDetectorMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )


@pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
def test_gru_io_shapes(batch_size: int) -> None:
    """``AnomalyDetectorGRU`` maps ``[B, seq_len*num_features]`` to ``[B, 1]``."""
    model = AnomalyDetectorGRU(
        num_features=4, seq_len=5, hidden_size=32, num_layers=1, dropout=0.3
    )
    model.eval()

    x = torch.randn(batch_size, 20)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {tuple(y.shape)}"
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all(), "GRU produced non-finite logits"


def test_gru_reshape_recovers_chronology() -> None:
    """The flatten -> view round-trip must preserve the sequence layout.

    We feed a deterministic 3D tensor ``[B, 5, 4]`` after flattening it
    row-major (exactly as :class:`ServidorDataset` does at index time).
    A forward hook on the GRU layer captures the tensor it actually
    receives; that tensor must match the original 3D input element-wise.
    """
    batch_size = 3
    seq_len = 5
    num_features = 4

    original_3d = torch.arange(
        batch_size * seq_len * num_features, dtype=torch.float32
    ).reshape(batch_size, seq_len, num_features)

    # Same memory layout used by ServidorDataset.__getitem__.
    flattened = original_3d.reshape(batch_size, -1)

    model = AnomalyDetectorGRU(
        num_features=num_features,
        seq_len=seq_len,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()

    captured: list[torch.Tensor] = []

    def _hook(_module: nn.Module, inputs: tuple, _output: object) -> None:
        captured.append(inputs[0].detach().clone())

    handle = model.gru.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(flattened)
    finally:
        handle.remove()

    assert len(captured) == 1
    seen_by_gru = captured[0]
    assert seen_by_gru.shape == original_3d.shape
    torch.testing.assert_close(seen_by_gru, original_3d, rtol=0, atol=0)


def test_gru_rejects_invalid_hyperparameters() -> None:
    """Constructor must reject non-positive sizes."""
    with pytest.raises(ValueError):
        AnomalyDetectorGRU(num_features=0)
    with pytest.raises(ValueError):
        AnomalyDetectorGRU(hidden_size=-1)
    with pytest.raises(ValueError):
        AnomalyDetectorGRU(dropout=1.2)


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


def test_inference_dataloaders_use_persisted_scaler(tmp_path: Path) -> None:
    """Inference must reuse the saved scaler rather than refitting a new one."""
    df = _synthetic_frame(num_rows=100)
    csv_path = tmp_path / "telemetry.csv"
    scaler_path = tmp_path / "scaler.joblib"
    df.to_csv(csv_path, index=False)

    train_df, _val_df, _test_df = temporal_split(df, train_ratio=0.70, val_ratio=0.15)
    scaler = fit_scaler(train_df)
    scaler.min_ = np.full_like(scaler.min_, 123.0)
    scaler.scale_ = np.full_like(scaler.scale_, 0.5)
    save_scaler(scaler, scaler_path)

    _train_loader, _val_loader, test_loader, meta = build_inference_dataloaders(
        csv_path=csv_path,
        scaler_path=scaler_path,
        batch_size=16,
        window_size=5,
        num_workers=0,
    )

    assert meta.test_windows == len(test_loader.dataset)
    x0, _y0 = test_loader.dataset[0]

    test_start = int(len(df) * 0.70) + int(len(df) * 0.15)
    expected_raw = df.iloc[test_start : test_start + 5][FEATURE_COLS].to_numpy(dtype=np.float64)
    expected_scaled = scaler.transform(expected_raw).reshape(-1).astype(np.float32)
    np.testing.assert_allclose(x0.numpy(), expected_scaled, rtol=0, atol=1e-6)


def test_end_to_end_train_and_evaluate_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training and evaluation CLIs should produce reusable artefacts end-to-end."""
    df = _synthetic_frame(num_rows=120)
    csv_path = tmp_path / "telemetry.csv"
    scaler_path = tmp_path / "scaler.joblib"
    model_path = tmp_path / "best_model.pth"
    curves_path = tmp_path / "curves.png"
    cm_path = tmp_path / "confusion.png"
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--csv-path",
            str(csv_path),
            "--scaler-path",
            str(scaler_path),
            "--model-path",
            str(model_path),
            "--curves-path",
            str(curves_path),
            "--epochs",
            "2",
            "--patience",
            "1",
            "--batch-size",
            "16",
            "--hidden-dims",
            "32",
            "16",
            "--lr",
            "0.001",
            "--dropout",
            "0.2",
            "--log-level",
            "ERROR",
        ],
    )
    train_main()

    assert scaler_path.exists()
    assert model_path.exists()
    assert curves_path.exists()

    model, ckpt = load_checkpoint(model_path, torch.device("cpu"))
    assert ckpt["model_type"] == "mlp"
    assert tuple(ckpt["hidden_dims"]) == (32, 16)
    assert model.training is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--csv-path",
            str(csv_path),
            "--scaler-path",
            str(scaler_path),
            "--checkpoint",
            str(model_path),
            "--cm-path",
            str(cm_path),
            "--batch-size",
            "16",
            "--log-level",
            "ERROR",
        ],
    )
    evaluate_main()

    assert cm_path.exists()
