"""Reusable inference helpers for snapshot and batch anomaly scoring."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from src.dataset import FEATURE_COLS, WINDOW_SIZE, load_scaler
from src.evaluate import DEFAULT_CHECKPOINT_PATH, DEFAULT_SCALER_PATH, load_checkpoint
from src.training_common import get_device

logger = logging.getLogger(__name__)

REQUIRED_UPLOAD_COLUMNS = ["timestamp", *FEATURE_COLS]


@dataclass(frozen=True)
class RuntimeResources:
    """Loaded model, scaler and metadata required for inference."""

    model: nn.Module | None
    scaler: MinMaxScaler | None
    threshold: float
    device: torch.device
    model_type: str
    ready: bool
    error: str | None = None


@dataclass(frozen=True)
class BatchInferenceResult:
    """Structured result for batch CSV scoring."""

    processed_df: pd.DataFrame
    summary: dict[str, object]
    records: list[dict[str, object]]
    warnings: list[str]


def load_runtime_resources(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    scaler_path: Path = DEFAULT_SCALER_PATH,
) -> RuntimeResources:
    """Load model checkpoint and scaler artifacts needed by the web app."""
    device = get_device()
    try:
        model, ckpt = load_checkpoint(checkpoint_path, device)
        scaler = load_scaler(scaler_path)
        threshold = float(ckpt.get("decision_threshold", 0.5))
        model_type = str(ckpt.get("model_type", "mlp"))
        logger.info(
            "Loaded runtime resources: model=%s device=%s threshold=%.3f",
            model_type,
            device,
            threshold,
        )
        return RuntimeResources(
            model=model,
            scaler=scaler,
            threshold=threshold,
            device=device,
            model_type=model_type,
            ready=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load runtime resources: %s", exc)
        return RuntimeResources(
            model=None,
            scaler=None,
            threshold=0.5,
            device=device,
            model_type="mlp",
            ready=False,
            error=str(exc),
        )


def ensure_runtime_ready(runtime: RuntimeResources) -> None:
    """Raise a helpful error if the model/scaler artifacts are not loaded."""
    if not runtime.ready or runtime.model is None or runtime.scaler is None:
        raise RuntimeError(
            "El modelo no está listo para inferencia. "
            f"Detalle: {runtime.error or 'artefactos no cargados'}"
        )


def build_snapshot_window(
    cpu_usage: float,
    mem_usage: float,
    network_traffic: float,
    cpu_temp: float,
) -> np.ndarray:
    """Repeat a single telemetry snapshot 5 times to mimic the training window."""
    row = np.array(
        [cpu_usage, mem_usage, network_traffic, cpu_temp],
        dtype=np.float64,
    )
    return np.repeat(row[None, :], WINDOW_SIZE, axis=0)


def scale_and_clip_features(
    features: np.ndarray,
    scaler: MinMaxScaler,
) -> tuple[np.ndarray, list[str]]:
    """Scale features with the trained scaler and warn if clipping is needed."""
    transformed = scaler.transform(features)
    warnings: list[str] = []
    below_zero = int(np.count_nonzero(transformed < 0.0))
    above_one = int(np.count_nonzero(transformed > 1.0))
    if below_zero or above_one:
        warnings.append(
            "Se detectaron valores fuera del rango esperado del scaler entrenado; "
            "se han recortado a [0, 1] para mantener estabilidad en inferencia."
        )
    clipped = np.clip(transformed, 0.0, 1.0).astype(np.float32)
    return clipped, warnings


def predict_window_matrix(
    window_matrix: np.ndarray,
    runtime: RuntimeResources,
) -> np.ndarray:
    """Score an already-windowed feature matrix and return probabilities."""
    ensure_runtime_ready(runtime)
    assert runtime.model is not None

    tensor = torch.from_numpy(window_matrix.astype(np.float32, copy=False)).to(
        runtime.device
    )
    with torch.no_grad():
        logits = runtime.model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return probs


def predict_snapshot(
    cpu_usage: float,
    mem_usage: float,
    network_traffic: float,
    cpu_temp: float,
    runtime: RuntimeResources,
) -> float:
    """Run snapshot inference by repeating the same point across the 5-step window."""
    ensure_runtime_ready(runtime)
    assert runtime.scaler is not None
    snapshot = build_snapshot_window(
        cpu_usage=cpu_usage,
        mem_usage=mem_usage,
        network_traffic=network_traffic,
        cpu_temp=cpu_temp,
    )
    scaled, _warnings = scale_and_clip_features(snapshot, runtime.scaler)
    return float(predict_window_matrix(scaled.reshape(1, -1), runtime)[0])


def predict_trend(
    readings: list[tuple[float, float, float, float]],
    runtime: RuntimeResources,
    window_size: int = WINDOW_SIZE,
) -> tuple[float, list[str]]:
    """Score a real temporal sequence built from up to ``window_size`` readings.

    Unlike :func:`predict_snapshot`, this preserves the temporal variation
    the model was trained on. When fewer than ``window_size`` readings
    are provided, the oldest reading is repeated on the left (causal
    padding), matching the batch inference behaviour.

    Args:
        readings: Chronological list of ``(cpu_usage, mem_usage,
            network_traffic, cpu_temp)`` tuples. Must be non-empty and
            contain at most ``window_size`` entries.
        runtime: Loaded model and scaler.
        window_size: Expected sequence length (default ``5``).

    Returns:
        Tuple ``(probability, warnings)``. ``warnings`` may include a
        scaler-out-of-range notice.

    Raises:
        ValueError: If ``readings`` is empty or longer than
            ``window_size``.
    """
    ensure_runtime_ready(runtime)
    assert runtime.scaler is not None

    if not readings:
        raise ValueError("readings must contain at least one entry")
    if len(readings) > window_size:
        raise ValueError(
            f"readings has {len(readings)} entries, expected at most {window_size}"
        )

    matrix = np.array(readings, dtype=np.float64)
    if matrix.shape[1] != len(FEATURE_COLS):
        raise ValueError(
            f"each reading must have {len(FEATURE_COLS)} values, got {matrix.shape[1]}"
        )
    if matrix.shape[0] < window_size:
        pad = np.repeat(matrix[:1], window_size - matrix.shape[0], axis=0)
        matrix = np.vstack([pad, matrix])

    scaled, warnings = scale_and_clip_features(matrix, runtime.scaler)
    probability = float(predict_window_matrix(scaled.reshape(1, -1), runtime)[0])
    return probability, warnings


def validate_upload_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the uploaded CSV dataframe."""
    missing = [column for column in REQUIRED_UPLOAD_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    normalized = df.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    if normalized["timestamp"].isna().any():
        raise ValueError("The 'timestamp' column contains invalid datetime values")

    normalized = normalized.sort_values("timestamp").reset_index(drop=True)
    for column in FEATURE_COLS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        if normalized[column].isna().any():
            raise ValueError(f"Column {column!r} contains non-numeric values")
    return normalized


def build_padded_windows(features: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Create one causal window per row, padding the left side with the first row."""
    if len(features) == 0:
        return np.zeros((0, window_size * len(FEATURE_COLS)), dtype=np.float32)

    windows: list[np.ndarray] = []
    for idx in range(len(features)):
        start = max(0, idx - window_size + 1)
        window = features[start : idx + 1]
        if len(window) < window_size:
            pad = np.repeat(window[:1], window_size - len(window), axis=0)
            window = np.vstack([pad, window])
        windows.append(window.reshape(-1))
    return np.asarray(windows, dtype=np.float32)


def summarize_processed_dataframe(processed_df: pd.DataFrame) -> dict[str, object]:
    """Build compact summary metrics for the upload response."""
    n_rows = int(len(processed_df))
    n_anomalies = int(processed_df["prediction"].sum())
    anomaly_rate = float(n_anomalies / n_rows) if n_rows else 0.0
    top_peaks = (
        processed_df.nlargest(3, "cpu_temp")[["timestamp", "cpu_temp"]]
        .assign(timestamp=lambda df: df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"))
        .to_dict(orient="records")
    )
    return {
        "n_rows": n_rows,
        "n_anomalies": n_anomalies,
        "anomaly_rate": anomaly_rate,
        "top_thermal_peaks": top_peaks,
    }


def infer_uploaded_dataframe(
    df: pd.DataFrame,
    runtime: RuntimeResources,
) -> BatchInferenceResult:
    """Run batch inference over an uploaded CSV dataframe."""
    ensure_runtime_ready(runtime)
    assert runtime.scaler is not None

    normalized = validate_upload_dataframe(df)
    scaled_features, warnings = scale_and_clip_features(
        normalized[FEATURE_COLS].to_numpy(dtype=np.float64),
        runtime.scaler,
    )
    window_matrix = build_padded_windows(scaled_features, window_size=WINDOW_SIZE)
    probabilities = predict_window_matrix(window_matrix, runtime)
    predictions = (probabilities > runtime.threshold).astype(np.int64)

    processed_df = normalized.copy()
    processed_df["probability"] = probabilities
    processed_df["prediction"] = predictions
    processed_df["idx"] = np.arange(len(processed_df), dtype=np.int64)

    summary = summarize_processed_dataframe(processed_df)
    records = (
        processed_df[
            ["idx", "timestamp", "cpu_usage", "cpu_temp", "probability", "prediction"]
        ]
        .assign(
            timestamp=lambda frame: frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            prob=lambda frame: frame["probability"],
        )
        .drop(columns=["probability"])
        .to_dict(orient="records")
    )
    return BatchInferenceResult(
        processed_df=processed_df,
        summary=summary,
        records=records,
        warnings=warnings,
    )


def compute_session_id(file_bytes: bytes) -> str:
    """Create a deterministic session id from uploaded CSV content."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]
