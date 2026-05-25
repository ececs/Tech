"""Hyperparameter sweep for the TechStream anomaly detector.

This script exhaustively trains :class:`AnomalyDetectorMLP` over a small
grid of hyperparameters (learning rate, dropout, hidden sizes, ``pos_weight``
scaling factor and batch size). For every trained model it computes the
**best-threshold** F1 on validation by scanning the precision-recall
curve, instead of using a fixed 0.5 decision threshold. Results are
appended atomically to a JSONL log so partial progress survives crashes.

Designed to run on the secondary CUDA workstation (RTX 4070 Super) but
falls back transparently to MPS or CPU. Launch from the repo root::

    python -m src.sweep --epochs 30 --patience 7 --output sweep_results.json
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import itertools
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import build_dataloaders
from src.model import AnomalyDetectorMLP
from src.train import (
    EarlyStopping,
    evaluate,
    get_device,
    set_seed,
    train_one_epoch,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "dataset_servidores.csv"
DEFAULT_SCALER_PATH: Path = PROJECT_ROOT / "artifacts" / "scaler.joblib"
DEFAULT_OUTPUT_PATH: Path = PROJECT_ROOT / "sweep_results.jsonl"


@dataclass
class SweepRun:
    """Single configuration outcome."""

    run_id: int
    lr: float
    dropout: float
    hidden_dims: tuple[int, int]
    pos_weight_scale: float
    batch_size: int
    weight_decay: float
    seed: int
    epochs_run: int
    best_val_f1_fixed: float
    best_val_f1_tuned: float
    best_threshold: float
    val_precision_at_best: float
    val_recall_at_best: float
    final_train_loss: float
    final_val_loss: float
    train_time_seconds: float


@torch.no_grad()
def collect_val_scores(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(scores, labels)`` arrays for the entire validation loader."""
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        scores.append(probs)
        labels.append(targets.numpy().reshape(-1))
    return np.concatenate(scores), np.concatenate(labels)


def best_threshold_f1(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, float]:
    """Find the threshold that maximizes F1 on the PR curve.

    Args:
        scores: Predicted positive probabilities.
        labels: Binary ground-truth labels.

    Returns:
        Tuple ``(best_f1, best_threshold, precision_at_best, recall_at_best)``.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_idx = int(np.argmax(f1))
    if best_idx >= len(thresholds):
        best_threshold = 1.0
    else:
        best_threshold = float(thresholds[best_idx])
    return (
        float(f1[best_idx]),
        best_threshold,
        float(precision[best_idx]),
        float(recall[best_idx]),
    )


def run_single_config(
    run_id: int,
    lr: float,
    dropout: float,
    hidden_dims: tuple[int, int],
    pos_weight_scale: float,
    batch_size: int,
    weight_decay: float,
    epochs: int,
    patience: int,
    seed: int,
    csv_path: Path,
    scaler_path: Path,
    device: torch.device,
) -> SweepRun:
    """Train a single configuration and return its measured outcome."""
    set_seed(seed)
    train_loader, val_loader, _test_loader, meta = build_dataloaders(
        csv_path=csv_path,
        scaler_path=scaler_path,
        batch_size=batch_size,
        num_workers=0,
    )
    model = AnomalyDetectorMLP(
        input_dim=meta["input_dim"],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    effective_pos_weight = meta["pos_weight"] * pos_weight_scale
    pos_weight_tensor = torch.tensor(
        [effective_pos_weight], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    stopper = EarlyStopping(patience=patience)

    last_train_loss = 0.0
    last_val_loss = 0.0
    epochs_run = 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        last_train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        last_val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        epochs_run = epoch
        stopper.step(val_f1, model)
        if stopper.should_stop:
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    scores, labels = collect_val_scores(model, val_loader, device)
    best_f1, best_thr, prec, rec = best_threshold_f1(scores, labels)
    elapsed = time.perf_counter() - t0

    return SweepRun(
        run_id=run_id,
        lr=lr,
        dropout=dropout,
        hidden_dims=hidden_dims,
        pos_weight_scale=pos_weight_scale,
        batch_size=batch_size,
        weight_decay=weight_decay,
        seed=seed,
        epochs_run=epochs_run,
        best_val_f1_fixed=float(stopper.best_score),
        best_val_f1_tuned=best_f1,
        best_threshold=best_thr,
        val_precision_at_best=prec,
        val_recall_at_best=rec,
        final_train_loss=last_train_loss,
        final_val_loss=last_val_loss,
        train_time_seconds=elapsed,
    )


def build_grid(
    smoke: bool,
) -> list[dict]:
    """Return the list of configurations to sweep."""
    if smoke:
        return [
            {"lr": 1e-3, "dropout": 0.3, "hidden_dims": (64, 32), "pos_weight_scale": 1.0, "batch_size": 128},
            {"lr": 1e-3, "dropout": 0.3, "hidden_dims": (64, 32), "pos_weight_scale": 0.25, "batch_size": 128},
            {"lr": 3e-4, "dropout": 0.2, "hidden_dims": (128, 64), "pos_weight_scale": 1.0, "batch_size": 128},
            {"lr": 3e-4, "dropout": 0.5, "hidden_dims": (32, 16), "pos_weight_scale": 0.5, "batch_size": 128},
        ]
    lrs = [3e-4, 1e-3, 3e-3]
    dropouts = [0.2, 0.3, 0.5]
    hidden = [(32, 16), (64, 32), (128, 64)]
    scales = [0.25, 0.5, 1.0]
    batches = [64, 128]
    configs = []
    for lr, dr, hd, sc, bs in itertools.product(lrs, dropouts, hidden, scales, batches):
        configs.append(
            {
                "lr": lr,
                "dropout": dr,
                "hidden_dims": hd,
                "pos_weight_scale": sc,
                "batch_size": bs,
            }
        )
    return configs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--smoke", action="store_true", help="Run a tiny 4-config grid"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """Run the full sweep."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    device = get_device()
    logger.info("Sweep device: %s", device)

    grid = build_grid(smoke=args.smoke)
    logger.info("Sweep grid size: %d", len(grid))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Silence dataset/training INFO chatter inside each run
    logging.getLogger("src.dataset").setLevel(logging.WARNING)
    logging.getLogger("src.train").setLevel(logging.WARNING)

    best_overall: SweepRun | None = None
    with args.output.open("w") as fh:
        for idx, config in enumerate(grid, start=1):
            logger.info("[%03d/%03d] %s", idx, len(grid), config)
            run = run_single_config(
                run_id=idx,
                lr=config["lr"],
                dropout=config["dropout"],
                hidden_dims=config["hidden_dims"],
                pos_weight_scale=config["pos_weight_scale"],
                batch_size=config["batch_size"],
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed,
                csv_path=args.csv_path,
                scaler_path=args.scaler_path,
                device=device,
            )
            fh.write(json.dumps(asdict(run)) + "\n")
            fh.flush()
            logger.info(
                "  → epochs=%d | f1@0.5=%.4f | f1*=%.4f thr*=%.3f P=%.3f R=%.3f "
                "(%.1fs)",
                run.epochs_run,
                run.best_val_f1_fixed,
                run.best_val_f1_tuned,
                run.best_threshold,
                run.val_precision_at_best,
                run.val_recall_at_best,
                run.train_time_seconds,
            )
            if best_overall is None or run.best_val_f1_tuned > best_overall.best_val_f1_tuned:
                best_overall = run
                logger.info("  ⭐ new best: f1*=%.4f", run.best_val_f1_tuned)

    if best_overall is not None:
        logger.info(
            "Sweep finished. Best run #%d: f1*=%.4f thr*=%.3f lr=%.0e dropout=%.2f "
            "hidden=%s pos_weight_scale=%.2f batch=%d",
            best_overall.run_id,
            best_overall.best_val_f1_tuned,
            best_overall.best_threshold,
            best_overall.lr,
            best_overall.dropout,
            best_overall.hidden_dims,
            best_overall.pos_weight_scale,
            best_overall.batch_size,
        )


if __name__ == "__main__":
    main()
