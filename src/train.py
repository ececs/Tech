"""Training entry point for the TechStream anomaly detector.

Run as a module from the repository root::

    python -m src.train --epochs 50 --batch-size 128

The script fits the data pipeline, trains :class:`AnomalyDetectorMLP`
with ``BCEWithLogitsLoss`` weighted by class imbalance, applies early
stopping on validation F1-score, and exports the best checkpoint,
the fitted scaler and the training curves plot.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import build_dataloaders
from src.model import AnomalyDetectorMLP

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "dataset_servidores.csv"
DEFAULT_SCALER_PATH: Path = PROJECT_ROOT / "artifacts" / "scaler.joblib"
DEFAULT_MODEL_PATH: Path = PROJECT_ROOT / "best_model.pth"
DEFAULT_CURVES_PATH: Path = PROJECT_ROOT / "training_curves.png"


@dataclass
class EpochMetrics:
    """Container for per-epoch metrics."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    val_precision: list[float] = field(default_factory=list)
    val_recall: list[float] = field(default_factory=list)


class EarlyStopping:
    """Early stopping that maximizes validation F1-score.

    Args:
        patience: Epochs to wait without improvement before stopping.
        min_delta: Minimum F1 increment considered as improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float = -float("inf")
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_state: dict | None = None

    def step(self, score: float, model: nn.Module) -> bool:
        """Update best score and return ``True`` when a new best is found.

        Args:
            score: Current epoch validation F1.
            model: Model whose state will be snapshot on improvement.

        Returns:
            ``True`` if ``score`` is a new best, ``False`` otherwise.
        """
        improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run a single training epoch and return the mean batch loss."""
    model.train()
    running_loss = 0.0
    running_count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_count += batch_size
    return running_loss / max(running_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, float, float, float]:
    """Evaluate the model and return ``(loss, f1, precision, recall)``."""
    model.eval()
    running_loss = 0.0
    running_count = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets_dev = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets_dev)
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_count += batch_size
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float().cpu().numpy().reshape(-1)
        all_preds.append(preds)
        all_targets.append(targets.numpy().reshape(-1))

    y_pred = np.concatenate(all_preds) if all_preds else np.zeros(0)
    y_true = np.concatenate(all_targets) if all_targets else np.zeros(0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0.0,
    )
    return running_loss / max(running_count, 1), float(f1), float(precision), float(recall)


def plot_curves(history: EpochMetrics, out_path: Path) -> None:
    """Save a two-panel figure with loss and F1 curves."""
    epochs = range(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history.train_loss, label="train", marker="o")
    axes[0].plot(epochs, history.val_loss, label="val", marker="s")
    axes[0].set_title("Loss (BCEWithLogits)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history.val_f1, label="val F1", marker="o", color="tab:green")
    axes[1].plot(
        epochs, history.val_precision, label="val precision", marker="^", color="tab:orange"
    )
    axes[1].plot(
        epochs, history.val_recall, label="val recall", marker="v", color="tab:red"
    )
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved training curves to %s", out_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train AnomalyDetectorMLP")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--curves-path", type=Path, default=DEFAULT_CURVES_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--pos-weight-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to neg/pos pos_weight (e.g. 0.25 to soften)",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Threshold used to compute reported P/R/F1; checkpoint stores it",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    set_seed(args.seed)

    device = get_device()
    logger.info("Using device: %s", device)

    train_loader, val_loader, _test_loader, meta = build_dataloaders(
        csv_path=args.csv_path,
        scaler_path=args.scaler_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(
        "Class balance (train windows): pos=%d neg=%d pos_weight=%.4f",
        meta["train_positives"],
        meta["train_negatives"],
        meta["pos_weight"],
    )

    model = AnomalyDetectorMLP(
        input_dim=meta["input_dim"],
        hidden_dims=(64, 32),
        dropout=args.dropout,
    ).to(device)

    effective_pos_weight = meta["pos_weight"] * args.pos_weight_scale
    pos_weight = torch.tensor(
        [effective_pos_weight], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(
        "Effective pos_weight=%.4f (raw=%.4f scale=%.2f)",
        effective_pos_weight,
        meta["pos_weight"],
        args.pos_weight_scale,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    stopper = EarlyStopping(patience=args.patience)

    history = EpochMetrics()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_precision, val_recall = evaluate(
            model, val_loader, criterion, device, threshold=args.decision_threshold
        )
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_f1.append(val_f1)
        history.val_precision.append(val_precision)
        history.val_recall.append(val_recall)

        improved = stopper.step(val_f1, model)
        marker = " *best" if improved else ""
        logger.info(
            "epoch=%03d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f "
            "| val_precision=%.4f | val_recall=%.4f%s",
            epoch,
            train_loss,
            val_loss,
            val_f1,
            val_precision,
            val_recall,
            marker,
        )
        if stopper.should_stop:
            logger.info(
                "Early stopping at epoch=%d (best val_f1=%.4f)",
                epoch,
                stopper.best_score,
            )
            break

    if stopper.best_state is None:
        logger.warning("Training finished without an improvement step; saving last state")
        best_state = model.state_dict()
    else:
        best_state = stopper.best_state

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": meta["input_dim"],
            "hidden_dims": (64, 32),
            "dropout": args.dropout,
            "window_size": meta["window_size"],
            "num_features": meta["num_features"],
            "best_val_f1": stopper.best_score,
            "decision_threshold": args.decision_threshold,
            "pos_weight_scale": args.pos_weight_scale,
            "effective_pos_weight": effective_pos_weight,
        },
        args.model_path,
    )
    logger.info(
        "Saved best checkpoint to %s (best val_f1=%.4f)",
        args.model_path,
        stopper.best_score,
    )

    plot_curves(history, args.curves_path)


if __name__ == "__main__":
    main()
