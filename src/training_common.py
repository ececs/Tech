"""Shared training utilities for anomaly detector entry points."""

from __future__ import annotations

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

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Container for per-epoch metrics."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    val_precision: list[float] = field(default_factory=list)
    val_recall: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class TrainLoopConfig:
    """Typed config shared by the MLP/GRU training loops."""

    epochs: int
    patience: int
    decision_threshold: float


class EarlyStopping:
    """Early stopping that maximizes validation F1-score."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float = -float("inf")
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_state: dict | None = None

    def step(self, score: float, model: nn.Module) -> bool:
        """Update best score and return ``True`` when a new best is found."""
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


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainLoopConfig,
    model_label: str,
) -> tuple[dict[str, torch.Tensor], float, EpochMetrics]:
    """Run the train/validation loop with early stopping."""
    stopper = EarlyStopping(patience=config.patience)
    history = EpochMetrics()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_precision, val_recall = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=config.decision_threshold,
        )
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_f1.append(val_f1)
        history.val_precision.append(val_precision)
        history.val_recall.append(val_recall)

        improved = stopper.step(val_f1, model)
        marker = " *best" if improved else ""
        logger.info(
            "%s epoch=%03d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f "
            "| val_precision=%.4f | val_recall=%.4f%s",
            model_label,
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
                "%s early stopping at epoch=%d (best val_f1=%.4f)",
                model_label,
                epoch,
                stopper.best_score,
            )
            break

    if stopper.best_state is None:
        logger.warning("%s finished without an improvement step; saving last state", model_label)
        best_state = copy.deepcopy(model.state_dict())
    else:
        best_state = stopper.best_state

    return best_state, stopper.best_score, history


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
