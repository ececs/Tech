"""Evaluate the trained AnomalyDetectorMLP on the temporal test split.

Loads ``best_model.pth`` (reconstructs the architecture from the metadata
saved at training time) and the persisted ``scaler.joblib``. Re-builds
the test :class:`DataLoader` from ``dataset_servidores.csv`` using the
same temporal split as training (last 15 % of rows), runs inference at
the calibrated decision threshold, prints classification metrics and
saves a confusion-matrix heatmap to ``test_confusion_matrix.png``.

Run from the repository root::

    python -m src.evaluate
    python -m src.evaluate --threshold 0.71 --checkpoint best_model.pth
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src.dataset import build_dataloaders
from src.model import AnomalyDetectorMLP
from src.train import get_device

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "dataset_servidores.csv"
DEFAULT_SCALER_PATH: Path = PROJECT_ROOT / "artifacts" / "scaler.joblib"
DEFAULT_CHECKPOINT_PATH: Path = PROJECT_ROOT / "best_model.pth"
DEFAULT_CM_PATH: Path = PROJECT_ROOT / "test_confusion_matrix.png"


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[AnomalyDetectorMLP, dict]:
    """Load the trained model and its metadata from a checkpoint file.

    Args:
        path: Path to ``best_model.pth``.
        device: Torch device where the model will be placed.

    Returns:
        Tuple ``(model_eval_mode, checkpoint_dict)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the checkpoint lacks the expected metadata keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    required_keys = {"model_state_dict", "input_dim", "hidden_dims", "dropout"}
    missing = required_keys - set(ckpt.keys())
    if missing:
        raise KeyError(f"Checkpoint missing keys: {sorted(missing)}")

    model = AnomalyDetectorMLP(
        input_dim=int(ckpt["input_dim"]),
        hidden_dims=tuple(ckpt["hidden_dims"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(
    model: AnomalyDetectorMLP,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return scores, hard predictions and labels.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader yielding ``(features, labels)`` batches.
        device: Inference device.
        threshold: Decision threshold applied to ``sigmoid(logits)``.

    Returns:
        Tuple ``(probabilities, predictions, labels)`` all of shape ``[N]``.
    """
    scores: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        scores.append(prob)
        preds.append((prob > threshold).astype(np.int64))
        labels.append(targets.numpy().reshape(-1).astype(np.int64))
    return (
        np.concatenate(scores),
        np.concatenate(preds),
        np.concatenate(labels),
    )


def plot_confusion_matrix(
    cm: np.ndarray, out_path: Path, title: str = "Test Confusion Matrix"
) -> None:
    """Save a seaborn heatmap of a 2x2 confusion matrix.

    Args:
        cm: Confusion matrix with rows = true, cols = predicted.
        out_path: Destination PNG path.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    annot = np.array(
        [
            [f"TN\n{cm[0, 0]}", f"FP\n{cm[0, 1]}"],
            [f"FN\n{cm[1, 0]}", f"TP\n{cm[1, 1]}"],
        ]
    )
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", out_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate AnomalyDetectorMLP")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--cm-path", type=Path, default=DEFAULT_CM_PATH)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold (defaults to the value stored in the checkpoint)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation on the test split."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    device = get_device()
    logger.info("Evaluation device: %s", device)

    model, ckpt = load_checkpoint(args.checkpoint, device)
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(ckpt.get("decision_threshold", 0.5))
    )
    logger.info(
        "Loaded checkpoint: input_dim=%d hidden_dims=%s dropout=%.2f "
        "best_val_f1=%.4f decision_threshold=%.3f",
        int(ckpt["input_dim"]),
        tuple(ckpt["hidden_dims"]),
        float(ckpt["dropout"]),
        float(ckpt.get("best_val_f1", float("nan"))),
        threshold,
    )

    _train_loader, _val_loader, test_loader, meta = build_dataloaders(
        csv_path=args.csv_path,
        scaler_path=args.scaler_path,
        batch_size=args.batch_size,
        window_size=int(ckpt.get("window_size", 5)),
        num_workers=0,
    )
    logger.info(
        "Test split: windows=%d positives_rate=%.4f",
        meta["test_windows"],
        np.mean([y.item() for _, y in test_loader.dataset]),
    )

    scores, preds, labels = predict(model, test_loader, device, threshold)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0.0
    )
    accuracy = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    logger.info("=" * 60)
    logger.info("TEST METRICS (threshold=%.3f)", threshold)
    logger.info("=" * 60)
    logger.info("F1-score : %.4f", f1)
    logger.info("Precision: %.4f", precision)
    logger.info("Recall   : %.4f", recall)
    logger.info("Accuracy : %.4f", accuracy)
    logger.info("Confusion matrix:")
    logger.info("                Pred 0   Pred 1")
    logger.info("    True 0    %7d  %7d   (TN, FP)", tn, fp)
    logger.info("    True 1    %7d  %7d   (FN, TP)", fn, tp)
    logger.info("=" * 60)
    logger.info(
        "\n%s",
        classification_report(
            labels, preds, target_names=["normal", "failure"], zero_division=0.0
        ),
    )

    plot_confusion_matrix(
        cm,
        args.cm_path,
        title=f"Test Confusion Matrix (threshold={threshold:.3f}, F1={f1:.3f})",
    )


if __name__ == "__main__":
    main()
