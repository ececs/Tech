"""Training entry point for the GRU variant of the anomaly detector.

Mirrors :mod:`src.train` but instantiates :class:`AnomalyDetectorGRU`
instead of the MLP. The MLP training script is intentionally left
untouched so its reproducibility guarantees are preserved. All shared
loop machinery (early stopping, per-epoch evaluation, curve plotting,
device selection, seeding) is imported from :mod:`src.train` to avoid
silent drift between the two pipelines.

Run from the repository root::

    python -m src.train_gru \\
        --epochs 50 --patience 7 --batch-size 64 --lr 3e-3 \\
        --dropout 0.3 --pos-weight-scale 0.25 --decision-threshold 0.5

Artefacts produced: ``best_model_gru.pth`` and ``training_curves_gru.png``.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

from src.dataset import build_dataloaders
from src.model import AnomalyDetectorGRU
from src.train import (
    EarlyStopping,
    EpochMetrics,
    evaluate,
    get_device,
    plot_curves,
    set_seed,
    train_one_epoch,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "dataset_servidores.csv"
DEFAULT_SCALER_PATH: Path = PROJECT_ROOT / "artifacts" / "scaler.joblib"
DEFAULT_MODEL_PATH: Path = PROJECT_ROOT / "best_model_gru.pth"
DEFAULT_CURVES_PATH: Path = PROJECT_ROOT / "training_curves_gru.png"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train AnomalyDetectorGRU")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--curves-path", type=Path, default=DEFAULT_CURVES_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="GRU hidden state dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of stacked GRU layers (>1 enables inter-layer dropout)",
    )
    parser.add_argument(
        "--pos-weight-scale",
        type=float,
        default=0.25,
        help="Multiplier applied to neg/pos pos_weight",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Threshold used to compute reported P/R/F1; checkpoint stores it",
    )
    parser.add_argument("--patience", type=int, default=7)
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
    """Run the full GRU training pipeline."""
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

    model = AnomalyDetectorGRU(
        num_features=meta["num_features"],
        seq_len=meta["window_size"],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "AnomalyDetectorGRU instantiated: num_features=%d seq_len=%d "
        "hidden_size=%d num_layers=%d dropout=%.2f params=%d",
        meta["num_features"],
        meta["window_size"],
        args.hidden_size,
        args.num_layers,
        args.dropout,
        num_params,
    )

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
            "model_type": "gru",
            "model_state_dict": best_state,
            "num_features": meta["num_features"],
            "seq_len": meta["window_size"],
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "best_val_f1": stopper.best_score,
            "decision_threshold": args.decision_threshold,
            "pos_weight_scale": args.pos_weight_scale,
            "effective_pos_weight": effective_pos_weight,
            "num_params": num_params,
        },
        args.model_path,
    )
    logger.info(
        "Saved best GRU checkpoint to %s (best val_f1=%.4f, params=%d)",
        args.model_path,
        stopper.best_score,
        num_params,
    )

    plot_curves(history, args.curves_path)


if __name__ == "__main__":
    main()
