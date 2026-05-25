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
import logging
from pathlib import Path
import torch
from torch import nn

from src.dataset import build_training_dataloaders
from src.model import AnomalyDetectorMLP
from src.training_common import (
    TrainLoopConfig,
    get_device,
    plot_curves,
    run_training_loop,
    set_seed,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "dataset_servidores.csv"
DEFAULT_SCALER_PATH: Path = PROJECT_ROOT / "artifacts" / "scaler.joblib"
DEFAULT_MODEL_PATH: Path = PROJECT_ROOT / "best_model.pth"
DEFAULT_CURVES_PATH: Path = PROJECT_ROOT / "training_curves.png"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train AnomalyDetectorMLP")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--curves-path", type=Path, default=DEFAULT_CURVES_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs=2,
        metavar=("H1", "H2"),
        default=(64, 32),
        help="Two hidden-layer sizes for the MLP architecture",
    )
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


def validate_args(args: argparse.Namespace) -> None:
    """Fail fast on invalid CLI argument combinations."""
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be > 0, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}")
    if any(dim <= 0 for dim in args.hidden_dims):
        raise ValueError(f"--hidden-dims must contain positive ints, got {args.hidden_dims}")
    if args.lr <= 0:
        raise ValueError(f"--lr must be > 0, got {args.lr}")
    if args.weight_decay < 0:
        raise ValueError(f"--weight-decay must be >= 0, got {args.weight_decay}")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(f"--dropout must be in [0, 1), got {args.dropout}")
    if args.pos_weight_scale <= 0:
        raise ValueError(
            f"--pos-weight-scale must be > 0, got {args.pos_weight_scale}"
        )
    if not 0.0 <= args.decision_threshold <= 1.0:
        raise ValueError(
            "--decision-threshold must be in [0, 1], "
            f"got {args.decision_threshold}"
        )
    if args.patience <= 0:
        raise ValueError(f"--patience must be > 0, got {args.patience}")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0, got {args.num_workers}")


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    validate_args(args)
    set_seed(args.seed)

    device = get_device()
    logger.info("Using device: %s", device)

    train_loader, val_loader, _test_loader, meta = build_training_dataloaders(
        csv_path=args.csv_path,
        scaler_path=args.scaler_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(
        "Class balance (train windows): pos=%d neg=%d pos_weight=%.4f",
        meta.train_positives,
        meta.train_negatives,
        meta.pos_weight,
    )

    model = AnomalyDetectorMLP(
        input_dim=meta.input_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "AnomalyDetectorMLP instantiated: input_dim=%d hidden_dims=%s dropout=%.2f params=%d",
        meta.input_dim,
        tuple(args.hidden_dims),
        args.dropout,
        num_params,
    )

    effective_pos_weight = meta.pos_weight * args.pos_weight_scale
    pos_weight = torch.tensor(
        [effective_pos_weight], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(
        "Effective pos_weight=%.4f (raw=%.4f scale=%.2f)",
        effective_pos_weight,
        meta.pos_weight,
        args.pos_weight_scale,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_state, best_val_f1, history = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=TrainLoopConfig(
            epochs=args.epochs,
            patience=args.patience,
            decision_threshold=args.decision_threshold,
        ),
        model_label="MLP",
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "mlp",
            "model_state_dict": best_state,
            "input_dim": meta.input_dim,
            "hidden_dims": tuple(args.hidden_dims),
            "dropout": args.dropout,
            "window_size": meta.window_size,
            "num_features": meta.num_features,
            "best_val_f1": best_val_f1,
            "decision_threshold": args.decision_threshold,
            "pos_weight_scale": args.pos_weight_scale,
            "effective_pos_weight": effective_pos_weight,
            "num_params": num_params,
        },
        args.model_path,
    )
    logger.info(
        "Saved best checkpoint to %s (best val_f1=%.4f)",
        args.model_path,
        best_val_f1,
    )

    plot_curves(history, args.curves_path)


if __name__ == "__main__":
    main()
