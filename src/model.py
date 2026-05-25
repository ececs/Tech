"""MLP architecture for binary server-failure detection.

The model consumes a flattened sliding window of telemetry features and
returns raw logits (no sigmoid). This is intentional: pair it with
:class:`torch.nn.BCEWithLogitsLoss` for numerical stability during
training, and apply :func:`torch.sigmoid` only at inference time when
explicit probabilities are required.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class AnomalyDetectorMLP(nn.Module):
    """Feed-forward classifier with two hidden blocks and dropout.

    Architecture:
        ``Linear(input_dim -> h1) -> BN -> ReLU -> Dropout ->``
        ``Linear(h1 -> h2) -> BN -> ReLU -> Dropout ->``
        ``Linear(h2 -> 1)`` returning logits.

    Args:
        input_dim: Flattened input dimension (default ``20`` = 5 * 4).
        hidden_dims: Sizes of the two hidden layers.
        dropout: Dropout probability applied after each hidden block.

    Raises:
        ValueError: If ``hidden_dims`` does not contain exactly two values.
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: tuple[int, int] = (64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 2:
            raise ValueError(
                f"hidden_dims must be a 2-tuple, got {hidden_dims!r}"
            )
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(h2, 1),
        )
        logger.debug(
            "Initialized AnomalyDetectorMLP: input_dim=%d hidden=%s dropout=%.2f",
            input_dim,
            hidden_dims,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw logits for a batch of flattened windows.

        Args:
            x: Tensor of shape ``[batch, input_dim]``.

        Returns:
            Tensor of shape ``[batch, 1]`` containing logits.
        """
        return self.net(x)
