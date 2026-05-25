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
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if len(hidden_dims) != 2:
            raise ValueError(
                f"hidden_dims must be a 2-tuple, got {hidden_dims!r}"
            )
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError(
                f"hidden_dims must contain positive values, got {hidden_dims!r}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
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


class AnomalyDetectorGRU(nn.Module):
    """GRU classifier that consumes the sliding window as a native sequence.

    The dataset emits each window flattened as a vector of length
    ``seq_len * num_features`` in row-major chronological order
    (``[t0_f0..t0_fN, t1_f0..t1_fN, ...]``). This module reshapes the
    flattened tensor back to ``[batch, seq_len, num_features]`` and lets
    a recurrent layer model the temporal dependencies natively, instead
    of forcing an MLP to recover them from a vector.

    Architecture::

        x [B, seq_len*num_features]
          -> view(-1, seq_len, num_features)
          -> GRU(input_size=num_features, hidden_size, num_layers)
          -> take last time-step: out[:, -1, :]
          -> Dropout
          -> Linear(hidden_size, 1)
          -> raw logits

    PyTorch's ``nn.GRU`` only applies the ``dropout`` argument *between*
    stacked layers, so we add an explicit ``nn.Dropout`` after the
    recurrent output to regularise the single-layer case as well.

    Args:
        num_features: Features per time step (default ``4``).
        seq_len: Window length used by the dataset (default ``5``).
        hidden_size: Hidden state dimension of the GRU.
        num_layers: Number of stacked GRU layers.
        dropout: Probability for the post-GRU dropout and the inter-layer
            dropout when ``num_layers > 1``.

    Raises:
        ValueError: If ``num_features``, ``seq_len``, ``hidden_size`` or
            ``num_layers`` are not positive.
    """

    def __init__(
        self,
        num_features: int = 4,
        seq_len: int = 5,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_features <= 0 or seq_len <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError(
                "num_features, seq_len, hidden_size and num_layers must be > 0; "
                f"got {num_features=}, {seq_len=}, {hidden_size=}, {num_layers=}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.num_features = num_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout

        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        logger.debug(
            "Initialized AnomalyDetectorGRU: num_features=%d seq_len=%d "
            "hidden_size=%d num_layers=%d dropout=%.2f",
            num_features,
            seq_len,
            hidden_size,
            num_layers,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw logits for a batch of flattened windows.

        Args:
            x: Tensor of shape ``[batch, seq_len * num_features]``.

        Returns:
            Tensor of shape ``[batch, 1]`` containing logits.
        """
        x = x.view(-1, self.seq_len, self.num_features)
        out, _ = self.gru(x)
        last = self.head_dropout(out[:, -1, :])
        return self.fc(last)
