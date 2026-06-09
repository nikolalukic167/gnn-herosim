"""
PointwiseEdgeMLP — pointwise scorer for Regime A batch tabular placement.

Architecture: Linear → LayerNorm → SiLU → Linear → LayerNorm → SiLU → Linear(→1)
Input: 22-d Option-B edge features (3 task + 14 platform + 5 edge).
Output: scalar logit per edge.  Used with grouped softmax / CE at training time
and argmax decode at inference time (same decode path as GNN/XGB).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.policy.tabular.constants import FEATURE_DIM


class PointwiseEdgeMLP(nn.Module):
    """Pointwise edge scorer for batch placement (Regime A).

    Args:
        input_dim: Feature dimension (default 22, must equal FEATURE_DIM).
        hidden_dim: Hidden layer width (default 64).
    """

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 64) -> None:
        super().__init__()
        if input_dim != FEATURE_DIM:
            raise ValueError(
                f"input_dim={input_dim} does not match FEATURE_DIM={FEATURE_DIM}"
            )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of edge feature vectors.

        Args:
            x: Float tensor of shape [N, input_dim].

        Returns:
            Score tensor of shape [N] (squeezed from [N, 1]).
        """
        return self.net(x).squeeze(-1)
