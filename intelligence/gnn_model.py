"""GAT Graph Attention Network model — implementation in Phase 4B (T-045).

Architecture (per ARCHITECTURE.md §4.2 + TECH_STACK.md §4.2):

    Input:  N × 13 feature tensor, edge_index COO tensor, edge_attr weights
    Layer 1: GATConv(in=13, out=64, heads=4, dropout=0.3, concat=True) → N × 256
             + ELU activation + BatchNorm1d(256)
    Layer 2: GATConv(in=256, out=32, heads=2, dropout=0.3, concat=False) → N × 32
             + ELU activation
    Output:  Linear(32 → 1) + Sigmoid → burnout_score per node [0, 1]

MC Dropout: 5 stochastic forward passes (model.train() during inference) for confidence.
Attention weights from Layer 2: used as feature attribution (which edges contribute).

Design Rationale (GAT over GCN/GraphSAGE):
- Attention weights are interpretable — explains which collaborations drive risk
- Handles heterogeneous edge weights natively
- Performs well on sparse org graphs (~15% edge density)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

logger = logging.getLogger(__name__)

# Architecture hyperparameters — locked in T-015 design review
HIDDEN_DIM = 64
HEADS_1 = 4
HEADS_2 = 2
DROPOUT = 0.3
IN_CHANNELS = 13      # Feature vector dimension (must match FEATURE_DIM in feature_extractor.py)
OUT_CHANNELS = 1      # Binary burnout risk per node
MC_PASSES = 5         # MC Dropout inference passes


class BurnoutGAT(nn.Module):
    """2-layer Graph Attention Network for burnout risk prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = GATConv(
            IN_CHANNELS,
            HIDDEN_DIM,
            heads=HEADS_1,
            dropout=DROPOUT,
            concat=True,
        )
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM * HEADS_1)  # 256
        self.conv2 = GATConv(
            HIDDEN_DIM * HEADS_1,
            32,
            heads=HEADS_2,
            dropout=DROPOUT,
            concat=False,
        )
        self.classifier = nn.Linear(32, OUT_CHANNELS)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,                    # [N, 13]
        edge_index: torch.Tensor,           # [2, E]
        edge_attr: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Node feature matrix of shape [N, IN_CHANNELS].
            edge_index: Graph connectivity in COO format, shape [2, E].
            edge_attr: Optional edge weight tensor, shape [E] or [E, F].
            return_attention: If True, also return Layer-2 attention weights.

        Returns:
            Burnout scores of shape [N, 1], or a (scores, attention_weights) tuple
            when return_attention is True.  Attention weights have shape [E, HEADS_2].
        """
        # Layer 1: GATConv → BN → ELU  (output: [N, 256])
        x = self.elu(self.bn1(self.conv1(x, edge_index)))

        if return_attention:
            # return_attention_weights=True makes conv2 return
            # (out, (edge_index_out, alpha))
            conv2_out: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
            conv2_out = self.conv2(x, edge_index, return_attention_weights=True)  # type: ignore[assignment]
            x_out, (_edge_index_out, alpha) = conv2_out
            x_out = self.elu(x_out)
            logits: torch.Tensor = self.classifier(x_out)
            return logits, alpha

        x = self.elu(self.conv2(x, edge_index))
        return self.classifier(x)  # raw logits — sigmoid applied by BCEWithLogitsLoss or caller

    @torch.no_grad()  # type: ignore[misc]
    def mc_dropout_predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        n_passes: int = MC_PASSES,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MC Dropout inference: n stochastic forward passes for confidence intervals.

        Temporarily re-enables dropout (model.train()) so each pass produces a
        different stochastic sample, then restores eval mode.

        Args:
            x: Node feature matrix of shape [N, IN_CHANNELS].
            edge_index: Graph connectivity in COO format, shape [2, E].
            n_passes: Number of MC Dropout forward passes (default: MC_PASSES).

        Returns:
            A 3-tuple (mean_scores, confidence_low, confidence_high), each of
            shape [N, 1], representing the mean prediction and the 95% CI bounds
            computed as mean ± 1.96 * std.
        """
        self.train()  # Enable dropout for stochastic sampling
        raw_passes: list[torch.Tensor] = []
        for _ in range(n_passes):
            pass_out = self.forward(x, edge_index)
            # forward() returns raw logits; apply sigmoid for probability output
            assert isinstance(pass_out, torch.Tensor)
            raw_passes.append(torch.sigmoid(pass_out))
        self.eval()

        passes = torch.stack(raw_passes)  # [n_passes, N, 1]
        mean = passes.mean(dim=0)
        std = passes.std(dim=0)
        return mean, mean - 1.96 * std, mean + 1.96 * std


# Trained architecture — matches final-v1 / csv-v2-tuned checkpoints
# 10 features, smaller capacity, dropout=0.1
SMALL_IN_CHANNELS = 10
SMALL_FEATURE_COLS = [
    "meeting_density",
    "after_hours_ratio",
    "response_latency_avg",
    "focus_time_blocks",
    "msg_volume_daily",
    "msg_response_time",
    "mention_load",
    "context_switch_rate",
    "hrv_avg",
    "sleep_score",
]


class SmallBurnoutGAT(nn.Module):
    """Compact 2-layer GAT — matches the final-v1 trained checkpoint.

    Architecture: GATConv(10→32×2) → BN → GATConv(64→16×1) → Linear(16→1)
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = GATConv(SMALL_IN_CHANNELS, 32, heads=2, dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = GATConv(64, 16, heads=1, dropout=dropout, concat=False)
        self.classifier = nn.Linear(16, 1)
        self.elu = nn.ELU()
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.elu(self.bn1(self.conv1(x, edge_index)))
        x = self.drop(x)

        if return_attention:
            conv2_out = self.conv2(x, edge_index, return_attention_weights=True)  # type: ignore[assignment]
            x_out, (_ei, alpha) = conv2_out  # type: ignore[misc]
            return self.classifier(self.elu(x_out)), alpha

        x = self.elu(self.conv2(x, edge_index))
        return self.classifier(x)

    @torch.no_grad()  # type: ignore[misc]
    def mc_dropout_predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        n_passes: int = MC_PASSES,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.train()
        raw_passes: list[torch.Tensor] = []
        for _ in range(n_passes):
            out = self.forward(x, edge_index)
            assert isinstance(out, torch.Tensor)
            raw_passes.append(torch.sigmoid(out))
        self.eval()
        passes = torch.stack(raw_passes)
        mean = passes.mean(dim=0)
        std = passes.std(dim=0)
        return mean, mean - 1.96 * std, mean + 1.96 * std
