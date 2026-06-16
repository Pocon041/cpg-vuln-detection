from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import (
    RGCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


class GatedBottleneck(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.value = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.activation(self.value(features))
        gate = torch.sigmoid(self.gate(features))
        return self.dropout(self.norm(value * gate))


class ResidualRGCNBlock(nn.Module):
    def __init__(self, *, hidden_dim: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        self.conv = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        conv = self.conv(hidden, edge_index, edge_type)
        hidden = self.norm(hidden + self.dropout(self.activation(conv)))
        return self.ffn_norm(hidden + self.dropout(self.ffn(hidden)))


class MultiPoolReadout(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.project = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(self.attention(hidden).squeeze(-1))
        weighted_sum = global_add_pool(hidden * scores.unsqueeze(-1), batch)
        weight_sum = global_add_pool(scores.unsqueeze(-1), batch).clamp_min(1e-6)
        attention_pool = weighted_sum / weight_sum
        mean_pool = global_mean_pool(hidden, batch)
        max_pool = global_max_pool(hidden, batch)
        return self.project(torch.cat((attention_pool, mean_pool, max_pool), dim=-1)), scores
