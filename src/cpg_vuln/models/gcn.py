from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

from cpg_vuln.models.common import ModelOutput


class GCNClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_node_types: int,
        hidden_dim: int = 128,
        node_type_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.node_types = nn.Embedding(num_node_types, node_type_dim)
        self.project = nn.Linear(input_dim + node_type_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, data) -> ModelOutput:
        features = torch.cat((data.x, self.node_types(data.node_type_id)), dim=-1)
        hidden = self.dropout(self.activation(self.project(features)))
        first = self.dropout(self.activation(self.norm1(self.conv1(hidden, data.edge_index))))
        second = self.dropout(self.activation(self.norm2(self.conv2(first, data.edge_index) + first)))
        pooled = torch.cat(
            (global_mean_pool(second, data.batch), global_max_pool(second, data.batch)),
            dim=-1,
        )
        return ModelOutput(logits=self.classifier(pooled))

