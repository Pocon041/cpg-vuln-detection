from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import RGCNConv, global_add_pool

from cpg_vuln.models.common import ModelOutput


class SelectiveFusionCPG(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        function_dim: int,
        num_node_types: int,
        num_relations: int,
        hidden_dim: int = 128,
        node_type_dim: int = 32,
        dropout: float = 0.3,
        use_semantics: bool = True,
    ) -> None:
        super().__init__()
        self.use_semantics = use_semantics
        self.node_types = nn.Embedding(num_node_types, node_type_dim)
        self.project = nn.Linear(input_dim + node_type_dim, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.attention = nn.Linear(hidden_dim, 1)
        self.semantic_project = nn.Sequential(
            nn.Linear(function_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, data) -> ModelOutput:
        features = torch.cat((data.x, self.node_types(data.node_type_id)), dim=-1)
        hidden = self.dropout(self.activation(self.project(features)))
        first = self.dropout(
            self.activation(self.norm1(self.conv1(hidden, data.edge_index, data.edge_type)))
        )
        second = self.dropout(
            self.activation(self.norm2(self.conv2(first, data.edge_index, data.edge_type) + first))
        )
        scores = torch.sigmoid(self.attention(second).squeeze(-1))
        weighted_sum = global_add_pool(second * scores.unsqueeze(-1), data.batch)
        weight_sum = global_add_pool(scores.unsqueeze(-1), data.batch).clamp_min(1e-6)
        graph_vector = weighted_sum / weight_sum
        if self.use_semantics:
            semantic_vector = self.semantic_project(data.function_x)
            gate = torch.sigmoid(self.gate(torch.cat((graph_vector, semantic_vector), dim=-1)))
            fused = gate * graph_vector + (1 - gate) * semantic_vector
        else:
            fused = graph_vector
        return ModelOutput(logits=self.classifier(fused), node_attention=scores)

