from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GatedGraphConv
from torch_geometric.utils import to_dense_batch

from cpg_vuln.models.common import ModelOutput


class DevignCPG(nn.Module):
    """Devign-style GGNN + Conv graph classifier for CPG inputs."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_node_types: int,
        hidden_dim: int = 128,
        node_type_dim: int = 32,
        steps: int = 6,
        dropout: float = 0.3,
        max_nodes: int = 205,
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive")
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        self.max_nodes = max_nodes
        self.node_types = nn.Embedding(num_node_types, node_type_dim)
        self.project = nn.Linear(input_dim + node_type_dim, hidden_dim)
        self.ggnn = GatedGraphConv(hidden_dim, num_layers=steps)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.value_conv = _DevignConvBranch(max_nodes=max_nodes, input_width=hidden_dim * 2)
        self.gate_conv = _DevignConvBranch(max_nodes=max_nodes, input_width=hidden_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, data) -> ModelOutput:
        features = torch.cat((data.x, self.node_types(data.node_type_id)), dim=-1)
        node_input = self.dropout(self.activation(self.project(features)))
        propagated = self.ggnn(node_input, data.edge_index)
        sequence, _ = to_dense_batch(
            torch.cat((propagated, node_input), dim=-1),
            data.batch,
            max_num_nodes=self.max_nodes,
        )
        hidden_sequence, _ = to_dense_batch(
            propagated,
            data.batch,
            max_num_nodes=self.max_nodes,
        )
        risk_logit = self.value_conv(sequence) * self.gate_conv(hidden_sequence)
        risk_logit = self.output_dropout(risk_logit)
        baseline_logit = torch.zeros_like(risk_logit)
        return ModelOutput(logits=torch.cat((baseline_logit, risk_logit), dim=-1))


class _DevignConvBranch(nn.Module):
    def __init__(self, *, max_nodes: int, input_width: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(max_nodes, 50, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(50, 20, kernel_size=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(20 * _devign_conv_output_width(input_width), 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.pool1(torch.relu(self.conv1(sequence)))
        hidden = self.pool2(self.conv2(hidden))
        return self.fc(hidden.flatten(start_dim=1))


def _devign_conv_output_width(input_width: int) -> int:
    width = _pool_width(input_width, kernel_size=3, stride=2)
    width = width + 2
    return _pool_width(width, kernel_size=2, stride=2)


def _pool_width(width: int, *, kernel_size: int, stride: int) -> int:
    return int((width - kernel_size) / stride + 1)
