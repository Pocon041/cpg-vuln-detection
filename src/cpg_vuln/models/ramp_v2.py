from __future__ import annotations

import torch
from torch import nn

from cpg_vuln.models.common import ModelOutput
from cpg_vuln.models.relational import (
    GatedBottleneck,
    MultiPoolReadout,
    ResidualRGCNBlock,
)


def _graph_level_function_features(data) -> torch.Tensor:
    function_x = data.function_x
    num_graphs = int(data.num_graphs)
    if function_x.ndim != 2:
        raise ValueError(
            "Expected batched function_x with shape "
            f"[num_graphs, function_dim], got {tuple(function_x.shape)}"
        )
    if function_x.shape[0] != num_graphs:
        raise ValueError(
            "function_x graph dimension mismatch: "
            f"num_graphs={num_graphs}, function_x.shape={tuple(function_x.shape)}"
        )
    return function_x


class RampV2CPG(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        function_dim: int,
        num_node_types: int,
        num_relations: int,
        hidden_dim: int = 256,
        node_type_dim: int = 32,
        layers: int = 3,
        dropout: float = 0.2,
        encoder: str = "rgcn",
        use_semantics: bool = True,
    ) -> None:
        super().__init__()
        if encoder != "rgcn":
            raise ValueError(f"unsupported fast-proof encoder: {encoder}")
        if layers < 1:
            raise ValueError("layers must be positive")
        self.use_semantics = use_semantics
        self.node_types = nn.Embedding(num_node_types, node_type_dim)
        self.project = GatedBottleneck(input_dim + node_type_dim, hidden_dim, dropout)
        self.blocks = nn.ModuleList(
            [
                ResidualRGCNBlock(
                    hidden_dim=hidden_dim,
                    num_relations=num_relations,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.jump_project = nn.Sequential(
            nn.Linear(hidden_dim * layers, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.readout = MultiPoolReadout(hidden_dim, dropout)
        self.semantic_project = nn.Sequential(
            nn.Linear(function_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def _graph_vector_and_attention(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = torch.cat((data.x, self.node_types(data.node_type_id)), dim=-1)
        hidden = self.project(node_features)
        states = []
        for block in self.blocks:
            hidden = block(hidden, data.edge_index, data.edge_type)
            states.append(hidden)
        hidden = self.jump_project(torch.cat(states, dim=-1))
        return self.readout(hidden, data.batch)

    def _semantic_vector(self, data) -> torch.Tensor:
        return self.semantic_project(_graph_level_function_features(data))

    def _fused_vector(
        self,
        graph_vector: torch.Tensor,
        semantic_vector: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat((graph_vector, semantic_vector), dim=-1))
        )
        return gate * graph_vector + (1.0 - gate) * semantic_vector

    def forward(self, data) -> ModelOutput:
        graph_vector, attention = self._graph_vector_and_attention(data)
        if self.use_semantics:
            graph_vector = self._fused_vector(graph_vector, self._semantic_vector(data))
        return ModelOutput(logits=self.classifier(graph_vector), node_attention=attention)


class RampV2DualHeadCPG(RampV2CPG):
    def __init__(
        self,
        *,
        input_dim: int,
        function_dim: int,
        num_node_types: int,
        num_relations: int,
        hidden_dim: int = 256,
        node_type_dim: int = 32,
        layers: int = 3,
        dropout: float = 0.2,
        encoder: str = "rgcn",
        use_semantics: bool = True,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            function_dim=function_dim,
            num_node_types=num_node_types,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            node_type_dim=node_type_dim,
            layers=layers,
            dropout=dropout,
            encoder=encoder,
            use_semantics=use_semantics,
        )
        self.graph_classifier = nn.Linear(hidden_dim, 2)
        self.semantic_classifier = nn.Linear(hidden_dim, 2)
        self.logit_weights = nn.Parameter(torch.zeros(3))

    def forward(self, data) -> ModelOutput:
        graph_vector, attention = self._graph_vector_and_attention(data)
        if not self.use_semantics:
            return ModelOutput(
                logits=self.graph_classifier(graph_vector),
                node_attention=attention,
            )

        semantic_vector = self._semantic_vector(data)
        fused_vector = self._fused_vector(graph_vector, semantic_vector)
        fusion_logits = self.classifier(fused_vector)
        graph_logits = self.graph_classifier(graph_vector)
        semantic_logits = self.semantic_classifier(semantic_vector)
        weights = torch.softmax(self.logit_weights, dim=0)
        logits = (
            weights[0] * fusion_logits
            + weights[1] * graph_logits
            + weights[2] * semantic_logits
        )
        return ModelOutput(logits=logits, node_attention=attention)
