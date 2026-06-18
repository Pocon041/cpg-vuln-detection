from __future__ import annotations

import math

import torch
from torch import nn

from cpg_vuln.models.common import ModelOutput
from cpg_vuln.models.relational import (
    GatedBottleneck,
    GatedResidualRGCNBlock,
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

    def _node_hidden(self, data) -> torch.Tensor:
        node_features = torch.cat((data.x, self.node_types(data.node_type_id)), dim=-1)
        hidden = self.project(node_features)
        states = []
        for block in self.blocks:
            hidden = block(hidden, data.edge_index, data.edge_type)
            states.append(hidden)
        return self.jump_project(torch.cat(states, dim=-1))

    def _graph_vector_and_attention(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._node_hidden(data)
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


class RampV2GatedRGCNCPG(RampV2CPG):
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
        dropout: float = 0.25,
        encoder: str = "rgcn",
        use_semantics: bool = True,
        gate_bias_init: float = -1.0,
        ffn_multiplier: int = 2,
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
        self.blocks = nn.ModuleList(
            [
                GatedResidualRGCNBlock(
                    hidden_dim=hidden_dim,
                    num_relations=num_relations,
                    dropout=dropout,
                    gate_bias_init=gate_bias_init,
                    ffn_multiplier=ffn_multiplier,
                )
                for _ in range(layers)
            ]
        )

    def _graph_vector_attention_diagnostics(
        self,
        data,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        node_features = torch.cat((data.x, self.node_types(data.node_type_id)), dim=-1)
        hidden = self.project(node_features)
        states = []
        gate_means = []
        gate_stds = []
        for block in self.blocks:
            hidden, stats = block(
                hidden,
                data.edge_index,
                data.edge_type,
                return_gate_stats=True,
            )
            states.append(hidden)
            gate_means.append(stats["gate_mean"])
            gate_stds.append(stats["gate_std"])
        hidden = self.jump_project(torch.cat(states, dim=-1))
        graph_vector, attention = self.readout(hidden, data.batch)
        diagnostics = {
            "encoder_gate_mean": torch.stack(gate_means).mean(),
            "encoder_gate_std": torch.stack(gate_stds).mean(),
            "encoder_gate_mean_by_layer": torch.stack(gate_means),
            "encoder_gate_std_by_layer": torch.stack(gate_stds),
        }
        return graph_vector, attention, diagnostics

    def forward(self, data) -> ModelOutput:
        graph_vector, attention, diagnostics = self._graph_vector_attention_diagnostics(data)
        if self.use_semantics:
            semantic_vector = self._semantic_vector(data)
            fusion_gate = torch.sigmoid(
                self.fusion_gate(torch.cat((graph_vector, semantic_vector), dim=-1))
            )
            diagnostics["fusion_gate_mean"] = fusion_gate.detach().mean()
            graph_vector = (
                fusion_gate * graph_vector + (1.0 - fusion_gate) * semantic_vector
            )
        return ModelOutput(
            logits=self.classifier(graph_vector),
            node_attention=attention,
            diagnostics=diagnostics,
        )


class RampV3SliceMILCPG(RampV2CPG):
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
        dropout: float = 0.25,
        encoder: str = "rgcn",
        use_semantics: bool = True,
        slice_top_k: int = 3,
        slice_temperature: float = 1.0,
        fusion_logit_init: tuple[float, float, float, float] = (4.0, 0.0, 0.0, -2.0),
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
        if slice_top_k < 1:
            raise ValueError("slice_top_k must be positive")
        if slice_temperature <= 0:
            raise ValueError("slice_temperature must be positive")
        self.slice_top_k = slice_top_k
        self.slice_temperature = slice_temperature
        self.graph_classifier = nn.Linear(hidden_dim, 2)
        self.semantic_classifier = nn.Linear(hidden_dim, 2)
        self.slice_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if len(fusion_logit_init) != 4:
            raise ValueError("fusion_logit_init must contain four values")
        self.logit_weights = nn.Parameter(
            torch.tensor(fusion_logit_init, dtype=torch.float32)
        )

    def forward(self, data) -> ModelOutput:
        hidden = self._node_hidden(data)
        graph_vector, attention = self.readout(hidden, data.batch)
        graph_logits = self.graph_classifier(graph_vector)
        slice_logits, slice_diagnostics = self._slice_mil_logits(hidden, data)
        if self.use_semantics:
            semantic_vector = self._semantic_vector(data)
            fused_vector = self._fused_vector(graph_vector, semantic_vector)
            semantic_logits = self.semantic_classifier(semantic_vector)
            fusion_logits = self.classifier(fused_vector)
        else:
            semantic_logits = graph_logits.new_zeros(graph_logits.shape)
            fusion_logits = graph_logits
        weights = torch.softmax(self.logit_weights, dim=0)
        logits = (
            weights[0] * fusion_logits
            + weights[1] * graph_logits
            + weights[2] * semantic_logits
            + weights[3] * slice_logits
        )
        diagnostics = {
            **slice_diagnostics,
            "fusion_weight_fused": weights[0].detach(),
            "fusion_weight_graph": weights[1].detach(),
            "fusion_weight_semantic": weights[2].detach(),
            "fusion_weight_slice": weights[3].detach(),
        }
        return ModelOutput(
            logits=logits,
            node_attention=attention,
            diagnostics=diagnostics,
            auxiliary_logits={
                "graph": graph_logits,
                "semantic": semantic_logits,
                "slice": slice_logits,
            },
            evidence_logits=slice_logits,
        )

    def _slice_mil_logits(
        self,
        hidden: torch.Tensor,
        data,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        node_scores = self.slice_score(hidden).squeeze(-1)
        graph_scores: list[torch.Tensor] = []
        candidate_counts: list[torch.Tensor] = []
        selected_counts: list[torch.Tensor] = []
        seed_mask = getattr(data, "slice_seed_mask", None)
        if seed_mask is not None:
            seed_mask = seed_mask.bool()
        for graph_index in range(int(data.num_graphs)):
            graph_mask = data.batch == graph_index
            candidate_mask = graph_mask
            if seed_mask is not None:
                seeded = graph_mask & seed_mask
                if bool(seeded.any()):
                    candidate_mask = seeded
            scores = node_scores[candidate_mask]
            top_k = min(self.slice_top_k, int(scores.numel()))
            selected = torch.topk(scores, k=top_k).values
            graph_score = (
                torch.logsumexp(selected / self.slice_temperature, dim=0)
                * self.slice_temperature
                - math.log(top_k)
            )
            graph_scores.append(graph_score)
            candidate_counts.append(scores.new_tensor(float(scores.numel())))
            selected_counts.append(scores.new_tensor(float(top_k)))
        risk = torch.stack(graph_scores)
        logits = torch.stack((-0.5 * risk, 0.5 * risk), dim=-1)
        diagnostics = {
            "slice_candidate_count_mean": torch.stack(candidate_counts).mean().detach(),
            "slice_selected_count_mean": torch.stack(selected_counts).mean().detach(),
            "slice_risk_logit_mean": risk.detach().mean(),
            "slice_risk_logit_std": risk.detach().std(unbiased=False),
        }
        return logits, diagnostics
