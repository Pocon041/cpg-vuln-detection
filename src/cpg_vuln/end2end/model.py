from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.utils import softmax

from cpg_vuln.models.common import ModelOutput


class RawCodeMILTransformer(nn.Module):
    """Transformer encoder with gated attention MIL over raw source chunks."""

    def __init__(
        self,
        *,
        model_name: str = "microsoft/codebert-base",
        encoder: nn.Module | None = None,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        self.encoder = encoder
        hidden_dim = int(self.encoder.config.hidden_size)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        self.value_attention = nn.Linear(hidden_dim, hidden_dim)
        self.gate_attention = nn.Linear(hidden_dim, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        output = nn.Linear(hidden_dim, 2)
        nn.init.normal_(output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(output.bias)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            output,
        )

    def forward(self, data) -> ModelOutput:
        input_ids = data.input_ids
        attention_mask = data.attention_mask
        chunk_batch = getattr(data, "batch", None)
        if chunk_batch is None:
            chunk_batch = torch.zeros(
                input_ids.shape[0],
                dtype=torch.long,
                device=input_ids.device,
            )
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        chunk_vectors = _masked_mean_pool(encoded.last_hidden_state, attention_mask)
        attention_features = torch.tanh(self.value_attention(chunk_vectors)) * torch.sigmoid(
            self.gate_attention(chunk_vectors)
        )
        scores = self.attention_score(attention_features).squeeze(-1)
        weights = softmax(scores, chunk_batch)
        attention_pool = global_add_pool(chunk_vectors * weights.unsqueeze(-1), chunk_batch)
        max_pool = global_max_pool(chunk_vectors, chunk_batch)
        graph_vectors = self.norm(torch.cat((attention_pool, max_pool), dim=-1))
        logits = self.classifier(self.dropout(graph_vectors))
        entropy = global_add_pool(
            -(weights * weights.clamp_min(1e-8).log()).unsqueeze(-1),
            chunk_batch,
        ).mean()
        return ModelOutput(
            logits=logits,
            node_attention=weights,
            diagnostics={
                "chunk_count_mean": torch.bincount(chunk_batch).float().mean(),
                "chunk_attention_entropy_mean": entropy,
            },
        )


def _masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
