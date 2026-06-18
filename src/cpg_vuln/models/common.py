from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    node_attention: torch.Tensor | None = None
    diagnostics: dict[str, torch.Tensor] | None = None
    auxiliary_logits: dict[str, torch.Tensor] | None = None
    evidence_logits: torch.Tensor | None = None
