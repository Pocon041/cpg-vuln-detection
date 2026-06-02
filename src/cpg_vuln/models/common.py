from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    node_attention: torch.Tensor | None = None

