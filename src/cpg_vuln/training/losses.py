from __future__ import annotations

import torch
import torch.nn.functional as F


def get_risk_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected logits with shape [B, 2], got {tuple(logits.shape)}")
    return logits[:, 1] - logits[:, 0]


def compute_ramp_loss(
    *,
    batch_logits: torch.Tensor,
    batch_targets: torch.Tensor,
    positive_logits: torch.Tensor,
    matched_negative_logits: torch.Tensor,
    margin: float,
    lambda_replay: float,
    lambda_rank: float,
    class_weight: tuple[float, float] | list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if batch_logits.ndim != 2 or batch_logits.shape[1] != 2:
        raise ValueError(f"Expected batch_logits [B, 2], got {tuple(batch_logits.shape)}")
    targets = batch_targets.long().view(-1)
    if batch_logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Batch mismatch: logits={tuple(batch_logits.shape)}, targets={tuple(targets.shape)}"
        )
    weight = _class_weight_tensor(class_weight, batch_logits)
    main_loss = F.cross_entropy(batch_logits, targets, weight=weight)
    if positive_logits.numel() == 0:
        replay_loss = batch_logits.new_zeros(())
        rank_loss = batch_logits.new_zeros(())
    else:
        if positive_logits.ndim != 2 or positive_logits.shape[1] != 2:
            raise ValueError(f"Expected positive_logits [P, 2], got {tuple(positive_logits.shape)}")
        if matched_negative_logits.ndim != 2 or matched_negative_logits.shape[1] != 2:
            raise ValueError(
                f"Expected matched_negative_logits [P, 2], got {tuple(matched_negative_logits.shape)}"
            )
        if positive_logits.shape[0] != matched_negative_logits.shape[0]:
            raise ValueError(
                f"Pair mismatch: positive={tuple(positive_logits.shape)}, "
                f"negative={tuple(matched_negative_logits.shape)}"
            )
        positive_targets = torch.ones(
            positive_logits.shape[0],
            dtype=torch.long,
            device=positive_logits.device,
        )
        negative_targets = torch.zeros(
            matched_negative_logits.shape[0],
            dtype=torch.long,
            device=matched_negative_logits.device,
        )
        replay_loss = 0.5 * (
            F.cross_entropy(positive_logits, positive_targets, weight=weight)
            + F.cross_entropy(matched_negative_logits, negative_targets, weight=weight)
        )
        positive_risk = get_risk_logits(positive_logits)
        negative_risk = get_risk_logits(matched_negative_logits)
        rank_loss = F.softplus(margin - positive_risk + negative_risk).mean()
    total_loss = main_loss + lambda_replay * replay_loss + lambda_rank * rank_loss
    return total_loss, {
        "loss": float(total_loss.detach()),
        "main_loss": float(main_loss.detach()),
        "replay_loss": float(replay_loss.detach()),
        "ranking_loss": float(rank_loss.detach()),
    }


def _class_weight_tensor(
    class_weight: tuple[float, float] | list[float] | None,
    logits: torch.Tensor,
) -> torch.Tensor | None:
    if class_weight is None:
        return None
    if len(class_weight) != 2:
        raise ValueError("class_weight must contain two values for labels 0 and 1")
    return torch.tensor(class_weight, dtype=logits.dtype, device=logits.device)
