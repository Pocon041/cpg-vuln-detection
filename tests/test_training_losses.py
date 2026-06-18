from __future__ import annotations

import pytest
import torch

from cpg_vuln.training.losses import compute_ramp_loss, get_risk_logits


def test_get_risk_logits_returns_positive_minus_negative_logit() -> None:
    logits = torch.tensor([[1.0, 4.0], [3.0, 2.0]], dtype=torch.float32)

    risk = get_risk_logits(logits)

    assert torch.allclose(risk, torch.tensor([3.0, -1.0]))


def test_get_risk_logits_rejects_non_two_class_logits() -> None:
    logits = torch.zeros((2, 3), dtype=torch.float32)

    try:
        get_risk_logits(logits)
    except ValueError as error:
        assert "Expected logits with shape [B, 2]" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_ramp_loss_is_cross_entropy_when_no_pairs_exist() -> None:
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    empty = torch.empty((0, 2), dtype=torch.float32)

    loss, parts = compute_ramp_loss(
        batch_logits=logits,
        batch_targets=targets,
        positive_logits=empty,
        matched_negative_logits=empty,
        margin=0.5,
        lambda_replay=0.5,
        lambda_rank=0.25,
    )

    expected = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(loss, expected)
    assert parts["replay_loss"] == 0.0
    assert parts["ranking_loss"] == 0.0


def test_ramp_loss_accepts_class_weight_for_main_classification_loss() -> None:
    logits = torch.tensor([[2.0, -1.0], [2.0, -1.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    empty = torch.empty((0, 2), dtype=torch.float32)

    loss, parts = compute_ramp_loss(
        batch_logits=logits,
        batch_targets=targets,
        positive_logits=empty,
        matched_negative_logits=empty,
        margin=0.5,
        lambda_replay=0.5,
        lambda_rank=0.25,
        class_weight=(1.0, 3.0),
    )

    expected = torch.nn.functional.cross_entropy(
        logits,
        targets,
        weight=torch.tensor([1.0, 3.0]),
    )
    assert torch.allclose(loss, expected)
    assert parts["main_loss"] == float(expected)


def test_ramp_loss_adds_auxiliary_branch_classification_loss() -> None:
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    auxiliary_logits = {
        "graph": torch.tensor([[-1.0, 2.0], [2.0, -1.0]], dtype=torch.float32),
        "slice": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
    }
    empty = torch.empty((0, 2), dtype=torch.float32)

    loss, parts = compute_ramp_loss(
        batch_logits=logits,
        batch_targets=targets,
        positive_logits=empty,
        matched_negative_logits=empty,
        margin=0.5,
        lambda_replay=0.5,
        lambda_rank=0.25,
        auxiliary_logits=auxiliary_logits,
        lambda_auxiliary=0.2,
    )

    main = torch.nn.functional.cross_entropy(logits, targets)
    auxiliary = 0.5 * sum(
        torch.nn.functional.cross_entropy(branch, targets)
        for branch in auxiliary_logits.values()
    )
    assert torch.allclose(loss, main + 0.2 * auxiliary)
    assert parts["auxiliary_loss"] == float(auxiliary)
    assert parts["weighted_auxiliary_loss"] == pytest.approx(float(0.2 * auxiliary))
    assert parts["pre_rank_loss"] == pytest.approx(float(main + 0.2 * auxiliary))


def test_ramp_loss_replay_term_changes_loss_when_ranking_is_disabled() -> None:
    batch_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([1, 0], dtype=torch.long)
    positive_logits = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    negative_logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    loss, parts = compute_ramp_loss(
        batch_logits=batch_logits,
        batch_targets=targets,
        positive_logits=positive_logits,
        matched_negative_logits=negative_logits,
        margin=0.5,
        lambda_replay=0.5,
        lambda_rank=0.0,
    )

    assert parts["replay_loss"] > 1.0
    assert parts["ranking_loss"] > 0.0
    assert loss > parts["main_loss"]


def test_ramp_loss_penalizes_negative_with_higher_risk_logit() -> None:
    batch_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([1, 0], dtype=torch.long)
    positive_logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    negative_logits = torch.tensor([[-1.0, 3.0]], dtype=torch.float32)

    loss, parts = compute_ramp_loss(
        batch_logits=batch_logits,
        batch_targets=targets,
        positive_logits=positive_logits,
        matched_negative_logits=negative_logits,
        margin=0.5,
        lambda_replay=0.5,
        lambda_rank=0.25,
    )

    assert parts["ranking_loss"] > 0.5
    assert loss > parts["main_loss"]


def test_ramp_loss_can_rank_evidence_logits_instead_of_final_logits() -> None:
    batch_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([1, 0], dtype=torch.long)
    positive_logits = torch.tensor([[0.0, 3.0]], dtype=torch.float32)
    negative_logits = torch.tensor([[3.0, 0.0]], dtype=torch.float32)
    positive_evidence = torch.tensor([[3.0, 0.0]], dtype=torch.float32)
    negative_evidence = torch.tensor([[0.0, 3.0]], dtype=torch.float32)

    loss, parts = compute_ramp_loss(
        batch_logits=batch_logits,
        batch_targets=targets,
        positive_logits=positive_logits,
        matched_negative_logits=negative_logits,
        positive_evidence_logits=positive_evidence,
        matched_negative_evidence_logits=negative_evidence,
        margin=0.5,
        lambda_replay=0.0,
        lambda_rank=0.25,
    )

    assert parts["ranking_loss"] > 6.0
    assert loss > parts["main_loss"]
