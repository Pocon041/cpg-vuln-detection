from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from cpg_vuln.models.ramp_v2 import (
    RampV2CPG,
    RampV2DualHeadCPG,
    RampV2GatedRGCNCPG,
    RampV3SliceMILCPG,
)
from cpg_vuln.models.relational import GatedResidualRGCNBlock


def _graph(label: int, *, flat_function_x: bool = False) -> Data:
    function_x = torch.randn(8) if flat_function_x else torch.randn(1, 8)
    return Data(
        x=torch.randn(4, 8),
        edge_index=torch.tensor([[0, 1, 2, 2, 3], [1, 2, 0, 3, 1]], dtype=torch.long),
        edge_type=torch.tensor([0, 1, 2, 3, 1], dtype=torch.long),
        node_type_id=torch.tensor([0, 1, 2, 1], dtype=torch.long),
        function_x=function_x,
        y=torch.tensor([label]),
        sample_id=f"sample-{label}",
        line_numbers=torch.tensor([1, 2, 3, 4], dtype=torch.long),
        slice_seed_mask=torch.tensor([False, True, True, False], dtype=torch.bool),
        slice_seed_type_id=torch.tensor([0, 3, 1, 0], dtype=torch.long),
    )


def test_ramp_v2_rgcn_returns_logits_and_attention() -> None:
    batch = Batch.from_data_list([_graph(0), _graph(1)])
    model = RampV2CPG(
        input_dim=8,
        function_dim=8,
        num_node_types=3,
        num_relations=4,
        hidden_dim=16,
        node_type_dim=4,
        layers=3,
        dropout=0.0,
        encoder="rgcn",
    )

    output = model(batch)

    assert output.logits.shape == (2, 2)
    assert output.node_attention.shape == (8,)
    assert torch.all(output.node_attention >= 0.0)
    assert torch.all(output.node_attention <= 1.0)


def test_ramp_v2_dual_head_returns_logits_and_attention() -> None:
    batch = Batch.from_data_list([_graph(0), _graph(1)])
    model = RampV2DualHeadCPG(
        input_dim=8,
        function_dim=8,
        num_node_types=3,
        num_relations=4,
        hidden_dim=16,
        node_type_dim=4,
        layers=3,
        dropout=0.0,
        encoder="rgcn",
    )

    output = model(batch)

    assert output.logits.shape == (2, 2)
    assert output.node_attention.shape == (8,)
    assert torch.all(output.node_attention >= 0.0)
    assert torch.all(output.node_attention <= 1.0)


def test_gated_rgcn_block_returns_hidden_and_gate_stats() -> None:
    block = GatedResidualRGCNBlock(
        hidden_dim=16,
        num_relations=4,
        dropout=0.0,
        gate_bias_init=-1.0,
    )
    hidden = torch.randn(5, 16)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
        dtype=torch.long,
    )
    edge_type = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)

    output, stats = block(hidden, edge_index, edge_type, return_gate_stats=True)

    assert output.shape == hidden.shape
    assert set(stats) == {"gate_mean", "gate_std"}
    assert 0.0 <= stats["gate_mean"].item() <= 1.0
    assert stats["gate_std"].item() >= 0.0


def test_ramp_v2_gated_rgcn_returns_logits_attention_and_diagnostics() -> None:
    batch = Batch.from_data_list([_graph(0), _graph(1)])
    model = RampV2GatedRGCNCPG(
        input_dim=8,
        function_dim=8,
        num_node_types=3,
        num_relations=4,
        hidden_dim=16,
        node_type_dim=4,
        layers=3,
        dropout=0.0,
        encoder="rgcn",
    )

    output = model(batch)

    assert output.logits.shape == (2, 2)
    assert output.node_attention.shape == (8,)
    assert output.diagnostics is not None
    assert "encoder_gate_mean" in output.diagnostics
    assert "fusion_gate_mean" in output.diagnostics


def test_ramp_v3_slice_mil_returns_branch_logits_and_diagnostics() -> None:
    batch = Batch.from_data_list([_graph(0), _graph(1)])
    model = RampV3SliceMILCPG(
        input_dim=8,
        function_dim=8,
        num_node_types=3,
        num_relations=4,
        hidden_dim=16,
        node_type_dim=4,
        layers=2,
        dropout=0.0,
        encoder="rgcn",
        slice_top_k=2,
    )

    output = model(batch)

    assert output.logits.shape == (2, 2)
    assert output.evidence_logits is not None
    assert output.evidence_logits.shape == (2, 2)
    assert output.auxiliary_logits is not None
    assert set(output.auxiliary_logits) == {"graph", "semantic", "slice"}
    assert output.auxiliary_logits["slice"].shape == (2, 2)
    assert output.diagnostics is not None
    assert "slice_candidate_count_mean" in output.diagnostics
    assert "fusion_weight_slice" in output.diagnostics


def test_ramp_v3_slice_branch_is_not_dominant_at_initialization() -> None:
    batch = Batch.from_data_list([_graph(0), _graph(1)])
    model = RampV3SliceMILCPG(
        input_dim=8,
        function_dim=8,
        num_node_types=3,
        num_relations=4,
        hidden_dim=16,
        node_type_dim=4,
        layers=2,
        dropout=0.0,
        encoder="rgcn",
    )

    output = model(batch)

    assert output.diagnostics is not None
    assert output.diagnostics["fusion_weight_fused"] > 0.8
    assert output.diagnostics["fusion_weight_slice"] < 0.05


def test_ramp_v3_slice_mil_falls_back_to_all_nodes_without_seed_mask() -> None:
    graphs = [_graph(0), _graph(1)]
    for graph in graphs:
        del graph.slice_seed_mask
        del graph.slice_seed_type_id
    batch = Batch.from_data_list(graphs)
    model = RampV3SliceMILCPG(
        input_dim=8,
        function_dim=8,
        num_node_types=3,
        num_relations=4,
        hidden_dim=16,
        node_type_dim=4,
        layers=2,
        dropout=0.0,
        encoder="rgcn",
        slice_top_k=2,
    )

    output = model(batch)

    assert output.evidence_logits is not None
    assert output.evidence_logits.shape == (2, 2)
    assert output.diagnostics is not None
    assert output.diagnostics["slice_candidate_count_mean"].item() == 4.0


def test_ramp_v2_rejects_flat_function_features() -> None:
    batch = Batch.from_data_list([_graph(0), _graph(1)])
    batch.function_x = batch.function_x.reshape(-1)
    model = RampV2CPG(
        input_dim=8,
        function_dim=8,
        num_node_types=3,
        num_relations=4,
        hidden_dim=16,
    )

    with pytest.raises(ValueError, match="Expected batched function_x"):
        model(batch)
