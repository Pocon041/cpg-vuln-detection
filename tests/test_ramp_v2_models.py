from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from cpg_vuln.models.ramp_v2 import RampV2CPG, RampV2DualHeadCPG


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
