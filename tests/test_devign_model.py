from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from cpg_vuln.models.devign import DevignCPG


def _graph(label: int, *, nodes: int = 4) -> Data:
    edges = []
    for node in range(max(nodes - 1, 0)):
        edges.append((node, node + 1))
        edges.append((node + 1, node))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(
        x=torch.randn(nodes, 8),
        edge_index=edge_index,
        edge_type=torch.arange(edge_index.shape[1], dtype=torch.long) % 4,
        node_type_id=torch.arange(nodes, dtype=torch.long) % 3,
        function_x=torch.randn(1, 8),
        y=torch.tensor([label]),
        sample_id=f"sample-{label}",
        line_numbers=torch.arange(1, nodes + 1, dtype=torch.long),
    )


def test_devign_cpg_returns_graph_logits_without_attention() -> None:
    batch = Batch.from_data_list([_graph(0), _graph(1)])
    model = DevignCPG(
        input_dim=8,
        num_node_types=3,
        hidden_dim=16,
        node_type_dim=4,
        steps=2,
        dropout=0.0,
    )

    output = model(batch)

    assert output.logits.shape == (2, 2)
    assert torch.isfinite(output.logits).all()
    assert output.node_attention is None


def test_devign_cpg_uses_fixed_node_slots_for_conv_readout() -> None:
    batch = Batch.from_data_list([_graph(0, nodes=3), _graph(1, nodes=7)])
    model = DevignCPG(
        input_dim=8,
        num_node_types=3,
        hidden_dim=16,
        node_type_dim=4,
        steps=2,
        dropout=0.0,
        max_nodes=5,
    )

    output = model(batch)

    assert output.logits.shape == (2, 2)
    assert torch.isfinite(output.logits).all()
    assert model.max_nodes == 5
