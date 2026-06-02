from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from cpg_vuln.models.gcn import GCNClassifier
from cpg_vuln.models.selective_fusion import SelectiveFusionCPG


def _graph(label: int) -> Data:
    return Data(
        x=torch.randn(3, 8),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_type=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        node_type_id=torch.tensor([0, 1, 2], dtype=torch.long),
        function_x=torch.randn(1, 8),
        y=torch.tensor([label]),
        sample_id=f"sample-{label}",
    )


def test_gcn_classifier_returns_graph_logits() -> None:
    batch = Batch.from_data_list([_graph(0), _graph(1)])
    model = GCNClassifier(input_dim=8, num_node_types=3, hidden_dim=16)

    output = model(batch)

    assert output.logits.shape == (2, 2)
    assert output.node_attention is None


def test_selective_fusion_returns_logits_and_node_attention() -> None:
    batch = Batch.from_data_list([_graph(0), _graph(1)])
    model = SelectiveFusionCPG(
        input_dim=8,
        function_dim=8,
        num_node_types=3,
        num_relations=4,
        hidden_dim=16,
    )

    output = model(batch)

    assert output.logits.shape == (2, 2)
    assert output.node_attention.shape == (6,)

