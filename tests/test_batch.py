from __future__ import annotations

from cpg_vuln.data.batch import DynamicBatchSampler, GraphSize


def test_dynamic_batch_sampler_isolates_oversized_graph() -> None:
    sampler = DynamicBatchSampler(
        [
            GraphSize("small-a", nodes=10, edges=20),
            GraphSize("huge", nodes=9000, edges=70000),
            GraphSize("small-b", nodes=10, edges=20),
        ],
        max_nodes=8000,
        max_edges=60000,
        shuffle=False,
    )

    assert list(sampler) == [[0], [1], [2]]

