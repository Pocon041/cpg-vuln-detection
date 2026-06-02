from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cpg_vuln.features.cache import MemmapFeatureCache
from cpg_vuln.features.codebert import (
    build_node_codebert_cache,
    masked_mean_pool,
    token_windows,
)
from cpg_vuln.features.text import NodeTextRegistry


class FakeEncoder:
    dim = 3

    def encode_texts(self, texts: list[str], *, max_length: int, batch_size: int) -> np.ndarray:
        return np.asarray([[len(text), max_length, batch_size] for text in texts], dtype=np.float32)


def test_masked_mean_pool_ignores_padding() -> None:
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])

    assert masked_mean_pool(hidden, mask).tolist() == [[2.0, 4.0]]


def test_token_windows_use_overlap_without_dropping_tail() -> None:
    windows = token_windows(list(range(11)), max_tokens=5, overlap=2)

    assert windows == [list(range(5)), list(range(3, 8)), list(range(6, 11))]


def test_node_codebert_cache_resumes_only_pending_rows(tmp_path: Path) -> None:
    registry = NodeTextRegistry(["a", "bb", "ccc"])
    cache_root = tmp_path / "codebert"
    partial = MemmapFeatureCache.create(cache_root, rows=3, dim=3)
    partial.write([0], np.asarray([[10, 10, 10]], dtype=np.float32))

    cache = build_node_codebert_cache(
        registry,
        cache_root,
        encoder=FakeEncoder(),
        max_length=64,
        batch_size=2,
    )

    assert cache.is_complete
    assert cache.read([0]).tolist() == [[10.0, 10.0, 10.0]]
    assert cache.read([1, 2]).tolist() == [[2.0, 64.0, 2.0], [3.0, 64.0, 2.0]]

