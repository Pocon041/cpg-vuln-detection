from __future__ import annotations

from pathlib import Path

import numpy as np

from cpg_vuln.features.cache import MemmapFeatureCache


def test_memmap_cache_can_resume_partial_writes(tmp_path: Path) -> None:
    cache = MemmapFeatureCache.create(tmp_path / "cache", rows=3, dim=2, dtype="float16")
    cache.write([1], np.asarray([[1.5, 2.5]], dtype=np.float32))

    resumed = MemmapFeatureCache.open(tmp_path / "cache")

    assert resumed.completed.tolist() == [False, True, False]
    assert resumed.pending_indices() == [0, 2]
    np.testing.assert_allclose(resumed.read([1]), [[1.5, 2.5]])

