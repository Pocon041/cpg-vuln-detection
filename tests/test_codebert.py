from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cpg_vuln.data.audit import ManifestRecord
from cpg_vuln.features.cache import FeatureCacheMetadata, MemmapFeatureCache
from cpg_vuln.features.codebert import (
    build_function_codebert_cache,
    build_node_codebert_cache,
    masked_mean_pool,
    token_windows,
)
from cpg_vuln.features.normalization import NormalizationSpec, sha256_json
from cpg_vuln.features.text import NodeTextRegistry


class FakeEncoder:
    dim = 3

    def encode_texts(self, texts: list[str], *, max_length: int, batch_size: int) -> np.ndarray:
        return np.asarray([[len(text), max_length, batch_size] for text in texts], dtype=np.float32)

    def encode_function(
        self,
        source: str,
        *,
        max_content_tokens: int = 510,
        overlap: int = 256,
        batch_size: int = 8,
    ) -> np.ndarray:
        return np.asarray([len(source), max_content_tokens, overlap], dtype=np.float32)


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
    spec = NormalizationSpec(mode="raw")
    partial = MemmapFeatureCache.create(
        cache_root,
        rows=3,
        dim=3,
        metadata=FeatureCacheMetadata(
            rows=3,
            dim=3,
            dtype="float16",
            normalization_key=spec.normalization_key,
            normalization_fingerprint=spec.fingerprint,
            text_registry_sha256=registry.sha256(),
            producer="node-codebert",
            producer_fingerprint=sha256_json(
                {"producer": "node-codebert", "model_name": "microsoft/codebert-base", "max_length": 64}
            ),
        ),
    )
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


def test_node_codebert_cache_records_normalization_metadata(tmp_path: Path) -> None:
    registry = NodeTextRegistry(["a", "bb"])
    spec = NormalizationSpec(mode="semantic-anon")

    cache = build_node_codebert_cache(
        registry,
        tmp_path / "codebert",
        encoder=FakeEncoder(),
        max_length=64,
        batch_size=2,
        normalization_spec=spec,
    )

    assert cache.metadata.normalization_key == "semantic-anon-v1"
    assert cache.metadata.normalization_fingerprint == spec.fingerprint
    assert cache.metadata.text_registry_sha256 == registry.sha256()
    assert cache.metadata.producer == "node-codebert"


def test_function_codebert_cache_records_raw_source_metadata(tmp_path: Path) -> None:
    source = tmp_path / "sample.c"
    source.write_text("int f(void) { return 0; }\n", encoding="utf-8")
    record = ManifestRecord("sample_1", 1, str(tmp_path / "sample.graphml"), str(source), "hash")

    cache, indices = build_function_codebert_cache(
        [record],
        tmp_path / "functions",
        encoder=FakeEncoder(),
        model_name="fake-codebert",
        max_content_tokens=32,
        overlap=8,
    )

    metadata = json.loads((tmp_path / "functions" / "source_normalization.json").read_text(encoding="utf-8"))
    assert cache.metadata.normalization_key == "function-source-raw"
    assert cache.metadata.producer == "function-codebert"
    assert metadata["function_source_normalization"] == "raw"
    assert indices == {"sample_1": 0}


def test_function_codebert_cache_can_record_normalized_source_metadata(tmp_path: Path) -> None:
    source = tmp_path / "sample.c"
    source.write_text("int f(void) { return secret_name; }\n", encoding="utf-8")
    record = ManifestRecord("sample_1", 1, str(tmp_path / "sample.graphml"), str(source), "hash")
    spec = NormalizationSpec(mode="full-anon")

    cache, indices = build_function_codebert_cache(
        [record],
        tmp_path / "functions",
        encoder=FakeEncoder(),
        model_name="fake-codebert",
        max_content_tokens=32,
        overlap=8,
        normalization_spec=spec,
        source_transform=lambda source_text, _record: "int FUNC_1 ( void ) { return VAR_1 ; }",
    )

    metadata = json.loads((tmp_path / "functions" / "source_normalization.json").read_text(encoding="utf-8"))
    assert cache.metadata.normalization_key == "function-source-full-anon-v1"
    assert cache.metadata.normalization_fingerprint == spec.fingerprint
    assert metadata["function_source_normalization"] == "full-anon-v1"
    assert cache.read([0]).tolist() == [[38.0, 32.0, 8.0]]
    assert indices == {"sample_1": 0}
