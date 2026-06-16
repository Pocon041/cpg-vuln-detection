from __future__ import annotations

import json
from pathlib import Path

import pytest
from gensim.models import Word2Vec

from cpg_vuln.features.cache import FeatureCacheMetadata, MemmapFeatureCache
from cpg_vuln.features.normalization import NormalizationSpec
from cpg_vuln.features.text import NodeTextRegistry
from cpg_vuln.features.word2vec import build_word2vec_cache


def test_word2vec_builds_feature_cache_for_registered_texts(tmp_path: Path) -> None:
    registry = NodeTextRegistry(["copy source buffer", "<BLOCK>", "return 0"])

    cache = build_word2vec_cache(
        registry,
        tmp_path / "word2vec",
        vector_size=8,
        epochs=2,
        seed=42,
    )

    restored = MemmapFeatureCache.open(tmp_path / "word2vec" / "features", read_only=True)
    assert cache.is_complete
    assert restored.metadata.rows == 3
    assert restored.metadata.dim == 8


def test_word2vec_force_rebuilds_cache_for_changed_registry(tmp_path: Path) -> None:
    output_dir = tmp_path / "word2vec"
    build_word2vec_cache(
        NodeTextRegistry(["copy source", "return 0"]),
        output_dir,
        vector_size=8,
        epochs=2,
        seed=42,
    )

    rebuilt = build_word2vec_cache(
        NodeTextRegistry(["copy source", "return 0", "free pointer"]),
        output_dir,
        vector_size=8,
        epochs=2,
        seed=42,
        force=True,
    )

    model = Word2Vec.load(str(output_dir / "word2vec.model"))
    assert rebuilt.is_complete
    assert rebuilt.metadata.rows == 3
    assert model.corpus_count == 3


def test_word2vec_writes_pending_features_in_batches(tmp_path: Path, monkeypatch) -> None:
    written_batch_sizes: list[int] = []
    original_write = MemmapFeatureCache.write

    def tracking_write(self, indices, values):
        written_batch_sizes.append(len(indices))
        original_write(self, indices, values)

    monkeypatch.setattr(MemmapFeatureCache, "write", tracking_write)

    build_word2vec_cache(
        NodeTextRegistry(["a", "b", "c", "d", "e"]),
        tmp_path / "word2vec",
        vector_size=8,
        epochs=2,
        seed=42,
        batch_size=2,
    )

    assert written_batch_sizes == [2, 2, 1]


def test_feature_cache_rejects_same_shape_with_different_text_registry_sha(tmp_path: Path) -> None:
    root = tmp_path / "features"
    MemmapFeatureCache.create(
        root,
        rows=2,
        dim=8,
        metadata=FeatureCacheMetadata(
            rows=2,
            dim=8,
            dtype="float16",
            normalization_key="semantic-anon-v1",
            normalization_fingerprint="fingerprint-a",
            text_registry_sha256="registry-a",
            producer="word2vec",
            producer_fingerprint="producer-a",
        ),
    )

    with pytest.raises(ValueError, match="cache metadata mismatch"):
        MemmapFeatureCache.create(
            root,
            rows=2,
            dim=8,
            metadata=FeatureCacheMetadata(
                rows=2,
                dim=8,
                dtype="float16",
                normalization_key="semantic-anon-v1",
                normalization_fingerprint="fingerprint-a",
                text_registry_sha256="registry-b",
                producer="word2vec",
                producer_fingerprint="producer-a",
            ),
        )


def test_word2vec_train_only_raises_not_implemented(tmp_path: Path) -> None:
    registry = NodeTextRegistry(["copy source", "return 0"])

    with pytest.raises(NotImplementedError, match="train-only"):
        build_word2vec_cache(
            registry,
            tmp_path / "word2vec",
            vector_size=8,
            epochs=1,
            seed=7,
            normalization_spec=NormalizationSpec(mode="semantic-anon"),
            training_scope="train-only",
        )


def test_word2vec_metadata_records_transductive_scope(tmp_path: Path) -> None:
    registry = NodeTextRegistry(["copy source", "return 0"])

    cache = build_word2vec_cache(
        registry,
        tmp_path / "word2vec",
        vector_size=8,
        epochs=1,
        seed=7,
        normalization_spec=NormalizationSpec(mode="semantic-anon"),
        training_scope="transductive",
    )

    scope = json.loads((tmp_path / "word2vec" / "training_scope.json").read_text(encoding="utf-8"))
    assert cache.metadata.normalization_key == "semantic-anon-v1"
    assert scope["word2vec_training_scope"] == "transductive"
    assert "complete node-text registry" in scope["word2vec_scope_note"]
