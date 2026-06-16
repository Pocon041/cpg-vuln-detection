from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

from cpg_vuln.features.cache import FeatureCacheMetadata, MemmapFeatureCache
from cpg_vuln.features.normalization import NormalizationSpec, sha256_json
from cpg_vuln.features.text import NodeTextRegistry, tokenize_c


def build_word2vec_cache(
    registry: NodeTextRegistry,
    output_dir: Path,
    *,
    vector_size: int = 128,
    epochs: int = 10,
    seed: int = 42,
    batch_size: int = 1024,
    force: bool = False,
    normalization_spec: NormalizationSpec | None = None,
    training_scope: str = "transductive",
) -> MemmapFeatureCache:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    validate_word2vec_training_scope(training_scope)
    normalization_spec = normalization_spec or NormalizationSpec(mode="raw")
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_scope.json").write_text(
        json.dumps(
            {
                "word2vec_training_scope": training_scope,
                "word2vec_scope_note": "Vocabulary and vectors were trained from the complete node-text registry.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    model_path = output_dir / "word2vec.model"
    sentences = [
        _tokens_or_empty(text)
        for text in tqdm(
            registry.values,
            desc="tokenize Word2Vec texts",
            unit="text",
            ascii=True,
        )
    ]
    if model_path.is_file():
        print(f"loading existing Word2Vec model from {model_path}")
        model = Word2Vec.load(str(model_path))
        if model.vector_size != vector_size:
            raise ValueError("existing Word2Vec model dimension does not match config")
    else:
        print(f"training Word2Vec model on {len(sentences)} unique node texts")
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            sg=1,
            min_count=1,
            workers=1,
            epochs=epochs,
            seed=seed,
        )
        model.save(str(model_path))
    cache = MemmapFeatureCache.create(
        output_dir / "features",
        rows=len(registry),
        dim=vector_size,
        metadata=FeatureCacheMetadata(
            rows=len(registry),
            dim=vector_size,
            dtype="float16",
            normalization_key=normalization_spec.normalization_key,
            normalization_fingerprint=normalization_spec.fingerprint,
            text_registry_sha256=registry.sha256(),
            producer="word2vec",
            producer_fingerprint=sha256_json(
                {
                    "producer": "word2vec",
                    "vector_size": vector_size,
                    "epochs": epochs,
                    "seed": seed,
                    "tokenizer_version": normalization_spec.tokenizer_version,
                    "training_scope": training_scope,
                }
            ),
        ),
    )
    pending = cache.pending_indices()
    for start in tqdm(
        range(0, len(pending), batch_size),
        desc="build Word2Vec features",
        unit="batch",
        ascii=True,
    ):
        indices = pending[start : start + batch_size]
        values = np.stack([_average_tokens(model, sentences[index]) for index in indices])
        cache.write(indices, values)
    return cache


def validate_word2vec_training_scope(training_scope: str) -> None:
    if training_scope == "transductive":
        return
    if training_scope == "train-only":
        raise NotImplementedError(
            "Word2Vec train-only mode requires train-split registry filtering "
            "and an explicit OOV strategy; it is not implemented in v1."
        )
    raise ValueError(f"unsupported Word2Vec training scope: {training_scope}")


def _tokens_or_empty(text: str) -> list[str]:
    return tokenize_c(text) or ["<EMPTY>"]


def _average_tokens(model: Word2Vec, tokens: list[str]) -> np.ndarray:
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0, dtype=np.float32)
