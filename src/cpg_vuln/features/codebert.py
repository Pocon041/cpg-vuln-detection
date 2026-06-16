from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
from tqdm import tqdm

from cpg_vuln.data.audit import ManifestRecord
from cpg_vuln.data.graphml import GraphMLParser, choose_primary_method
from cpg_vuln.features.cache import FeatureCacheMetadata, MemmapFeatureCache
from cpg_vuln.features.normalization import (
    NormalizationSpec,
    build_scope_context,
    normalize_source_text,
    sha256_json,
)
from cpg_vuln.features.text import NodeTextRegistry


class TextEncoder(Protocol):
    dim: int

    def encode_texts(
        self, texts: list[str], *, max_length: int, batch_size: int
    ) -> np.ndarray: ...


class CodeBertEncoder:
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        *,
        device: str | None = None,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode_texts(
        self, texts: list[str], *, max_length: int, batch_size: int
    ) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**encoded).last_hidden_state
            pooled = masked_mean_pool(hidden, encoded["attention_mask"])
            batches.append(pooled.cpu().numpy())
        return np.concatenate(batches) if batches else np.empty((0, self.dim), dtype=np.float32)

    @torch.inference_mode()
    def encode_function(
        self,
        source: str,
        *,
        max_content_tokens: int = 510,
        overlap: int = 256,
        batch_size: int = 8,
    ) -> np.ndarray:
        token_ids = self.tokenizer.encode(source, add_special_tokens=False)
        windows = token_windows(token_ids, max_tokens=max_content_tokens, overlap=overlap)
        if not windows:
            windows = [[]]
        vectors: list[np.ndarray] = []
        for start in range(0, len(windows), batch_size):
            prepared = [
                self.tokenizer.prepare_for_model(
                    window,
                    add_special_tokens=True,
                    max_length=max_content_tokens + 2,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                )
                for window in windows[start : start + batch_size]
            ]
            input_ids = torch.tensor([item["input_ids"] for item in prepared], device=self.device)
            mask = torch.tensor([item["attention_mask"] for item in prepared], device=self.device)
            hidden = self.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
            vectors.append(masked_mean_pool(hidden, mask).cpu().numpy())
        return np.concatenate(vectors).mean(axis=0, dtype=np.float32)


def masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def token_windows(token_ids: list[int], *, max_tokens: int, overlap: int) -> list[list[int]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("overlap must satisfy 0 <= overlap < max_tokens")
    windows: list[list[int]] = []
    step = max_tokens - overlap
    start = 0
    while start < len(token_ids):
        window = token_ids[start : start + max_tokens]
        windows.append(window)
        if start + max_tokens >= len(token_ids):
            break
        start += step
    return windows


def build_node_codebert_cache(
    registry: NodeTextRegistry,
    output_dir: Path,
    *,
    encoder: TextEncoder | None = None,
    model_name: str = "microsoft/codebert-base",
    max_length: int = 64,
    batch_size: int = 64,
    normalization_spec: NormalizationSpec | None = None,
) -> MemmapFeatureCache:
    encoder = encoder or CodeBertEncoder(model_name)
    normalization_spec = normalization_spec or NormalizationSpec(mode="raw")
    cache = MemmapFeatureCache.create(
        output_dir,
        rows=len(registry),
        dim=encoder.dim,
        metadata=FeatureCacheMetadata(
            rows=len(registry),
            dim=encoder.dim,
            dtype="float16",
            normalization_key=normalization_spec.normalization_key,
            normalization_fingerprint=normalization_spec.fingerprint,
            text_registry_sha256=registry.sha256(),
            producer="node-codebert",
            producer_fingerprint=sha256_json(
                {
                    "producer": "node-codebert",
                    "model_name": model_name,
                    "max_length": max_length,
                }
            ),
        ),
    )
    pending = cache.pending_indices()
    for start in tqdm(
        range(0, len(pending), batch_size),
        desc="build CodeBERT node features",
        unit="batch",
        ascii=True,
    ):
        indices = pending[start : start + batch_size]
        values = encoder.encode_texts(
            [registry.values[index] for index in indices],
            max_length=max_length,
            batch_size=batch_size,
        )
        cache.write(indices, values)
    return cache


def build_function_codebert_cache(
    records: list[ManifestRecord],
    output_dir: Path,
    *,
    encoder: CodeBertEncoder | None = None,
    model_name: str = "microsoft/codebert-base",
    max_content_tokens: int = 510,
    overlap: int = 256,
    batch_size: int = 8,
    normalization_spec: NormalizationSpec | None = None,
    source_transform: Callable[[str, ManifestRecord], str] | None = None,
) -> tuple[MemmapFeatureCache, dict[str, int]]:
    encoder = encoder or CodeBertEncoder(model_name)
    normalization_spec = normalization_spec or NormalizationSpec(mode="raw")
    source_normalization = (
        "raw"
        if normalization_spec.mode == "raw"
        else normalization_spec.normalization_key
    )
    normalization_fingerprint = (
        "function-source-raw"
        if normalization_spec.mode == "raw"
        else normalization_spec.fingerprint
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    function_producer_payload = {
        "producer": "function-codebert",
        "model_name": model_name,
        "max_content_tokens": max_content_tokens,
        "overlap": overlap,
        "function_source_normalization": source_normalization,
        "normalization_fingerprint": normalization_fingerprint,
    }
    producer_fingerprint = sha256_json(function_producer_payload)
    (output_dir / "source_normalization.json").write_text(
        json.dumps(
            {
                "function_source_normalization": source_normalization,
                "normalization_fingerprint": normalization_fingerprint,
                "producer_fingerprint": producer_fingerprint,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    sample_ids = [record.sample_id for record in records]
    indices = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    (output_dir / "function_indices.json").write_text(
        json.dumps(indices, indent=2) + "\n",
        encoding="utf-8",
    )
    cache = MemmapFeatureCache.create(
        output_dir / "features",
        rows=len(records),
        dim=encoder.dim,
        metadata=FeatureCacheMetadata(
            rows=len(records),
            dim=encoder.dim,
            dtype="float16",
            normalization_key=f"function-source-{source_normalization}",
            normalization_fingerprint=normalization_fingerprint,
            text_registry_sha256="",
            producer="function-codebert",
            producer_fingerprint=producer_fingerprint,
        ),
    )
    by_id = {record.sample_id: record for record in records}
    for index in tqdm(
        cache.pending_indices(),
        desc="build CodeBERT function features",
        unit="function",
        ascii=True,
    ):
        sample_id = sample_ids[index]
        source = Path(by_id[sample_id].source_path).read_text(encoding="utf-8", errors="replace")
        source = transform_function_source(
            source,
            by_id[sample_id],
            normalization_spec=normalization_spec,
            source_transform=source_transform,
        )
        vector = encoder.encode_function(
            source,
            max_content_tokens=max_content_tokens,
            overlap=overlap,
            batch_size=batch_size,
        )
        cache.write([index], vector[None, :])
    return cache, indices


def transform_function_source(
    source: str,
    record: ManifestRecord,
    *,
    normalization_spec: NormalizationSpec,
    source_transform: Callable[[str, ManifestRecord], str] | None = None,
) -> str:
    if source_transform is not None:
        return source_transform(source, record)
    if normalization_spec.mode == "raw":
        return source
    graph = GraphMLParser().parse(Path(record.graphml_path))
    root = choose_primary_method(graph)
    scope = build_scope_context(graph, root, normalization_spec)
    return normalize_source_text(source, scope, normalization_spec)
