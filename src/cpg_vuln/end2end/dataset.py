from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from cpg_vuln.data.audit import ManifestRecord
from cpg_vuln.data.batch import GraphSize


class SourceTokenizer(Protocol):
    def encode(self, source: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def prepare_for_model(
        self,
        token_ids: list[int],
        *,
        add_special_tokens: bool,
        max_length: int,
        padding: str,
        truncation: bool,
        return_attention_mask: bool,
    ) -> dict[str, list[int]]: ...


@dataclass(frozen=True)
class EncodedSource:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    chunk_start: torch.Tensor


class RawCodeDataset(Sequence[Data]):
    """Pre-tokenized raw source dataset for function-level end-to-end training."""

    def __init__(
        self,
        records: list[ManifestRecord],
        sample_ids: list[str],
        *,
        tokenizer: SourceTokenizer,
        max_length: int,
        stride: int,
        max_chunks: int,
        desc: str = "tokenize raw code",
    ) -> None:
        if max_length < 4:
            raise ValueError("max_length must be at least 4")
        if max_chunks < 1:
            raise ValueError("max_chunks must be positive")
        max_content_tokens = max_length - 2
        if stride < 1 or stride > max_content_tokens:
            raise ValueError("stride must satisfy 1 <= stride <= max_length - 2")
        by_id = {record.sample_id: record for record in records}
        self._items: list[Data] = []
        self.graph_sizes: list[GraphSize] = []
        for sample_id in tqdm(sample_ids, desc=desc, unit="function", ascii=True):
            record = by_id[sample_id]
            source = Path(record.source_path).read_text(encoding="utf-8", errors="replace")
            encoded = encode_source(
                source,
                tokenizer=tokenizer,
                max_length=max_length,
                stride=stride,
                max_chunks=max_chunks,
            )
            data = self.data_from_encoded(
                sample_id=record.sample_id,
                label=record.label,
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                chunk_start=encoded.chunk_start,
            )
            self._items.append(data)
            self.graph_sizes.append(
                GraphSize(
                    record.sample_id,
                    nodes=int(encoded.input_ids.shape[0]),
                    edges=0,
                )
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Data:
        return self._items[index]

    @staticmethod
    def data_from_encoded(
        *,
        sample_id: str,
        label: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_start: torch.Tensor,
    ) -> Data:
        chunk_count = int(input_ids.shape[0])
        return Data(
            x=torch.zeros((chunk_count, 1), dtype=torch.float32),
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
            chunk_start=chunk_start.long(),
            line_numbers=chunk_start.long(),
            y=torch.tensor([int(label)], dtype=torch.long),
            sample_id=sample_id,
        )


def encode_source(
    source: str,
    *,
    tokenizer: SourceTokenizer,
    max_length: int,
    stride: int,
    max_chunks: int,
) -> EncodedSource:
    token_ids = tokenizer.encode(source, add_special_tokens=False)
    max_content_tokens = max_length - 2
    windows = _token_windows(token_ids, max_tokens=max_content_tokens, stride=stride)
    windows = _head_tail_windows(windows, max_chunks=max_chunks)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    starts: list[int] = []
    for start, window in windows:
        prepared = tokenizer.prepare_for_model(
            window,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        input_ids.append(prepared["input_ids"])
        attention_masks.append(prepared["attention_mask"])
        starts.append(start)
    return EncodedSource(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_masks, dtype=torch.long),
        chunk_start=torch.tensor(starts, dtype=torch.long),
    )


def _token_windows(
    token_ids: list[int],
    *,
    max_tokens: int,
    stride: int,
) -> list[tuple[int, list[int]]]:
    if not token_ids:
        return [(0, [])]
    windows: list[tuple[int, list[int]]] = []
    start = 0
    while start < len(token_ids):
        windows.append((start, token_ids[start : start + max_tokens]))
        if start + max_tokens >= len(token_ids):
            break
        start += stride
    return windows


def _head_tail_windows(
    windows: list[tuple[int, list[int]]],
    *,
    max_chunks: int,
) -> list[tuple[int, list[int]]]:
    if len(windows) <= max_chunks:
        return windows
    front_count = (max_chunks + 1) // 2
    tail_count = max_chunks - front_count
    selected = list(windows[:front_count])
    if tail_count:
        selected.extend(windows[-tail_count:])
    deduped: dict[int, list[int]] = {}
    for start, window in selected:
        deduped[start] = window
    return [(start, deduped[start]) for start in sorted(deduped)]
