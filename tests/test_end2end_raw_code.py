from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch_geometric.data import Batch

from cpg_vuln.data.audit import ManifestRecord
from cpg_vuln.end2end.dataset import RawCodeDataset
from cpg_vuln.end2end.model import RawCodeMILTransformer


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def encode(self, source: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [int(token) for token in source.split()]

    def prepare_for_model(
        self,
        token_ids: list[int],
        *,
        add_special_tokens: bool,
        max_length: int,
        padding: str,
        truncation: bool,
        return_attention_mask: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is True
        assert padding == "max_length"
        assert truncation is True
        assert return_attention_mask is True
        values = [self.cls_token_id, *token_ids[: max_length - 2], self.sep_token_id]
        values = values[:max_length]
        mask = [1] * len(values)
        while len(values) < max_length:
            values.append(self.pad_token_id)
            mask.append(0)
        return {"input_ids": values, "attention_mask": mask}


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 6) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(256, hidden_size)

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_raw_code_dataset_tokenizes_long_sources_as_head_tail_chunks(tmp_path: Path) -> None:
    source = tmp_path / "sample.c"
    source.write_text(" ".join(str(index) for index in range(20)), encoding="utf-8")
    record = ManifestRecord(
        sample_id="sample",
        label=1,
        graphml_path=str(tmp_path / "sample.graphml"),
        source_path=str(source),
        source_hash="hash",
    )

    dataset = RawCodeDataset(
        [record],
        ["sample"],
        tokenizer=FakeTokenizer(),
        max_length=6,
        stride=2,
        max_chunks=3,
        desc="test tokenize",
    )

    item = dataset[0]

    assert item.sample_id == "sample"
    assert item.y.tolist() == [1]
    assert item.num_nodes == 3
    assert item.input_ids.shape == (3, 6)
    assert item.attention_mask.shape == (3, 6)
    assert item.input_ids[:, 0].tolist() == [101, 101, 101]
    assert item.input_ids[:, -1].tolist() == [102, 102, 102]
    assert item.chunk_start.tolist() == [0, 2, 16]
    assert dataset.graph_sizes[0].nodes == 3
    assert dataset.graph_sizes[0].edges == 0


def test_raw_code_dataset_respects_single_chunk_limit(tmp_path: Path) -> None:
    source = tmp_path / "sample.c"
    source.write_text(" ".join(str(index) for index in range(20)), encoding="utf-8")
    record = ManifestRecord(
        sample_id="sample",
        label=0,
        graphml_path=str(tmp_path / "sample.graphml"),
        source_path=str(source),
        source_hash="hash",
    )

    dataset = RawCodeDataset(
        [record],
        ["sample"],
        tokenizer=FakeTokenizer(),
        max_length=6,
        stride=2,
        max_chunks=1,
        desc="test tokenize",
    )

    item = dataset[0]

    assert item.input_ids.shape == (1, 6)
    assert item.chunk_start.tolist() == [0]


def test_raw_code_mil_transformer_aggregates_chunks_per_function() -> None:
    samples = []
    for index in range(2):
        samples.append(
            RawCodeDataset.data_from_encoded(
                sample_id=f"sample-{index}",
                label=index,
                input_ids=torch.tensor(
                    [
                        [101, 10 + index, 102, 0],
                        [101, 20 + index, 102, 0],
                    ],
                    dtype=torch.long,
                ),
                attention_mask=torch.tensor(
                    [[1, 1, 1, 0], [1, 1, 1, 0]],
                    dtype=torch.long,
                ),
                chunk_start=torch.tensor([0, 2], dtype=torch.long),
            )
        )
    batch = Batch.from_data_list(samples)
    model = RawCodeMILTransformer(
        encoder=TinyEncoder(hidden_size=6),
        dropout=0.0,
    )

    output = model(batch)

    assert output.logits.shape == (2, 2)
    assert output.node_attention is None
    assert output.diagnostics is not None
    assert output.diagnostics["chunk_count_mean"].item() == 2.0
    assert output.diagnostics["chunk_attention_entropy_mean"].item() > 0.0
