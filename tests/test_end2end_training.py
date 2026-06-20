from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch_geometric.nn import global_add_pool

from cpg_vuln.config import load_config
from cpg_vuln.end2end import training as end2end_training
from cpg_vuln.models.common import ModelOutput
from tests.test_end2end_raw_code import FakeTokenizer


class TinyEnd2EndModel(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 2)
        self.kwargs = kwargs

    def forward(self, data) -> ModelOutput:
        chunk_values = data.input_ids.float().sum(dim=1, keepdim=True)
        graph_values = global_add_pool(chunk_values, data.batch)
        return ModelOutput(
            logits=self.linear(graph_values),
            node_attention=None,
            diagnostics={
                "chunk_count_mean": torch.bincount(data.batch).float().mean(),
            },
        )


def test_train_end2end_uses_raw_sources_and_writes_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifacts = tmp_path / "artifacts"
    outputs = tmp_path / "outputs"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    records = []
    split = {"train": [], "val": [], "test": []}
    for index in range(6):
        sample_id = f"sample-{index}"
        source_path = source_root / f"{sample_id}.c"
        source_path.write_text(f"{index} {index + 1} {index + 2}", encoding="utf-8")
        records.append(
            {
                "sample_id": sample_id,
                "label": index % 2,
                "graphml_path": str(tmp_path / f"{sample_id}.graphml"),
                "source_path": str(source_path),
                "source_hash": f"hash-{index}",
            }
        )
    split["train"] = ["sample-0", "sample-1"]
    split["val"] = ["sample-2", "sample-3"]
    split["test"] = ["sample-4", "sample-5"]
    data_dir = artifacts / "data"
    split_dir = data_dir / "splits"
    split_dir.mkdir(parents=True)
    with (data_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (split_dir / "strict.json").write_text(json.dumps(split), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  artifacts_dir: {artifacts.as_posix()}
  outputs_dir: {outputs.as_posix()}
training:
  epochs: 1
  patience: 1
  weight_decay: 0.0
  gradient_clip: 1.0
  max_nodes: 8
  max_edges: 0
  seed: 7
  device: cpu
end2end:
  model_name: fake-codebert
  max_length: 8
  stride: 4
  max_chunks: 2
  max_batch_chunks: 4
  learning_rate: 0.001
  dropout: 0.0
  positive_class_weight: 1.65
  checkpoint_min_ppr: 0.25
  checkpoint_max_ppr: 0.75
  checkpoint_max_recall: 0.90
""",
        encoding="utf-8",
    )
    config = load_config(config_path)

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda model_name, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(end2end_training, "RawCodeMILTransformer", TinyEnd2EndModel)

    end2end_training.train_end2end(
        config,
        split="strict",
        run_name="end2end-test",
        checkpoint_metric="f1",
        threshold_strategy="fixed_0_5",
        evaluate_test=True,
        epochs=1,
        force=True,
    )

    run_dir = outputs / "runs" / "raw-v1" / "end2end-test"
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "predictions.csv").is_file()
    assert metrics["run_metadata"]["uses_intermediate_representation"] is False
    assert metrics["run_metadata"]["representation"] == "raw-source"
    assert metrics["config"]["class_weight"] == [1.0, 1.65]
    assert metrics["config"]["checkpoint_min_ppr"] == 0.25
    assert metrics["config"]["checkpoint_max_ppr"] == 0.75
    assert metrics["config"]["checkpoint_max_recall"] == 0.90
    assert metrics["test_fixed_0_5"]["samples"] == 2
