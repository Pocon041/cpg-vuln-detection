from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cpg_vuln.features.normalization import NormalizationSpec


@dataclass(frozen=True)
class ArtifactLayout:
    artifacts_root: Path
    outputs_root: Path
    spec: NormalizationSpec

    @property
    def normalization_root(self) -> Path:
        return self.artifacts_root / "normalization" / self.spec.normalization_key

    @property
    def topology_dir(self) -> Path:
        return self.normalization_root / "topologies"

    @property
    def word2vec_dir(self) -> Path:
        return self.normalization_root / "features" / "word2vec"

    @property
    def node_codebert_dir(self) -> Path:
        return self.normalization_root / "features" / "codebert" / "nodes"

    @property
    def function_codebert_dir(self) -> Path:
        if self.spec.mode == "raw":
            return self.artifacts_root / "features" / "codebert" / "functions-raw"
        return self.normalization_root / "features" / "codebert" / "functions"

    @property
    def retrieval_dir(self) -> Path:
        return self.normalization_root / "retrieval"

    @property
    def run_root(self) -> Path:
        return self.outputs_root / "runs" / self.spec.normalization_key

    @property
    def report_root(self) -> Path:
        return self.outputs_root / "reports" / self.spec.normalization_key

    @property
    def explanation_root(self) -> Path:
        return self.outputs_root / "reports" / "explanations" / self.spec.normalization_key
