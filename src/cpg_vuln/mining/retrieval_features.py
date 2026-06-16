from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cpg_vuln.data.store import load_topology
from cpg_vuln.mining.motif import RiskMotif, extract_risk_motif


def build_risk_motifs(topology_paths: list[Path]) -> list[RiskMotif]:
    return [extract_risk_motif(load_topology(path)) for path in topology_paths]


def retrieval_vector_map(motifs: list[RiskMotif]) -> dict[str, np.ndarray]:
    return {motif.sample_id: motif.to_vector() for motif in motifs}


def write_motifs(path: Path, motifs: list[RiskMotif]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for motif in motifs:
            handle.write(
                json.dumps(
                    {
                        "sample_id": motif.sample_id,
                        "label": motif.label,
                        "features": motif.features,
                    }
                )
                + "\n"
            )
