from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from cpg_vuln.cli import main

from .helpers import write_graphml


def test_cli_audit_and_build_topologies_on_small_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    graphml = dataset / "graphml"
    sources = tmp_path / "sources"
    graphml.mkdir(parents=True)
    sources.mkdir()
    write_graphml(graphml / "sample_1.graphml")
    write_graphml(graphml / "11_1-checkpoint.graphml")
    (sources / "sample_1.c").write_text("int target(char *src) { return src[0]; }\n", encoding="utf-8")
    (sources / "11_1-checkpoint.c").write_text(
        "int residual(void) { return 0; }\n",
        encoding="utf-8",
    )
    with (dataset / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "base_id", "label", "graphml_path"])
        writer.writerow(["sample_1", "sample", "1", "graphml/sample_1.graphml"])
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "dataset_root": str(dataset),
                    "metadata_csv": str(dataset / "labels.csv"),
                    "source_root": str(sources),
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "outputs_dir": str(tmp_path / "outputs"),
                },
                "source_mapping": {
                    "source_map_path": str(
                        tmp_path / "artifacts" / "manifests" / "source_map.csv"
                    ),
                    "overrides_path": str(tmp_path / "missing-overrides.csv"),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="override file does not exist"):
        main(["--config", str(config), "audit"])
    main(["--config", str(config), "build-topologies", "--limit", "1"])

    assert (tmp_path / "artifacts" / "data" / "manifest.jsonl").is_file()
    source_map = tmp_path / "artifacts" / "manifests" / "source_map.csv"
    assert source_map.is_file()
    with source_map.open("r", encoding="utf-8", newline="") as handle:
        assert [row["sample_id"] for row in csv.DictReader(handle)] == ["sample_1"]
    topology_root = tmp_path / "artifacts" / "normalization" / "raw-v1" / "topologies"
    assert (topology_root / "ast" / "sample_1.pt").is_file()
    assert (topology_root / "core-cpg" / "sample_1.pt").is_file()
    assert (topology_root / "completed" / "sample_1.json").is_file()
