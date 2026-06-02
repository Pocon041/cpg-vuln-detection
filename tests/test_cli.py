from __future__ import annotations

import csv
from pathlib import Path

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
    (sources / "sample_1.c").write_text("int target(char *src) { return src[0]; }\n", encoding="utf-8")
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
                }
            }
        ),
        encoding="utf-8",
    )

    main(["--config", str(config), "audit"])
    main(["--config", str(config), "build-topologies", "--limit", "1"])

    assert (tmp_path / "artifacts" / "data" / "manifest.jsonl").is_file()
    assert (tmp_path / "artifacts" / "topologies" / "ast" / "sample_1.pt").is_file()
    assert (tmp_path / "artifacts" / "topologies" / "core-cpg" / "sample_1.pt").is_file()
    assert (
        tmp_path / "artifacts" / "topologies" / "completed" / "sample_1.json"
    ).is_file()
