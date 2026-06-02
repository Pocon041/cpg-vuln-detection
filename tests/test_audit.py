from __future__ import annotations

import csv
from pathlib import Path

from cpg_vuln.data.audit import CONFLICT_SAMPLE_IDS, audit_dataset

from .helpers import write_graphml


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_audit_uses_metadata_and_excludes_conflicting_samples(tmp_path: Path) -> None:
    graph_root = tmp_path / "graphml"
    source_root = tmp_path / "source"
    graph_root.mkdir()
    source_root.mkdir()
    write_graphml(graph_root / "ok_0.graphml")
    write_graphml(graph_root / "3569_0.graphml")
    write_graphml(graph_root / "11_1-checkpoint.graphml")
    (source_root / "ok_0.c").write_text("int ok(void) { return 0; }\n", encoding="utf-8")
    (source_root / "3569_0.c").write_text("int conflict(void) { return 0; }\n", encoding="utf-8")
    metadata = tmp_path / "labels.csv"
    _write_csv(
        metadata,
        [
            {"sample_id": "ok_0", "base_id": "ok", "label": "0", "graphml_path": "graphml/ok_0.graphml"},
            {"sample_id": "3569_0", "base_id": "3569", "label": "0", "graphml_path": "graphml/3569_0.graphml"},
        ],
    )

    report = audit_dataset(metadata, tmp_path, source_root)

    assert [record.sample_id for record in report.included] == ["ok_0"]
    assert report.excluded["3569_0"] == "conflicting_normalized_source_label"
    assert "11_1-checkpoint" not in {record.sample_id for record in report.included}
    assert "3569_0" in CONFLICT_SAMPLE_IDS

