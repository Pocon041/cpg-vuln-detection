from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CONFLICT_SAMPLE_IDS = frozenset({"3569_0", "6309_1", "3835_0", "955_1"})
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\r\n]*")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    label: int
    graphml_path: str
    source_path: str
    source_hash: str


@dataclass(frozen=True)
class AuditReport:
    included: list[ManifestRecord]
    excluded: dict[str, str]
    metadata_rows: int

    def to_dict(self) -> dict[str, object]:
        return {
            "metadata_rows": self.metadata_rows,
            "included_samples": len(self.included),
            "excluded_samples": len(self.excluded),
            "excluded": self.excluded,
        }

    def write(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "audit_report.json", self.to_dict())
        with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.included:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        with (output_dir / "excluded.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "reason"])
            writer.writerows(sorted(self.excluded.items()))


def normalize_source(source: str) -> str:
    source = _BLOCK_COMMENT.sub("", source)
    source = _LINE_COMMENT.sub("", source)
    return _WHITESPACE.sub("", source)


def normalized_source_hash(source: str) -> str:
    return hashlib.sha256(normalize_source(source).encode("utf-8")).hexdigest()


def audit_dataset(
    metadata_csv: Path,
    dataset_root: Path,
    source_root: Path,
    *,
    excluded_csv: Path | None = None,
    conflict_sample_ids: Iterable[str] = CONFLICT_SAMPLE_IDS,
) -> AuditReport:
    conflict_ids = set(conflict_sample_ids)
    excluded = _read_known_exclusions(excluded_csv)
    included: list[ManifestRecord] = []
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        sample_id = row["sample_id"].strip()
        if sample_id in excluded:
            continue
        if sample_id in conflict_ids:
            excluded[sample_id] = "conflicting_normalized_source_label"
            continue
        graphml_path = dataset_root / row["graphml_path"]
        source_path = source_root / f"{sample_id}.c"
        if not graphml_path.is_file():
            excluded[sample_id] = "missing_graphml"
            continue
        if not source_path.is_file():
            excluded[sample_id] = "missing_source"
            continue
        source = source_path.read_text(encoding="utf-8", errors="replace")
        included.append(
            ManifestRecord(
                sample_id=sample_id,
                label=int(row["label"]),
                graphml_path=str(graphml_path.resolve()),
                source_path=str(source_path.resolve()),
                source_hash=normalized_source_hash(source),
            )
        )
    return AuditReport(included=included, excluded=excluded, metadata_rows=len(rows))


def read_manifest(path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(ManifestRecord(**json.loads(line)))
    return records


def _read_known_exclusions(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["sample_id"].strip(): row["reason"].strip()
            for row in csv.DictReader(handle)
        }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

