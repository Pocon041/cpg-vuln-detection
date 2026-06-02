from __future__ import annotations

from cpg_vuln.data.audit import ManifestRecord
from cpg_vuln.data.split import grouped_stratified_split, stratified_split


def _records() -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for index in range(40):
        records.append(
            ManifestRecord(
                sample_id=f"{index}_{index % 2}",
                label=index % 2,
                graphml_path=f"graphml/{index}_{index % 2}.graphml",
                source_path=f"{index}_{index % 2}.c",
                source_hash=f"group-{index % 2}-{index // 4}",
            )
        )
    return records


def test_stratified_split_is_deterministic_and_approximately_8_1_1() -> None:
    first = stratified_split(_records(), seed=42)
    second = stratified_split(_records(), seed=42)

    assert first == second
    assert len(first["train"]) == 32
    assert len(first["val"]) == 4
    assert len(first["test"]) == 4


def test_grouped_split_never_leaks_source_hash_across_sets() -> None:
    records = _records()
    splits = grouped_stratified_split(records, seed=42)
    by_id = {record.sample_id: record for record in records}
    seen: dict[str, str] = {}
    for split_name, sample_ids in splits.items():
        for sample_id in sample_ids:
            source_hash = by_id[sample_id].source_hash
            assert source_hash not in seen or seen[source_hash] == split_name
            seen[source_hash] = split_name
