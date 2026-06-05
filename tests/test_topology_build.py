from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import cpg_vuln.data.build as build_module
from cpg_vuln.data.audit import ManifestRecord
from cpg_vuln.data.build import CANONICAL_VIEWS, build_topologies
from cpg_vuln.data.store import NodeTypeRegistry, load_topology
from cpg_vuln.features.text import NodeTextRegistry

from .helpers import write_graphml


def _record(tmp_path: Path, sample_id: str = "sample_1") -> ManifestRecord:
    graph = tmp_path / f"{sample_id}.graphml"
    source = tmp_path / f"{sample_id}.c"
    write_graphml(graph)
    source.write_text("int target(char *src) { return src[0]; }\n", encoding="utf-8")
    return ManifestRecord(sample_id, 1, str(graph), str(source), "hash")


def _marker(output_dir: Path, sample_id: str = "sample_1") -> dict:
    return json.loads(
        (output_dir / "completed" / f"{sample_id}.json").read_text(encoding="utf-8")
    )


def _index(output_dir: Path) -> list[dict]:
    return json.loads((output_dir / "index.json").read_text(encoding="utf-8"))


def _assert_valid_transaction(output_dir: Path, sample_id: str = "sample_1") -> None:
    marker = _marker(output_dir, sample_id)
    assert marker["schema_version"] == 1
    assert marker["sample_id"] == sample_id
    assert marker["views"] == sorted(CANONICAL_VIEWS)
    assert marker["text_registry_size_at_commit"] <= len(
        NodeTextRegistry.read(output_dir / "text_registry.json")
    )
    assert marker["node_type_registry_size_at_commit"] <= len(
        NodeTypeRegistry.read(output_dir / "node_type_registry.json")
    )
    assert marker["topology_files"] == {
        view: str((output_dir / view / f"{sample_id}.pt").resolve())
        for view in CANONICAL_VIEWS
    }
    index = [
        item for item in _index(output_dir) if item["sample_id"] == sample_id
    ]
    assert len(index) == len(CANONICAL_VIEWS)
    assert {(item["sample_id"], item["view"]) for item in index} == {
        (sample_id, view) for view in CANONICAL_VIEWS
    }
    for item in index:
        payload = load_topology(Path(item["path"]))
        assert item["commit_id"] == marker["commit_id"] == payload["commit_id"]
        assert payload["cache_schema_version"] == 1


def test_build_topologies_commits_canonical_views_with_marker_last(tmp_path: Path) -> None:
    output_dir = tmp_path / "topologies"

    build_topologies([_record(tmp_path)], output_dir)

    assert "slice-cpg" in CANONICAL_VIEWS
    _assert_valid_transaction(output_dir)


def test_topology_skip_requires_completed_marker(tmp_path: Path) -> None:
    output_dir = tmp_path / "topologies"
    record = _record(tmp_path)
    build_topologies([record], output_dir)
    marker = output_dir / "completed" / "sample_1.json"
    marker.unlink()
    old_commit_ids = {
        load_topology(output_dir / view / "sample_1.pt")["commit_id"]
        for view in CANONICAL_VIEWS
    }

    build_topologies([record], output_dir)

    new_marker = _marker(output_dir)
    assert new_marker["commit_id"] not in old_commit_ids
    _assert_valid_transaction(output_dir)


def test_resume_rejects_payload_commit_id_mismatch(tmp_path: Path) -> None:
    output_dir = tmp_path / "topologies"
    record = _record(tmp_path)
    build_topologies([record], output_dir)
    path = output_dir / "ast" / "sample_1.pt"
    payload = load_topology(path)
    payload["commit_id"] = "stale-payload"
    torch.save(payload, path)

    build_topologies([record], output_dir)

    assert _marker(output_dir)["commit_id"] != "stale-payload"
    _assert_valid_transaction(output_dir)


def test_resume_rebuilds_corrupt_topology_payload(tmp_path: Path) -> None:
    output_dir = tmp_path / "topologies"
    record = _record(tmp_path)
    build_topologies([record], output_dir)
    path = output_dir / "ast" / "sample_1.pt"
    path.write_bytes(b"not a torch payload")

    build_topologies([record], output_dir)

    _assert_valid_transaction(output_dir)


def test_resume_rejects_index_commit_id_mismatch(tmp_path: Path) -> None:
    output_dir = tmp_path / "topologies"
    record = _record(tmp_path)
    build_topologies([record], output_dir)
    index = _index(output_dir)
    index[0]["commit_id"] = "stale-index"
    (output_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")

    build_topologies([record], output_dir)

    assert all(item["commit_id"] != "stale-index" for item in _index(output_dir))
    _assert_valid_transaction(output_dir)


def test_ids_within_registry_accepts_empty_tensor() -> None:
    assert build_module.ids_within_registry(torch.empty(0, dtype=torch.long), 0)


def test_resume_after_registry_commit_preserves_orphans_without_reindexing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "topologies"
    record = _record(tmp_path)
    original_write_payloads = build_module._write_topology_payloads
    failures = 0

    def interrupt_after_registry_commit(*args, **kwargs):
        nonlocal failures
        failures += 1
        if failures == 1:
            raise RuntimeError("interrupt after registry commit")
        return original_write_payloads(*args, **kwargs)

    monkeypatch.setattr(build_module, "_write_topology_payloads", interrupt_after_registry_commit)
    with pytest.raises(RuntimeError, match="interrupt after registry commit"):
        build_topologies([record], output_dir)

    orphan_values = NodeTextRegistry.read(output_dir / "text_registry.json").values
    monkeypatch.setattr(build_module, "_write_topology_payloads", original_write_payloads)
    build_topologies([record], output_dir)

    assert NodeTextRegistry.read(output_dir / "text_registry.json").values[: len(orphan_values)] == orphan_values
    _assert_valid_transaction(output_dir)


def test_resume_replaces_stale_index_records_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "topologies"
    record = _record(tmp_path)
    original_marker_write = build_module._write_completion_marker

    def interrupt_before_marker(*args, **kwargs):
        raise RuntimeError("interrupt before marker")

    monkeypatch.setattr(build_module, "_write_completion_marker", interrupt_before_marker)
    with pytest.raises(RuntimeError, match="interrupt before marker"):
        build_topologies([record], output_dir)

    assert not (output_dir / "completed" / "sample_1.json").exists()
    monkeypatch.setattr(build_module, "_write_completion_marker", original_marker_write)
    build_topologies([record], output_dir)

    keys = [(item["sample_id"], item["view"]) for item in _index(output_dir)]
    assert len(keys) == len(set(keys)) == len(CANONICAL_VIEWS)
    _assert_valid_transaction(output_dir)


def test_completion_marker_is_written_after_index_and_topologies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "topologies"
    original_marker_write = build_module._write_completion_marker

    def assert_committed_state_before_marker(path: Path, marker: dict[str, object]) -> None:
        assert (output_dir / "index.json").is_file()
        assert all((output_dir / view / "sample_1.pt").is_file() for view in CANONICAL_VIEWS)
        original_marker_write(path, marker)

    monkeypatch.setattr(build_module, "_write_completion_marker", assert_committed_state_before_marker)

    build_topologies([_record(tmp_path)], output_dir)

    _assert_valid_transaction(output_dir)


def test_topology_write_preserves_replace_error_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_paths = {
        view: tmp_path / view / "sample_1.pt"
        for view in CANONICAL_VIEWS
    }
    payloads = {view: {} for view in CANONICAL_VIEWS}

    def save_payload(path: Path, payload: dict[str, object]) -> None:
        path.write_bytes(b"payload")

    def fail_replace(temporary: Path, final: Path) -> None:
        raise RuntimeError("replace failed")

    def fail_unlink(self: Path, *, missing_ok: bool = False) -> None:
        raise PermissionError("cleanup failed")

    monkeypatch.setattr(build_module, "save_topology_payload", save_payload)
    monkeypatch.setattr(build_module, "replace_file_atomic", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(RuntimeError, match="replace failed"):
        build_module._write_topology_payloads(final_paths, payloads)


def test_rebuild_invalidates_old_marker_before_replacing_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "topologies"
    record = _record(tmp_path)
    build_topologies([record], output_dir)
    original_write_payloads = build_module._write_topology_payloads

    def assert_marker_removed(*args, **kwargs):
        assert not (output_dir / "completed" / "sample_1.json").exists()
        raise RuntimeError("interrupt after marker invalidation")

    monkeypatch.setattr(build_module, "_write_topology_payloads", assert_marker_removed)
    with pytest.raises(RuntimeError, match="interrupt after marker invalidation"):
        build_topologies([record], output_dir, force=True)
    monkeypatch.setattr(build_module, "_write_topology_payloads", original_write_payloads)


def test_dangling_text_id_triggers_rebuild(tmp_path: Path) -> None:
    output_dir = tmp_path / "topologies"
    record = _record(tmp_path)
    build_topologies([record], output_dir)
    path = output_dir / "ast" / "sample_1.pt"
    payload = load_topology(path)
    payload["text_id"][0] = 999999
    torch.save(payload, path)

    build_topologies([record], output_dir)

    rebuilt = load_topology(path)
    assert int(rebuilt["text_id"].max()) < len(
        NodeTextRegistry.read(output_dir / "text_registry.json")
    )


def test_dangling_node_type_id_triggers_rebuild(tmp_path: Path) -> None:
    output_dir = tmp_path / "topologies"
    record = _record(tmp_path)
    build_topologies([record], output_dir)
    path = output_dir / "ast" / "sample_1.pt"
    payload = load_topology(path)
    payload["node_type_id"][0] = 999999
    torch.save(payload, path)

    build_topologies([record], output_dir)

    rebuilt = load_topology(path)
    assert int(rebuilt["node_type_id"].max()) < len(
        NodeTypeRegistry.read(output_dir / "node_type_registry.json")
    )


def test_concurrent_topology_writer_is_rejected(tmp_path: Path) -> None:
    output_dir = tmp_path / "topologies"
    output_dir.mkdir()

    with build_module._topology_build_lock(output_dir, break_stale_lock=False):
        with pytest.raises(RuntimeError, match="topology build lock already exists"):
            with build_module._topology_build_lock(output_dir, break_stale_lock=False):
                raise AssertionError("second writer unexpectedly acquired the lock")


def test_existing_lock_requires_explicit_break_flag(tmp_path: Path) -> None:
    output_dir = tmp_path / "topologies"
    output_dir.mkdir()
    lock = output_dir / ".build-topologies.lock"
    lock.write_text(
        '{"pid":123,"hostname":"host","created_at":"2026-06-02T12:00:00Z"}\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="--break-stale-lock"):
        build_topologies([_record(tmp_path)], output_dir)

    build_topologies([_record(tmp_path)], output_dir, break_stale_lock=True)

    assert not lock.exists()
    _assert_valid_transaction(output_dir)


def test_resume_rejects_partial_view_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "topologies"
    record = _record(tmp_path)
    build_topologies([record], output_dir)
    original_replace = build_module.replace_file_atomic
    topology_replacements = 0

    def interrupt_after_one_topology_replace(temporary: Path, final: Path) -> None:
        nonlocal topology_replacements
        original_replace(temporary, final)
        if final.suffix == ".pt":
            topology_replacements += 1
            if topology_replacements == 1:
                raise RuntimeError("interrupt after one topology replace")

    monkeypatch.setattr(build_module, "replace_file_atomic", interrupt_after_one_topology_replace)
    with pytest.raises(RuntimeError, match="interrupt after one topology replace"):
        build_topologies([record], output_dir, force=True)
    assert not (output_dir / "completed" / "sample_1.json").exists()

    monkeypatch.setattr(build_module, "replace_file_atomic", original_replace)
    build_topologies([record], output_dir)

    _assert_valid_transaction(output_dir)
