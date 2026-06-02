from __future__ import annotations

import json
import os
import pickle
import socket
import sys
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

import torch
from tqdm import tqdm

from cpg_vuln.data.audit import ManifestRecord
from cpg_vuln.data.graphml import GraphMLParser, choose_primary_method
from cpg_vuln.data.store import (
    TOPOLOGY_CACHE_SCHEMA_VERSION,
    NodeTypeRegistry,
    build_topology_payload,
    load_topology,
    save_topology_payload,
    topology_index_record,
)
from cpg_vuln.data.topology import VIEW_RELATIONS, build_view
from cpg_vuln.features.text import NodeTextRegistry
from cpg_vuln.utils.fingerprint import (
    replace_file_atomic,
    temporary_path_for,
    write_json_atomic,
)


CANONICAL_VIEWS = tuple(sorted(VIEW_RELATIONS))
COMPLETION_SCHEMA_VERSION = 1
LOCK_FILENAME = ".build-topologies.lock"


def build_topologies(
    records: list[ManifestRecord],
    output_dir: Path,
    *,
    limit: int | None = None,
    force: bool = False,
    break_stale_lock: bool = False,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts_path = output_dir / "text_registry.json"
    types_path = output_dir / "node_type_registry.json"
    index_path = output_dir / "index.json"
    with _topology_build_lock(output_dir, break_stale_lock=break_stale_lock):
        texts = NodeTextRegistry.read(texts_path)
        node_types = NodeTypeRegistry.read(types_path)
        index = _read_index(index_path)
        selected = records[:limit] if limit is not None else records
        parser = GraphMLParser()
        for record in tqdm(selected, desc="build topologies"):
            if not force and _sample_is_complete(
                record.sample_id, output_dir, index, texts, node_types
            ):
                continue
            index = _commit_sample(
                record,
                output_dir,
                index,
                texts,
                node_types,
                parser,
            )
        return index


def _commit_sample(
    record: ManifestRecord,
    output_dir: Path,
    index: list[dict[str, object]],
    texts: NodeTextRegistry,
    node_types: NodeTypeRegistry,
    parser: GraphMLParser,
) -> list[dict[str, object]]:
    marker_path = _marker_path(output_dir, record.sample_id)
    marker_path.unlink(missing_ok=True)
    commit_id = uuid.uuid4().hex
    graph = parser.parse(Path(record.graphml_path))
    root = choose_primary_method(graph)
    payloads = {
        view: build_topology_payload(
            build_view(graph, root, view),
            record.sample_id,
            record.label,
            texts,
            node_types,
            commit_id=commit_id,
        )
        for view in CANONICAL_VIEWS
    }
    texts.write(output_dir / "text_registry.json")
    node_types.write(output_dir / "node_type_registry.json")
    final_paths = {
        view: output_dir / view / f"{record.sample_id}.pt"
        for view in CANONICAL_VIEWS
    }
    _write_topology_payloads(final_paths, payloads)
    new_records = [
        topology_index_record(final_paths[view], payloads[view])
        for view in CANONICAL_VIEWS
    ]
    index = _replace_sample_index(index, record.sample_id, new_records)
    _write_index(output_dir / "index.json", index)
    _write_completion_marker(
        marker_path,
        {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "sample_id": record.sample_id,
            "commit_id": commit_id,
            "views": list(CANONICAL_VIEWS),
            "text_registry_size_at_commit": len(texts),
            "node_type_registry_size_at_commit": len(node_types),
            "topology_files": {
                view: str(final_paths[view].resolve()) for view in CANONICAL_VIEWS
            },
        },
    )
    return index


def _write_topology_payloads(
    final_paths: dict[str, Path], payloads: dict[str, dict[str, object]]
) -> None:
    temporaries: list[tuple[Path, Path]] = []
    try:
        for view in CANONICAL_VIEWS:
            final_path = final_paths[view]
            temporary = temporary_path_for(final_path)
            temporaries.append((temporary, final_path))
            save_topology_payload(temporary, payloads[view])
        for temporary, final_path in temporaries:
            replace_file_atomic(temporary, final_path)
    finally:
        has_primary_exception = sys.exc_info()[0] is not None
        for temporary, _ in temporaries:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                if not has_primary_exception:
                    raise


def ids_within_registry(values: torch.Tensor, registry_size: int) -> bool:
    if values.numel() == 0:
        return True
    return int(values.min()) >= 0 and int(values.max()) < registry_size


def _sample_is_complete(
    sample_id: str,
    output_dir: Path,
    index: list[dict[str, object]],
    texts: NodeTextRegistry,
    node_types: NodeTypeRegistry,
) -> bool:
    marker_path = _marker_path(output_dir, sample_id)
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(marker, dict):
        return False
    if marker.get("schema_version") != COMPLETION_SCHEMA_VERSION:
        return False
    if marker.get("sample_id") != sample_id:
        return False
    if marker.get("views") != list(CANONICAL_VIEWS):
        return False
    if not _valid_registry_size(marker.get("text_registry_size_at_commit"), len(texts)):
        return False
    if not _valid_registry_size(
        marker.get("node_type_registry_size_at_commit"), len(node_types)
    ):
        return False
    topology_files = marker.get("topology_files")
    if not isinstance(topology_files, dict) or set(topology_files) != set(CANONICAL_VIEWS):
        return False
    records = [
        item
        for item in index
        if isinstance(item, dict) and item.get("sample_id") == sample_id
    ]
    if len(records) != len(CANONICAL_VIEWS):
        return False
    if {item.get("view") for item in records} != set(CANONICAL_VIEWS):
        return False
    commit_id = marker.get("commit_id")
    if not isinstance(commit_id, str) or not commit_id:
        return False
    records_by_view = {str(item["view"]): item for item in records}
    for view in CANONICAL_VIEWS:
        record = records_by_view[view]
        expected_path = output_dir / view / f"{sample_id}.pt"
        topology_path = topology_files[view]
        index_path = record.get("path")
        if not isinstance(topology_path, str) or not isinstance(index_path, str):
            return False
        path = Path(topology_path)
        if path.resolve() != expected_path.resolve():
            return False
        if not path.is_file() or Path(index_path).resolve() != path.resolve():
            return False
        try:
            payload = load_topology(path)
        except (
            EOFError,
            OSError,
            pickle.UnpicklingError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return False
        if not isinstance(payload, dict):
            return False
        if record.get("commit_id") != commit_id or payload.get("commit_id") != commit_id:
            return False
        if payload.get("sample_id") != sample_id or payload.get("view") != view:
            return False
        if payload.get("cache_schema_version") != TOPOLOGY_CACHE_SCHEMA_VERSION:
            return False
        text_ids = payload.get("text_id")
        node_type_ids = payload.get("node_type_id")
        if not isinstance(text_ids, torch.Tensor) or not isinstance(node_type_ids, torch.Tensor):
            return False
        if not ids_within_registry(text_ids, len(texts)):
            return False
        if not ids_within_registry(node_type_ids, len(node_types)):
            return False
    return True


def _valid_registry_size(value: object, current_size: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= current_size


def _replace_sample_index(
    index: list[dict[str, object]],
    sample_id: str,
    new_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_key = {
        (str(item["sample_id"]), str(item["view"])): item
        for item in index
        if str(item["sample_id"]) != sample_id
    }
    for item in new_records:
        by_key[(str(item["sample_id"]), str(item["view"]))] = item
    return sorted(by_key.values(), key=lambda item: (str(item["sample_id"]), str(item["view"])))


def _read_index(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _write_index(path: Path, index: list[dict[str, object]]) -> None:
    write_json_atomic(
        path,
        sorted(index, key=lambda item: (str(item["sample_id"]), str(item["view"]))),
    )


def _write_completion_marker(path: Path, marker: dict[str, object]) -> None:
    write_json_atomic(path, marker)


def _marker_path(output_dir: Path, sample_id: str) -> Path:
    return output_dir / "completed" / f"{sample_id}.json"


@contextmanager
def _topology_build_lock(
    output_dir: Path, *, break_stale_lock: bool
) -> Iterator[None]:
    path = output_dir / LOCK_FILENAME
    if break_stale_lock:
        with _stale_lock_break_guard(output_dir):
            path.unlink(missing_ok=True)
            descriptor = _acquire_lock(path)
    else:
        descriptor = _acquire_lock(path)
    try:
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _acquire_lock(path: Path) -> int:
    metadata = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        diagnostics = path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"topology build lock already exists at {path}: {diagnostics}. "
            "If no writer is active, rerun with --break-stale-lock."
        ) from error
    try:
        os.write(descriptor, (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8"))
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return descriptor


@contextmanager
def _stale_lock_break_guard(output_dir: Path) -> Iterator[None]:
    path = output_dir / ".build-topologies.break-lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("another process is already breaking a topology lock") from error
    try:
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)
