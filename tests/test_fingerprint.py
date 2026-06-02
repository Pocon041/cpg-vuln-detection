from __future__ import annotations

import os
from pathlib import Path

import pytest

from cpg_vuln.utils import fingerprint
from cpg_vuln.utils.fingerprint import (
    replace_file_atomic,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_ordered_strings,
    sha256_text,
    stable_json_dumps,
    temporary_path_for,
    write_json_atomic,
)


def test_stable_json_dumps_sorts_keys_and_preserves_list_order() -> None:
    payload = {"z": ["beta", "alpha"], "a": {"b": 2, "a": 1}}

    assert stable_json_dumps(payload) == '{"a":{"a":1,"b":2},"z":["beta","alpha"]}'


def test_sha256_helpers_hash_bytes_text_files_json_and_ordered_strings(tmp_path: Path) -> None:
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    path = tmp_path / "payload.txt"
    path.write_bytes(b"abc")

    assert sha256_bytes(b"abc") == expected
    assert sha256_text("abc") == expected
    assert sha256_file(path) == expected
    assert sha256_json({"b": 2, "a": 1}) == sha256_text('{"a":1,"b":2}')
    assert sha256_ordered_strings(["alpha", "beta"]) == sha256_json(["alpha", "beta"])
    assert sha256_ordered_strings(["alpha", "beta"]) != sha256_ordered_strings(["beta", "alpha"])


def test_write_json_atomic_replaces_from_same_directory_without_temp_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "nested" / "registry.json"
    path.parent.mkdir()
    path.write_text('{"old":true}\n', encoding="utf-8")
    replacements: list[tuple[Path, Path]] = []
    real_replace = fingerprint.os.replace

    def record_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(fingerprint.os, "replace", record_replace)

    write_json_atomic(path, {"value": "updated"})

    assert path.read_text(encoding="utf-8") == '{"value":"updated"}\n'
    assert len(replacements) == 1
    temp_path, final_path = replacements[0]
    assert temp_path.parent == path.parent
    assert final_path == path
    assert list(path.parent.glob(f".{path.name}.tmp.*")) == []


def test_write_json_atomic_preserves_replace_error_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_replace(temp_path: Path, final_path: Path) -> None:
        raise RuntimeError("replace failed")

    def fail_unlink(self: Path, *, missing_ok: bool = False) -> None:
        raise PermissionError("cleanup failed")

    monkeypatch.setattr(fingerprint, "replace_file_atomic", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(RuntimeError, match="replace failed"):
        write_json_atomic(tmp_path / "registry.json", {"value": "updated"})


def test_replace_file_atomic_rejects_cross_directory_replacement(tmp_path: Path) -> None:
    temp_path = tmp_path / "temporary" / "payload.tmp"
    final_path = tmp_path / "final" / "payload.json"
    temp_path.parent.mkdir()
    final_path.parent.mkdir()
    temp_path.write_text("temporary", encoding="utf-8")

    with pytest.raises(ValueError, match="same directory"):
        replace_file_atomic(temp_path, final_path)

    assert temp_path.is_file()
    assert not final_path.exists()


def test_temporary_path_for_uses_unique_same_directory_name(tmp_path: Path) -> None:
    final_path = tmp_path / "nested" / "registry.json"

    first = temporary_path_for(final_path)
    second = temporary_path_for(final_path)

    assert final_path.parent.is_dir()
    assert first.parent == final_path.parent
    assert first.name.startswith(f".{final_path.name}.tmp.{os.getpid()}.")
    assert first != second
