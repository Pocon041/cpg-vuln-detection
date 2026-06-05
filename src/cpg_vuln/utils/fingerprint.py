from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from collections.abc import Iterable
from pathlib import Path


_HASH_CHUNK_SIZE = 1024 * 1024
_REPLACE_PERMISSION_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.5, 1.0)


def stable_json_dumps(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    return sha256_text(stable_json_dumps(payload))


def sha256_ordered_strings(values: Iterable[str]) -> str:
    return sha256_json(list(values))


def temporary_path_for(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    return final_path.parent / f".{final_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"


def replace_file_atomic(temp_path: Path, final_path: Path) -> None:
    if temp_path.parent.resolve() != final_path.parent.resolve():
        raise ValueError("atomic replacement requires files in the same directory")
    for delay in (*_REPLACE_PERMISSION_RETRY_DELAYS, None):
        try:
            os.replace(temp_path, final_path)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temporary_path_for(path)
    try:
        temp_path.write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")
        replace_file_atomic(temp_path, path)
    finally:
        has_primary_exception = sys.exc_info()[0] is not None
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            if not has_primary_exception:
                raise
