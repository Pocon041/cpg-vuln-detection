from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cpg_vuln.data.graphml import GraphNode
from cpg_vuln.features.lexer import lex_c
from cpg_vuln.utils.fingerprint import write_json_atomic


class NodeTextRegistry:
    def __init__(self, values: list[str] | None = None) -> None:
        self.values = list(values or [])
        self._ids = {text: index for index, text in enumerate(self.values)}

    def __len__(self) -> int:
        return len(self.values)

    def add(self, text: str) -> int:
        existing = self._ids.get(text)
        if existing is not None:
            return existing
        index = len(self.values)
        self.values.append(text)
        self._ids[text] = index
        return index

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for value in self.values:
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def write(self, path: Path) -> None:
        write_json_atomic(path, self.values)

    @classmethod
    def read(cls, path: Path) -> "NodeTextRegistry":
        if not path.is_file():
            return cls()
        return cls(json.loads(path.read_text(encoding="utf-8")))


def normalize_node_text(node: GraphNode) -> str:
    if node.label == "METHOD":
        value = f"{node.attrs.get('NAME', '')} {node.attrs.get('SIGNATURE', '')}".strip()
        return value or "<METHOD>"
    if node.label == "BLOCK":
        return "<BLOCK>"
    return (
        node.attrs.get("CODE", "").strip()
        or node.attrs.get("NAME", "").strip()
        or f"<{node.label}>"
    )


def tokenize_c(text: str) -> list[str]:
    return [token.text for token in lex_c(text)]
