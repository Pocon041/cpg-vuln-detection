from __future__ import annotations

import json
import re
from pathlib import Path

from cpg_vuln.data.graphml import GraphNode


_TOKEN = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*|0[xX][0-9A-Fa-f]+|\d+(?:\.\d+)?|"
    r"==|!=|<=|>=|->|\+\+|--|&&|\|\||<<|>>|"
    r"[{}()\[\];,.?:~!%^&*+\-=/<>|]"
)


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

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.values, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

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
    return _TOKEN.findall(text)

