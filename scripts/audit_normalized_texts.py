from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from cpg_vuln.features.text import NodeTextRegistry


STRING_LEAK = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])\'')
PATH_LEAK = re.compile(r"([A-Za-z]:\\|/home/|/tmp/|/usr/|\\\\)")


def audit_normalized_values(values: list[str]) -> dict[str, int]:
    return {
        "total_texts": len(values),
        "string_literal_leaks": sum(1 for value in values if STRING_LEAK.search(value)),
        "path_like_leaks": sum(1 for value in values if PATH_LEAK.search(value)),
        "unknown_identifier_texts": sum(1 for value in values if "UNKNOWN_ID_" in value),
        "semantic_tag_texts": sum(1 for value in values if "SEM_" in value),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit normalized node text registry")
    parser.add_argument("registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry = NodeTextRegistry.read(args.registry)
    stats = audit_normalized_values(registry.values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
