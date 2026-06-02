from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from cpg_vuln.data.audit import ManifestRecord
from cpg_vuln.data.graphml import GraphMLParser, choose_primary_method
from cpg_vuln.data.store import NodeTypeRegistry, save_topology
from cpg_vuln.data.topology import VIEW_RELATIONS, build_view
from cpg_vuln.features.text import NodeTextRegistry


def build_topologies(
    records: list[ManifestRecord],
    output_dir: Path,
    *,
    views: tuple[str, ...] = tuple(VIEW_RELATIONS),
    limit: int | None = None,
    force: bool = False,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts_path = output_dir / "text_registry.json"
    types_path = output_dir / "node_type_registry.json"
    index_path = output_dir / "index.json"
    texts = NodeTextRegistry.read(texts_path)
    node_types = NodeTypeRegistry.read(types_path)
    index = _read_index(index_path)
    selected = records[:limit] if limit is not None else records
    parser = GraphMLParser()
    for record in tqdm(selected, desc="build topologies"):
        expected = [output_dir / view / f"{record.sample_id}.pt" for view in views]
        if not force and all(path.is_file() for path in expected):
            continue
        graph = parser.parse(Path(record.graphml_path))
        root = choose_primary_method(graph)
        for view, path in zip(views, expected, strict=True):
            if force or not path.is_file():
                index = [
                    item
                    for item in index
                    if not (item["sample_id"] == record.sample_id and item["view"] == view)
                ]
                index.append(
                    save_topology(
                        path,
                        build_view(graph, root, view),
                        record.sample_id,
                        record.label,
                        texts,
                        node_types,
                    )
                )
        texts.write(texts_path)
        node_types.write(types_path)
        _write_index(index_path, index)
    return index


def _read_index(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _write_index(path: Path, index: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            sorted(index, key=lambda item: (str(item["sample_id"]), str(item["view"]))),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

