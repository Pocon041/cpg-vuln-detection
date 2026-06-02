from __future__ import annotations

import csv
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

from cpg_vuln.data.audit import ManifestRecord
from cpg_vuln.data.graphml import GraphMLParser, ast_closure, choose_primary_method
from cpg_vuln.features.text import tokenize_c
from cpg_vuln.utils.fingerprint import replace_file_atomic, temporary_path_for


_OVERRIDE_FIELDS = ["sample_id", "line_offset", "notes"]
_OUTPUT_FIELDS = [
    "sample_id",
    "raw_source_path",
    "prepared_source_path",
    "line_offset",
    "offset_source",
    "mapping_status",
    "notes",
]
_KEY_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$|^\d+$")


@dataclass(frozen=True)
class SourceMapConfig:
    default_line_offset: int
    prepared_source_root: Path | None
    overrides_path: Path | None
    validate_offsets: bool
    allow_sample_overrides: bool
    max_sampled_nodes: int
    context_radius: int
    minimum_token_match_ratio: float


@dataclass(frozen=True)
class SourceMapRow:
    sample_id: str
    raw_source_path: str
    prepared_source_path: str
    line_offset: int
    offset_source: str
    mapping_status: str
    notes: str


def build_source_map(
    records: list[ManifestRecord], *, config: SourceMapConfig
) -> list[SourceMapRow]:
    _validate_config(config)
    overrides = _read_overrides(
        config.overrides_path,
        records,
        enabled=config.allow_sample_overrides,
    )
    return [
        _build_row(record, config=config, override=overrides.get(record.sample_id))
        for record in records
    ]


def write_source_map(path: Path, rows: list[SourceMapRow]) -> None:
    temporary = temporary_path_for(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        replace_file_atomic(temporary, path)
    finally:
        has_primary_exception = sys.exc_info()[0] is not None
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if not has_primary_exception:
                raise


def _read_overrides(
    path: Path | None,
    records: list[ManifestRecord],
    *,
    enabled: bool,
) -> dict[str, tuple[int, str]]:
    if not enabled:
        return {}
    if path is None or not path.is_file():
        warnings.warn("source-map override file does not exist", UserWarning)
        return {}
    known_ids = {record.sample_id for record in records}
    overrides: dict[str, tuple[int, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _OVERRIDE_FIELDS:
            raise ValueError(
                "source-map override header must be sample_id,line_offset,notes"
            )
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"malformed source-map override row {reader.line_num}")
            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id not in known_ids:
                raise ValueError(f"unknown source-map override sample ID: {sample_id!r}")
            if sample_id in overrides:
                raise ValueError(f"duplicate source-map override sample ID: {sample_id}")
            try:
                line_offset = int(row["line_offset"])
            except ValueError as error:
                raise ValueError(f"invalid source-map offset for {sample_id}") from error
            if line_offset < 0:
                raise ValueError(
                    f"source-map offset must be non-negative for {sample_id}"
                )
            overrides[sample_id] = (line_offset, row["notes"].strip())
    return overrides


def _build_row(
    record: ManifestRecord,
    *,
    config: SourceMapConfig,
    override: tuple[int, str] | None,
) -> SourceMapRow:
    raw_source = Path(record.source_path)
    prepared_source = ""
    if config.prepared_source_root is not None:
        candidate = config.prepared_source_root / f"{record.sample_id}.c"
        if candidate.is_file():
            prepared_source = str(candidate)
        else:
            warnings.warn(f"prepared source does not exist: {candidate}", UserWarning)
    if override is None:
        line_offset, notes = config.default_line_offset, ""
        offset_source = "default"
    else:
        line_offset, notes = override
        offset_source = "override"
    status = (
        "raw_source_missing"
        if not raw_source.is_file()
        else _mapping_status(
            Path(record.graphml_path),
            raw_source,
            line_offset=line_offset,
            offset_source=offset_source,
            config=config,
        )
    )
    return SourceMapRow(
        sample_id=record.sample_id,
        raw_source_path=str(raw_source),
        prepared_source_path=prepared_source,
        line_offset=line_offset,
        offset_source=offset_source,
        mapping_status=status,
        notes=notes,
    )


def _mapping_status(
    graphml_path: Path,
    raw_source_path: Path,
    *,
    line_offset: int,
    offset_source: str,
    config: SourceMapConfig,
) -> str:
    validated = f"validated_{offset_source}"
    suspicious = f"suspicious_{offset_source}"
    graph = GraphMLParser().parse(graphml_path)
    try:
        closure = ast_closure(graph, choose_primary_method(graph).node_id)
    except ValueError:
        return "no_line_evidence"
    evidence: list[tuple[int, str, list[str]]] = []
    for node_id in closure:
        node = graph.nodes[node_id]
        code = node.attrs.get("CODE", "").strip()
        try:
            graphml_line = int(node.attrs.get("LINE_NUMBER", ""))
        except ValueError:
            continue
        tokens = _key_tokens(code)
        if code and code.lower() != "<empty>" and tokens:
            evidence.append((graphml_line, node_id, tokens))
    evidence = _evenly_sample(
        sorted(evidence, key=lambda item: (item[0], item[1])),
        config.max_sampled_nodes,
    )
    if not evidence:
        return "no_line_evidence"
    source_lines = raw_source_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    matches = 0
    for graphml_line, _, tokens in evidence:
        raw_line = graphml_line - line_offset
        if raw_line < 1 or raw_line > len(source_lines):
            return suspicious
        if not config.validate_offsets:
            continue
        start = max(0, raw_line - 1 - config.context_radius)
        stop = min(len(source_lines), raw_line + config.context_radius)
        context_tokens = set(_key_tokens("\n".join(source_lines[start:stop])))
        if any(token in context_tokens for token in tokens):
            matches += 1
    if not config.validate_offsets:
        return validated
    return (
        validated
        if matches / len(evidence) >= config.minimum_token_match_ratio
        else suspicious
    )


def _key_tokens(code: str) -> list[str]:
    return [
        token
        for token in tokenize_c(code)
        if len(token) >= 2 and _KEY_TOKEN.fullmatch(token)
    ]


def _evenly_sample(values: list, limit: int) -> list:
    if limit <= 0:
        raise ValueError("max_sampled_nodes must be positive")
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[0]]
    return [
        values[round(index * (len(values) - 1) / (limit - 1))]
        for index in range(limit)
    ]


def _validate_config(config: SourceMapConfig) -> None:
    _require_nonnegative_integer(
        "default_line_offset",
        config.default_line_offset,
    )
    if (
        not isinstance(config.max_sampled_nodes, int)
        or isinstance(config.max_sampled_nodes, bool)
        or config.max_sampled_nodes <= 0
    ):
        raise ValueError("max_sampled_nodes must be positive")
    _require_nonnegative_integer("context_radius", config.context_radius)
    ratio = config.minimum_token_match_ratio
    if (
        not isinstance(ratio, (int, float))
        or isinstance(ratio, bool)
        or not 0 <= ratio <= 1
    ):
        raise ValueError("minimum_token_match_ratio must be between 0 and 1")


def _require_nonnegative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
