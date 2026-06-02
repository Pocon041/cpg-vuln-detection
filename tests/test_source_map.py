from __future__ import annotations

import csv
import warnings
from pathlib import Path

import pytest

import cpg_vuln.data.source_map as source_map_module
from cpg_vuln.data.audit import ManifestRecord
from cpg_vuln.data.source_map import (
    SourceMapConfig,
    _evenly_sample,
    build_source_map,
    write_source_map,
)

from .helpers import write_graphml


def _record(
    tmp_path: Path, sample_id: str = "sample_1", *, graph_line: int = 65
) -> ManifestRecord:
    graph = tmp_path / f"{sample_id}.graphml"
    source = tmp_path / f"{sample_id}.c"
    write_graphml(graph, line_number=graph_line)
    source.write_text("strcpy(dst, src);\n", encoding="utf-8")
    return ManifestRecord(sample_id, 1, str(graph), str(source), "hash")


def _config(tmp_path: Path, **overrides: object) -> SourceMapConfig:
    overrides_path = tmp_path / "overrides.csv"
    if "overrides_path" not in overrides:
        overrides_path.write_text("sample_id,line_offset,notes\n", encoding="utf-8")
    values = {
        "default_line_offset": 64,
        "prepared_source_root": None,
        "overrides_path": overrides_path,
        "validate_offsets": True,
        "allow_sample_overrides": True,
        "max_sampled_nodes": 32,
        "context_radius": 2,
        "minimum_token_match_ratio": 0.5,
    }
    values.update(overrides)
    return SourceMapConfig(**values)


def test_source_map_uses_default_offset_and_allows_empty_prepared_path(
    tmp_path: Path,
) -> None:
    row = build_source_map([_record(tmp_path)], config=_config(tmp_path))[0]

    assert row.sample_id == "sample_1"
    assert row.prepared_source_path == ""
    assert row.line_offset == 64
    assert row.offset_source == "default"
    assert row.mapping_status == "validated_default"


def test_source_map_override_takes_precedence(tmp_path: Path) -> None:
    overrides = tmp_path / "overrides.csv"
    overrides.write_text(
        "sample_id,line_offset,notes\nsample_1,69,extra vector shim\n",
        encoding="utf-8",
    )

    row = build_source_map(
        [_record(tmp_path, graph_line=70)],
        config=_config(tmp_path, overrides_path=overrides),
    )[0]

    assert row.line_offset == 69
    assert row.offset_source == "override"
    assert row.mapping_status == "validated_override"
    assert row.notes == "extra vector shim"


def test_source_map_missing_override_file_warns_only(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="override file does not exist") as caught:
        row = build_source_map(
            [_record(tmp_path)],
            config=_config(tmp_path, overrides_path=tmp_path / "missing.csv"),
        )[0]

    assert len(caught) == 1
    assert row.mapping_status == "validated_default"


def test_source_map_malformed_override_header_fails(tmp_path: Path) -> None:
    overrides = tmp_path / "overrides.csv"
    overrides.write_text("sample_id,offset\nsample_1,64\n", encoding="utf-8")

    with pytest.raises(ValueError, match="header"):
        build_source_map(
            [_record(tmp_path)],
            config=_config(tmp_path, overrides_path=overrides),
        )


def test_source_map_marks_out_of_range_lines_suspicious(tmp_path: Path) -> None:
    row = build_source_map(
        [_record(tmp_path, graph_line=500)],
        config=_config(tmp_path),
    )[0]

    assert row.mapping_status == "suspicious_default"


def test_source_map_marks_weak_context_suspicious(tmp_path: Path) -> None:
    record = _record(tmp_path)
    Path(record.source_path).write_text("return 0;\n", encoding="utf-8")

    row = build_source_map([record], config=_config(tmp_path))[0]

    assert row.mapping_status == "suspicious_default"


def test_source_map_context_radius_matches_adjacent_lines(tmp_path: Path) -> None:
    record = _record(tmp_path)
    Path(record.source_path).write_text(
        "return 0;\nstrcpy(dst, src);\n",
        encoding="utf-8",
    )

    row = build_source_map(
        [record],
        config=_config(tmp_path, context_radius=1),
    )[0]

    assert row.mapping_status == "validated_default"


def test_source_map_sampling_is_deterministic(tmp_path: Path) -> None:
    record = _record(tmp_path)

    first = build_source_map([record], config=_config(tmp_path))
    second = build_source_map([record], config=_config(tmp_path))

    assert first == second


def test_write_source_map_atomically_writes_expected_csv_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "nested" / "source_map.csv"
    replacements: list[tuple[Path, Path]] = []
    real_replace = source_map_module.replace_file_atomic

    def record_replace(temporary: Path, final: Path) -> None:
        replacements.append((temporary, final))
        real_replace(temporary, final)

    monkeypatch.setattr(source_map_module, "replace_file_atomic", record_replace)

    write_source_map(
        path,
        build_source_map([_record(tmp_path)], config=_config(tmp_path)),
    )

    with path.open("r", encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle)) == [
            "sample_id",
            "raw_source_path",
            "prepared_source_path",
            "line_offset",
            "offset_source",
            "mapping_status",
            "notes",
        ]
    assert len(replacements) == 1
    temporary, final = replacements[0]
    assert temporary.parent == path.parent
    assert final == path
    assert list(path.parent.glob(f".{path.name}.tmp.*")) == []


def test_write_source_map_preserves_replace_error_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_replace(temporary: Path, final: Path) -> None:
        raise RuntimeError("replace failed")

    def fail_unlink(self: Path, *, missing_ok: bool = False) -> None:
        raise PermissionError("cleanup failed")

    monkeypatch.setattr(source_map_module, "replace_file_atomic", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(RuntimeError, match="replace failed"):
        write_source_map(tmp_path / "source_map.csv", [])


def test_source_map_marks_graph_without_line_evidence(tmp_path: Path) -> None:
    record = _record(tmp_path)
    graph = Path(record.graphml_path)
    graph.write_text(
        graph.read_text(encoding="utf-8")
        .replace('<data key="node__CALL__LINE_NUMBER">65</data>', "")
        .replace('<data key="node__IDENTIFIER__LINE_NUMBER">65</data>', ""),
        encoding="utf-8",
    )

    row = build_source_map([record], config=_config(tmp_path))[0]

    assert row.mapping_status == "no_line_evidence"


def test_source_map_validation_ignores_nodes_outside_primary_ast(tmp_path: Path) -> None:
    record = _record(tmp_path)
    graph = Path(record.graphml_path)
    graph.write_text(
        graph.read_text(encoding="utf-8").replace(
            "</graph>",
            """
    <node id="999">
      <data key="labelV">CALL</data>
      <data key="node__CALL__CODE">outside_primary_ast</data>
      <data key="node__CALL__LINE_NUMBER">999999</data>
    </node>
  </graph>
""",
        ),
        encoding="utf-8",
    )

    row = build_source_map([record], config=_config(tmp_path))[0]

    assert row.mapping_status == "validated_default"


def test_source_map_marks_missing_raw_source(tmp_path: Path) -> None:
    record = _record(tmp_path)
    Path(record.source_path).unlink()

    row = build_source_map([record], config=_config(tmp_path))[0]

    assert row.mapping_status == "raw_source_missing"


def test_source_map_missing_configured_prepared_source_warns_and_writes_empty_path(
    tmp_path: Path,
) -> None:
    prepared_root = tmp_path / "prepared"

    with pytest.warns(UserWarning, match="prepared source does not exist") as caught:
        row = build_source_map(
            [_record(tmp_path)],
            config=_config(tmp_path, prepared_source_root=prepared_root),
        )[0]

    assert len(caught) == 1
    assert row.prepared_source_path == ""
    assert row.mapping_status == "validated_default"


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("sample_id,line_offset,notes\nunknown,64,\n", "unknown"),
        ("sample_id,line_offset,notes\nsample_1,invalid,\n", "invalid"),
        ("sample_id,line_offset,notes\nsample_1,-1,\n", "non-negative"),
        (
            "sample_id,line_offset,notes\nsample_1,64,\nsample_1,69,\n",
            "duplicate",
        ),
    ],
)
def test_source_map_invalid_override_rows_fail(
    tmp_path: Path, contents: str, message: str
) -> None:
    overrides = tmp_path / "overrides.csv"
    overrides.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_source_map(
            [_record(tmp_path)],
            config=_config(tmp_path, overrides_path=overrides),
        )


def test_source_map_disabled_overrides_ignore_path(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        row = build_source_map(
            [_record(tmp_path)],
            config=_config(
                tmp_path,
                overrides_path=tmp_path / "missing.csv",
                allow_sample_overrides=False,
            ),
        )[0]

    assert row.line_offset == 64
    assert row.offset_source == "default"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"default_line_offset": -1}, "default_line_offset"),
        ({"max_sampled_nodes": 0}, "max_sampled_nodes"),
        ({"context_radius": -1}, "context_radius"),
        ({"minimum_token_match_ratio": -0.01}, "minimum_token_match_ratio"),
        ({"minimum_token_match_ratio": 1.01}, "minimum_token_match_ratio"),
    ],
)
def test_source_map_invalid_config_fails_even_without_records(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_source_map([], config=_config(tmp_path, **override))


def test_evenly_sample_supports_limit_one_deterministically() -> None:
    values = ["first", "middle", "last"]

    assert _evenly_sample(values, 1) == ["first"]
    assert _evenly_sample(values, 1) == ["first"]
