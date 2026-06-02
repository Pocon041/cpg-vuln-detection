# P0-A Transactional Topology Cache and Source Map Design

## Goal

Make topology cache recovery safe after interruption and add a conservative
source-line mapping layer for the existing GraphML batch. This batch establishes
shared atomic-write and fingerprint primitives that later feature-cache and
training-run work can reuse.

## Scope

P0-A includes:

- stable JSON and SHA-256 helpers;
- same-directory temporary files and atomic replacement;
- transactional per-sample topology commits;
- verifiable per-sample completion markers;
- idempotent topology index updates;
- single-writer enforcement for topology builds;
- source-map generation during `audit`;
- default source-line offset `64`;
- optional per-sample source-line offset overrides;
- conservative source-map validation and status reporting;
- focused unit and temporary-directory fault-injection tests.

## Atomic Write Utilities

Add `src/cpg_vuln/utils/fingerprint.py` with:

```python
stable_json_dumps(payload) -> str
sha256_bytes(data) -> str
sha256_text(text) -> str
sha256_file(path) -> str
sha256_json(payload) -> str
sha256_ordered_strings(values) -> str
write_json_atomic(path, payload) -> None
replace_file_atomic(temp_path, final_path) -> None
```

Stable JSON uses UTF-8, sorted object keys, and preserved list ordering.
`write_json_atomic()` writes a uniquely named temporary file in the target
directory and commits it with `os.replace()`. Temporary topology `.pt` files
must also live next to their final files, for example:

```text
artifacts/topologies/ast/.0_0.pt.tmp.<pid>.<uuid>
artifacts/topologies/ast/0_0.pt
```

The utilities are intentionally small. Later batches may use them for feature
and run fingerprints, but P0-A does not change feature caches or training runs.

## Transactional Topology Commit

`build-topologies` remains a single-writer operation. Parsing may be parallelized
in a future batch, but registry, index, and completion-marker commits must remain
serial unless explicit locking or a two-phase merge design is added.

Use a lock file under:

```text
artifacts/topologies/.build-topologies.lock
```

Opening the lock with exclusive creation rejects a concurrent writer with a
clear error. The lock is removed in a `finally` block.

For each sample:

1. Build every requested view and update in-memory registries.
2. Atomically write `text_registry.json`.
3. Atomically write `node_type_registry.json`.
4. Write each topology payload to a same-directory temporary `.pt` file.
5. Atomically replace every final `.pt` file.
6. Replace existing index entries with the same `(sample_id, view)` keys.
7. Atomically write the complete, stably sorted `index.json`.
8. Atomically write the completion marker last.

Registry files intentionally commit before topology files. If interruption
occurs after registry commit but before topology commit, orphan registry entries
may remain. This is acceptable: orphan entries waste limited space but cannot
create invalid references. Recovery must never compact, reorder, or renumber
registry entries because existing topology files rely on stable IDs.

The index is keyed by `(sample_id, view)`. Rebuilding a sample overwrites old
records with the same key and writes a stable `sample_id`, `view` ordering. It
must never append duplicate records.

Completion markers live at:

```text
artifacts/topologies/completed/<sample_id>.json
```

Each marker contains:

```json
{
  "schema_version": 1,
  "sample_id": "0_0",
  "views": ["ast", "cfg", "pdg", "core-cpg", "dataflow-cpg"],
  "text_registry_size_at_commit": 12345,
  "node_type_registry_size_at_commit": 22,
  "topology_files": {
    "ast": "artifacts/topologies/ast/0_0.pt"
  }
}
```

The marker records commit-time registry sizes for diagnostics. Recovery does not
require current registry sizes to equal those values because subsequent samples
append entries.

## Resume Validation

A sample may be skipped only when all checks pass:

1. Its completion marker exists and has schema version `1`.
2. The marker `sample_id` matches the current sample.
3. The marker contains exactly the requested views.
4. Every requested topology file exists.
5. The index contains exactly one entry for every requested `(sample_id, view)`.
6. Each topology payload has `max(text_id) < len(text_registry)`.
7. Each topology payload has `max(node_type_id) < len(node_type_registry)`.

Any failed condition triggers a full rebuild of the requested views for that
sample. A stale or partial completion marker is overwritten only after the
replacement topology files and idempotent index update commit successfully.

## Source Mapping Configuration

The original source root has one authority:

```yaml
paths:
  source_root: F&Q/F&Q
```

Its meaning is: the directory containing original `.c` files before Joern
compatibility-header injection.

Add:

```yaml
source_mapping:
  default_line_offset: 64
  source_map_path: artifacts/manifests/source_map.csv
  prepared_source_root: null
  overrides_path: configs/source_map_overrides.csv
  validate_offsets: true
  allow_sample_overrides: true
  validation:
    max_sampled_nodes: 32
    context_radius: 2
    minimum_token_match_ratio: 0.5
```

`prepared_source_root` is optional. When it is `null`, source-map generation
writes an empty `prepared_source_path`. If configured but a prepared file is
missing, generation writes an empty path, records a warning, and preserves the
mapping result for the original source.

`configs/source_map_overrides.csv` has:

```csv
sample_id,line_offset,notes
```

When overrides are enabled and the file is missing, generation emits a warning
and continues. When the file exists but its header, offset, or sample IDs are
invalid, generation raises an explicit error. Offset selection priority is:

```text
per-sample override
default_line_offset
```

## Source Map Output

During `audit`, generate the configured `source_map.csv`:

```csv
sample_id,raw_source_path,prepared_source_path,line_offset,offset_source,mapping_status,notes
0_0,F&Q/F&Q/0_0.c,,64,default,validated_default,
661_1,F&Q/F&Q/661_1.c,,69,override,validated_override,extra file-specific vector shim
```

Allowed `mapping_status` values:

| Status | Meaning |
| --- | --- |
| `validated_default` | Default offset passed lightweight validation. |
| `validated_override` | Override offset passed lightweight validation. |
| `suspicious_default` | Default offset produced weak or out-of-range evidence. |
| `suspicious_override` | Override offset produced weak or out-of-range evidence. |
| `raw_source_missing` | Original source file is unavailable. |
| `no_line_evidence` | GraphML does not contain enough usable line evidence. |

The validation is diagnostic, not an alignment algorithm. It must never modify
an offset automatically.

For each sample, conservatively:

1. Parse GraphML nodes with `LINE_NUMBER` and non-empty `CODE`.
2. Sample at most `max_sampled_nodes`.
3. Compute `raw_line = graphml_line - line_offset`.
4. Require mapped lines to stay within original-source bounds.
5. Ignore empty text, `<empty>`, pure punctuation, and overly short tokens.
6. Search for at least one key token within `raw_line +/- context_radius`.
7. Mark the row validated only when the match ratio meets
   `minimum_token_match_ratio`.

Future GraphML regeneration should insert:

```c
#line 1 "sample_id.c"
```

after the compatibility header and set the default offset to `0`. P0-A does not
rerun Joern or regenerate GraphML.

## Tests

Add focused tests for:

- atomic JSON replacement;
- interruption after registry commit but before topology commit;
- interruption after some view files commit but before index commit;
- interruption after index commit but before marker commit;
- orphan registry entries surviving recovery without reindexing;
- stale index entries being replaced, not duplicated;
- skip requiring a completion marker;
- dangling `text_id` triggering rebuild;
- dangling `node_type_id` triggering rebuild;
- completion marker being written last;
- concurrent topology writers being rejected;
- source map using `paths.source_root`;
- default offset `64`;
- override precedence;
- empty prepared-source paths when `prepared_source_root` is `null`;
- missing override file warning;
- malformed override file failure;
- out-of-range source mapping status;
- no-line-evidence source mapping status.

Tests use temporary directories and small synthetic GraphML fixtures. They must
not write repository `artifacts/` or `outputs/`.

## Out of Scope

P0-A does not:

- add Word2Vec split-aware fitting;
- modify Word2Vec or CodeBERT caches;
- add feature-cache fingerprints;
- modify the training runner;
- add run fingerprints;
- wire source maps into explanation output;
- clear or overwrite existing `artifacts/`;
- clear or overwrite existing `outputs/`;
- run full topology builds;
- run full training;
- infer line offsets automatically;
- parallelize topology registry commits;
- modify the enhanced model architecture.

## Acceptance Criteria

P0-A is complete when:

1. New and existing related unit tests pass.
2. Temporary-directory fault-injection recovery tests pass.
3. Repository `artifacts/` and `outputs/` remain unmodified.
4. `audit` can generate `source_map.csv` in a temporary directory.
5. Source maps support default offset `64`.
6. Overrides take precedence over the default offset.
7. `prepared_source_root: null` produces valid rows with empty prepared paths.
8. Recovery leaves no dangling `text_id` or `node_type_id`.
9. Index output contains no duplicate `(sample_id, view)` keys.
10. Completion markers commit last.
11. Concurrent topology writers receive an explicit error.

