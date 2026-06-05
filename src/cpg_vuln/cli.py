from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpg_vuln.config import load_config
from cpg_vuln.data.audit import audit_dataset, read_manifest
from cpg_vuln.data.build import build_topologies
from cpg_vuln.data.source_map import SourceMapConfig, build_source_map, write_source_map
from cpg_vuln.data.split import grouped_stratified_split, stratified_split
from cpg_vuln.features.codebert import (
    CodeBertEncoder,
    build_function_codebert_cache,
    build_node_codebert_cache,
)
from cpg_vuln.features.text import NodeTextRegistry
from cpg_vuln.features.word2vec import build_word2vec_cache
from cpg_vuln.training.runner import train_baselines, train_enhanced
from cpg_vuln.visualization.explain import export_top_k_explanations
from cpg_vuln.visualization.report import summarize_runs


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    paths = config["paths"]
    artifacts = Path(paths["artifacts_dir"])
    outputs = Path(paths["outputs_dir"])
    if args.command == "audit":
        report = audit_dataset(
            Path(paths["metadata_csv"]),
            Path(paths["dataset_root"]),
            Path(paths["source_root"]),
            excluded_csv=_optional_path(paths.get("excluded_csv")),
        )
        report.write(artifacts / "data")
        source_mapping = config["source_mapping"]
        validation = source_mapping["validation"]
        write_source_map(
            Path(source_mapping["source_map_path"]),
            build_source_map(
                report.included,
                config=SourceMapConfig(
                    default_line_offset=source_mapping["default_line_offset"],
                    prepared_source_root=_optional_path(
                        source_mapping["prepared_source_root"]
                    ),
                    overrides_path=_optional_path(source_mapping["overrides_path"]),
                    validate_offsets=source_mapping["validate_offsets"],
                    allow_sample_overrides=source_mapping["allow_sample_overrides"],
                    max_sampled_nodes=validation["max_sampled_nodes"],
                    context_radius=validation["context_radius"],
                    minimum_token_match_ratio=validation[
                        "minimum_token_match_ratio"
                    ],
                ),
            ),
        )
        split_dir = artifacts / "data" / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        seed = config["training"]["seed"]
        _write_json(split_dir / "course.json", stratified_split(report.included, seed=seed))
        _write_json(split_dir / "strict.json", grouped_stratified_split(report.included, seed=seed))
        print(json.dumps(report.to_dict(), indent=2))
    elif args.command == "build-topologies":
        records = read_manifest(artifacts / "data" / "manifest.jsonl")
        build_topologies(
            records,
            artifacts / "topologies",
            limit=args.limit,
            force=args.force,
            break_stale_lock=args.break_stale_lock,
        )
    elif args.command == "build-word2vec":
        registry = NodeTextRegistry.read(artifacts / "topologies" / "text_registry.json")
        build_word2vec_cache(
            registry,
            artifacts / "features" / "word2vec",
            vector_size=config["features"]["word2vec_dim"],
            epochs=config["features"]["word2vec_epochs"],
            seed=config["training"]["seed"],
            batch_size=config["features"]["word2vec_batch_size"],
            force=args.force,
        )
    elif args.command == "build-codebert-cache":
        registry = NodeTextRegistry.read(artifacts / "topologies" / "text_registry.json")
        records = read_manifest(artifacts / "data" / "manifest.jsonl")
        encoder = CodeBertEncoder(config["features"]["codebert_model"])
        build_node_codebert_cache(
            registry,
            artifacts / "features" / "codebert" / "nodes",
            encoder=encoder,
            model_name=config["features"]["codebert_model"],
            max_length=config["features"]["codebert_node_max_length"],
            batch_size=config["features"]["codebert_batch_size"],
        )
        build_function_codebert_cache(
            records,
            artifacts / "features" / "codebert" / "functions",
            encoder=encoder,
            model_name=config["features"]["codebert_model"],
            max_content_tokens=config["features"]["function_max_tokens"],
            overlap=config["features"]["function_overlap"],
        )
    elif args.command == "train-baselines":
        train_baselines(
            config,
            views=tuple(args.views),
            embeddings=tuple(args.embeddings),
            splits=tuple(args.splits),
            epochs=args.epochs,
            force=args.force,
        )
    elif args.command == "train-enhanced":
        train_enhanced(
            config,
            splits=tuple(args.splits),
            variants=tuple(args.variants),
            epochs=args.epochs,
            force=args.force,
        )
    elif args.command == "summarize":
        summarize_runs(
            outputs / "runs",
            outputs / "reports",
            topology_index=artifacts / "topologies" / "index.json",
        )
    elif args.command == "explain":
        attention = outputs / "runs" / args.run / "node_attention.json"
        export_top_k_explanations(attention, outputs / "reports" / "explanations" / args.run, top_k=args.top_k)
    else:
        parser.error(f"unknown command: {args.command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPG vulnerability detection experiment CLI")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    topologies = commands.add_parser("build-topologies")
    topologies.add_argument("--limit", type=int)
    topologies.add_argument("--force", action="store_true")
    topologies.add_argument("--break-stale-lock", action="store_true")
    word2vec = commands.add_parser("build-word2vec")
    word2vec.add_argument("--force", action="store_true")
    commands.add_parser("build-codebert-cache")
    baseline = commands.add_parser("train-baselines")
    baseline.add_argument("--epochs", type=int)
    baseline.add_argument("--views", nargs="+", choices=("ast", "cfg", "pdg"), default=("ast", "cfg", "pdg"))
    baseline.add_argument("--embeddings", nargs="+", choices=("word2vec", "codebert"), default=("word2vec", "codebert"))
    baseline.add_argument("--splits", nargs="+", choices=("course", "strict"), default=("course", "strict"))
    baseline.add_argument("--force", action="store_true")
    enhanced = commands.add_parser("train-enhanced")
    enhanced.add_argument("--epochs", type=int)
    enhanced.add_argument(
        "--variants",
        nargs="+",
        choices=("selective-fusion", "no-semantics", "dataflow-only", "slice-fusion"),
        default=("selective-fusion", "no-semantics", "dataflow-only", "slice-fusion"),
    )
    enhanced.add_argument("--splits", nargs="+", choices=("course", "strict"), default=("course", "strict"))
    enhanced.add_argument("--force", action="store_true")
    commands.add_parser("summarize")
    explain = commands.add_parser("explain")
    explain.add_argument("--run", default="enhanced-selective-fusion-course")
    explain.add_argument("--top-k", type=int, default=10)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None
