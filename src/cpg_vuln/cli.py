from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpg_vuln.config import load_config
from cpg_vuln.data.audit import audit_dataset, read_manifest
from cpg_vuln.data.build import build_topologies
from cpg_vuln.data.layout import ArtifactLayout
from cpg_vuln.data.source_map import SourceMapConfig, build_source_map, write_source_map
from cpg_vuln.data.split import grouped_stratified_split, stratified_split
from cpg_vuln.features.normalization import NormalizationSpec
from cpg_vuln.features.codebert import (
    CodeBertEncoder,
    build_function_codebert_cache,
    build_node_codebert_cache,
)
from cpg_vuln.features.text import NodeTextRegistry
from cpg_vuln.features.word2vec import build_word2vec_cache
from cpg_vuln.mining.hard_negative_bank import write_pair_audit_sample
from cpg_vuln.mining.weak_baselines import run_weak_baselines
try:
    from cpg_vuln.training.runner import (
        evaluate_ramp,
        train_baselines,
        train_devign,
        train_enhanced,
        train_ramp,
    )
except ModuleNotFoundError as error:
    if error.name != "torch_geometric":
        raise

    def _missing_torch_geometric(*args, **kwargs):
        raise ModuleNotFoundError(
            "torch_geometric is required for training commands"
        ) from error

    evaluate_ramp = train_baselines = train_devign = train_enhanced = train_ramp = _missing_torch_geometric
from cpg_vuln.visualization.explain import export_attention_dashboard, export_top_k_explanations
from cpg_vuln.visualization.report import summarize_runs


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    paths = config["paths"]
    artifacts = Path(paths["artifacts_dir"])
    outputs = Path(paths["outputs_dir"])
    spec = _normalization_spec(config, override_mode=getattr(args, "normalization_mode", None))
    layout = ArtifactLayout(artifacts_root=artifacts, outputs_root=outputs, spec=spec)
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
            layout.topology_dir,
            limit=args.limit,
            force=args.force,
            break_stale_lock=args.break_stale_lock,
            normalization_spec=spec,
        )
    elif args.command == "build-word2vec":
        registry = NodeTextRegistry.read(layout.topology_dir / "text_registry.json")
        build_word2vec_cache(
            registry,
            layout.word2vec_dir,
            vector_size=config["features"]["word2vec_dim"],
            epochs=config["features"]["word2vec_epochs"],
            seed=config["training"]["seed"],
            batch_size=config["features"]["word2vec_batch_size"],
            force=args.force,
            normalization_spec=spec,
            training_scope=config["features"].get("word2vec_training_scope", "transductive"),
        )
    elif args.command == "build-codebert-cache":
        registry = NodeTextRegistry.read(layout.topology_dir / "text_registry.json")
        records = read_manifest(artifacts / "data" / "manifest.jsonl")
        encoder = CodeBertEncoder(config["features"]["codebert_model"])
        build_node_codebert_cache(
            registry,
            layout.node_codebert_dir,
            encoder=encoder,
            model_name=config["features"]["codebert_model"],
            max_length=config["features"]["codebert_node_max_length"],
            batch_size=config["features"]["codebert_batch_size"],
            normalization_spec=spec,
        )
        build_function_codebert_cache(
            records,
            layout.function_codebert_dir,
            encoder=encoder,
            model_name=config["features"]["codebert_model"],
            max_content_tokens=config["features"]["function_max_tokens"],
            overlap=config["features"]["function_overlap"],
            normalization_spec=spec,
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
    elif args.command == "train-devign":
        train_devign(
            config,
            split=args.split,
            view=args.view,
            embedding=args.embedding,
            run_name=args.run_name,
            checkpoint_metric=args.checkpoint_metric,
            threshold_strategy=args.threshold_strategy,
            learning_rate=args.learning_rate,
            positive_class_weight=args.positive_class_weight,
            evaluate_test=not args.defer_test,
            epochs=args.epochs,
            force=args.force,
        )
    elif args.command == "weak-baselines":
        run_weak_baselines(config, split=args.split, view=args.view)
    elif args.command == "train-ramp":
        train_ramp(
            config,
            experiment=args.experiment,
            split=args.split,
            view=args.view,
            model_name=args.model,
            run_name=args.run_name,
            lambda_replay=args.lambda_replay,
            lambda_rank=args.lambda_rank,
            margin=args.margin,
            max_pairs_per_positive=args.max_pairs_per_positive,
            checkpoint_metric=args.checkpoint_metric,
            threshold_strategy=args.threshold_strategy,
            learning_rate=args.learning_rate,
            positive_class_weight=args.positive_class_weight,
            checkpoint_min_ppr=args.checkpoint_min_ppr,
            checkpoint_max_ppr=args.checkpoint_max_ppr,
            checkpoint_max_recall=args.checkpoint_max_recall,
            evaluate_test=not args.defer_test,
            epochs=args.epochs,
            force=args.force,
        )
    elif args.command == "evaluate-ramp":
        evaluate_ramp(
            config,
            run=args.run,
            split=args.split,
            export_attention=args.export_attention,
        )
    elif args.command == "audit-hard-pairs":
        write_pair_audit_sample(
            layout.retrieval_dir / args.split / args.experiment / "bank.jsonl",
            layout.report_root / f"pair_audit_{args.split}_{args.experiment}.json",
            limit=args.limit,
            seed=args.seed,
        )
    elif args.command == "summarize":
        summarize_runs(
            layout.run_root,
            layout.report_root,
            topology_index=layout.topology_dir / "index.json",
        )
    elif args.command == "explain":
        attention = layout.run_root / args.run / "node_attention.json"
        export_top_k_explanations(attention, layout.explanation_root / args.run, top_k=args.top_k)
    elif args.command == "visualize-attention":
        run_dir = layout.run_root / args.run
        export_attention_dashboard(
            run_dir / "node_attention.json",
            run_dir / "predictions.csv",
            Path(config["source_mapping"]["source_map_path"]),
            layout.report_root / "attention" / args.run,
            run_name=args.run,
            top_samples=args.top_samples,
            top_lines=args.top_lines,
            context_radius=args.context,
        )
    else:
        parser.error(f"unknown command: {args.command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPG vulnerability detection experiment CLI")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    topologies = commands.add_parser("build-topologies")
    topologies.add_argument("--normalization-mode", choices=("raw", "semantic-anon", "full-anon"))
    topologies.add_argument("--limit", type=int)
    topologies.add_argument("--force", action="store_true")
    topologies.add_argument("--break-stale-lock", action="store_true")
    word2vec = commands.add_parser("build-word2vec")
    word2vec.add_argument("--normalization-mode", choices=("raw", "semantic-anon", "full-anon"))
    word2vec.add_argument("--force", action="store_true")
    codebert = commands.add_parser("build-codebert-cache")
    codebert.add_argument("--normalization-mode", choices=("raw", "semantic-anon", "full-anon"))
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
    devign = commands.add_parser("train-devign")
    devign.add_argument("--split", choices=("course", "strict"), default="strict")
    devign.add_argument("--view", default="core-cpg")
    devign.add_argument("--embedding", choices=("word2vec", "codebert"), default="codebert")
    devign.add_argument("--run-name")
    devign.add_argument(
        "--checkpoint-metric",
        choices=("loss", "f1", "roc_auc", "pr_auc", "mcc", "balanced_accuracy"),
    )
    devign.add_argument("--threshold-strategy", choices=("fixed_0_5", "val_f1", "val_mcc"))
    devign.add_argument("--learning-rate", type=float)
    devign.add_argument("--positive-class-weight", type=float)
    devign.add_argument("--defer-test", action="store_true")
    devign.add_argument("--epochs", type=int)
    devign.add_argument("--force", action="store_true")
    weak = commands.add_parser("weak-baselines")
    weak.add_argument("--split", choices=("course", "strict"), default="strict")
    weak.add_argument("--view", default="core-cpg")
    ramp = commands.add_parser("train-ramp")
    ramp.add_argument("--experiment", choices=("E0", "E1", "E2", "E3", "E4"), default="E4")
    ramp.add_argument("--split", choices=("course", "strict"), default="strict")
    ramp.add_argument("--view", default="core-cpg")
    ramp.add_argument(
        "--model",
        choices=(
            "selective-fusion",
            "ramp-v2-rgcn",
            "ramp-v2-dual",
            "ramp-v2-gated-rgcn",
            "ramp-v3-slice-mil",
        ),
        default="selective-fusion",
    )
    ramp.add_argument("--run-name")
    ramp.add_argument("--lambda-replay", type=float)
    ramp.add_argument("--lambda-rank", type=float)
    ramp.add_argument("--margin", type=float)
    ramp.add_argument("--max-pairs-per-positive", type=int)
    ramp.add_argument(
        "--checkpoint-metric",
        choices=("loss", "f1", "roc_auc", "pr_auc", "mcc", "balanced_accuracy"),
    )
    ramp.add_argument("--threshold-strategy", choices=("fixed_0_5", "val_f1", "val_mcc"))
    ramp.add_argument("--learning-rate", type=float)
    ramp.add_argument("--positive-class-weight", type=float)
    ramp.add_argument("--checkpoint-min-ppr", type=float)
    ramp.add_argument("--checkpoint-max-ppr", type=float)
    ramp.add_argument("--checkpoint-max-recall", type=float)
    ramp.add_argument("--defer-test", action="store_true")
    ramp.add_argument("--epochs", type=int)
    ramp.add_argument("--force", action="store_true")
    evaluate = commands.add_parser("evaluate-ramp")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--split", choices=("course", "strict"), default="strict")
    evaluate.add_argument("--export-attention", action="store_true")
    audit_pairs = commands.add_parser("audit-hard-pairs")
    audit_pairs.add_argument("--split", choices=("course", "strict"), default="strict")
    audit_pairs.add_argument("--experiment", choices=("E1", "E2", "E3", "E4"), default="E3")
    audit_pairs.add_argument("--limit", type=int, default=100)
    audit_pairs.add_argument("--seed", type=int, default=42)
    commands.add_parser("summarize")
    explain = commands.add_parser("explain")
    explain.add_argument("--run", default="enhanced-selective-fusion-course")
    explain.add_argument("--top-k", type=int, default=10)
    attention = commands.add_parser("visualize-attention")
    attention.add_argument("--run", default="ramp-E4-strict")
    attention.add_argument("--top-samples", type=int, default=24)
    attention.add_argument("--top-lines", type=int, default=10)
    attention.add_argument("--context", type=int, default=2)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _normalization_spec(config: dict, *, override_mode: str | None = None) -> NormalizationSpec:
    values = config["features"].get("node_text_normalization", {})
    return NormalizationSpec(
        mode=override_mode or values.get("mode", "raw"),
        version=int(values.get("version", 1)),
        api_taxonomy_version=int(values.get("api_taxonomy_version", 1)),
        tokenizer_version=int(values.get("tokenizer_version", 2)),
    )


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None
