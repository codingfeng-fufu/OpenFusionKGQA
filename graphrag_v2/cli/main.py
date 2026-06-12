"""CLI entry point for OpenFusionKGQA."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from graphrag_v2.artifacts.run_observability import (
    format_run_report,
    inspect_run_summary,
    redact_secrets,
)
from graphrag_v2.config import GraphRagConfig, load_config
from graphrag_v2.config.defaults import DEFAULT_CHAT_MODEL_ID
from graphrag_v2.config.loader import create_default_config
from graphrag_v2.community import Neo4jCommunityStore
from graphrag_v2.graph_store import GraphStoreError, create_graph_store
from graphrag_v2.llm import create_chat_provider
from graphrag_v2.qa import (
    GraphGroundedQA,
    LLMAnswerer,
    MockAnswerer,
    format_qa_result,
    format_qa_result_json,
)
from graphrag_v2.indexing import (
    index_documents_only,
    index_extraction_only,
    index_fusion_only,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kgqa",
        description="OpenFusionKGQA command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="Create a default settings file.",
    )
    init_parser.add_argument(
        "--output",
        default="settings.yaml",
        help="Path to write the settings file. Defaults to settings.yaml.",
    )
    init_parser.set_defaults(func=run_init)

    index_parser = subparsers.add_parser(
        "index",
        help="Index documents into graph artifacts.",
    )
    index_parser.add_argument("input", help="Input document file or directory.")
    index_parser.add_argument(
        "--output",
        required=True,
        help="Artifact output directory.",
    )
    index_parser.add_argument(
        "--config",
        default=None,
        help="Optional settings YAML/JSON path.",
    )
    index_parser.add_argument(
        "--extractor",
        choices=["mock", "llm"],
        default=None,
        help="Extraction provider. Defaults to config extraction.extractor_provider.",
    )
    index_parser.add_argument(
        "--graph-store",
        choices=["json", "neo4j"],
        default="json",
        help="Graph store provider. Defaults to json.",
    )
    index_parser.add_argument(
        "--community",
        action="store_true",
        help="Enable community detection and community reports.",
    )
    index_parser.add_argument(
        "--mode",
        choices=["full", "documents-only", "extraction-only", "fusion-only"],
        default="full",
        help="Indexing mode. Defaults to full.",
    )
    index_parser.add_argument(
        "--strict-neo4j",
        action="store_true",
        help=(
            "Require Neo4j as the production graph store and fail before "
            "indexing if Neo4j is unavailable."
        ),
    )
    index_parser.set_defaults(func=run_index)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question against an existing index.",
    )
    ask_parser.add_argument("question", help="Question to answer.")
    ask_parser.add_argument(
        "--index",
        required=True,
        help="Artifact/index directory.",
    )
    ask_parser.add_argument(
        "--config",
        default=None,
        help="Optional settings YAML/JSON path.",
    )
    ask_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format. Defaults to text.",
    )
    ask_parser.add_argument(
        "--answerer",
        choices=["mock", "llm"],
        default="mock",
        help="Answer generation provider. Defaults to mock.",
    )
    ask_parser.add_argument(
        "--strict-neo4j",
        action="store_true",
        help=(
            "Fail instead of falling back to local artifacts when the index "
            "metadata expects Neo4j."
        ),
    )
    ask_parser.set_defaults(func=run_ask)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect generated artifacts.",
    )
    inspect_parser.add_argument(
        "target",
        choices=[
            "run",
            "entities",
            "relationships",
            "graph",
            "communities",
            "community-reports",
            "rejected",
        ],
        help="Artifact group to inspect.",
    )
    inspect_parser.add_argument(
        "--index",
        required=True,
        help="Artifact/index directory.",
    )
    inspect_parser.add_argument(
        "--graph-store",
        choices=["json", "neo4j"],
        default="json",
        help="Graph store provider. Defaults to json.",
    )
    inspect_parser.add_argument(
        "--config",
        default=None,
        help="Optional settings YAML/JSON path.",
    )
    inspect_parser.set_defaults(func=run_inspect)

    return parser


def run_init(args: argparse.Namespace) -> int:
    return _run_cli_action("Settings initialization", lambda: _run_init(args))


def run_index(args: argparse.Namespace) -> int:
    return _run_cli_action("Indexing", lambda: _run_index(args))


def _run_index(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else GraphRagConfig()
    _validate_index_mode(args)
    extractor_name = args.extractor or config.extraction.extractor_provider

    if args.mode == "documents-only":
        metadata = index_documents_only(
            input_path=args.input,
            output_path=args.output,
            config=config,
            mode=args.mode,
        )
        print(
            "Indexed documents: "
            f"{metadata['num_documents']} documents, "
            f"{metadata['num_text_units']} text units"
        )
        print(f"Artifacts written to: {args.output}")
        return 0

    if args.mode == "extraction-only":
        metadata = asyncio.run(
            index_extraction_only(
                input_path=args.input,
                output_path=args.output,
                config=config,
                extractor_name=extractor_name,
                mode=args.mode,
            )
        )
        print(
            "Extracted candidates: "
            f"{metadata['num_candidate_entities']} entities, "
            f"{metadata['num_candidate_relationships']} relationships, "
            f"{metadata['num_candidate_triples']} triples"
        )
        print(f"Artifacts written to: {args.output}")
        return 0

    if args.mode == "fusion-only":
        metadata = asyncio.run(
            index_fusion_only(
                input_path=args.input,
                output_path=args.output,
                config=config,
                extractor_name=extractor_name,
                mode=args.mode,
                graph_store_provider=args.graph_store,
                enable_community=args.community,
                strict_neo4j=args.strict_neo4j,
            )
        )
        print(
            "Fused graph: "
            f"{metadata['num_entities']} entities, "
            f"{metadata['num_relationships']} relationships, "
            f"{metadata['num_rejected_triples']} rejected triples"
        )
        if args.community:
            print(
                "Detected communities: "
                f"{metadata['num_communities']} communities, "
                f"{metadata['num_community_reports']} reports"
            )
        print(f"Artifacts written to: {args.output}")
        return 0

    metadata = asyncio.run(
        index_fusion_only(
            input_path=args.input,
            output_path=args.output,
            config=config,
            extractor_name=extractor_name,
            mode=args.mode,
            graph_store_provider=args.graph_store,
            enable_community=args.community,
            strict_neo4j=args.strict_neo4j,
        )
    )

    print(
        "Indexed graph: "
        f"{metadata['num_entities']} entities, "
        f"{metadata['num_relationships']} relationships, "
        f"{metadata['num_rejected_triples']} rejected triples"
    )
    if args.community:
        print(
            "Detected communities: "
            f"{metadata['num_communities']} communities, "
            f"{metadata['num_community_reports']} reports"
        )
    print(f"Artifacts written to: {args.output}")
    return 0


def _validate_index_mode(args: argparse.Namespace) -> None:
    if getattr(args, "strict_neo4j", False) and args.graph_store != "neo4j":
        raise ValueError("--strict-neo4j requires --graph-store neo4j.")


def run_ask(args: argparse.Namespace) -> int:
    return _run_cli_action("Question answering", lambda: _run_ask(args))


def _run_ask(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else GraphRagConfig()
    answerer = _create_answerer(args.answerer, config)
    qa = GraphGroundedQA.from_index(
        args.index,
        graph_store_config=config.graph_store,
        answerer=answerer,
        allow_neo4j_fallback=not args.strict_neo4j,
    )
    result = qa.ask(args.question)

    if args.format == "json":
        print(format_qa_result_json(result))
    else:
        print(format_qa_result(result))
    return 0


def run_inspect(args: argparse.Namespace) -> int:
    return _run_cli_action("Inspection", lambda: _run_inspect(args))


def _run_inspect(args: argparse.Namespace) -> int:
    if args.target == "run":
        summary, summary_status = inspect_run_summary(args.index)
        print(
            format_run_report(
                args.index,
                summary=summary,
                summary_status=summary_status,
            )
        )
        return 0

    config = load_config(args.config) if args.config else GraphRagConfig()
    if args.target == "graph":
        graph_store = create_graph_store(
            provider=args.graph_store,
            index_path=args.index,
            config=config.graph_store,
        )
        stats = graph_store.get_stats()

        print(_format_graph_store_stats(stats))
        return 0

    if args.target in {"communities", "community-reports"}:
        _inspect_community_artifacts(
            index_path=Path(args.index),
            target=args.target,
            config=config,
            graph_store_provider=args.graph_store,
        )
        return 0

    print(
        "kgqa inspect is available as a CLI skeleton, but artifact inspection "
        "is not implemented in this phase.",
        file=sys.stderr,
    )
    print(f"Requested target: {args.target}", file=sys.stderr)
    print(f"Requested index: {args.index}", file=sys.stderr)
    return 2


def _run_init(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Settings file already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    create_default_config(output)
    print(f"Created settings file: {output}")
    return 0


def _create_answerer(answerer_name: str, config: GraphRagConfig) -> MockAnswerer | LLMAnswerer:
    if answerer_name == "mock":
        return MockAnswerer()
    if answerer_name == "llm":
        model_config = config.get_language_model_config(DEFAULT_CHAT_MODEL_ID)
        llm_client = create_chat_provider(
            provider=config.extraction.llm_provider,
            model_config=model_config,
            require_real=True,
        )
        return LLMAnswerer(llm_client=llm_client)
    raise ValueError(f"Unsupported answerer: {answerer_name}")


def _format_schema_names(names: list[str] | None) -> str:
    if names is None:
        return "None"
    if not names:
        return "0 []"
    return f"{len(names)} [{', '.join(names)}]"


def _format_graph_store_stats(stats) -> str:
    lines = [
        "Graph Store:",
        f"  provider: {stats.provider}",
        f"  health_status: {stats.health_status}",
        f"  index_id: {stats.index_id}",
        f"  database: {stats.database}",
        f"  text_units: {stats.num_text_units}",
        f"  entities: {stats.num_entities}",
        f"  relationships: {stats.num_relationships}",
        f"  rejected_triples: {stats.num_rejected_triples}",
        f"  schema_version: {stats.schema_version}",
        f"  schema_ready: {stats.schema_ready}",
        "  schema_constraints: "
        f"{_format_schema_names(stats.schema_constraints)}",
        "  schema_indexes: "
        f"{_format_schema_names(stats.schema_indexes)}",
        "  expected_schema_constraints: "
        f"{_format_schema_names(stats.expected_schema_constraints)}",
        "  expected_schema_indexes: "
        f"{_format_schema_names(stats.expected_schema_indexes)}",
        "  missing_schema_constraints: "
        f"{_format_schema_names(stats.missing_schema_constraints)}",
        "  missing_schema_indexes: "
        f"{_format_schema_names(stats.missing_schema_indexes)}",
        f"  graph_path: {stats.graph_path}",
        f"  metadata_path: {stats.metadata_path}",
    ]
    if stats.write_strategy is not None:
        lines.append(f"  write_strategy: {stats.write_strategy}")
    if stats.staging_index_id is not None:
        lines.append(f"  staging_index_id: {stats.staging_index_id}")
    return "\n".join(lines)


def _run_cli_action(label: str, action: Callable[[], int]) -> int:
    try:
        return action()
    except _EXPECTED_CLI_EXCEPTIONS as exc:
        print(
            f"{label} failed: {_sanitize_error_message(str(exc))}",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            f"{label} failed unexpectedly: "
            f"{exc.__class__.__name__}: {_sanitize_error_message(str(exc))}",
            file=sys.stderr,
        )
        return 1


_EXPECTED_CLI_EXCEPTIONS = (
    FileNotFoundError,
    GraphStoreError,
    OSError,
    ValueError,
)


def _sanitize_error_message(message: str) -> str:
    return str(redact_secrets(message))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _inspect_community_artifacts(
    index_path: Path,
    target: str,
    config: GraphRagConfig,
    graph_store_provider: str,
) -> None:
    artifact_name = (
        "communities.parquet"
        if target == "communities"
        else "community_reports.parquet"
    )
    artifact_path = index_path / artifact_name
    df = pd.read_parquet(artifact_path) if artifact_path.exists() else pd.DataFrame()

    stats = None
    if graph_store_provider == "neo4j":
        stats = Neo4jCommunityStore(config.graph_store, index_path=index_path).get_stats()

    print("Community Store:")
    print(f"  provider: {graph_store_provider}")
    if target == "communities":
        count = (
            stats["num_communities"]
            if stats is not None
            else len(df)
        )
        print(f"  communities: {count}")
    else:
        count = (
            stats["num_community_reports"]
            if stats is not None
            else len(df)
        )
        print(f"  community_reports: {count}")
    print(f"  artifact_path: {artifact_path.resolve()}")
    if len(df) == 0:
        return

    sample = df.head(3)
    for _, row in sample.iterrows():
        if target == "communities":
            print(
                f"  - {row['id']}: size={row['size']}, rank={row['rank']}, "
                f"title={row['title']}"
            )
        else:
            print(
                f"  - {row['id']}: community={row['community_id']}, "
                f"rank={row['rank']}, title={row['title']}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
