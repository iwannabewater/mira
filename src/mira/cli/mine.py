"""Run deterministic offline hard-negative mining over encoded bundles."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path  # noqa: TC003 - Typer resolves command annotations at runtime
from typing import Annotated

import typer

from mira.cli._common import (
    emit_json,
    encoded_template_checksum,
    load_encoded_collection,
    model_source_checksum,
    require_matching_encoded_provenance,
    validate_encoded_provenance,
    write_new_artifacts,
)
from mira.config import resolve_config
from mira.core.exceptions import ContractError
from mira.core.serialization import strict_json_loads
from mira.mining import (
    MetadataFalseNegativeFilter,
    MiningPipeline,
    MiningRequest,
    PositiveAwareSimilarityFilter,
)


def mine_command(  # noqa: PLR0913 - explicit artifact paths are the CLI contract
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    queries: Annotated[Path, typer.Option("--queries", exists=True, readable=True)],
    corpus: Annotated[Path, typer.Option("--corpus", exists=True, readable=True)],
    positives: Annotated[Path, typer.Option("--positives", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    manifest: Annotated[Path, typer.Option("--manifest")],
    model_checkpoint: Annotated[str, typer.Option("--model-checkpoint")],
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 8,
    retrieve_k: Annotated[int, typer.Option("--retrieve-k", min=1)] = 100,
    metadata_key: Annotated[list[str] | None, typer.Option("--metadata-key")] = None,
    similarity_alpha: Annotated[float | None, typer.Option("--similarity-alpha")] = None,
    seed: Annotated[int, typer.Option(min=0)] = 42,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    expected_queries_manifest_sha256: Annotated[
        str | None,
        typer.Option("--expected-queries-manifest-sha256"),
    ] = None,
    expected_corpus_manifest_sha256: Annotated[
        str | None,
        typer.Option("--expected-corpus-manifest-sha256"),
    ] = None,
) -> None:
    """Retrieve, filter, stratify, and write versioned hard negatives."""
    resolved = resolve_config(config)
    query_collection = load_encoded_collection(
        queries,
        expected_manifest_sha256=expected_queries_manifest_sha256,
        expected_role="query",
    )
    corpus_collection = load_encoded_collection(
        corpus,
        expected_manifest_sha256=expected_corpus_manifest_sha256,
        expected_role="document",
    )
    require_matching_encoded_provenance(query_collection, corpus_collection)
    instruction_template_checksum = encoded_template_checksum(
        query_collection,
        resolved.config,
    )
    for collection in (query_collection, corpus_collection):
        validate_encoded_provenance(
            collection,
            model_checkpoint,
            resolved.config,
            config_checksum=resolved.checksum,
            template_checksum=instruction_template_checksum,
        )
    positive_mapping = _load_positive_mapping(positives)
    metadata_filter = MetadataFalseNegativeFilter(tuple(metadata_key)) if metadata_key else None
    similarity_filter = (
        PositiveAwareSimilarityFilter(similarity_alpha) if similarity_alpha is not None else None
    )
    pipeline = MiningPipeline(
        metadata_filter=metadata_filter,
        similarity_filter=similarity_filter,
    )
    result = pipeline.run(
        MiningRequest(
            queries=query_collection,
            corpus=corpus_collection,
            positives=positive_mapping,
            top_k=top_k,
            retrieve_k=retrieve_k,
            manifest_values={
                "adapter_checksum": None,
                "instruction_template_checksum": instruction_template_checksum,
                "model_checksum": model_source_checksum(
                    model_checkpoint,
                    resolved.config,
                ),
                "model_path": model_checkpoint,
                "normalization_rule": "l2_normalized_inner_product",
                "random_seed": seed,
                "run_id": run_id or f"mine-{uuid.uuid4().hex}",
                "teacher_version": None,
            },
        )
    )

    def write_negatives(temporary: Path) -> None:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            for row in result.negatives:
                stream.write(
                    json.dumps(
                        asdict(row),
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
                stream.write("\n")
        result.manifest.verify_output_file(temporary)

    try:
        write_new_artifacts(
            (
                (output, write_negatives),
                (manifest, result.manifest.write),
            )
        )
    except FileExistsError as error:
        raise ContractError("mining output and manifest paths must not already exist") from error
    emit_json(
        {
            "config_checksum": resolved.checksum,
            "manifest": str(manifest),
            "manifest_checksum": result.manifest.checksum(),
            "negatives": len(result.negatives),
            "output": str(output),
            "status": "complete",
        }
    )


def _load_positive_mapping(path: Path) -> dict[str, tuple[str, ...]]:
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"), context="positive mapping")
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read positive mapping: {path}") from error
    except ValueError as error:
        raise ContractError(str(error)) from error
    if not isinstance(payload, dict):
        raise ContractError("positive mapping root must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for query_id, candidates in payload.items():
        if not isinstance(query_id, str) or not isinstance(candidates, list):
            raise ContractError("positive mapping values must be candidate ID lists")
        if any(not isinstance(candidate, str) for candidate in candidates):
            raise ContractError("positive candidate IDs must be strings")
        result[query_id] = tuple(candidates)
    return result
