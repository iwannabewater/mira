"""Evaluate encoded query/candidate bundles with exact retrieval."""

from __future__ import annotations

import re
from pathlib import Path  # noqa: TC003 - Typer resolves command annotations at runtime
from typing import TYPE_CHECKING, Annotated, Any

import torch
import typer

from mira.cli._common import (
    emit_json,
    encoded_template_checksum,
    file_sha256,
    load_encoded_collection,
    model_source_checksum,
    require_matching_encoded_provenance,
    validate_encoded_provenance,
    write_json_file,
)
from mira.config import resolve_config
from mira.core.exceptions import ContractError
from mira.evaluation import ExactSearchEvaluator, build_slice_masks, slice_metrics
from mira.models.pooling import truncate_and_normalize

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mira.mining import EncodedCollection

_CUTOFF = re.compile(r"^[a-z]+@(\d+)$")


def evaluate_command(  # noqa: PLR0913 - explicit artifact paths are the CLI contract
    checkpoint: Annotated[str, typer.Option("--checkpoint")],
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    queries: Annotated[Path, typer.Option("--queries", exists=True, readable=True)],
    candidates: Annotated[Path, typer.Option("--candidates", exists=True, readable=True)],
    relevance: Annotated[Path, typer.Option("--relevance", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    dimension: Annotated[int | None, typer.Option("--dimension", min=1)] = None,
    expected_queries_manifest_sha256: Annotated[
        str | None,
        typer.Option("--expected-queries-manifest-sha256"),
    ] = None,
    expected_candidates_manifest_sha256: Annotated[
        str | None,
        typer.Option("--expected-candidates-manifest-sha256"),
    ] = None,
) -> None:
    """Run exact cosine retrieval and write aggregate plus per-query evidence."""
    resolved = resolve_config(config)
    allowed_dimensions = resolved.config.evaluation.dimensions
    if dimension is not None and dimension not in allowed_dimensions:
        raise ContractError(
            f"evaluation dimension override {dimension} is not one of the configured "
            f"trained evaluation dimensions: {allowed_dimensions}"
        )
    configured_dimensions = (dimension,) if dimension is not None else allowed_dimensions
    query_collection = load_encoded_collection(
        queries,
        expected_manifest_sha256=expected_queries_manifest_sha256,
        expected_role="query",
    )
    candidate_collection = load_encoded_collection(
        candidates,
        expected_manifest_sha256=expected_candidates_manifest_sha256,
        expected_role="document",
    )
    require_matching_encoded_provenance(query_collection, candidate_collection)
    template_checksum = encoded_template_checksum(query_collection, resolved.config)
    for collection in (query_collection, candidate_collection):
        validate_encoded_provenance(
            collection,
            checkpoint,
            resolved.config,
            config_checksum=resolved.checksum,
            template_checksum=template_checksum,
        )
    relevance_tensor = _load_relevance(relevance, query_collection, candidate_collection)
    available_dimension = int(query_collection.embeddings.shape[1])
    if candidate_collection.embeddings.shape[1] != available_dimension:
        raise ContractError("query and candidate encoded dimensions differ")
    oversized = tuple(value for value in configured_dimensions if value > available_dimension)
    if oversized:
        raise ContractError(
            f"evaluation dimensions exceed encoded width {available_dimension}: {oversized}"
        )
    cutoffs = _metric_cutoffs(
        resolved.config.evaluation.metrics, len(candidate_collection.item_ids)
    )
    dimension_reports: dict[str, Any] = {}
    for value in configured_dimensions:
        query_embeddings = truncate_and_normalize(query_collection.embeddings, value)
        candidate_embeddings = truncate_and_normalize(candidate_collection.embeddings, value)
        report = ExactSearchEvaluator("cosine").evaluate(
            query_embeddings,
            candidate_embeddings,
            relevance_tensor,
            ks=cutoffs,
        )
        selected_metrics = {
            name: report.metrics[_metric_result_name(name)]
            for name in resolved.config.evaluation.metrics
        }
        selected_per_query = {
            name: report.per_query[_per_query_result_name(name)]
            for name in resolved.config.evaluation.metrics
        }
        slices = _evaluation_slices(
            selected_per_query,
            query_collection.metadata,
            resolved.config.evaluation.slices,
        )
        dimension_reports[str(value)] = {
            "metrics": selected_metrics,
            "per_query": {name: item.tolist() for name, item in selected_per_query.items()},
            "slices": slices,
            "top_indices": report.search.indices.tolist(),
        }
    largest_report = dimension_reports[str(configured_dimensions[-1])]
    payload: dict[str, Any] = {
        "checkpoint_checksum": model_source_checksum(checkpoint, resolved.config),
        "candidate_collection_checksum": candidate_collection.checksum(),
        "config_checksum": resolved.checksum,
        "dimension": configured_dimensions[-1],
        "dimensions": dimension_reports,
        "metrics": largest_report["metrics"],
        "per_query": largest_report["per_query"],
        "query_ids": query_collection.item_ids,
        "query_collection_checksum": query_collection.checksum(),
        "relevance_checksum": file_sha256(relevance),
        "status": "complete",
        "top_indices": largest_report["top_indices"],
    }
    write_json_file(payload, output)
    emit_json({"metrics": payload["metrics"], "output": str(output), "status": "complete"})


def _load_relevance(
    path: Path,
    queries: EncodedCollection,
    candidates: EncodedCollection,
) -> torch.Tensor:
    """Load an ID-aligned relevance matrix bound to exact encoded collections."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ContractError(f"cannot read relevance artifact: {path}") from error
    expected_keys = {
        "schema_version",
        "query_ids",
        "candidate_ids",
        "query_collection_checksum",
        "candidate_collection_checksum",
        "relevance",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ContractError(
            "relevance artifact must be a strict versioned mapping with aligned IDs and "
            "collection checksums"
        )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ContractError("unsupported relevance artifact schema_version")
    query_ids = payload["query_ids"]
    candidate_ids = payload["candidate_ids"]
    if not isinstance(query_ids, (list, tuple)) or any(
        not isinstance(item, str) for item in query_ids
    ):
        raise ContractError("relevance query_ids must be a string sequence")
    if not isinstance(candidate_ids, (list, tuple)) or any(
        not isinstance(item, str) for item in candidate_ids
    ):
        raise ContractError("relevance candidate_ids must be a string sequence")
    if tuple(query_ids) != queries.item_ids:
        raise ContractError("relevance query_ids do not exactly match encoded query order")
    if tuple(candidate_ids) != candidates.item_ids:
        raise ContractError("relevance candidate_ids do not exactly match encoded candidate order")
    if payload["query_collection_checksum"] != queries.checksum():
        raise ContractError("relevance query collection checksum mismatch")
    if payload["candidate_collection_checksum"] != candidates.checksum():
        raise ContractError("relevance candidate collection checksum mismatch")
    value = payload["relevance"]
    if not isinstance(value, torch.Tensor):
        raise ContractError("relevance artifact relevance value must be a tensor")
    return value


def _metric_cutoffs(metrics: tuple[str, ...], candidate_count: int) -> tuple[int, ...]:
    values = {
        int(match.group(1))
        for metric in metrics
        if (match := _CUTOFF.fullmatch(metric)) is not None
    }
    oversized = tuple(sorted(value for value in values if value > candidate_count))
    if oversized:
        raise ContractError(f"metric cutoffs exceed candidate count {candidate_count}: {oversized}")
    return tuple(sorted(values or {candidate_count}))


def _evaluation_slices(
    per_query: dict[str, torch.Tensor],
    metadata: tuple[Mapping[str, Any], ...],
    configured_slices: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    if not configured_slices:
        return {}
    missing = tuple(
        field for field in configured_slices if any(field not in values for values in metadata)
    )
    if missing:
        raise ContractError(f"encoded query metadata is missing evaluation slices: {missing}")
    masks = build_slice_masks(
        {field: tuple(values[field] for values in metadata) for field in configured_slices}
    )
    return {name: dict(values) for name, values in slice_metrics(per_query, masks).items()}


def _metric_result_name(name: str) -> str:
    return name.replace("hitrate@", "hit_rate@")


def _per_query_result_name(name: str) -> str:
    if name == "map":
        return "average_precision"
    return _metric_result_name(name)
