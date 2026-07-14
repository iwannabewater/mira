"""Deterministic exact-search reference benchmark."""

from __future__ import annotations

import time
from pathlib import Path  # noqa: TC003 - Typer resolves command annotations at runtime
from typing import Annotated

import torch
import typer

from mira.cli._common import emit_json, write_json_file
from mira.config import resolve_config
from mira.evaluation import ExactSearchEvaluator
from mira.observability import analyze_embedding_health


def benchmark_command(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    queries: Annotated[int, typer.Option(min=1)] = 64,
    candidates: Annotated[int, typer.Option(min=1)] = 512,
) -> None:
    """Benchmark the CPU exact-search reference at the configured dimension."""
    resolved = resolve_config(config)
    dimension = resolved.config.model.hidden_size or max(resolved.config.evaluation.dimensions)
    generator = torch.Generator().manual_seed(resolved.config.project.seed)
    query_embeddings = torch.randn(queries, dimension, generator=generator)
    candidate_embeddings = torch.randn(candidates, dimension, generator=generator)
    start = time.perf_counter()
    result = ExactSearchEvaluator("cosine").search(
        query_embeddings,
        candidate_embeddings,
        k=min(10, candidates),
    )
    elapsed = time.perf_counter() - start
    health = analyze_embedding_health(candidate_embeddings)
    report = {
        "candidate_count": candidates,
        "config_checksum": resolved.checksum,
        "device": "cpu",
        "dimension": dimension,
        "elapsed_seconds": elapsed,
        "queries_per_second": queries / elapsed,
        "query_count": queries,
        "result_checksum": float(result.scores.sum().item()),
        "status": "measured",
        "candidate_health": {
            "collapse_score": health.collapse_score,
            "effective_rank": health.effective_rank,
            "norm_mean": health.norm_mean,
        },
    }
    if output is not None:
        write_json_file(report, output)
    emit_json(report)
