"""Migrate a legacy encoded tensor bundle into the v2 artifact contract."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Typer resolves command annotations at runtime
from typing import Annotated, Literal

import typer

from mira.cli._common import emit_json, load_encoded_collection
from mira.core.exceptions import ContractError
from mira.inference import write_encoded_artifact


def migrate_encoded_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, readable=True, help="Legacy v1 tensor bundle."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New mira.encoded.v2 artifact directory."),
    ],
    role: Annotated[
        Literal["query", "document"],
        typer.Option("--role", help="Retrieval-axis role to bind into the v2 manifest."),
    ],
    normalization: Annotated[
        Literal["normalized", "unnormalized"],
        typer.Option(
            "--normalization",
            help="Explicit vector normalization semantics for the legacy bundle.",
        ),
    ],
    shard_size: Annotated[
        int,
        typer.Option("--shard-size", min=1, help="Maximum vectors per v2 shard."),
    ] = 4096,
) -> None:
    """Convert a safe legacy bundle while making missing v2 semantics explicit."""
    if input_path.is_dir():
        raise ContractError("migration input must be a legacy v1 file, not a v2 directory")
    collection = load_encoded_collection(input_path)
    normalized = normalization == "normalized"
    claimed_rule = collection.provenance.get("normalization_rule")
    if claimed_rule == "l2_normalized_inner_product" and not normalized:
        raise ContractError(
            "legacy provenance claims L2-normalized vectors; use --normalization normalized"
        )
    summary = write_encoded_artifact(
        collection,
        output,
        role=role,
        normalized=normalized,
        shard_size=shard_size,
    )
    emit_json(
        {
            "dimension": summary.embedding_dimension,
            "items": summary.item_count,
            "manifest_sha256": summary.manifest_sha256,
            "normalization": normalization,
            "output": str(summary.path),
            "role": role,
            "shards": summary.shards,
            "status": "complete",
        }
    )
