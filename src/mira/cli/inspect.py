"""Inspect a fully resolved experiment configuration."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Typer resolves command annotations at runtime
from typing import Annotated

import typer

from mira.cli._common import emit_json
from mira.config import resolve_config


def inspect_command(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    overrides: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override one dotted config path as YAML."),
    ] = None,
) -> None:
    """Resolve, validate, and summarize an experiment without loading a model."""
    resolved = resolve_config(config, overrides=tuple(overrides or ()))
    value = resolved.config
    emit_json(
        {
            "checksum": resolved.checksum,
            "datasets": [source.name for source in value.data.train],
            "model": value.model.name_or_path,
            "objectives": [objective.type for objective in value.objectives],
            "project": value.project.name,
            "schema_version": value.schema_version,
            "source_files": [str(path) for path in resolved.source_files],
            "status": "valid",
        }
    )
