"""Validate complete datasets and build representative group layouts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mira.cli._common import emit_json
from mira.cli._media import validate_record_media
from mira.config import resolve_config
from mira.core.exceptions import DataValidationError
from mira.data import build_group_layout, read_jsonl

if TYPE_CHECKING:
    from mira.data.schema import EmbeddingRecord


def validate_data_command(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
    check_media: Annotated[  # noqa: FBT002 - Typer boolean flag contract
        bool,
        typer.Option("--check-media/--skip-media", help="Check local media existence/checksums."),
    ] = True,
) -> None:
    """Validate every configured JSONL record and its local media references."""
    resolved = resolve_config(config, overrides=tuple(overrides or ()))
    reports: list[dict[str, object]] = []
    all_records: list[EmbeddingRecord] = []
    sources: list[tuple[str, str | None, Path]] = [
        (source.name, source.task_id, Path(source.path)) for source in resolved.config.data.train
    ]
    validation = resolved.config.data.validation
    sources.append((validation.name or "validation", validation.task_id, Path(validation.path)))
    for name, task_id, path in sources:
        records = read_jsonl(path)
        if not records:
            raise DataValidationError(f"dataset {name!r} contains no records")
        if task_id is not None:
            mismatched = tuple(record.sample_id for record in records if record.task_id != task_id)
            if mismatched:
                raise DataValidationError(
                    f"dataset {name!r} contains task IDs different from {task_id!r}: {mismatched}"
                )
        if check_media:
            for record in records:
                validate_record_media(record, path.parent)
        representative = records[: resolved.config.runtime.train_batch_size_per_device]
        layout = build_group_layout(representative).layout
        reports.append(
            {
                "candidates_in_representative_batch": layout.num_candidates,
                "name": name,
                "path": str(path),
                "records": len(records),
                "representative_queries": layout.num_queries,
            }
        )
        all_records.extend(records)
    sample_ids = [record.sample_id for record in all_records]
    if len(set(sample_ids)) != len(sample_ids):
        raise DataValidationError("sample_id values must be unique across configured datasets")
    emit_json(
        {
            "config_checksum": resolved.checksum,
            "datasets": reports,
            "records": len(all_records),
            "status": "valid",
        }
    )
