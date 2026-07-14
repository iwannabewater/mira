"""Strict streaming I/O for the versioned MIRA JSONL format."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from mira.core.serialization import strict_json_loads
from mira.data.errors import JsonlDataError
from mira.data.schema import EmbeddingRecord

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class MiraJsonlReader:
    """Reusable strict reader for one MIRA JSONL path."""

    def __init__(self, path: str | Path) -> None:
        """Bind the reader to one filesystem path."""
        self.path = Path(path)

    def __iter__(self) -> Iterator[EmbeddingRecord]:
        """Yield strict records from the bound path."""
        return iter_jsonl(self.path)

    def read(self) -> tuple[EmbeddingRecord, ...]:
        """Read and validate all records."""
        return read_jsonl(self.path)


def iter_jsonl(path: str | Path) -> Iterator[EmbeddingRecord]:
    """Yield validated records from ``path``.

    Raises:
        JsonlDataError: If a line is blank, malformed, invalid, or reuses a
            previous ``sample_id``.
    """
    source = Path(path)
    seen_sample_ids: set[str] = set()
    try:
        handle = source.open(encoding="utf-8")
    except OSError as error:
        raise JsonlDataError(source, 0, f"cannot open dataset: {error}") from error

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise JsonlDataError(source, line_number, "blank JSONL lines are not allowed")
            try:
                payload = strict_json_loads(line, context="JSONL record")
            except json.JSONDecodeError as error:
                raise JsonlDataError(
                    source,
                    line_number,
                    f"invalid JSON at column {error.colno}: {error.msg}",
                ) from error
            except ValueError as error:
                raise JsonlDataError(source, line_number, f"invalid JSON: {error}") from error
            if not isinstance(payload, dict):
                raise JsonlDataError(source, line_number, "each line must be a JSON object")
            try:
                record = EmbeddingRecord.model_validate(payload)
            except (ValidationError, ValueError) as error:
                raise JsonlDataError(source, line_number, str(error)) from error
            if record.sample_id in seen_sample_ids:
                raise JsonlDataError(
                    source, line_number, f"duplicate sample_id {record.sample_id!r}"
                )
            seen_sample_ids.add(record.sample_id)
            yield record


def read_jsonl(path: str | Path) -> tuple[EmbeddingRecord, ...]:
    """Read and validate an entire MIRA JSONL dataset."""
    return tuple(iter_jsonl(path))


def write_jsonl(
    records: Iterable[EmbeddingRecord], path: str | Path, *, overwrite: bool = False
) -> None:
    """Write records in canonical compact UTF-8 JSONL form.

    Raises:
        FileExistsError: If the destination exists and ``overwrite`` is false.
        ValueError: If ``records`` contains duplicate sample IDs.
    """
    destination = Path(path)
    seen_sample_ids: set[str] = set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                if record.sample_id in seen_sample_ids:
                    raise ValueError(f"duplicate sample_id {record.sample_id!r}")
                seen_sample_ids.add(record.sample_id)
                handle.write(
                    json.dumps(
                        record.model_dump(mode="python", exclude_none=True),
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            temporary.replace(destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise FileExistsError(destination) from None
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
