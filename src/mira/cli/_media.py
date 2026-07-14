"""Shared fail-fast validation for local multimodal assets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from mira.core.exceptions import DataValidationError
from mira.core.filesystem import open_confined_regular_file

if TYPE_CHECKING:
    from mira.data.schema import EmbeddingRecord, MediaRef, MultimodalItem

_HASH_CHUNK_SIZE = 1024 * 1024


def validate_record_media(record: EmbeddingRecord, base: Path) -> None:
    """Validate every local media reference in one retrieval record."""
    resolve_record_media(record, base)


def validate_item_media(item: MultimodalItem, base: Path) -> None:
    """Validate every local media reference in one independently encoded item."""
    resolve_item_media(item, base)


def validate_media_reference(reference: MediaRef, base: Path) -> None:
    """Check local existence and an optional tagged SHA-256 checksum."""
    _resolve_media_reference(reference, base)


def resolve_record_media(record: EmbeddingRecord, base: Path) -> EmbeddingRecord:
    """Return a record whose verified media paths are canonical absolute paths."""
    return record.model_copy(
        update={
            "query": resolve_item_media(record.query, base),
            "positives": tuple(resolve_item_media(item, base) for item in record.positives),
            "negatives": tuple(resolve_item_media(item, base) for item in record.negatives),
        }
    )


def resolve_item_media(item: MultimodalItem, base: Path) -> MultimodalItem:
    """Return an item whose verified media paths are canonical absolute paths."""
    return item.model_copy(
        update={
            field: tuple(_resolve_media_reference(reference, base) for reference in references)
            for field, references in (
                ("images", item.images),
                ("videos", item.videos),
                ("audios", item.audios),
            )
        }
    )


def _resolve_media_reference(reference: MediaRef, base: Path) -> MediaRef:
    parsed = urlparse(reference.uri)
    if parsed.scheme:
        raise DataValidationError(
            f"remote and data media URIs are disabled in MIRA 0.1: {reference.uri}"
        )
    path = Path(reference.uri)
    if path.is_absolute():
        raise DataValidationError(f"absolute media paths are disabled: {path}")
    digest = hashlib.sha256()
    try:
        stream, canonical_path = open_confined_regular_file(base, path)
        with stream:
            for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as error:
        raise DataValidationError(
            "local media path is missing, non-regular, symlinked, or outside the dataset directory"
        ) from error
    observed = f"sha256:{digest.hexdigest()}"
    if reference.checksum is not None and observed != reference.checksum:
        raise DataValidationError(
            f"media checksum mismatch for {canonical_path}: "
            f"expected {reference.checksum}, got {observed}"
        )
    return reference.model_copy(update={"uri": str(canonical_path), "checksum": observed})
