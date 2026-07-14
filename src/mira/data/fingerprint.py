"""Portable semantic fingerprints for validated retrieval records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mira.core.exceptions import DataValidationError

if TYPE_CHECKING:
    from mira.data.schema import EmbeddingRecord


def canonical_record_payload(
    record: EmbeddingRecord,
    *,
    media_root: Path | None = None,
) -> dict[str, Any]:
    """Return a JSON payload with host-specific media roots removed."""
    # Keep opaque values in their original Python representation until the
    # strict ``json.dumps(..., allow_nan=False)`` fingerprint boundary. Pydantic
    # JSON mode rewrites non-finite values to ``None`` before that guard runs.
    payload = record.model_dump(mode="python")
    if media_root is None:
        return payload
    root = media_root.resolve(strict=True)
    for item in (payload["query"], *payload["positives"], *payload["negatives"]):
        for field in ("images", "videos", "audios"):
            for reference in item[field]:
                uri = Path(reference["uri"])
                if uri.is_absolute():
                    resolved_uri = uri.resolve(strict=False)
                    try:
                        reference["uri"] = resolved_uri.relative_to(root).as_posix()
                    except ValueError as error:
                        raise DataValidationError(
                            f"resolved media path is outside fingerprint root: {resolved_uri}"
                        ) from error
    return payload


def records_checksum(
    records: tuple[EmbeddingRecord, ...],
    *,
    media_root: Path | None = None,
) -> str:
    """Hash ordered record semantics, tagged media digests, and portable paths."""
    canonical = json.dumps(
        [canonical_record_payload(record, media_root=media_root) for record in records],
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()
