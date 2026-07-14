"""Data-layer exceptions with actionable record context."""

from __future__ import annotations

from pathlib import Path

from mira.core.exceptions import DataValidationError, MediaValidationError


class DataError(DataValidationError):
    """Base class for invalid MIRA data or data-pipeline state."""


class RecordValidationError(DataError):
    """Raised when a retrieval record violates a semantic invariant."""


class PlaceholderAlignmentError(RecordValidationError, MediaValidationError):
    """Raised when message placeholders and media references do not align."""


class SwiftConversionError(DataError):
    """Raised when a Swift sample cannot be converted without ambiguity or loss."""


class SamplerStateError(DataError):
    """Raised when deterministic sampler state cannot be restored exactly."""


class JsonlDataError(DataError):
    """An error tied to an exact JSONL source location."""

    def __init__(self, path: str | Path, line_number: int, detail: str) -> None:
        """Attach the source path, one-based line number, and validation detail."""
        self.path = Path(path)
        self.line_number = line_number
        self.detail = detail
        super().__init__(f"{self.path}:{self.line_number}: {detail}")
