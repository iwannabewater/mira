"""Composable false-negative filters for offline retrieval candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mira.core.exceptions import ConfigurationValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

_DEFAULT_IDENTITY_KEYS = (
    "canonical_product_id",
    "spu_id",
    "sku_id",
    "product_family_id",
    "image_hash_group",
    "duplicate_listing_group",
)


@dataclass(frozen=True, slots=True)
class MetadataFalseNegativeFilter:
    """Reject candidates sharing any declared non-empty identity value."""

    keys: tuple[str, ...] = _DEFAULT_IDENTITY_KEYS

    def __post_init__(self) -> None:
        """Require unique, non-empty metadata keys."""
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ConfigurationValidationError("metadata filter keys must be non-empty and unique")
        if any(not key.strip() for key in self.keys):
            raise ConfigurationValidationError("metadata filter keys must not be blank")

    def keep(self, query: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
        """Return false when metadata identifies a likely positive."""
        for key in self.keys:
            query_value = query.get(key)
            if query_value is not None and query_value == candidate.get(key):
                return False
        return True


@dataclass(frozen=True, slots=True)
class PositiveAwareSimilarityFilter:
    """Require candidate similarity to stay below ``alpha * best_positive``."""

    alpha: float

    def __post_init__(self) -> None:
        """Validate a finite positive similarity ratio."""
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ConfigurationValidationError("positive-aware alpha must be finite and positive")

    def keep(self, candidate_similarity: float, best_positive_similarity: float) -> bool:
        """Apply the explicit positive-relative threshold."""
        if not math.isfinite(candidate_similarity) or not math.isfinite(best_positive_similarity):
            raise ValueError("similarity filter inputs must be finite")
        return candidate_similarity < self.alpha * best_positive_similarity


@dataclass(frozen=True, slots=True)
class TeacherScoreFilter:
    """Reject candidates that a teacher rates above a relevance threshold."""

    maximum_relevance: float

    def __post_init__(self) -> None:
        """Require a finite teacher threshold."""
        if not math.isfinite(self.maximum_relevance):
            raise ConfigurationValidationError("teacher threshold must be finite")

    def keep(self, teacher_score: float) -> bool:
        """Return true only for candidates at or below the threshold."""
        if not math.isfinite(teacher_score):
            raise ValueError("teacher score must be finite")
        return teacher_score <= self.maximum_relevance
