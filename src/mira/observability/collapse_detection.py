"""Reference embedding-health diagnostics with explicit alert thresholds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from mira.core.exceptions import InvalidEmbeddingError

_MATRIX_RANK = 2


@dataclass(frozen=True, slots=True)
class CollapseThresholds:
    """Thresholds that convert continuous health metrics into alerts."""

    minimum_dimension_std: float = 1e-4
    minimum_effective_rank_ratio: float = 0.1
    maximum_pairwise_cosine: float = 0.98
    maximum_norm_coefficient_of_variation: float = 1.0

    def __post_init__(self) -> None:
        """Validate finite, non-negative thresholds."""
        for name in (
            "minimum_dimension_std",
            "minimum_effective_rank_ratio",
            "maximum_pairwise_cosine",
            "maximum_norm_coefficient_of_variation",
        ):
            value = _finite_threshold(getattr(self, name), name=name)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class EmbeddingHealthReport:
    """Scalar embedding diagnostics, spectrum, and triggered alerts."""

    norm_mean: float
    norm_std: float
    dimension_std_mean: float
    dimension_std_min: float
    pairwise_cosine_mean: float
    pairwise_cosine_std: float
    effective_rank: float
    effective_rank_ratio: float
    collapse_score: float
    singular_values: tuple[float, ...]
    alerts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _HealthScalars:
    norm_mean: float
    norm_std: float
    dimension_std_min: float
    effective_rank_ratio: float
    pairwise_cosine_mean: float


def analyze_embedding_health(
    embeddings: Tensor,
    *,
    thresholds: CollapseThresholds | None = None,
) -> EmbeddingHealthReport:
    """Compute finite float32 diagnostics without changing caller gradients."""
    if embeddings.ndim != _MATRIX_RANK or min(embeddings.shape) <= 0:
        raise InvalidEmbeddingError("health analysis requires non-empty [items, dimension]")
    if not embeddings.is_floating_point():
        raise InvalidEmbeddingError("health analysis requires floating-point embeddings")
    values = embeddings.detach().float()
    if not torch.isfinite(values).all().item():
        raise InvalidEmbeddingError("health analysis received NaN or infinite embeddings")
    norms = torch.linalg.vector_norm(values, dim=1)
    if not torch.isfinite(norms).all().item():
        raise InvalidEmbeddingError("health analysis received a non-finite embedding L2 norm")
    if torch.any(norms == 0).item():
        raise InvalidEmbeddingError("health analysis received a zero-norm embedding")
    dimension_std = values.std(dim=0, unbiased=False)
    if not torch.isfinite(dimension_std).all().item():
        raise InvalidEmbeddingError("health analysis produced non-finite dimension statistics")
    normalized = values / norms.unsqueeze(1)
    pairwise = _off_diagonal(normalized @ normalized.transpose(0, 1))
    if not torch.isfinite(pairwise).all().item():
        raise InvalidEmbeddingError("health analysis produced non-finite cosine statistics")
    centered = values - values.mean(dim=0, keepdim=True)
    if not torch.isfinite(centered).all().item():
        raise InvalidEmbeddingError("health analysis produced non-finite centered embeddings")
    singular_values = torch.linalg.svdvals(centered)
    if not torch.isfinite(singular_values).all().item():
        raise InvalidEmbeddingError("health analysis produced non-finite singular values")
    effective_rank = _effective_rank(singular_values)
    maximum_rank = max(1, min(max(values.shape[0] - 1, 1), values.shape[1]))
    effective_rank_ratio = effective_rank / maximum_rank
    cosine_mean = float(pairwise.mean().item()) if pairwise.numel() else 1.0
    cosine_std = float(pairwise.std(unbiased=False).item()) if pairwise.numel() else 0.0
    collapse_score = max(
        0.0,
        min(1.0, 0.5 * (1.0 - effective_rank_ratio) + 0.5 * max(cosine_mean, 0.0)),
    )
    resolved = thresholds or CollapseThresholds()
    norm_mean = float(norms.mean().item())
    norm_std = float(norms.std(unbiased=False).item())
    dimension_std_mean = float(dimension_std.mean().item())
    dimension_std_min = float(dimension_std.min().item())
    reported_scalars = (
        norm_mean,
        norm_std,
        dimension_std_mean,
        dimension_std_min,
        cosine_mean,
        cosine_std,
        effective_rank,
        effective_rank_ratio,
        collapse_score,
    )
    if not all(math.isfinite(value) for value in reported_scalars):
        raise InvalidEmbeddingError("health analysis produced a non-finite report")
    scalars = _HealthScalars(
        norm_mean,
        norm_std,
        dimension_std_min,
        effective_rank_ratio,
        cosine_mean,
    )
    alerts = _alerts(resolved, scalars)
    return EmbeddingHealthReport(
        norm_mean=norm_mean,
        norm_std=norm_std,
        dimension_std_mean=dimension_std_mean,
        dimension_std_min=dimension_std_min,
        pairwise_cosine_mean=cosine_mean,
        pairwise_cosine_std=cosine_std,
        effective_rank=effective_rank,
        effective_rank_ratio=effective_rank_ratio,
        collapse_score=collapse_score,
        singular_values=tuple(float(value.item()) for value in singular_values),
        alerts=alerts,
    )


def _off_diagonal(matrix: Tensor) -> Tensor:
    if matrix.shape[0] <= 1:
        return matrix.new_empty(0)
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def _effective_rank(singular_values: Tensor) -> float:
    power = singular_values.double().square()
    total = power.sum()
    if total <= torch.finfo(power.dtype).eps:
        return 0.0
    probabilities = power / total
    nonzero = probabilities[probabilities > 0]
    entropy = -(nonzero * nonzero.log()).sum()
    return float(entropy.exp().item())


def _alerts(
    thresholds: CollapseThresholds,
    values: _HealthScalars,
) -> tuple[str, ...]:
    alerts: list[str] = []
    if values.dimension_std_min < thresholds.minimum_dimension_std:
        alerts.append("dimension_variance_below_threshold")
    if values.effective_rank_ratio < thresholds.minimum_effective_rank_ratio:
        alerts.append("effective_rank_below_threshold")
    if values.pairwise_cosine_mean > thresholds.maximum_pairwise_cosine:
        alerts.append("pairwise_cosine_above_threshold")
    coefficient = values.norm_std / values.norm_mean
    if coefficient > thresholds.maximum_norm_coefficient_of_variation:
        alerts.append("norm_variation_above_threshold")
    return tuple(alerts)


def _finite_threshold(value: object, *, name: str) -> float:
    """Return one finite threshold without accepting bool or string coercion."""
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be a built-in real number, not {type(value).__name__}")
    try:
        normalized = float(cast("int | float", value))
    except OverflowError as error:
        raise ValueError("collapse thresholds must be finite and non-negative") from error
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError("collapse thresholds must be finite and non-negative")
    return normalized
