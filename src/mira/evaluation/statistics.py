"""Query-level slicing and bootstrap uncertainty estimates."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mira.core.exceptions import ConfigurationError, NumericalError

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

_MIN_BOOTSTRAP_VALUES = 2


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Percentile bootstrap confidence interval for a mean."""

    estimate: float
    low: float
    high: float
    confidence: float
    n_resamples: int


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """Paired bootstrap estimate for ``candidate - baseline``."""

    delta: float
    low: float
    high: float
    p_value: float
    confidence: float
    n_resamples: int


def bootstrap_confidence_interval(
    values: Tensor,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Estimate a percentile confidence interval over query means."""
    samples = _bootstrap_means(
        _validate_values("values", values), n_resamples=n_resamples, seed=seed
    )
    low, high = _interval(samples, confidence)
    return ConfidenceInterval(
        estimate=float(values.to(dtype=torch.float64).mean().item()),
        low=low,
        high=high,
        confidence=confidence,
        n_resamples=n_resamples,
    )


def paired_bootstrap(
    baseline: Tensor,
    candidate: Tensor,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Compare aligned query-level values with a paired bootstrap.

    The two-sided p-value uses the bootstrap sign distribution with a one-sample
    correction, so it never reports an impossible zero probability.
    """
    baseline_values = _validate_values("baseline", baseline)
    candidate_values = _validate_values("candidate", candidate)
    if baseline_values.shape != candidate_values.shape:
        raise ConfigurationError(
            "paired bootstrap inputs must have identical query alignment; "
            f"got {tuple(baseline_values.shape)} and {tuple(candidate_values.shape)}"
        )
    differences = candidate_values - baseline_values
    samples = _bootstrap_means(differences, n_resamples=n_resamples, seed=seed)
    low, high = _interval(samples, confidence)
    non_positive = int((samples <= 0).sum().item())
    non_negative = int((samples >= 0).sum().item())
    p_value = min(
        1.0,
        2
        * min(
            (non_positive + 1) / (n_resamples + 1),
            (non_negative + 1) / (n_resamples + 1),
        ),
    )
    return PairedBootstrapResult(
        delta=float(differences.mean().item()),
        low=low,
        high=high,
        p_value=p_value,
        confidence=confidence,
        n_resamples=n_resamples,
    )


def build_slice_masks(
    metadata: Mapping[str, Sequence[Hashable]],
) -> Mapping[str, Tensor]:
    """Build deterministic ``field=value`` query masks from metadata columns."""
    if not metadata:
        raise ConfigurationError("slice metadata must not be empty")
    lengths = {len(values) for values in metadata.values()}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ConfigurationError("slice metadata columns must have one non-zero length")
    masks: dict[str, Tensor] = {}
    for field in sorted(metadata):
        values = metadata[field]
        unique = sorted(set(values), key=lambda item: (type(item).__name__, repr(item)))
        for value in unique:
            masks[f"{field}={value}"] = torch.tensor(
                tuple(item == value for item in values), dtype=torch.bool
            )
    return MappingProxyType(masks)


def slice_metrics(
    per_query: Mapping[str, Tensor],
    slice_masks: Mapping[str, Tensor],
) -> Mapping[str, Mapping[str, float]]:
    """Aggregate query-level retrieval evidence for named slices."""
    if not per_query or not slice_masks:
        raise ConfigurationError("per_query metrics and slice_masks must not be empty")
    invalid_metrics = tuple(
        name
        for name, values in per_query.items()
        if values.ndim != 1 or not bool(torch.isfinite(values).all().item())
    )
    if invalid_metrics:
        raise NumericalError(f"per-query metrics must be finite rank-1 tensors: {invalid_metrics}")
    query_counts = {int(values.numel()) for values in per_query.values()}
    if len(query_counts) != 1:
        raise ConfigurationError("all per-query metric vectors must have equal length")
    query_count = next(iter(query_counts))
    result: dict[str, Mapping[str, float]] = {}
    for name, mask in slice_masks.items():
        if mask.dtype is not torch.bool or mask.ndim != 1 or mask.numel() != query_count:
            raise ConfigurationError(
                f"slice {name!r} must be a boolean vector with {query_count} entries"
            )
        if not bool(mask.any().item()):
            raise ConfigurationError(f"slice {name!r} selects no queries")
        aggregates: dict[str, float] = {}
        for metric, values in per_query.items():
            selected = values[mask.to(device=values.device)]
            if metric == "average_precision":
                aggregates["map"] = float(selected.mean().item())
            elif metric == "first_relevant_rank":
                aggregates["mean_rank"] = float(selected.mean().item())
                aggregates["median_rank"] = float(selected.quantile(0.5).item())
            else:
                aggregates[metric] = float(selected.mean().item())
        result[name] = MappingProxyType(aggregates)
    return MappingProxyType(result)


def _validate_values(name: str, values: Tensor) -> Tensor:
    if values.ndim != 1 or values.numel() < _MIN_BOOTSTRAP_VALUES:
        raise ConfigurationError(f"{name} must contain at least two query-level values")
    result = values.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(result).all().item()):
        raise NumericalError(f"{name} contains NaN or infinite values")
    return result


def _bootstrap_means(values: Tensor, *, n_resamples: int, seed: int) -> Tensor:
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, int)
        or n_resamples < _MIN_BOOTSTRAP_VALUES
    ):
        raise ConfigurationError("n_resamples must be an integer of at least 2")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    query_count = values.numel()
    chunks: list[Tensor] = []
    remaining = n_resamples
    while remaining:
        batch_size = min(remaining, 256)
        indices = torch.randint(
            query_count,
            (batch_size, query_count),
            generator=generator,
            device="cpu",
        )
        chunks.append(values[indices].mean(dim=1))
        remaining -= batch_size
    return torch.cat(chunks)


def _interval(samples: Tensor, confidence: float) -> tuple[float, float]:
    if not 0 < confidence < 1:
        raise ConfigurationError("confidence must be strictly between 0 and 1")
    alpha = (1 - confidence) / 2
    bounds = torch.quantile(samples, torch.tensor((alpha, 1 - alpha), dtype=torch.float64))
    return float(bounds[0].item()), float(bounds[1].item())
