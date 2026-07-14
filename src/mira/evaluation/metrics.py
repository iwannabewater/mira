"""Retrieval metrics for binary, multi-positive, and graded relevance."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mira.core.exceptions import ConfigurationError, NumericalError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_RELEVANCE_RANK = 2


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Aggregate metrics and their query-level contributions."""

    metrics: Mapping[str, float]
    per_query: Mapping[str, Tensor]


def compute_retrieval_metrics(  # noqa: PLR0915 - formulas stay together for review
    ranked_relevance: Tensor,
    *,
    ks: Sequence[int],
    ideal_relevance: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> MetricResult:
    """Compute retrieval metrics from a complete ranked relevance matrix.

    Relevance is binary when values are 0 or 1 and graded when positive values
    carry different gains. Multiple positive candidates are supported. mAP,
    mean rank, and median rank require a complete ranking: every relevant item
    in ``ideal_relevance`` must appear in ``ranked_relevance``.

    Args:
        ranked_relevance: Relevance ordered by predicted rank, shaped ``[Q, C]``.
        ks: Positive cutoffs no larger than the ranking width.
        ideal_relevance: Unordered relevance for the evaluated candidate pool.
            When omitted, the complete ranked matrix is also used as the pool.
        valid_mask: Boolean mask for real ranked entries. Invalid padded entries
            are excluded from rank positions and precision denominators.

    Returns:
        Aggregate and per-query Recall, Precision, HitRate, MRR, NDCG, AP, and
        first-positive rank values.

    Raises:
        ConfigurationError: If shapes, cutoffs, relevance, or completeness fail.
        NumericalError: If relevance contains NaN or infinite values.
    """
    relevance = _validate_relevance("ranked_relevance", ranked_relevance)
    _, ranking_width = relevance.shape
    cutoffs = _validate_cutoffs(ks, ranking_width)
    if ideal_relevance is not None and ideal_relevance.device != relevance.device:
        raise ConfigurationError("ideal and ranked relevance must share a device")
    ideal = (
        relevance
        if ideal_relevance is None
        else _validate_relevance("ideal_relevance", ideal_relevance)
    )
    if ideal.shape != relevance.shape:
        raise ConfigurationError(
            "ideal_relevance must match the complete ranked_relevance shape; "
            f"got {tuple(ideal.shape)} and {tuple(relevance.shape)}"
        )
    valid = _validate_valid_mask(valid_mask, relevance)
    binary = (relevance > 0) & valid
    ideal_binary = ideal > 0
    total_relevant = ideal_binary.sum(dim=1)
    if bool((total_relevant == 0).any().item()):
        missing_rows = torch.nonzero(total_relevant == 0, as_tuple=False).flatten().tolist()
        raise ConfigurationError(
            f"queries without relevant candidates are not evaluable: {missing_rows}"
        )

    retrieved_relevant = binary.sum(dim=1)
    if bool((retrieved_relevant != total_relevant).any().item()):
        incomplete_rows = torch.nonzero(
            retrieved_relevant != total_relevant, as_tuple=False
        ).flatten()
        details = tuple(
            (
                int(row.item()),
                int(retrieved_relevant[row].item()),
                int(total_relevant[row].item()),
            )
            for row in incomplete_rows
        )
        raise ConfigurationError(
            "complete rankings must contain every relevant candidate; "
            f"query/retrieved/expected={details}"
        )

    dtype = relevance.dtype
    binary_float = binary.to(dtype=dtype)
    valid_float = valid.to(dtype=dtype)
    cumulative_hits = binary_float.cumsum(dim=1)
    effective_positions = valid_float.cumsum(dim=1)
    per_query: dict[str, Tensor] = {}

    gains = torch.where(valid, torch.exp2(relevance) - 1, torch.zeros_like(relevance))
    discounts = torch.log2(torch.arange(2, ranking_width + 2, dtype=dtype, device=relevance.device))
    ideal_sorted = torch.sort(ideal, dim=1, descending=True, stable=True).values
    ideal_gains = torch.exp2(ideal_sorted) - 1
    if not bool(torch.isfinite(gains).all().item()) or not bool(
        torch.isfinite(ideal_gains).all().item()
    ):
        raise NumericalError("graded relevance is too large for finite NDCG gain")

    for cutoff in cutoffs:
        end = cutoff - 1
        hits_at_k = cumulative_hits[:, end]
        valid_at_k = effective_positions[:, end]
        recall = hits_at_k / total_relevant.to(dtype=dtype)
        precision = hits_at_k / valid_at_k.clamp_min(1)
        hit_rate = (hits_at_k > 0).to(dtype=dtype)

        first_hit = _first_relevant_rank(binary[:, :cutoff], valid[:, :cutoff])
        reciprocal_rank = torch.where(
            torch.isfinite(first_hit), first_hit.reciprocal(), torch.zeros_like(first_hit)
        )
        dcg = (gains[:, :cutoff] / discounts[:cutoff]).sum(dim=1)
        ideal_dcg = (ideal_gains[:, :cutoff] / discounts[:cutoff]).sum(dim=1)
        ndcg = dcg / ideal_dcg.clamp_min(torch.finfo(dtype).tiny)

        per_query[f"recall@{cutoff}"] = recall
        per_query[f"precision@{cutoff}"] = precision
        per_query[f"hit_rate@{cutoff}"] = hit_rate
        per_query[f"mrr@{cutoff}"] = reciprocal_rank
        per_query[f"ndcg@{cutoff}"] = ndcg

    precision_at_rank = cumulative_hits / effective_positions.clamp_min(1)
    average_precision = (precision_at_rank * binary_float).sum(dim=1) / total_relevant.to(
        dtype=dtype
    )
    first_relevant_rank = _first_relevant_rank(binary, valid)
    per_query["average_precision"] = average_precision
    per_query["first_relevant_rank"] = first_relevant_rank

    metrics = {name: float(values.mean().item()) for name, values in per_query.items()}
    del metrics["average_precision"]
    del metrics["first_relevant_rank"]
    metrics["map"] = float(average_precision.mean().item())
    metrics["mean_rank"] = float(first_relevant_rank.mean().item())
    metrics["median_rank"] = float(first_relevant_rank.quantile(0.5).item())
    return MetricResult(
        metrics=MappingProxyType(metrics),
        per_query=MappingProxyType(per_query),
    )


def _validate_relevance(name: str, value: Tensor) -> Tensor:
    if value.ndim != _RELEVANCE_RANK or value.shape[0] == 0 or value.shape[1] == 0:
        raise ConfigurationError(f"{name} must be a non-empty [Q, C] tensor")
    if value.dtype is torch.bool:
        result = value.to(dtype=torch.float64)
    elif value.is_floating_point():
        result = value
    else:
        result = value.to(dtype=torch.float64)
    if not bool(torch.isfinite(result).all().item()):
        raise NumericalError(f"{name} contains NaN or infinite values")
    if bool((result < 0).any().item()):
        raise ConfigurationError(f"{name} must contain non-negative relevance values")
    return result


def _validate_cutoffs(ks: Sequence[int], width: int) -> tuple[int, ...]:
    if not ks:
        raise ConfigurationError("at least one retrieval cutoff is required")
    cutoffs = tuple(sorted(set(ks)))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in cutoffs):
        raise ConfigurationError("retrieval cutoffs must be integers")
    if cutoffs[0] <= 0 or cutoffs[-1] > width:
        raise ConfigurationError(f"retrieval cutoffs must be within [1, {width}], got {cutoffs}")
    return cutoffs


def _validate_valid_mask(mask: Tensor | None, relevance: Tensor) -> Tensor:
    if mask is None:
        return torch.ones_like(relevance, dtype=torch.bool)
    if mask.dtype is not torch.bool or mask.shape != relevance.shape:
        raise ConfigurationError(
            "valid_mask must be boolean and match ranked_relevance shape; "
            f"got dtype={mask.dtype}, shape={tuple(mask.shape)}"
        )
    if mask.device != relevance.device:
        raise ConfigurationError("valid_mask and ranked_relevance must share a device")
    if bool((mask.sum(dim=1) == 0).any().item()):
        rows = torch.nonzero(mask.sum(dim=1) == 0, as_tuple=False).flatten().tolist()
        raise ConfigurationError(f"rankings contain no valid candidates for queries {rows}")
    return mask


def _first_relevant_rank(binary: Tensor, valid: Tensor) -> Tensor:
    effective_positions = valid.to(dtype=torch.float64).cumsum(dim=1)
    infinity = torch.full_like(effective_positions, float("inf"))
    return torch.where(binary, effective_positions, infinity).min(dim=1).values
