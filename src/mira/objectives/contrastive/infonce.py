"""Numerically stable multi-positive InfoNCE over explicit group layouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from mira.core.contracts import ObjectiveResult
from mira.core.exceptions import InvalidEmbeddingError, ObjectiveError
from mira.objectives.base import EmbeddingObjective

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mira.core.contracts import EmbeddingBatch, EmbeddingOutput, ObjectiveContext
    from mira.core.layout import GroupLayout

_EMBEDDING_RANK = 2


@dataclass(frozen=True, slots=True)
class _DirectionalResult:
    loss: Tensor
    metrics: dict[str, Tensor | float]
    diagnostics: dict[str, object]


class MultiPositiveInfoNCE(EmbeddingObjective):
    r"""Multi-positive weighted InfoNCE with optional symmetric direction.

    For every query ``i`` the numerator is
    ``logsumexp(log(w_ip) + s_ip / temperature)`` over explicit positives; the
    denominator is an unweighted ``logsumexp`` over ``layout.candidate_mask``.
    Positive weights are used verbatim and are never normalized.
    """

    def __init__(  # noqa: PLR0913
        self,
        temperature: float = 0.07,
        *,
        learnable_temperature: bool = False,
        min_temperature: float = 1e-4,
        max_temperature: float = 1.0,
        task_temperatures: Mapping[str, float] | None = None,
        symmetric: bool = False,
        logit_clip: float | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        """Configure temperature, normalization, clipping, and directionality."""
        super().__init__()
        _validate_temperature("temperature", temperature)
        _validate_temperature("min_temperature", min_temperature)
        _validate_temperature("max_temperature", max_temperature)
        if min_temperature > max_temperature:
            raise ValueError("min_temperature cannot exceed max_temperature")
        if not min_temperature <= temperature <= max_temperature:
            raise ValueError("temperature must lie within [min_temperature, max_temperature]")
        if logit_clip is not None and (not math.isfinite(logit_clip) or logit_clip <= 0):
            raise ValueError("logit_clip must be finite and positive")
        task_values = dict(task_temperatures or {})
        for task_id, value in task_values.items():
            if not task_id:
                raise ValueError("task temperature IDs must not be empty")
            _validate_temperature(f"task temperature {task_id!r}", value)
            if not min_temperature <= value <= max_temperature:
                raise ValueError(
                    f"task temperature {task_id!r} must lie within "
                    "[min_temperature, max_temperature]"
                )
        if learnable_temperature and task_values:
            raise ValueError(
                "learnable_temperature and fixed task_temperatures are mutually exclusive"
            )

        initial = torch.tensor(math.log(temperature), dtype=torch.float32)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(initial)
        else:
            self.register_buffer("log_temperature", initial)
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.task_temperatures = task_values
        self.symmetric = symmetric
        self.logit_clip = logit_clip
        self.normalize_embeddings = normalize_embeddings

    @property
    def temperature(self) -> Tensor:
        """Return the effective scalar default temperature."""
        return self.log_temperature.exp().clamp(min=self.min_temperature, max=self.max_temperature)

    def forward(
        self,
        output: EmbeddingOutput,
        batch: EmbeddingBatch,
        context: ObjectiveContext,
    ) -> ObjectiveResult:
        """Compute InfoNCE from a flattened model output."""
        del context
        queries, candidates, layout = self.extract_global_embeddings(output, batch)
        return self.compute(queries, candidates, layout)

    def compute(
        self,
        query_embeddings: Tensor,
        candidate_embeddings: Tensor,
        layout: GroupLayout,
    ) -> ObjectiveResult:
        """Compute the loss from explicit query and candidate tensors."""
        queries, candidates = _validate_and_prepare_embeddings(
            query_embeddings,
            candidate_embeddings,
            layout,
            normalize=self.normalize_embeddings,
        )
        similarities = queries @ candidates.transpose(0, 1)
        embedding_norm = torch.cat(
            (
                torch.linalg.vector_norm(query_embeddings.float(), dim=-1),
                torch.linalg.vector_norm(candidate_embeddings.float(), dim=-1),
            )
        ).mean()
        temperatures = self._temperatures(layout.task_ids, similarities.device)
        forward_result = self._direction(
            similarities=similarities,
            positive_mask=layout.positive_mask,
            denominator_mask=layout.candidate_mask,
            positive_weights=layout.positive_weights,
            temperatures=temperatures,
            direction="query_to_candidate",
        )

        loss = forward_result.loss
        metrics = dict(forward_result.metrics)
        diagnostics: dict[str, object] = {
            "query_to_candidate": forward_result.diagnostics,
            "symmetric": self.symmetric,
            "query_shape": tuple(query_embeddings.shape),
            "candidate_shape": tuple(candidate_embeddings.shape),
            "logsumexp_dtype": "float32",
            "positive_weights_normalized": False,
        }
        if self.symmetric:
            reverse_result = self._reverse_direction(similarities, layout, temperatures)
            loss = (loss + reverse_result.loss) * 0.5
            metrics = {key: (metrics[key] + reverse_result.metrics[key]) * 0.5 for key in metrics}
            diagnostics["candidate_to_query"] = reverse_result.diagnostics
        metrics["embedding_norm"] = embedding_norm.detach()
        diagnostics["input_embedding_norm_mean"] = embedding_norm.detach()
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise ObjectiveError("InfoNCE produced a non-scalar or non-finite loss")
        return ObjectiveResult(loss=loss, metrics=metrics, diagnostics=diagnostics)

    def _direction(  # noqa: PLR0913
        self,
        *,
        similarities: Tensor,
        positive_mask: Tensor,
        denominator_mask: Tensor,
        positive_weights: Tensor,
        temperatures: Tensor,
        direction: str,
    ) -> _DirectionalResult:
        if similarities.dtype != torch.float32:
            raise ObjectiveError("InfoNCE similarity computation must use float32")
        shape = tuple(similarities.shape)
        for name, tensor in (
            ("positive_mask", positive_mask),
            ("denominator_mask", denominator_mask),
            ("positive_weights", positive_weights),
        ):
            if tuple(tensor.shape) != shape:
                raise ObjectiveError(f"{direction} {name} shape does not match logits")
            if tensor.device != similarities.device:
                raise ObjectiveError(f"{direction} {name} is on a different device")
        if tuple(temperatures.shape) != (shape[0],):
            raise ObjectiveError(f"{direction} temperatures must have shape [{shape[0]}]")
        if not denominator_mask.any(dim=1).all().item():
            rows = torch.nonzero(~denominator_mask.any(dim=1), as_tuple=False).flatten().tolist()
            raise ObjectiveError(f"{direction} has all candidates masked for rows {rows}")
        effective_positive_mask = positive_mask & denominator_mask
        if not effective_positive_mask.any(dim=1).all().item():
            rows = (
                torch.nonzero(~effective_positive_mask.any(dim=1), as_tuple=False)
                .flatten()
                .tolist()
            )
            raise ObjectiveError(f"{direction} has no valid positive for rows {rows}")
        selected_weights = positive_weights[effective_positive_mask]
        if (
            not torch.isfinite(selected_weights).all().item()
            or torch.any((selected_weights <= 0) | (selected_weights > 1)).item()
        ):
            raise ObjectiveError("positive weights must be finite and satisfy 0 < weight <= 1")

        logits = similarities / temperatures[:, None]
        if self.logit_clip is not None:
            logits = logits.clamp(min=-self.logit_clip, max=self.logit_clip)
        if not torch.isfinite(logits).all().item():
            raise ObjectiveError(f"{direction} logits contain NaN or infinity")
        denominator_logits = logits.masked_fill(~denominator_mask, -torch.inf)
        log_denominator = torch.logsumexp(denominator_logits, dim=1)
        safe_weights = torch.where(effective_positive_mask, positive_weights, 1.0)
        numerator_logits = (logits + safe_weights.log()).masked_fill(
            ~effective_positive_mask, -torch.inf
        )
        log_numerator = torch.logsumexp(numerator_logits, dim=1)
        per_query_loss = log_denominator - log_numerator
        if not torch.isfinite(per_query_loss).all().item():
            raise ObjectiveError(f"{direction} per-query loss contains NaN or infinity")

        negative_mask = denominator_mask & ~positive_mask
        negative_count = negative_mask.sum(dim=1)
        has_negative = negative_count > 0
        positive_mean = similarities[effective_positive_mask].mean()
        negative_mean = _masked_mean_or_zero(similarities, negative_mask)
        hardest_by_row = similarities.masked_fill(~negative_mask, -torch.inf).max(dim=1).values
        hardest_negative = (
            hardest_by_row[has_negative].mean()
            if has_negative.any().item()
            else similarities.new_zeros(())
        )
        best_positive = (
            similarities.masked_fill(~effective_positive_mask, -torch.inf).max(dim=1).values
        )
        margin = (
            (best_positive[has_negative] - hardest_by_row[has_negative]).mean()
            if has_negative.any().item()
            else similarities.new_zeros(())
        )
        metrics: dict[str, Tensor | float] = {
            "positive_similarity": positive_mean.detach(),
            "negative_similarity": negative_mean.detach(),
            "hardest_negative": hardest_negative.detach(),
            "margin": margin.detach(),
            "valid_pair_count": float(denominator_mask.sum().item()),
            "masked_pair_count": float(denominator_mask.numel() - denominator_mask.sum().item()),
            "skipped_query_count": float((~has_negative).sum().item()),
            "temperature": temperatures.mean().detach(),
            "embedding_norm": torch.tensor(1.0, device=similarities.device),
            "nan_inf_count": 0.0,
        }
        diagnostics: dict[str, object] = {
            "per_query_loss": per_query_loss.detach(),
            "positive_count": effective_positive_mask.sum(dim=1).detach(),
            "negative_count": negative_count.detach(),
            "valid_pair_count": int(denominator_mask.sum().item()),
            "masked_pair_count": int(denominator_mask.numel() - denominator_mask.sum().item()),
            "skipped_query_count": int((~has_negative).sum().item()),
            "temperature": temperatures.detach(),
            "logit_min": float(logits.min().detach().item()),
            "logit_max": float(logits.max().detach().item()),
            "finite": True,
        }
        return _DirectionalResult(
            loss=per_query_loss.mean(), metrics=metrics, diagnostics=diagnostics
        )

    def _reverse_direction(
        self, similarities: Tensor, layout: GroupLayout, forward_temperatures: Tensor
    ) -> _DirectionalResult:
        active_candidates = layout.positive_mask.any(dim=0)
        if not active_candidates.any().item():
            raise ObjectiveError("symmetric InfoNCE has no candidate with a positive target")
        reverse_positive = layout.positive_mask.transpose(0, 1)[active_candidates]
        reverse_denominator = layout.candidate_mask.transpose(0, 1)[active_candidates]
        reverse_weights = layout.positive_weights.transpose(0, 1)[active_candidates]
        reverse_temperatures: list[Tensor] = []
        active_columns = torch.nonzero(active_candidates, as_tuple=False).flatten()
        for column in active_columns:
            query_rows = torch.nonzero(layout.positive_mask[:, column], as_tuple=False).flatten()
            candidate_temperatures = forward_temperatures.index_select(0, query_rows)
            if not torch.allclose(
                candidate_temperatures,
                candidate_temperatures[0].expand_as(candidate_temperatures),
            ):
                raise ObjectiveError(
                    "a symmetric candidate is positive for tasks with different temperatures"
                )
            reverse_temperatures.append(candidate_temperatures[0])
        return self._direction(
            similarities=similarities.transpose(0, 1)[active_candidates],
            positive_mask=reverse_positive,
            denominator_mask=reverse_denominator,
            positive_weights=reverse_weights,
            temperatures=torch.stack(reverse_temperatures),
            direction="candidate_to_query",
        )

    def _temperatures(self, task_ids: Sequence[str], device: torch.device) -> Tensor:
        if self.task_temperatures:
            missing = sorted(set(task_ids).difference(self.task_temperatures))
            if missing:
                names = ", ".join(missing)
                raise ObjectiveError(f"no fixed temperature configured for task(s): {names}")
            values = [self.task_temperatures[task_id] for task_id in task_ids]
            return torch.tensor(values, dtype=torch.float32, device=device)
        return self.temperature.to(device=device).expand(len(task_ids))


class SymmetricMultiPositiveInfoNCE(MultiPositiveInfoNCE):
    """Convenience variant with symmetric query/candidate training enabled."""

    def __init__(self, temperature: float = 0.07, **kwargs: Any) -> None:
        """Force the symmetric direction while retaining all base options."""
        kwargs["symmetric"] = True
        super().__init__(temperature, **kwargs)


def _validate_and_prepare_embeddings(
    queries: Tensor,
    candidates: Tensor,
    layout: GroupLayout,
    *,
    normalize: bool,
) -> tuple[Tensor, Tensor]:
    if queries.ndim != _EMBEDDING_RANK or candidates.ndim != _EMBEDDING_RANK:
        raise InvalidEmbeddingError("InfoNCE embeddings must have shape [Q,D] and [C,D]")
    if queries.shape[0] != layout.num_queries or candidates.shape[0] != layout.num_candidates:
        raise InvalidEmbeddingError(
            "embedding axes do not match GroupLayout query/candidate counts"
        )
    if queries.shape[1] != candidates.shape[1] or queries.shape[1] == 0:
        raise InvalidEmbeddingError(
            "query and candidate embedding dimensions must match and be > 0"
        )
    if not queries.dtype.is_floating_point or not candidates.dtype.is_floating_point:
        raise InvalidEmbeddingError("embeddings must use floating-point dtypes")
    if queries.device != candidates.device or queries.device != layout.positive_mask.device:
        raise InvalidEmbeddingError("embeddings and GroupLayout must be on the same device")
    queries32 = queries.float()
    candidates32 = candidates.float()
    if not torch.isfinite(queries32).all().item() or not torch.isfinite(candidates32).all().item():
        raise InvalidEmbeddingError("embeddings contain NaN or infinity")
    query_norms = torch.linalg.vector_norm(queries32, dim=-1)
    candidate_norms = torch.linalg.vector_norm(candidates32, dim=-1)
    if (
        not torch.isfinite(query_norms).all().item()
        or not torch.isfinite(candidate_norms).all().item()
    ):
        raise InvalidEmbeddingError("embedding norms overflowed float32")
    if torch.any(query_norms == 0).item() or torch.any(candidate_norms == 0).item():
        raise InvalidEmbeddingError("embeddings contain a zero-norm vector")
    if normalize:
        queries32 = torch.nn.functional.normalize(queries32, p=2, dim=-1)
        candidates32 = torch.nn.functional.normalize(candidates32, p=2, dim=-1)
    return queries32, candidates32


def _masked_mean_or_zero(values: Tensor, mask: Tensor) -> Tensor:
    if mask.any().item():
        return values[mask].mean()
    return values.new_zeros(())


def _validate_temperature(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
