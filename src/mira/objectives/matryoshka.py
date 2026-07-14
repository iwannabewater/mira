"""Matryoshka Representation Learning over independently normalized slices."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import TYPE_CHECKING

import torch

from mira.core.contracts import (
    EmbeddingBatch,
    EmbeddingOutput,
    ObjectiveContext,
    ObjectiveResult,
)
from mira.core.exceptions import InvalidEmbeddingError, ObjectiveError
from mira.objectives.base import EmbeddingObjective

if TYPE_CHECKING:
    from torch import Tensor

    from mira.core.layout import GroupLayout

_EMBEDDING_RANK = 2


class MatryoshkaObjective(EmbeddingObjective):
    r"""Apply one retrieval objective independently at each embedding dimension.

    Each slice is passed to the child objective independently. The child
    objective's explicit normalization therefore occurs *after* slicing, and
    the final loss is the unnormalized weighted sum ``sum(alpha_k * loss_k)``.
    """

    def __init__(
        self,
        objective: EmbeddingObjective,
        dimensions: Sequence[int] | Mapping[int, float],
        weights: Sequence[float] | None = None,
    ) -> None:
        """Validate ordered dimensions and their unnormalized aggregation weights."""
        super().__init__()
        if isinstance(dimensions, Mapping):
            if weights is not None:
                raise ValueError("weights cannot accompany a dimension-to-weight mapping")
            dimension_values = tuple(dimensions)
            weight_values = tuple(dimensions.values())
        else:
            dimension_values = tuple(dimensions)
            weight_values = (
                tuple(1.0 for _ in dimension_values) if weights is None else tuple(weights)
            )
        if not dimension_values:
            raise ValueError("MRL requires at least one dimension")
        if any(not isinstance(dimension, int) or dimension <= 0 for dimension in dimension_values):
            raise ValueError("MRL dimensions must be positive integers")
        if any(left >= right for left, right in pairwise(dimension_values)):
            raise ValueError("MRL dimensions must be strictly increasing")
        if len(weight_values) != len(dimension_values):
            raise ValueError("MRL dimensions and weights must have equal length")
        if any(not math.isfinite(weight) or weight < 0 for weight in weight_values):
            raise ValueError("MRL weights must be finite and non-negative")
        if not any(weight > 0 for weight in weight_values):
            raise ValueError("MRL weights must not all be zero")
        self.objective = objective
        self.dimensions = dimension_values
        self.weights = weight_values

    def forward(
        self,
        output: EmbeddingOutput,
        batch: EmbeddingBatch,
        context: ObjectiveContext,
    ) -> ObjectiveResult:
        """Compute MRL from a flattened global embedding output."""
        del context
        queries, candidates, layout = self.extract_global_embeddings(output, batch)
        return self.compute(queries, candidates, layout)

    def compute(
        self,
        query_embeddings: Tensor,
        candidate_embeddings: Tensor,
        layout: GroupLayout,
    ) -> ObjectiveResult:
        """Slice, independently normalize through the child, and aggregate."""
        if query_embeddings.ndim != _EMBEDDING_RANK or candidate_embeddings.ndim != _EMBEDDING_RANK:
            raise InvalidEmbeddingError("MRL embeddings must have shape [Q,D] and [C,D]")
        hidden_size = query_embeddings.shape[-1]
        if candidate_embeddings.shape[-1] != hidden_size:
            raise InvalidEmbeddingError("MRL query and candidate hidden sizes must match")
        if self.dimensions[-1] > hidden_size:
            raise InvalidEmbeddingError(
                f"MRL dimension {self.dimensions[-1]} exceeds hidden size {hidden_size}"
            )

        total: Tensor | None = None
        metrics: dict[str, Tensor | float] = {}
        per_dimension: dict[int, object] = {}
        largest_metrics: Mapping[str, Tensor | float] | None = None
        for dimension, weight in zip(self.dimensions, self.weights, strict=True):
            try:
                query_slice = _normalize_slice(query_embeddings[:, :dimension], dimension)
                candidate_slice = _normalize_slice(candidate_embeddings[:, :dimension], dimension)
                result = self.objective.compute(
                    query_slice,
                    candidate_slice,
                    layout,
                )
            except InvalidEmbeddingError as error:
                raise InvalidEmbeddingError(f"MRL dimension {dimension}: {error}") from error
            weighted_loss = result.loss * weight
            total = weighted_loss if total is None else total + weighted_loss
            metrics[f"mrl/d{dimension}/loss"] = result.loss.detach()
            metrics[f"mrl/d{dimension}/weight"] = float(weight)
            per_dimension[dimension] = {
                "weight": weight,
                "loss": result.loss.detach(),
                "metrics": result.metrics,
                "diagnostics": result.diagnostics,
                "slice_norm_rule": "L2 normalize after slicing",
            }
            largest_metrics = result.metrics
        if total is None or not total.isfinite().item():
            raise ObjectiveError("MRL produced no finite scalar loss")
        if largest_metrics is not None:
            for name, value in largest_metrics.items():
                metrics.setdefault(name, value)
        return ObjectiveResult(
            loss=total,
            metrics=metrics,
            diagnostics={
                "dimensions": self.dimensions,
                "weights": self.weights,
                "weights_normalized": False,
                "per_dimension": per_dimension,
            },
        )


MatryoshkaLoss = MatryoshkaObjective
MRLObjective = MatryoshkaObjective


def _normalize_slice(embeddings: Tensor, dimension: int) -> Tensor:
    values = embeddings.float()
    if not torch.isfinite(values).all().item():
        raise InvalidEmbeddingError(f"MRL dimension {dimension} contains NaN or infinite values")
    norms = torch.linalg.vector_norm(values, dim=-1)
    if not torch.isfinite(norms).all().item():
        raise InvalidEmbeddingError(f"MRL dimension {dimension} contains a non-finite L2 norm")
    if torch.any(norms == 0).item():
        raise InvalidEmbeddingError(f"MRL dimension {dimension} contains a zero-norm vector")
    return torch.div(values, norms.unsqueeze(-1))
