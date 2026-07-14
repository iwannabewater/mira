"""One finite, observable optimizer step over MIRA contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self, cast

import torch
from torch import nn

from mira.core.exceptions import InvalidEmbeddingError, ObjectiveError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor
    from torch.optim import Optimizer

    from mira.core.capabilities import ModelCapabilities
    from mira.core.contracts import (
        EmbeddingBatch,
        EmbeddingOutput,
        ObjectiveContext,
        ObjectiveResult,
        RepresentationRequest,
    )
    from mira.models.base import AdapterMode
    from mira.objectives.base import EmbeddingObjective


class TrainableAdapter(Protocol):
    """Structural interface needed by the single-step engine."""

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capability declarations."""

    def prepare_inputs(
        self,
        batch: EmbeddingBatch,
        mode: AdapterMode,
    ) -> Mapping[str, object]:
        """Prepare one structured batch."""

    def forward_representations(
        self,
        inputs: Mapping[str, object],
        request: RepresentationRequest,
    ) -> EmbeddingOutput:
        """Return requested embeddings."""

    def train(self) -> Self:
        """Set module training mode."""

    def parameters(self) -> Iterator[nn.Parameter]:
        """Iterate trainable module parameters."""


@dataclass(frozen=True, slots=True)
class StepResult:
    """Detached evidence from one completed optimizer step."""

    loss: float
    gradient_norm: float
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        """Normalize finite numeric evidence while rejecting boolean coercion."""
        loss = _finite_builtin_number(self.loss, name="StepResult loss")
        gradient_norm = _finite_builtin_number(
            self.gradient_norm,
            name="StepResult gradient_norm",
        )
        if gradient_norm < 0:
            raise ValueError("StepResult gradient_norm must be non-negative")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("StepResult metrics must be a mapping")
        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if type(name) is not str:
                raise TypeError("StepResult metric names must be strings")
            if not name.strip():
                raise ValueError("StepResult metric names must be non-empty")
            metrics[name] = _finite_builtin_number(
                value,
                name=f"StepResult metric {name!r}",
            )
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "gradient_norm", gradient_norm)
        object.__setattr__(self, "metrics", metrics)


class TrainingStep:
    """Execute forward, objective, backward, finite checks, and optimizer step."""

    def __init__(
        self,
        adapter: TrainableAdapter,
        objective: EmbeddingObjective,
        optimizer: Optimizer,
        request: RepresentationRequest,
        *,
        max_gradient_norm: float | None = None,
    ) -> None:
        """Bind the explicit components used by every step."""
        if max_gradient_norm is not None:
            max_gradient_norm = _finite_builtin_number(
                max_gradient_norm,
                name="max_gradient_norm",
            )
            if max_gradient_norm <= 0:
                raise ValueError("max_gradient_norm must be positive when supplied")
        self.adapter = adapter
        self.objective = objective
        self.optimizer = optimizer
        self.request = request
        self.max_gradient_norm = max_gradient_norm

    def run(self, batch: EmbeddingBatch, context: ObjectiveContext) -> StepResult:
        """Complete one step or fail before mutating parameters with invalid gradients."""
        self.adapter.train()
        self.objective.train()
        self.optimizer.zero_grad(set_to_none=True)
        inputs = self.adapter.prepare_inputs(batch, "train")
        output = self.adapter.forward_representations(inputs, self.request)
        result = self.objective(output, batch, context)
        if not torch.isfinite(result.loss).item():
            raise ObjectiveError("training step received a non-finite objective loss")
        metrics = _scalar_metrics(result)
        result.loss.backward()
        parameters = _trainable_parameters(self.adapter, self.objective)
        gradient_norm = _gradient_norm(parameters)
        if not torch.isfinite(gradient_norm).item():
            self.optimizer.zero_grad(set_to_none=True)
            raise InvalidEmbeddingError("training gradients contain NaN or infinite values")
        if self.max_gradient_norm is not None:
            clipped = nn.utils.clip_grad_norm_(
                parameters,
                max_norm=self.max_gradient_norm,
                error_if_nonfinite=True,
            )
            gradient_norm = clipped.detach()
        self.optimizer.step()
        return StepResult(
            loss=float(result.loss.detach().item()),
            gradient_norm=float(gradient_norm.item()),
            metrics=metrics,
        )


def _trainable_parameters(
    adapter: TrainableAdapter,
    objective: EmbeddingObjective,
) -> tuple[nn.Parameter, ...]:
    unique: dict[int, nn.Parameter] = {}
    for parameter in (*adapter.parameters(), *objective.parameters()):
        if parameter.requires_grad:
            unique[id(parameter)] = parameter
    return tuple(unique.values())


def _gradient_norm(parameters: tuple[nn.Parameter, ...]) -> Tensor:
    gradients = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        raise InvalidEmbeddingError("training step produced no parameter gradients")
    return cast("Tensor", torch.linalg.vector_norm(torch.stack(gradients)))


def _scalar_metrics(result: ObjectiveResult) -> dict[str, float]:
    if not isinstance(result.metrics, Mapping):
        raise ObjectiveError("objective metrics must be a mapping")
    metrics: dict[str, float] = {}
    for name, value in result.metrics.items():
        if type(name) is not str or not name.strip():
            raise ObjectiveError("objective metric names must be non-empty strings")
        scalar: object = value
        if isinstance(value, torch.Tensor):
            if value.ndim != 0:
                raise ObjectiveError(f"objective metric {name!r} must be scalar")
            scalar = value.detach().item()
        try:
            metrics[name] = _finite_builtin_number(
                scalar,
                name=f"objective metric {name!r}",
            )
        except (TypeError, ValueError) as error:
            raise ObjectiveError(
                f"objective metric {name!r} must be a finite numeric scalar"
            ) from error
    return metrics


def _finite_builtin_number(value: object, *, name: str) -> float:
    """Return one finite built-in number as float without accepting booleans."""
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be a built-in real number, not {type(value).__name__}")
    try:
        normalized = float(cast("int | float", value))
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized
