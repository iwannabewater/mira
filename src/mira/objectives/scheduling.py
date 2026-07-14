"""Deterministic objective-weight schedules."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from mira.core.exceptions import ObjectiveError

if TYPE_CHECKING:
    from mira.core.contracts import ObjectiveContext

_Boundary = TypeVar("_Boundary", int, float)


class WeightSchedule(ABC):
    """Resolve one non-negative objective weight from explicit progress state."""

    @abstractmethod
    def value(self, context: ObjectiveContext, task_ids: Sequence[str]) -> float:
        """Return the weight for this step and homogeneous task batch."""


@dataclass(frozen=True, slots=True)
class ConstantSchedule(WeightSchedule):
    """A constant non-negative weight."""

    weight: float

    def __post_init__(self) -> None:
        """Validate the constant weight."""
        _validate_weight(self.weight)

    def value(self, context: ObjectiveContext, task_ids: Sequence[str]) -> float:
        """Return the configured constant."""
        del context, task_ids
        return self.weight


@dataclass(frozen=True, slots=True)
class LinearWarmupSchedule(WeightSchedule):
    """Linearly interpolate from ``start`` to ``end`` over warmup steps."""

    start: float
    end: float
    warmup_steps: int

    def __post_init__(self) -> None:
        """Validate endpoints and the positive warmup duration."""
        _validate_weight(self.start)
        _validate_weight(self.end)
        if self.warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")

    def value(self, context: ObjectiveContext, task_ids: Sequence[str]) -> float:
        """Resolve the clipped linear interpolation at this step."""
        del task_ids
        progress = min(context.global_step / self.warmup_steps, 1.0)
        return self.start + (self.end - self.start) * progress


@dataclass(frozen=True, slots=True)
class CosineSchedule(WeightSchedule):
    """Warm up linearly, then cosine-decay from ``peak`` to ``end``."""

    peak: float
    end: float
    total_steps: int
    warmup_steps: int = 0
    start: float = 0.0

    def __post_init__(self) -> None:
        """Validate warmup and cosine interval parameters."""
        for value in (self.start, self.peak, self.end):
            _validate_weight(value)
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.warmup_steps < 0 or self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")

    def value(self, context: ObjectiveContext, task_ids: Sequence[str]) -> float:
        """Resolve the warmup or cosine-decay segment."""
        del task_ids
        step = min(context.global_step, self.total_steps)
        if self.warmup_steps and step < self.warmup_steps:
            progress = step / self.warmup_steps
            return self.start + (self.peak - self.start) * progress
        decay_steps = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.end + (self.peak - self.end) * cosine


class StepSchedule(WeightSchedule):
    """Piecewise-constant weights selected by global step boundaries."""

    def __init__(self, values: Mapping[int, float] | Sequence[tuple[int, float]]) -> None:
        """Create a step schedule beginning at step zero."""
        entries = _ordered_unique_entries(values, "step")
        _validate_entries(entries, "step")
        self.entries = entries

    def value(self, context: ObjectiveContext, task_ids: Sequence[str]) -> float:
        """Return the last weight whose step boundary has passed."""
        del task_ids
        return _piecewise_value(self.entries, context.global_step)


class EpochSchedule(WeightSchedule):
    """Piecewise-constant weights selected by fractional epoch boundaries."""

    def __init__(self, values: Mapping[float, float] | Sequence[tuple[float, float]]) -> None:
        """Create an epoch schedule beginning at epoch zero."""
        entries = _ordered_unique_entries(values, "epoch")
        _validate_entries(entries, "epoch")
        self.entries = entries

    def value(self, context: ObjectiveContext, task_ids: Sequence[str]) -> float:
        """Return the last weight whose epoch boundary has passed."""
        del task_ids
        return _piecewise_value(self.entries, context.epoch)


class PerTaskSchedule(WeightSchedule):
    """Select a weight by task while rejecting mixed-weight batches."""

    def __init__(self, values: Mapping[str, float], *, default: float | None = None) -> None:
        """Validate task weights and an optional fallback."""
        if not values:
            raise ValueError("per-task schedule requires at least one task")
        if any(not task_id for task_id in values):
            raise ValueError("per-task schedule IDs must not be empty")
        for value in values.values():
            _validate_weight(value)
        if default is not None:
            _validate_weight(default)
        self.values = dict(values)
        self.default = default

    def value(self, context: ObjectiveContext, task_ids: Sequence[str]) -> float:
        """Resolve one common weight or reject a mixed-weight batch."""
        del context
        resolved: list[float] = []
        for task_id in task_ids:
            if task_id in self.values:
                resolved.append(self.values[task_id])
            elif self.default is not None:
                resolved.append(self.default)
            else:
                raise ObjectiveError(f"no objective weight configured for task {task_id!r}")
        unique = set(resolved)
        if len(unique) != 1:
            raise ObjectiveError(
                "a mixed-task batch resolves to different objective weights; "
                "use task-homogeneous batches"
            )
        return resolved[0]


def as_schedule(value: float | WeightSchedule | Mapping[str, float]) -> WeightSchedule:
    """Coerce user-facing weight syntax into a validated schedule."""
    if isinstance(value, WeightSchedule):
        return value
    if isinstance(value, Mapping):
        return PerTaskSchedule(value)
    return ConstantSchedule(float(value))


def _validate_weight(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError("objective weights must be finite and non-negative")


def _validate_entries(entries: Sequence[tuple[float, float]], axis: str) -> None:
    if not entries:
        raise ValueError(f"{axis} schedule must not be empty")
    if entries[0][0] != 0:
        raise ValueError(f"{axis} schedule must start at 0")
    if any(not math.isfinite(boundary) or boundary < 0 for boundary, _ in entries):
        raise ValueError(f"{axis} boundaries must be finite and non-negative")
    for _, weight in entries:
        _validate_weight(weight)


def _piecewise_value(entries: Sequence[tuple[float, float]], position: float) -> float:
    selected = entries[0][1]
    for boundary, value in entries:
        if position < boundary:
            break
        selected = value
    return selected


def _ordered_unique_entries(
    values: Mapping[_Boundary, float] | Sequence[tuple[_Boundary, float]], axis: str
) -> tuple[tuple[_Boundary, float], ...]:
    entries = tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
    boundaries = [boundary for boundary, _ in entries]
    if len(set(boundaries)) != len(boundaries):
        raise ValueError(f"{axis} schedule boundaries must be unique")
    return tuple(sorted(entries, key=lambda entry: float(entry[0])))


ConstantWeight = ConstantSchedule
LinearWarmup = LinearWarmupSchedule
CosineWeightSchedule = CosineSchedule
