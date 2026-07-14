"""Composable objective graph with schedules and explicit applicability gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from torch import Tensor, nn

from mira.core.contracts import (
    ObjectiveContext,
    ObjectiveResult,
)
from mira.core.exceptions import ObjectiveError
from mira.objectives.base import EmbeddingObjective
from mira.objectives.scheduling import WeightSchedule, as_schedule

if TYPE_CHECKING:
    from mira.core.contracts import EmbeddingBatch, EmbeddingOutput
    from mira.core.layout import GroupLayout


class ObjectiveTerm(nn.Module):
    """One named objective, schedule, and set of activation conditions."""

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        objective: EmbeddingObjective,
        weight: float | WeightSchedule | Mapping[str, float] = 1.0,
        *,
        applies_to: Sequence[str] = ("*",),
        metadata_conditions: Mapping[str, Any] | None = None,
        curriculum_stages: Sequence[str] | None = None,
    ) -> None:
        """Bind one objective to its activation and weighting policy."""
        super().__init__()
        if not name:
            raise ValueError("objective term name must not be empty")
        if not applies_to or any(not task for task in applies_to):
            raise ValueError("applies_to must contain non-empty task IDs or '*'")
        self.name = name
        self.objective = objective
        self.weight_schedule = as_schedule(weight)
        self.applies_to = frozenset(applies_to)
        self.metadata_conditions = dict(metadata_conditions or {})
        self.curriculum_stages = None if curriculum_stages is None else frozenset(curriculum_stages)

    def active(self, layout: GroupLayout, context: ObjectiveContext) -> bool:
        """Return whether all explicit task and metadata gates pass."""
        if "*" not in self.applies_to:
            allowed = [task_id in self.applies_to for task_id in layout.task_ids]
            if any(allowed) and not all(allowed):
                raise ObjectiveError(
                    f"objective {self.name!r} applies to only part of a mixed-task batch"
                )
            if not any(allowed):
                return False
        if self.curriculum_stages is not None:
            stage = context.extras.get("curriculum_stage")
            if stage not in self.curriculum_stages:
                return False
        return all(
            _resolve_condition(context.extras, path) == expected
            for path, expected in self.metadata_conditions.items()
        )

    def weight(self, layout: GroupLayout, context: ObjectiveContext) -> float:
        """Resolve this term's non-negative weight."""
        return self.weight_schedule.value(context, layout.task_ids)


class ObjectiveComposer(EmbeddingObjective):
    """Aggregate enabled objective terms without trainer-side type branching."""

    def __init__(self, terms: Sequence[ObjectiveTerm]) -> None:
        """Register a non-empty sequence of uniquely named terms."""
        super().__init__()
        if not terms:
            raise ValueError("ObjectiveComposer requires at least one term")
        names = [term.name for term in terms]
        if len(set(names)) != len(names):
            raise ValueError("objective term names must be unique")
        self.terms = nn.ModuleList(terms)

    def forward(
        self,
        output: EmbeddingOutput,
        batch: EmbeddingBatch,
        context: ObjectiveContext,
    ) -> ObjectiveResult:
        """Evaluate enabled terms and return their scheduled weighted sum."""
        return self._compose(
            batch.layout,
            context,
            lambda term: term.objective(output, batch, context),
        )

    def compute(
        self,
        query_embeddings: Tensor,
        candidate_embeddings: Tensor,
        layout: GroupLayout,
    ) -> ObjectiveResult:
        """Compute at initial progress for direct mathematical tests."""
        context = ObjectiveContext(global_step=0, epoch=0.0)
        return self.compute_with_context(query_embeddings, candidate_embeddings, layout, context)

    def compute_with_context(
        self,
        query_embeddings: Tensor,
        candidate_embeddings: Tensor,
        layout: GroupLayout,
        context: ObjectiveContext,
    ) -> ObjectiveResult:
        """Compute directly with explicit schedule progress."""
        return self._compose(
            layout,
            context,
            lambda term: term.objective.compute(query_embeddings, candidate_embeddings, layout),
        )

    def _compose(
        self,
        layout: GroupLayout,
        context: ObjectiveContext,
        evaluate: Any,
    ) -> ObjectiveResult:
        total: Tensor | None = None
        metrics: dict[str, Tensor | float] = {}
        diagnostics: dict[str, Any] = {"terms": {}}
        active_count = 0
        for module in self.terms:
            if not isinstance(module, ObjectiveTerm):
                raise TypeError("ObjectiveComposer contains an invalid module")
            if not module.active(layout, context):
                diagnostics["terms"][module.name] = {"active": False}
                continue
            weight = module.weight(layout, context)
            result = evaluate(module)
            contribution = result.loss * weight
            total = contribution if total is None else total + contribution
            active_count += 1
            metrics[f"{module.name}/loss"] = result.loss.detach()
            metrics[f"{module.name}/weight"] = weight
            for name, value in result.metrics.items():
                metrics[f"{module.name}/{name}"] = value
            diagnostics["terms"][module.name] = {
                "active": True,
                "weight": weight,
                "diagnostics": result.diagnostics,
            }
        if total is None or active_count == 0:
            raise ObjectiveError("no objective term is active for this batch")
        metrics["objective_count"] = float(active_count)
        diagnostics["active_objective_count"] = active_count
        return ObjectiveResult(loss=total, metrics=metrics, diagnostics=diagnostics)


def _resolve_condition(values: Mapping[str, Any], path: str) -> Any:
    current: Any = values
    normalized = path.removeprefix("metadata.")
    for part in normalized.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current
