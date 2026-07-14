"""Adversarial objective, MRL, composer, and schedule coverage."""

from __future__ import annotations

import pytest
import torch

from mira.core.contracts import (
    BatchMetadata,
    EmbeddingBatch,
    EmbeddingOutput,
    ObjectiveContext,
)
from mira.core.exceptions import InvalidEmbeddingError, ObjectiveError
from mira.core.layout import GroupLayout
from mira.objectives import (
    ConstantSchedule,
    CosineSchedule,
    EpochSchedule,
    LinearWarmupSchedule,
    MatryoshkaObjective,
    MultiPositiveInfoNCE,
    ObjectiveComposer,
    ObjectiveTerm,
    PerTaskSchedule,
    StepSchedule,
    SymmetricMultiPositiveInfoNCE,
)
from mira.objectives.contrastive.infonce import _masked_mean_or_zero
from mira.objectives.output import ObjectiveResult


def _layout(
    positive_mask: torch.Tensor,
    *,
    denominator_mask: torch.Tensor | None = None,
) -> GroupLayout:
    query_count, candidate_count = positive_mask.shape
    if denominator_mask is None:
        denominator_mask = torch.ones_like(positive_mask)
    negatives = denominator_mask & ~positive_mask
    return GroupLayout(
        query_indices=torch.arange(query_count),
        candidate_indices=torch.arange(query_count, query_count + candidate_count),
        positive_mask=positive_mask,
        candidate_mask=denominator_mask,
        explicit_negative_mask=negatives,
        in_batch_negative_mask=torch.zeros_like(negatives),
        false_negative_mask=torch.zeros_like(negatives),
        task_ids=("task",) * query_count,
        dataset_ids=("dataset",) * query_count,
    )


def _tensors() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([[1.0, 0.2]]), torch.tensor([[0.8, 0.1], [0.1, 0.9]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.0},
        {"temperature": float("nan")},
        {"temperature": 0.5, "min_temperature": 0.6},
        {"temperature": 0.5, "min_temperature": 1.0, "max_temperature": 0.1},
        {"logit_clip": 0.0},
        {"task_temperatures": {"": 0.1}},
        {"task_temperatures": {"task": -0.1}},
        {"learnable_temperature": True, "task_temperatures": {"task": 0.1}},
    ],
)
def test_infonce_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r".+"):
        MultiPositiveInfoNCE(**kwargs)


def test_infonce_forward_uses_flattened_contract_indices() -> None:
    queries, candidates = _tensors()
    layout = _layout(torch.tensor([[True, False]]))
    flat = torch.cat((queries, candidates))
    batch = EmbeddingBatch(
        model_inputs={"ids": torch.ones(1)},
        layout=layout,
        metadata=BatchMetadata(("s",), ("task",), ("dataset",)),
    )
    output = EmbeddingOutput(global_embeddings=flat)
    result = MultiPositiveInfoNCE().forward(
        output, batch, ObjectiveContext(global_step=0, epoch=0.0)
    )
    assert result.loss.ndim == 0
    assert result.metrics["embedding_norm"] > 0


def test_forward_rejects_missing_global_or_out_of_bounds_layout() -> None:
    layout = _layout(torch.tensor([[True, False]]))
    batch = EmbeddingBatch(
        model_inputs={"ids": torch.ones(1)},
        layout=layout,
        metadata=BatchMetadata(("s",), ("task",), ("dataset",)),
    )
    with pytest.raises(InvalidEmbeddingError, match="requires global_embeddings"):
        MultiPositiveInfoNCE().forward(
            EmbeddingOutput(logits=torch.ones(3, 1)),
            batch,
            ObjectiveContext(0, 0.0),
        )
    with pytest.raises(InvalidEmbeddingError, match="exceeds embedding row count"):
        MultiPositiveInfoNCE().forward(
            EmbeddingOutput(global_embeddings=torch.ones(1, 2)),
            batch,
            ObjectiveContext(0, 0.0),
        )


@pytest.mark.parametrize(
    ("queries", "candidates", "match"),
    [
        (torch.ones(2), torch.ones(2, 2), "shape"),
        (torch.ones(1, 3), torch.ones(2, 2), "dimensions must match"),
        (torch.ones(1, 2, dtype=torch.int64), torch.ones(2, 2), "floating-point"),
        (torch.tensor([[float("nan"), 1.0]]), torch.ones(2, 2), "NaN"),
        (torch.zeros(1, 2), torch.ones(2, 2), "zero-norm"),
    ],
)
def test_infonce_rejects_invalid_embeddings(
    queries: torch.Tensor, candidates: torch.Tensor, match: str
) -> None:
    with pytest.raises(InvalidEmbeddingError, match=match):
        MultiPositiveInfoNCE().compute(queries, candidates, _layout(torch.tensor([[True, False]])))


def test_infonce_rejects_axis_mismatch_and_missing_task_temperature() -> None:
    queries, candidates = _tensors()
    with pytest.raises(InvalidEmbeddingError, match="query/candidate counts"):
        MultiPositiveInfoNCE().compute(
            torch.cat((queries, queries)),
            candidates,
            _layout(torch.tensor([[True, False]])),
        )
    with pytest.raises(ObjectiveError, match="no fixed temperature"):
        MultiPositiveInfoNCE(task_temperatures={"other": 0.1}).compute(
            queries, candidates, _layout(torch.tensor([[True, False]]))
        )
    huge = torch.full((1, 2), torch.finfo(torch.float32).max)
    with pytest.raises(InvalidEmbeddingError, match="norms overflowed"):
        MultiPositiveInfoNCE().compute(
            huge,
            torch.cat((huge, huge)),
            _layout(torch.tensor([[True, False]])),
        )


def test_infonce_logit_clipping_and_no_negative_diagnostics() -> None:
    objective = MultiPositiveInfoNCE(temperature=0.01, logit_clip=1.0, normalize_embeddings=False)
    layout = _layout(torch.tensor([[True]]), denominator_mask=torch.tensor([[True]]))
    result = objective.compute(torch.tensor([[10.0]]), torch.tensor([[10.0]]), layout)
    assert result.loss == 0
    assert result.metrics["skipped_query_count"] == 1
    diagnostics = result.diagnostics["query_to_candidate"]
    assert diagnostics["logit_max"] == 1.0
    assert _masked_mean_or_zero(torch.ones(1, 1), torch.tensor([[False]])) == 0


def test_mutated_layout_weight_and_positive_fail_independently() -> None:
    queries, candidates = _tensors()
    layout = _layout(torch.tensor([[True, False]]))
    layout.positive_weights[0, 0] = 0
    with pytest.raises(ObjectiveError, match="positive weights"):
        MultiPositiveInfoNCE().compute(queries, candidates, layout)
    layout = _layout(torch.tensor([[True, False]]))
    layout.positive_mask.fill_(value=False)
    with pytest.raises(ObjectiveError, match="no valid positive"):
        MultiPositiveInfoNCE().compute(queries, candidates, layout)


def test_symmetric_per_task_temperature_conflict_is_explicit() -> None:
    positive = torch.tensor([[True], [True]])
    layout = _layout(positive)
    object.__setattr__(layout, "task_ids", ("a", "b"))
    objective = MultiPositiveInfoNCE(task_temperatures={"a": 0.1, "b": 0.2}, symmetric=True)
    with pytest.raises(ObjectiveError, match="different temperatures"):
        objective.compute(torch.ones(2, 2), torch.ones(1, 2), layout)


def test_symmetric_convenience_and_defensive_direction_contracts() -> None:
    layout = _layout(torch.tensor([[True, False]]))
    objective = SymmetricMultiPositiveInfoNCE()
    assert objective.symmetric is True
    result = objective.compute(torch.tensor([[1.0, 0.0]]), torch.eye(2), layout)
    assert isinstance(result, ObjectiveResult)

    base = MultiPositiveInfoNCE()
    with pytest.raises(ObjectiveError, match="must use float32"):
        base._direction(
            similarities=torch.ones(1, 2, dtype=torch.float64),
            positive_mask=layout.positive_mask,
            denominator_mask=layout.candidate_mask,
            positive_weights=layout.positive_weights,
            temperatures=torch.ones(1),
            direction="test",
        )
    with pytest.raises(ObjectiveError, match="shape does not match"):
        base._direction(
            similarities=torch.ones(1, 2),
            positive_mask=torch.ones(1, 1, dtype=torch.bool),
            denominator_mask=layout.candidate_mask,
            positive_weights=layout.positive_weights,
            temperatures=torch.ones(1),
            direction="test",
        )
    with pytest.raises(ObjectiveError, match="temperatures must have shape"):
        base._direction(
            similarities=torch.ones(1, 2),
            positive_mask=layout.positive_mask,
            denominator_mask=layout.candidate_mask,
            positive_weights=layout.positive_weights,
            temperatures=torch.ones(2),
            direction="test",
        )


def test_objective_base_validate_and_meta_device_mismatch() -> None:
    objective = MultiPositiveInfoNCE()
    objective.validate(None, None, None)
    layout = _layout(torch.tensor([[True, False]]))
    with pytest.raises(InvalidEmbeddingError, match="same device"):
        objective.compute(
            torch.ones(1, 2, device="meta"),
            torch.ones(2, 2, device="meta"),
            layout,
        )


@pytest.mark.parametrize(
    ("dimensions", "weights"),
    [
        ((), None),
        ((0,), None),
        ((2, 2), None),
        ((2, 4), (1.0,)),
        ((2,), (float("nan"),)),
    ],
)
def test_mrl_rejects_additional_invalid_configurations(
    dimensions: tuple[int, ...], weights: tuple[float, ...] | None
) -> None:
    with pytest.raises(ValueError, match=r".+"):
        MatryoshkaObjective(MultiPositiveInfoNCE(), dimensions, weights)


def test_mrl_mapping_conflict_rank_hidden_mismatch_and_forward() -> None:
    with pytest.raises(ValueError, match="cannot accompany"):
        MatryoshkaObjective(MultiPositiveInfoNCE(), {2: 1.0}, (1.0,))
    objective = MatryoshkaObjective(MultiPositiveInfoNCE(), {2: 1.0})
    layout = _layout(torch.tensor([[True, False]]))
    with pytest.raises(InvalidEmbeddingError, match="shape"):
        objective.compute(torch.ones(2), torch.ones(2, 2), layout)
    with pytest.raises(InvalidEmbeddingError, match="hidden sizes"):
        objective.compute(torch.ones(1, 2), torch.ones(2, 3), layout)

    queries, candidates = _tensors()
    flat = torch.cat((queries, candidates))
    batch = EmbeddingBatch(
        {"ids": torch.ones(1)},
        layout,
        BatchMetadata(("s",), ("task",), ("dataset",)),
    )
    result = objective.forward(
        EmbeddingOutput(global_embeddings=flat), batch, ObjectiveContext(0, 0.0)
    )
    assert result.loss.ndim == 0


def test_all_schedule_families_and_boundaries() -> None:
    context = ObjectiveContext(global_step=5, epoch=1.5)
    assert ConstantSchedule(0.3).value(context, ("t",)) == 0.3
    assert LinearWarmupSchedule(0.0, 1.0, 10).value(context, ("t",)) == 0.5
    assert LinearWarmupSchedule(0.0, 1.0, 2).value(context, ("t",)) == 1.0
    assert CosineSchedule(1.0, 0.0, total_steps=10).value(context, ("t",)) == 0.5
    assert CosineSchedule(1.0, 0.0, total_steps=10, warmup_steps=5).value(
        ObjectiveContext(2, 0.0), ("t",)
    ) == pytest.approx(0.4)
    assert StepSchedule({0: 0.1, 5: 0.5}).value(context, ("t",)) == 0.5
    assert EpochSchedule({0.0: 0.1, 1.0: 0.4}).value(context, ("t",)) == 0.4
    assert PerTaskSchedule({"t": 0.7}).value(context, ("t", "t")) == 0.7
    assert PerTaskSchedule({"a": 0.1}, default=0.2).value(context, ("b",)) == 0.2


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ConstantSchedule(-1.0),
        lambda: LinearWarmupSchedule(0.0, 1.0, 0),
        lambda: CosineSchedule(1.0, 0.0, 0),
        lambda: CosineSchedule(1.0, 0.0, 10, warmup_steps=10),
        lambda: StepSchedule({1: 0.5}),
        lambda: EpochSchedule({-1.0: 0.5}),
        lambda: PerTaskSchedule({}),
        lambda: PerTaskSchedule({"": 1.0}),
    ],
)
def test_schedules_reject_invalid_configuration(factory: object) -> None:
    with pytest.raises(ValueError, match=r".+"):
        factory()  # type: ignore[operator]


def test_schedules_reject_duplicate_boundaries_in_sequences() -> None:
    with pytest.raises(ValueError, match="boundaries must be unique"):
        StepSchedule(((0, 0.1), (0, 0.2)))
    with pytest.raises(ValueError, match="boundaries must be unique"):
        EpochSchedule(((0.0, 0.1), (0.0, 0.2)))


def test_per_task_schedule_rejects_missing_and_mixed_weights() -> None:
    schedule = PerTaskSchedule({"a": 0.1, "b": 0.2})
    context = ObjectiveContext(0, 0.0)
    with pytest.raises(ObjectiveError, match="no objective weight"):
        schedule.value(context, ("missing",))
    with pytest.raises(ObjectiveError, match="mixed-task"):
        schedule.value(context, ("a", "b"))


def test_composer_forward_conditions_curriculum_and_errors() -> None:
    objective = MultiPositiveInfoNCE()
    with pytest.raises(ValueError, match="at least one term"):
        ObjectiveComposer(())
    with pytest.raises(ValueError, match="unique"):
        ObjectiveComposer((ObjectiveTerm("x", objective), ObjectiveTerm("x", objective)))
    with pytest.raises(ValueError, match="name"):
        ObjectiveTerm("", objective)
    with pytest.raises(ValueError, match="applies_to"):
        ObjectiveTerm("x", objective, applies_to=())

    term = ObjectiveTerm(
        "conditional",
        objective,
        {"task": 0.5},
        applies_to=("task",),
        metadata_conditions={"metadata.has_teacher_scores": True},
        curriculum_stages=("hard",),
    )
    composer = ObjectiveComposer((term,))
    layout = _layout(torch.tensor([[True, False]]))
    queries, candidates = _tensors()
    with pytest.raises(ObjectiveError, match="no objective term"):
        composer.compute_with_context(queries, candidates, layout, ObjectiveContext(0, 0.0))
    context = ObjectiveContext(
        0,
        0.0,
        extras={"curriculum_stage": "hard", "has_teacher_scores": True},
    )
    result = composer.compute_with_context(queries, candidates, layout, context)
    assert result.metrics["conditional/weight"] == 0.5
    assert result.diagnostics["terms"]["conditional"]["active"] is True


def test_composer_rejects_partial_mixed_task_applicability() -> None:
    layout = _layout(torch.tensor([[True, False], [False, True]]))
    object.__setattr__(layout, "task_ids", ("a", "b"))
    composer = ObjectiveComposer(
        (ObjectiveTerm("only-a", MultiPositiveInfoNCE(), applies_to=("a",)),)
    )
    with pytest.raises(ObjectiveError, match="only part"):
        composer.compute(torch.eye(2), torch.eye(2), layout)
