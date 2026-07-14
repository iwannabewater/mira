"""Focused adversarial coverage for stable runtime contract boundaries."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast

import pytest
import torch
from torch import nn

from mira.core.capabilities import ModelCapabilities
from mira.core.contracts import (
    BatchMetadata,
    EmbeddingBatch,
    EmbeddingOutput,
    ObjectiveContext,
    ObjectiveResult,
    RepresentationRequest,
    RepresentationType,
)
from mira.core.exceptions import ContractError, InvalidEmbeddingError, ObjectiveError
from mira.core.layout import GroupLayout
from mira.data.schema import Message, MultimodalItem
from mira.integrations.ms_swift.arguments import SwiftTemplateSpec
from mira.integrations.ms_swift.template_bridge import (
    SwiftTemplateBridge,
    validate_qwen_message_roles,
)
from mira.objectives import MultiPositiveInfoNCE, ObjectiveComposer, ObjectiveTerm
from mira.objectives.base import EmbeddingObjective
from mira.observability import CollapseThresholds, analyze_embedding_health
from mira.training import StepResult, TrainingStep

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from mira.models.base import AdapterMode


class _IntSubclass(int):
    pass


class _FloatSubclass(float):
    pass


def _layout() -> GroupLayout:
    positive = torch.tensor([[True, False]])
    negative = torch.tensor([[False, True]])
    return GroupLayout(
        query_indices=torch.tensor([0]),
        candidate_indices=torch.tensor([1, 2]),
        positive_mask=positive,
        candidate_mask=positive | negative,
        explicit_negative_mask=negative,
        in_batch_negative_mask=torch.zeros_like(negative),
        false_negative_mask=torch.zeros_like(negative),
        task_ids=("retrieval",),
        dataset_ids=("runtime",),
    )


def _batch() -> EmbeddingBatch:
    return EmbeddingBatch(
        model_inputs={"input_ids": torch.ones(3, 1, dtype=torch.long)},
        layout=_layout(),
        metadata=BatchMetadata(
            sample_ids=("sample",),
            task_ids=("retrieval",),
            dataset_ids=("runtime",),
        ),
    )


@pytest.mark.parametrize(
    ("sample_ids", "match"),
    [
        ("sample", "tuple of strings"),
        (["sample"], "tuple of strings"),
        ((1,), "non-empty strings"),
    ],
)
def test_batch_metadata_requires_exact_string_tuples(
    sample_ids: object,
    match: str,
) -> None:
    with pytest.raises(ContractError, match=match):
        BatchMetadata(
            sample_ids=sample_ids,  # type: ignore[arg-type]
            task_ids=("retrieval",),
            dataset_ids=("runtime",),
        )


@pytest.mark.parametrize(
    ("extras", "match"),
    [
        ([("key", 1)], "must be a mapping"),
        ({1: "value"}, "keys must be strings"),
    ],
)
def test_batch_metadata_requires_mapping_extras_with_string_keys(
    extras: object,
    match: str,
) -> None:
    with pytest.raises(ContractError, match=match):
        BatchMetadata(
            sample_ids=("sample",),
            task_ids=("retrieval",),
            dataset_ids=("runtime",),
            extras=extras,  # type: ignore[arg-type]
        )


def test_embedding_batch_requires_mapping_and_typed_contracts() -> None:
    metadata = BatchMetadata(("sample",), ("retrieval",), ("runtime",))
    layout = _layout()

    with pytest.raises(ContractError, match="model_inputs must be a mapping"):
        EmbeddingBatch(  # type: ignore[arg-type]
            model_inputs=[("input_ids", torch.ones(1))],
            layout=layout,
            metadata=metadata,
        )
    with pytest.raises(ContractError, match="layout must be a GroupLayout"):
        EmbeddingBatch(
            model_inputs={"input_ids": torch.ones(1)},
            layout=SimpleNamespace(),  # type: ignore[arg-type]
            metadata=metadata,
        )
    with pytest.raises(ContractError, match="metadata must be BatchMetadata"):
        EmbeddingBatch(
            model_inputs={"input_ids": torch.ones(1)},
            layout=layout,
            metadata=SimpleNamespace(),  # type: ignore[arg-type]
        )
    with pytest.raises(ContractError, match="map non-empty"):
        EmbeddingBatch(
            model_inputs={1: torch.ones(1)},  # type: ignore[dict-item]
            layout=layout,
            metadata=metadata,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"representation": "global"}, "RepresentationType"),
        ({"dimension": True}, "exact integer"),
        ({"dimension": 1.5}, "exact integer"),
        ({"dimension": _IntSubclass(1)}, "exact integer"),
        ({"hidden_layers": [0]}, "tuple of exact integers"),
        ({"hidden_layers": (False,)}, "tuple of exact integers"),
        ({"hidden_layers": (1.5,)}, "tuple of exact integers"),
        ({"hidden_layers": (_IntSubclass(0),)}, "tuple of exact integers"),
        ({"normalize": 1}, "exact boolean"),
    ],
)
def test_representation_request_rejects_runtime_coercion(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ContractError, match=match):
        RepresentationRequest(**kwargs)  # type: ignore[arg-type]


def test_representation_request_preserves_exact_enum_and_integer_values() -> None:
    request = RepresentationRequest(
        representation=RepresentationType.HYBRID,
        dimension=2,
        hidden_layers=(-1, 0),
        normalize=False,
    )

    assert request.representation is RepresentationType.HYBRID
    assert request.hidden_layers == (-1, 0)


def test_embedding_extraction_rejects_cross_device_layout() -> None:
    output = cast(
        "EmbeddingOutput",
        SimpleNamespace(global_embeddings=torch.empty(3, 2, device="meta")),
    )

    with pytest.raises(InvalidEmbeddingError, match="same device"):
        EmbeddingObjective.extract_global_embeddings(output, _batch())


def test_embedding_output_rejects_cross_device_token_mask() -> None:
    embeddings = torch.ones(1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool, device="meta")

    with pytest.raises(InvalidEmbeddingError, match="token embedding device"):
        EmbeddingOutput._validate_token_mask(embeddings, mask)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"global_embeddings": "not-a-tensor"},
        {"token_embeddings": "not-a-tensor"},
        {"token_mask": "not-a-tensor"},
        {"logits": "not-a-tensor"},
    ],
)
def test_embedding_output_rejects_non_tensor_optional_fields(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(InvalidEmbeddingError, match="must be a Tensor"):
        EmbeddingOutput(**kwargs)  # type: ignore[arg-type]


def test_embedding_output_requires_exact_normalization_and_layer_types() -> None:
    with pytest.raises(InvalidEmbeddingError, match="normalized must be an exact boolean"):
        EmbeddingOutput(
            global_embeddings=torch.tensor([[1.0, 0.0]]),
            normalized=1,  # type: ignore[arg-type]
        )
    with pytest.raises(InvalidEmbeddingError, match="layer_embeddings must be a mapping"):
        EmbeddingOutput(
            layer_embeddings=[(0, torch.ones(1, 2))]  # type: ignore[arg-type]
        )
    for layer in (True, _IntSubclass(1)):
        with pytest.raises(InvalidEmbeddingError, match="keys must be integers"):
            EmbeddingOutput(
                layer_embeddings={layer: torch.ones(1, 2)}  # type: ignore[dict-item]
            )
    with pytest.raises(InvalidEmbeddingError, match="values must be tensors"):
        EmbeddingOutput(layer_embeddings={0: object()})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("auxiliary", "match"),
    [
        ([("key", 1)], "must be a mapping"),
        ({1: "value"}, "keys must be strings"),
    ],
)
def test_embedding_output_requires_mapping_auxiliary_with_string_keys(
    auxiliary: object,
    match: str,
) -> None:
    with pytest.raises(InvalidEmbeddingError, match=match):
        EmbeddingOutput(
            logits=torch.ones(1, 1),
            auxiliary=auxiliary,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"global_step": True}, "global_step must be an exact integer"),
        ({"global_step": 1.0}, "global_step must be an exact integer"),
        ({"global_step": _IntSubclass(1)}, "global_step must be an exact integer"),
        ({"rank": False}, "rank must be an exact integer"),
        ({"world_size": 1.0}, "world_size must be an exact integer"),
        ({"training": 1}, "training must be an exact boolean"),
        ({"epoch": True}, "epoch must be a built-in real number"),
        ({"epoch": "0"}, "epoch must be a built-in real number"),
        ({"epoch": _FloatSubclass(0.5)}, "epoch must be a built-in real number"),
        ({"epoch": 10**10000}, "epoch must be finite"),
    ],
)
def test_objective_context_rejects_scalar_coercion(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values: dict[str, object] = {"global_step": 0, "epoch": 0.0}
    values.update(kwargs)

    with pytest.raises(ContractError, match=match):
        ObjectiveContext(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("extras", "match"),
    [
        ([("key", 1)], "must be a mapping"),
        ({1: "value"}, "keys must be strings"),
    ],
)
def test_objective_context_requires_mapping_extras_with_string_keys(
    extras: object,
    match: str,
) -> None:
    with pytest.raises(ContractError, match=match):
        ObjectiveContext(
            global_step=0,
            epoch=0,
            extras=extras,  # type: ignore[arg-type]
        )


def test_objective_context_normalizes_builtin_integer_epoch() -> None:
    context = ObjectiveContext(global_step=0, epoch=1)

    assert context.epoch == 1.0
    assert type(context.epoch) is float


@pytest.mark.parametrize("loss", [1.0, "1", True, object()])
def test_objective_result_domain_rejects_non_tensor_loss(loss: object) -> None:
    with pytest.raises(ObjectiveError, match="loss must be a Tensor"):
        ObjectiveResult(loss=loss)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("metrics", [("metric", 1.0)], "metrics must be a mapping"),
        ("diagnostics", [("detail", 1.0)], "diagnostics must be a mapping"),
    ],
)
def test_objective_result_requires_mapping_containers(
    field: str,
    value: object,
    match: str,
) -> None:
    kwargs = {field: value}
    with pytest.raises(ObjectiveError, match=match):
        ObjectiveResult(loss=torch.tensor(1.0), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", [1, " "])
def test_objective_result_requires_non_empty_string_metric_names(name: object) -> None:
    with pytest.raises(ObjectiveError, match="non-empty strings"):
        ObjectiveResult(
            loss=torch.tensor(1.0),
            metrics={name: 1.0},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize(
    "metric",
    [_IntSubclass(1), _FloatSubclass(1.0), 10**10000],
    ids=("int-subclass", "float-subclass", "overflow"),
)
def test_objective_result_rejects_numeric_subclasses_and_overflow(metric: object) -> None:
    with pytest.raises(ObjectiveError, match="finite real number"):
        ObjectiveResult(
            loss=torch.tensor(1.0),
            metrics={"invalid": metric},  # type: ignore[dict-item]
        )


def test_composer_forward_uses_the_batch_layout() -> None:
    composer = ObjectiveComposer(
        (ObjectiveTerm("retrieval", MultiPositiveInfoNCE(temperature=0.2)),)
    )
    output = EmbeddingOutput(global_embeddings=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]))

    result = composer(output, _batch(), ObjectiveContext(global_step=3, epoch=0.5))

    assert result.loss > 0
    assert result.metrics["objective_count"] == 1.0


def test_objective_term_rejects_absent_task_and_nested_metadata() -> None:
    layout = _layout()
    context = ObjectiveContext(global_step=0, epoch=0.0)
    wrong_task = ObjectiveTerm(
        "other-task",
        MultiPositiveInfoNCE(),
        applies_to=("classification",),
    )
    nested_condition = ObjectiveTerm(
        "teacher-gated",
        MultiPositiveInfoNCE(),
        metadata_conditions={"metadata.teacher.available": True},
    )

    assert not wrong_task.active(layout, context)
    assert not nested_condition.active(
        layout,
        ObjectiveContext(global_step=0, epoch=0.0, extras={"teacher": {}}),
    )


def test_composer_defends_against_invalid_registered_module() -> None:
    composer = ObjectiveComposer((ObjectiveTerm("valid", MultiPositiveInfoNCE()),))
    composer.terms[0] = nn.Identity()

    with pytest.raises(TypeError, match="invalid module"):
        composer.compute(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            _layout(),
        )


@pytest.mark.parametrize(
    "thresholds",
    [
        {"minimum_dimension_std": -1.0},
        {"maximum_pairwise_cosine": float("inf")},
    ],
)
def test_collapse_thresholds_require_finite_non_negative_values(
    thresholds: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        CollapseThresholds(**thresholds)


@pytest.mark.parametrize(
    "value",
    [True, "0.1", object(), _IntSubclass(1), _FloatSubclass(0.1)],
)
def test_collapse_thresholds_reject_boolean_and_non_numeric_types(value: object) -> None:
    with pytest.raises(TypeError, match="built-in real number"):
        CollapseThresholds(minimum_dimension_std=value)  # type: ignore[arg-type]


def test_collapse_thresholds_reject_overflow_and_normalize_integers() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        CollapseThresholds(minimum_dimension_std=10**10000)

    thresholds = CollapseThresholds(minimum_dimension_std=1)
    assert thresholds.minimum_dimension_std == 1.0
    assert type(thresholds.minimum_dimension_std) is float


@pytest.mark.parametrize(
    ("embeddings", "match"),
    [
        (torch.ones(3), "non-empty"),
        (torch.empty(0, 2), "non-empty"),
        (torch.ones(2, 2, dtype=torch.int64), "floating-point"),
        (torch.tensor([[float("nan"), 1.0]]), "NaN or infinite"),
        (torch.tensor([[1e38, 1e38]]), "non-finite embedding L2 norm"),
        (torch.tensor([[2e38], [2e38]]), "non-finite dimension statistics"),
        (torch.tensor([[0.0, 0.0], [1.0, 0.0]]), "zero-norm"),
    ],
)
def test_embedding_health_rejects_invalid_input(embeddings: Tensor, match: str) -> None:
    with pytest.raises(InvalidEmbeddingError, match=match):
        analyze_embedding_health(embeddings)


def test_embedding_health_covers_singleton_and_healthy_spectra() -> None:
    singleton = analyze_embedding_health(torch.tensor([[1.0, 2.0]]))
    healthy = analyze_embedding_health(torch.eye(3))
    high_magnitude = analyze_embedding_health(torch.eye(3).repeat(40, 1) * 1e19)

    assert singleton.pairwise_cosine_mean == 1.0
    assert singleton.pairwise_cosine_std == 0.0
    assert singleton.effective_rank == 0.0
    assert healthy.effective_rank == pytest.approx(2.0)
    assert high_magnitude.effective_rank == pytest.approx(2.0, rel=1e-5)
    assert healthy.effective_rank_ratio == pytest.approx(1.0)
    assert healthy.alerts == ()
    assert all(
        math.isfinite(value)
        for value in (
            healthy.norm_mean,
            healthy.norm_std,
            healthy.dimension_std_mean,
            healthy.dimension_std_min,
            healthy.pairwise_cosine_mean,
            healthy.pairwise_cosine_std,
            healthy.effective_rank,
            healthy.effective_rank_ratio,
            healthy.collapse_score,
            *healthy.singular_values,
        )
    )


def test_embedding_health_reports_excess_norm_variation() -> None:
    report = analyze_embedding_health(
        torch.tensor([[1.0, 0.0], [3.0, 0.0]]),
        thresholds=CollapseThresholds(maximum_norm_coefficient_of_variation=0.1),
    )

    assert "norm_variation_above_threshold" in report.alerts


class _ScalarAdapter(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))
        self.frozen = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(supports_global_embedding=True)

    def prepare_inputs(
        self,
        batch: EmbeddingBatch,
        mode: AdapterMode,
    ) -> Mapping[str, object]:
        assert batch.layout.num_queries == 1
        assert mode == "train"
        return {}

    def forward_representations(
        self,
        inputs: Mapping[str, object],
        request: RepresentationRequest,
    ) -> EmbeddingOutput:
        assert not inputs
        assert request.normalize
        return EmbeddingOutput(logits=self.weight.reshape(1, 1))


class _RuntimeObjective(EmbeddingObjective):
    def __init__(self, behavior: Literal["square", "sqrt", "disconnected", "nonfinite"]) -> None:
        super().__init__()
        self.behavior = behavior

    def forward(
        self,
        output: EmbeddingOutput,
        batch: EmbeddingBatch,
        context: ObjectiveContext,
    ) -> ObjectiveResult:
        del batch, context
        assert output.logits is not None
        if self.behavior == "nonfinite":
            return cast(
                "ObjectiveResult",
                SimpleNamespace(
                    loss=torch.tensor(float("nan"), requires_grad=True),
                    metrics={},
                ),
            )
        if self.behavior == "sqrt":
            return ObjectiveResult(loss=output.logits.sum().sqrt())
        if self.behavior == "disconnected":
            return ObjectiveResult(loss=torch.tensor(1.0, requires_grad=True))
        loss = output.logits.square().sum()
        return ObjectiveResult(
            loss=loss,
            metrics={"tensor_metric": torch.tensor(2.0), "float_metric": 3.0},
        )

    def compute(
        self,
        query_embeddings: Tensor,
        candidate_embeddings: Tensor,
        layout: GroupLayout,
    ) -> ObjectiveResult:
        del query_embeddings, candidate_embeddings, layout
        raise AssertionError("direct compute is not used by TrainingStep")


class _UncheckedMetricObjective(_RuntimeObjective):
    def __init__(self, metric: object) -> None:
        super().__init__("square")
        self.metric = metric

    def forward(
        self,
        output: EmbeddingOutput,
        batch: EmbeddingBatch,
        context: ObjectiveContext,
    ) -> ObjectiveResult:
        del batch, context
        assert output.logits is not None
        return cast(
            "ObjectiveResult",
            SimpleNamespace(
                loss=output.logits.square().sum(),
                metrics={"invalid": self.metric},
            ),
        )


class _UncheckedMetricsObjective(_RuntimeObjective):
    def __init__(self, metrics: object) -> None:
        super().__init__("square")
        self.metrics = metrics

    def forward(
        self,
        output: EmbeddingOutput,
        batch: EmbeddingBatch,
        context: ObjectiveContext,
    ) -> ObjectiveResult:
        del batch, context
        assert output.logits is not None
        return cast(
            "ObjectiveResult",
            SimpleNamespace(
                loss=output.logits.square().sum(),
                metrics=self.metrics,
            ),
        )


def _training_step(
    value: float,
    behavior: Literal["square", "sqrt", "disconnected", "nonfinite"],
    *,
    max_gradient_norm: float | None = None,
) -> tuple[_ScalarAdapter, TrainingStep]:
    adapter = _ScalarAdapter(value)
    objective = _RuntimeObjective(behavior)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    return adapter, TrainingStep(
        adapter,
        objective,
        optimizer,
        RepresentationRequest(),
        max_gradient_norm=max_gradient_norm,
    )


@pytest.mark.parametrize(
    ("limit", "match"),
    [
        (0.0, "positive"),
        (-1.0, "positive"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (10**10000, "finite"),
    ],
    ids=("zero", "negative", "nan", "infinite", "overflow"),
)
def test_training_step_rejects_non_positive_or_non_finite_clip_limit(
    limit: float,
    match: str,
) -> None:
    adapter = _ScalarAdapter(1.0)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)

    with pytest.raises(ValueError, match=match):
        TrainingStep(
            adapter,
            _RuntimeObjective("square"),
            optimizer,
            RepresentationRequest(),
            max_gradient_norm=limit,
        )


@pytest.mark.parametrize(
    "limit",
    [True, "1", object(), _IntSubclass(1), _FloatSubclass(1.0)],
)
def test_training_step_rejects_boolean_and_non_numeric_clip_limits(limit: object) -> None:
    adapter = _ScalarAdapter(1.0)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)

    with pytest.raises(TypeError, match="built-in real number"):
        TrainingStep(
            adapter,
            _RuntimeObjective("square"),
            optimizer,
            RepresentationRequest(),
            max_gradient_norm=limit,  # type: ignore[arg-type]
        )


def test_training_step_rejects_non_finite_loss_before_backward() -> None:
    adapter, step = _training_step(1.0, "nonfinite")
    before = adapter.weight.detach().clone()

    with pytest.raises(ObjectiveError, match="non-finite objective loss"):
        step.run(_batch(), ObjectiveContext(global_step=0, epoch=0.0))

    torch.testing.assert_close(adapter.weight, before)
    assert adapter.weight.grad is None


def test_training_step_clears_non_finite_gradients_without_updating() -> None:
    adapter, step = _training_step(0.0, "sqrt")
    before = adapter.weight.detach().clone()

    with pytest.raises(InvalidEmbeddingError, match="gradients contain"):
        step.run(_batch(), ObjectiveContext(global_step=0, epoch=0.0))

    torch.testing.assert_close(adapter.weight, before)
    assert adapter.weight.grad is None


def test_training_step_requires_a_parameter_gradient() -> None:
    _adapter, step = _training_step(1.0, "disconnected")

    with pytest.raises(InvalidEmbeddingError, match="no parameter gradients"):
        step.run(_batch(), ObjectiveContext(global_step=0, epoch=0.0))


def test_training_step_clips_gradients_and_normalizes_scalar_metrics() -> None:
    adapter, step = _training_step(2.0, "square", max_gradient_norm=0.25)

    result = step.run(_batch(), ObjectiveContext(global_step=0, epoch=0.0))

    assert result.loss == pytest.approx(4.0)
    assert result.gradient_norm == pytest.approx(4.0)
    assert result.metrics == {"tensor_metric": 2.0, "float_metric": 3.0}
    assert adapter.weight.grad is not None
    assert adapter.weight.grad.item() == pytest.approx(0.25)
    assert adapter.weight.item() == pytest.approx(1.975)


def test_training_step_updates_without_gradient_clipping() -> None:
    adapter, step = _training_step(2.0, "square")

    result = step.run(_batch(), ObjectiveContext(global_step=0, epoch=0.0))

    assert result.gradient_norm == pytest.approx(4.0)
    assert adapter.weight.grad is not None
    assert adapter.weight.grad.item() == pytest.approx(4.0)
    assert adapter.weight.item() == pytest.approx(1.6)


@pytest.mark.parametrize(
    "metric",
    [
        True,
        "1",
        float("inf"),
        torch.tensor(data=True),
        torch.tensor(float("nan")),
        torch.ones(2),
    ],
)
def test_training_step_defensively_rejects_unchecked_invalid_metrics(metric: object) -> None:
    adapter = _ScalarAdapter(1.0)
    objective = _UncheckedMetricObjective(metric)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    step = TrainingStep(adapter, objective, optimizer, RepresentationRequest())
    before = adapter.weight.detach().clone()

    with pytest.raises(ObjectiveError, match=r"objective metric.*finite numeric scalar|scalar"):
        step.run(_batch(), ObjectiveContext(global_step=0, epoch=0.0))

    torch.testing.assert_close(adapter.weight, before)
    assert adapter.weight.grad is None


@pytest.mark.parametrize(
    ("metrics", "match"),
    [
        ([("metric", 1.0)], "metrics must be a mapping"),
        ({1: 1.0}, "names must be non-empty strings"),
        ({" ": 1.0}, "names must be non-empty strings"),
    ],
)
def test_training_step_defensively_rejects_unchecked_metric_container_and_keys(
    metrics: object,
    match: str,
) -> None:
    adapter = _ScalarAdapter(1.0)
    objective = _UncheckedMetricsObjective(metrics)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    step = TrainingStep(adapter, objective, optimizer, RepresentationRequest())
    before = adapter.weight.detach().clone()

    with pytest.raises(ObjectiveError, match=match):
        step.run(_batch(), ObjectiveContext(global_step=0, epoch=0.0))

    torch.testing.assert_close(adapter.weight, before)
    assert adapter.weight.grad is None


@pytest.mark.parametrize(
    "metric",
    [
        True,
        "1",
        1 + 2j,
        torch.tensor(data=True),
        torch.tensor(1 + 2j),
        torch.tensor(float("inf")),
        torch.ones(2),
    ],
)
def test_objective_result_rejects_non_real_or_non_finite_metrics(metric: object) -> None:
    with pytest.raises(ObjectiveError, match="finite real"):
        ObjectiveResult(
            loss=torch.tensor(1.0),
            metrics={"invalid": metric},  # type: ignore[dict-item]
        )


def test_objective_result_normalizes_python_integer_metric() -> None:
    result = ObjectiveResult(loss=torch.tensor(1.0), metrics={"count": 2})

    assert result.metrics == {"count": 2.0}
    assert type(result.metrics["count"]) is float


@pytest.mark.parametrize(
    ("kwargs", "exception", "match"),
    [
        ({"loss": True}, TypeError, "loss.*built-in real number"),
        ({"loss": "1"}, TypeError, "loss.*built-in real number"),
        ({"loss": _IntSubclass(1)}, TypeError, "loss.*built-in real number"),
        ({"gradient_norm": True}, TypeError, "gradient_norm.*built-in real number"),
        (
            {"gradient_norm": _FloatSubclass(1.0)},
            TypeError,
            "gradient_norm.*built-in real number",
        ),
        ({"metrics": {"flag": True}}, TypeError, "metric.*built-in real number"),
        (
            {"metrics": {"count": _IntSubclass(1)}},
            TypeError,
            "metric.*built-in real number",
        ),
        ({"metrics": [("count", 1.0)]}, TypeError, "metrics must be a mapping"),
        ({"metrics": {1: 1.0}}, TypeError, "metric names must be strings"),
        ({"metrics": {" ": 1.0}}, ValueError, "metric names must be non-empty"),
        ({"loss": float("nan")}, ValueError, "loss.*finite"),
        ({"loss": 10**10000}, ValueError, "loss.*finite"),
        ({"gradient_norm": -1.0}, ValueError, "gradient_norm.*non-negative"),
        ({"metrics": {"bad": float("inf")}}, ValueError, "metric.*finite"),
    ],
)
def test_step_result_enforces_finite_real_scalar_evidence(
    kwargs: dict[str, object],
    exception: type[Exception],
    match: str,
) -> None:
    values: dict[str, object] = {
        "loss": 1.0,
        "gradient_norm": 2.0,
        "metrics": {"count": 3.0},
    }
    values.update(kwargs)

    with pytest.raises(exception, match=match):
        StepResult(**values)  # type: ignore[arg-type]


def test_step_result_normalizes_integer_evidence() -> None:
    result = StepResult(loss=1, gradient_norm=2, metrics={"count": 3})

    assert result.loss == 1.0
    assert result.gradient_norm == 2.0
    assert result.metrics == {"count": 3.0}


def test_template_bridge_requires_callable_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mira.integrations.ms_swift.template_bridge.import_module",
        lambda _name: SimpleNamespace(get_template=None),
    )

    with pytest.raises(ContractError, match="does not expose"):
        SwiftTemplateBridge(object(), SwiftTemplateSpec())


@pytest.mark.parametrize(
    "messages",
    [
        (Message(role="assistant", content="answer"),),
        (
            Message(role="user", content="question"),
            Message(role="system", content="replacement instruction"),
        ),
    ],
)
def test_qwen_role_validation_rejects_instruction_bypass(
    messages: tuple[Message, ...],
) -> None:
    item = MultimodalItem(item_id="roles", messages=messages)

    with pytest.raises(ContractError, match="must start with user"):
        validate_qwen_message_roles(item)
