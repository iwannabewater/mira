"""Mathematical and gradient evidence for Phase-1 objectives."""

from __future__ import annotations

import math

import pytest
import torch

from mira.core.contracts import ObjectiveContext
from mira.core.exceptions import InvalidEmbeddingError, ObjectiveError
from mira.core.layout import GroupLayout
from mira.objectives import (
    LinearWarmupSchedule,
    MatryoshkaObjective,
    MultiPositiveInfoNCE,
    ObjectiveComposer,
    ObjectiveTerm,
)


def _layout(
    positive_mask: torch.Tensor,
    *,
    denominator_mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> GroupLayout:
    query_count, candidate_count = positive_mask.shape
    if denominator_mask is None:
        denominator_mask = torch.ones_like(positive_mask)
    negative_mask = denominator_mask & ~positive_mask
    return GroupLayout(
        query_indices=torch.arange(query_count),
        candidate_indices=torch.arange(query_count, query_count + candidate_count),
        positive_mask=positive_mask,
        candidate_mask=denominator_mask,
        explicit_negative_mask=negative_mask,
        in_batch_negative_mask=torch.zeros_like(negative_mask),
        false_negative_mask=torch.zeros_like(negative_mask),
        task_ids=("task",) * query_count,
        dataset_ids=("dataset",) * query_count,
        positive_weights=(positive_mask.float() if weights is None else weights),
    )


def test_positive_similarity_increase_lowers_infonce() -> None:
    objective = MultiPositiveInfoNCE(temperature=0.2, normalize_embeddings=False)
    layout = _layout(torch.tensor([[True, False]]))
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    low = objective.compute(torch.tensor([[0.2, 1.0]]), candidates, layout).loss
    high = objective.compute(torch.tensor([[1.0, 0.2]]), candidates, layout).loss
    assert high < low


def test_negative_similarity_increase_raises_infonce() -> None:
    objective = MultiPositiveInfoNCE(temperature=0.2, normalize_embeddings=False)
    layout = _layout(torch.tensor([[True, False]]))
    query = torch.tensor([[1.0, 0.0]])
    low = objective.compute(query, torch.tensor([[1.0, 0.0], [-1.0, 0.0]]), layout).loss
    high = objective.compute(query, torch.tensor([[1.0, 0.0], [0.9, 0.0]]), layout).loss
    assert high > low


def test_negative_permutation_is_invariant() -> None:
    objective = MultiPositiveInfoNCE(temperature=0.3)
    layout = _layout(torch.tensor([[True, False, False]]))
    query = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    original = objective.compute(query, candidates, layout).loss
    permutation = torch.tensor([0, 2, 1])
    permuted_layout = _layout(layout.positive_mask[:, permutation])
    permuted = objective.compute(query, candidates[permutation], permuted_layout).loss
    torch.testing.assert_close(original, permuted)


def test_adding_high_similarity_positive_does_not_worsen_loss() -> None:
    objective = MultiPositiveInfoNCE(temperature=0.2)
    query = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor([[0.8, 0.2], [1.0, 0.0], [0.0, 1.0]])
    one_positive = _layout(torch.tensor([[True, False, False]]))
    two_positives = _layout(torch.tensor([[True, True, False]]))
    loss_one = objective.compute(query, candidates, one_positive).loss
    loss_two = objective.compute(query, candidates, two_positives).loss
    assert loss_two <= loss_one


def test_all_masked_candidates_raise_explicit_error() -> None:
    layout = _layout(torch.tensor([[True, False]]))
    layout.candidate_mask.fill_(value=False)
    objective = MultiPositiveInfoNCE()
    with pytest.raises(ObjectiveError, match="all candidates masked"):
        objective.compute(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            layout,
        )


def test_symmetric_loss_matches_manual_logsumexp() -> None:
    temperature = 0.5
    objective = MultiPositiveInfoNCE(
        temperature=temperature, symmetric=True, normalize_embeddings=False
    )
    positive = torch.eye(2, dtype=torch.bool)
    layout = _layout(positive)
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    logits = queries @ candidates.T / temperature
    manual_forward = (torch.logsumexp(logits, dim=1) - logits.diagonal()).mean()
    manual_reverse = (torch.logsumexp(logits.T, dim=1) - logits.diagonal()).mean()
    result = objective.compute(queries, candidates, layout)
    torch.testing.assert_close(result.loss, (manual_forward + manual_reverse) / 2)


def test_graded_positive_weights_are_used_without_normalization() -> None:
    weights = torch.tensor([[0.25, 1.0, 0.0]])
    layout = _layout(torch.tensor([[True, True, False]]), weights=weights)
    objective = MultiPositiveInfoNCE(temperature=1.0, normalize_embeddings=False)
    query = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    result = objective.compute(query, candidates, layout)
    logits = torch.tensor([1.0, 0.0, -1.0])
    expected = torch.logsumexp(logits, dim=0) - torch.logsumexp(
        torch.tensor([1.0 + math.log(0.25), 0.0]), dim=0
    )
    torch.testing.assert_close(result.loss, expected)
    assert result.diagnostics["positive_weights_normalized"] is False


def test_infonce_has_nonzero_embedding_and_temperature_gradients() -> None:
    objective = MultiPositiveInfoNCE(temperature=0.4, learnable_temperature=True)
    layout = _layout(torch.tensor([[True, False], [False, True]]))
    queries = torch.tensor([[1.0, 0.2], [0.1, 1.0]], requires_grad=True)
    candidates = torch.tensor([[0.8, 0.1], [0.2, 0.9]], requires_grad=True)
    result = objective.compute(queries, candidates, layout)
    result.loss.backward()
    assert queries.grad is not None
    assert torch.count_nonzero(queries.grad)
    assert candidates.grad is not None
    assert torch.count_nonzero(candidates.grad)
    assert objective.log_temperature.grad is not None
    assert torch.count_nonzero(objective.log_temperature.grad)


def test_mrl_slices_renormalize_and_share_gradient() -> None:
    base = MultiPositiveInfoNCE(temperature=0.2)
    objective = MatryoshkaObjective(base, {2: 0.4, 4: 0.6})
    layout = _layout(torch.tensor([[True, False], [False, True]]))
    queries = torch.tensor([[3.0, 4.0, 10.0, 0.0], [4.0, 3.0, 0.0, 10.0]], requires_grad=True)
    candidates = torch.tensor([[4.0, 3.0, 10.0, 0.0], [3.0, 4.0, 0.0, 10.0]], requires_grad=True)
    result = objective.compute(queries, candidates, layout)
    result.loss.backward()
    assert queries.grad is not None
    assert torch.count_nonzero(queries.grad[:, :2])
    assert torch.count_nonzero(queries.grad[:, 2:])
    per_dimension = result.diagnostics["per_dimension"]
    assert set(per_dimension) == {2, 4}


def test_mrl_normalizes_even_when_child_normalization_is_disabled() -> None:
    base = MultiPositiveInfoNCE(temperature=0.5, normalize_embeddings=False)
    objective = MatryoshkaObjective(base, (2,))
    layout = _layout(torch.tensor([[True, False]]))
    queries = torch.tensor([[3.0, 4.0, 99.0]])
    candidates = torch.tensor([[4.0, 3.0, 99.0], [-3.0, -4.0, 99.0]])
    result = objective.compute(queries, candidates, layout)
    expected = base.compute(
        torch.nn.functional.normalize(queries[:, :2], dim=-1),
        torch.nn.functional.normalize(candidates[:, :2], dim=-1),
        layout,
    )
    torch.testing.assert_close(result.loss, expected.loss)


@pytest.mark.parametrize(
    ("dimensions", "weights", "match"),
    [
        ((4, 2), (1.0, 1.0), "strictly increasing"),
        ((2, 4), (-1.0, 1.0), "non-negative"),
        ((2, 4), (0.0, 0.0), "not all be zero"),
    ],
)
def test_mrl_rejects_invalid_configuration(
    dimensions: tuple[int, ...], weights: tuple[float, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        MatryoshkaObjective(MultiPositiveInfoNCE(), dimensions, weights)


def test_mrl_rejects_dimension_above_hidden_size_and_zero_norm_slice() -> None:
    layout = _layout(torch.tensor([[True, False]]))
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(InvalidEmbeddingError, match="exceeds hidden size"):
        MatryoshkaObjective(MultiPositiveInfoNCE(), (3,)).compute(
            torch.tensor([[1.0, 0.0]]), candidates, layout
        )
    with pytest.raises(InvalidEmbeddingError, match=r"dimension 1.*zero-norm"):
        MatryoshkaObjective(MultiPositiveInfoNCE(), (1,)).compute(
            torch.tensor([[0.0, 1.0]]), candidates, layout
        )
    with pytest.raises(InvalidEmbeddingError, match=r"dimension 2.*non-finite L2 norm"):
        MatryoshkaObjective(MultiPositiveInfoNCE(), (2,)).compute(
            torch.tensor([[1e38, 1e38]]), candidates, layout
        )


def test_composer_applies_schedule_and_reports_each_term() -> None:
    objective = MultiPositiveInfoNCE(temperature=0.2)
    composer = ObjectiveComposer(
        (
            ObjectiveTerm(
                "retrieval",
                objective,
                LinearWarmupSchedule(start=0.0, end=1.0, warmup_steps=10),
            ),
        )
    )
    layout = _layout(torch.tensor([[True, False]]))
    result = composer.compute_with_context(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.8, 0.2], [0.0, 1.0]]),
        layout,
        ObjectiveContext(global_step=5, epoch=0.5),
    )
    assert result.metrics["retrieval/weight"] == 0.5
    assert result.metrics["objective_count"] == 1.0
