"""Property tests across random groups, masks, and MRL dimensions."""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mira.core.layout import GroupLayout
from mira.objectives import MatryoshkaObjective, MultiPositiveInfoNCE


@settings(max_examples=30, deadline=None)
@given(
    query_count=st.integers(min_value=1, max_value=5),
    extra_candidates=st.integers(min_value=1, max_value=6),
    dimension=st.integers(min_value=2, max_value=16),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_random_group_mask_and_dimension_is_finite_and_permutation_invariant(
    query_count: int, extra_candidates: int, dimension: int, seed: int
) -> None:
    generator = torch.Generator().manual_seed(seed)
    candidate_count = query_count + extra_candidates
    positive = torch.zeros((query_count, candidate_count), dtype=torch.bool)
    positive[torch.arange(query_count), torch.arange(query_count)] = True
    denominator = torch.rand((query_count, candidate_count), generator=generator) > 0.25
    denominator |= positive
    negatives = denominator & ~positive
    layout = GroupLayout(
        query_indices=torch.arange(query_count),
        candidate_indices=torch.arange(query_count, query_count + candidate_count),
        positive_mask=positive,
        candidate_mask=denominator,
        explicit_negative_mask=negatives,
        in_batch_negative_mask=torch.zeros_like(negatives),
        false_negative_mask=torch.zeros_like(negatives),
        task_ids=("task",) * query_count,
        dataset_ids=("data",) * query_count,
    )
    queries = torch.randn((query_count, dimension), generator=generator, requires_grad=True)
    candidates = torch.randn((candidate_count, dimension), generator=generator, requires_grad=True)
    objective = MultiPositiveInfoNCE(temperature=0.3)
    result = objective.compute(queries, candidates, layout)
    assert torch.isfinite(result.loss)

    permutation = torch.randperm(candidate_count, generator=generator)
    permuted_layout = GroupLayout(
        query_indices=layout.query_indices,
        candidate_indices=layout.candidate_indices,
        positive_mask=positive[:, permutation],
        candidate_mask=denominator[:, permutation],
        explicit_negative_mask=negatives[:, permutation],
        in_batch_negative_mask=torch.zeros_like(negatives),
        false_negative_mask=torch.zeros_like(negatives),
        task_ids=layout.task_ids,
        dataset_ids=layout.dataset_ids,
        positive_weights=positive[:, permutation].float(),
    )
    permuted = objective.compute(queries, candidates[permutation], permuted_layout)
    torch.testing.assert_close(result.loss, permuted.loss)

    mrl = MatryoshkaObjective(objective, (dimension,))
    mrl_result = mrl.compute(queries, candidates, layout)
    mrl_result.loss.backward()
    assert queries.grad is not None
    assert torch.isfinite(queries.grad).all()
