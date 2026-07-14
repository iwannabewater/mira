from __future__ import annotations

import pytest
import torch

from mira.core import CrossDeviceLayoutMetadata, GroupLayout, LayoutValidationError


def _layout(**changes: object) -> GroupLayout:
    values: dict[str, object] = {
        "query_indices": torch.tensor([0, 1]),
        "candidate_indices": torch.tensor([2, 3, 4]),
        "positive_mask": torch.tensor([[True, False, False], [False, True, False]]),
        "candidate_mask": torch.tensor([[True, False, True], [True, True, True]]),
        "explicit_negative_mask": torch.tensor([[False, True, False], [True, False, False]]),
        "in_batch_negative_mask": torch.tensor([[False, False, True], [False, False, True]]),
        "false_negative_mask": torch.tensor([[False, True, False], [False, False, False]]),
        "task_ids": ("image_to_text", "image_to_text"),
        "dataset_ids": ("catalog", "catalog"),
    }
    values.update(changes)
    return GroupLayout(**values)  # type: ignore[arg-type]


def test_layout_preserves_explicit_denominator_semantics() -> None:
    layout = _layout()

    assert torch.equal(layout.denominator_mask, layout.candidate_mask)
    assert torch.equal(
        layout.valid_negative_mask,
        torch.tensor([[False, False, True], [True, False, True]]),
    )
    assert layout.false_negative_mask[0, 1]
    assert not layout.denominator_mask[0, 1]
    assert torch.equal(
        layout.positive_weights,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert layout.positive_columns(0).tolist() == [0]
    assert layout.denominator_columns(0).tolist() == [0, 2]
    assert layout.num_queries == 2
    assert layout.num_candidates == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_is_positive": ()},
        {"query_positive_keys": ()},
        {"candidate_keys": (frozenset(),)},
    ],
)
def test_cross_device_layout_metadata_rejects_unaligned_or_empty_keys(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "candidate_is_positive": (True,),
        "candidate_keys": (frozenset({"item:candidate"}),),
        "query_positive_keys": (frozenset({"item:candidate"}),),
    }
    values.update(changes)
    with pytest.raises(LayoutValidationError, match="cross-device"):
        CrossDeviceLayoutMetadata(**values)  # type: ignore[arg-type]


def test_layout_uses_graded_positive_weights_without_normalizing() -> None:
    weights = torch.tensor([[0.25, 0.0, 0.0], [0.0, 0.75, 0.0]])
    layout = _layout(positive_weights=weights)

    assert torch.equal(layout.positive_weights, weights)
    assert layout.positive_weights.sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("weight", [0.0, -0.1, 1.1, float("nan")])
def test_layout_rejects_invalid_positive_weights(weight: float) -> None:
    weights = torch.tensor([[weight, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(LayoutValidationError, match=r"positive_weights|positive weights"):
        _layout(positive_weights=weights)


def test_layout_rejects_positive_missing_from_denominator() -> None:
    candidate_mask = torch.tensor([[False, False, True], [True, True, True]])
    with pytest.raises(LayoutValidationError, match="candidate_mask"):
        _layout(candidate_mask=candidate_mask)


def test_layout_rejects_false_negative_in_denominator() -> None:
    candidate_mask = torch.tensor([[True, True, True], [True, True, True]])
    with pytest.raises(LayoutValidationError, match="false-negative"):
        _layout(candidate_mask=candidate_mask)


def test_layout_rejects_source_overlap_and_missing_positive() -> None:
    overlapping = torch.tensor([[False, True, True], [False, False, True]])
    with pytest.raises(LayoutValidationError, match="source masks must be disjoint"):
        _layout(in_batch_negative_mask=overlapping)

    no_positive = torch.zeros((2, 3), dtype=torch.bool)
    with pytest.raises(LayoutValidationError, match="at least one positive"):
        _layout(
            positive_mask=no_positive,
            candidate_mask=(
                torch.tensor([[False, True, True], [True, False, True]])
                & ~torch.tensor([[False, True, False], [False, False, False]])
            ),
        )


def test_layout_rejects_shape_dtype_index_and_metadata_errors() -> None:
    with pytest.raises(LayoutValidationError, match="shape"):
        _layout(positive_mask=torch.ones((1, 3), dtype=torch.bool))
    with pytest.raises(LayoutValidationError, match=r"torch\.bool"):
        _layout(positive_mask=torch.ones((2, 3)))
    with pytest.raises(LayoutValidationError, match="unique"):
        _layout(query_indices=torch.tensor([0, 0]))
    with pytest.raises(LayoutValidationError, match="task_ids"):
        _layout(task_ids=("only-one",))


def test_layout_scores_must_be_finite_on_denominator_only() -> None:
    relevance = torch.tensor([[1.0, float("nan"), 0.0], [0.0, 1.0, 0.0]])
    assert _layout(relevance_scores=relevance).relevance_scores is relevance

    invalid = relevance.clone()
    invalid[0, 0] = float("nan")
    with pytest.raises(LayoutValidationError, match="finite"):
        _layout(relevance_scores=invalid)


def test_layout_query_helpers_are_bounds_checked_and_to_preserves_semantics() -> None:
    layout = _layout()
    moved = layout.to("cpu")
    assert torch.equal(moved.candidate_mask, layout.candidate_mask)
    assert moved.task_ids == layout.task_ids
    with pytest.raises(IndexError, match="outside"):
        layout.positive_columns(2)
    with pytest.raises(IndexError, match="outside"):
        layout.denominator_columns(-1)


def test_layout_rejects_index_and_identifier_edge_cases() -> None:
    with pytest.raises(LayoutValidationError, match="non-empty rank-1"):
        _layout(query_indices=torch.tensor([], dtype=torch.int64))
    with pytest.raises(LayoutValidationError, match="int32 or int64"):
        _layout(query_indices=torch.tensor([0.0, 1.0]))
    with pytest.raises(LayoutValidationError, match="negative flattened"):
        _layout(query_indices=torch.tensor([-1, 0]))
    with pytest.raises(LayoutValidationError, match="non-empty"):
        _layout(dataset_ids=("catalog", " "))


def test_layout_rejects_positive_and_negative_semantic_conflicts() -> None:
    false_positive = torch.tensor([[True, True, False], [False, False, False]])
    with pytest.raises(LayoutValidationError, match="positive pair"):
        _layout(false_negative_mask=false_positive)

    negative_positive = torch.tensor([[True, True, False], [True, False, False]])
    with pytest.raises(LayoutValidationError, match="positive and negative"):
        _layout(explicit_negative_mask=negative_positive)

    unrelated_false_negative = torch.tensor([[False, True, False], [False, False, False]])
    no_source = torch.tensor([[False, False, False], [True, False, False]])
    with pytest.raises(LayoutValidationError, match="must be a subset"):
        _layout(
            explicit_negative_mask=no_source,
            false_negative_mask=unrelated_false_negative,
        )


def test_layout_rejects_weight_shape_dtype_and_nonpositive_positions() -> None:
    with pytest.raises(LayoutValidationError, match="positive_weights has shape"):
        _layout(positive_weights=torch.ones((1, 3)))
    with pytest.raises(LayoutValidationError, match="floating-point"):
        _layout(positive_weights=torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.int64))
    with pytest.raises(LayoutValidationError, match="zero outside"):
        _layout(positive_weights=torch.tensor([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0]]))


def test_layout_rejects_optional_score_shape_and_dtype() -> None:
    with pytest.raises(LayoutValidationError, match="teacher_scores has shape"):
        _layout(teacher_scores=torch.ones((1, 3)))
    with pytest.raises(LayoutValidationError, match="floating-point"):
        _layout(teacher_scores=torch.ones((2, 3), dtype=torch.int64))


def test_layout_rejects_cross_device_tensors() -> None:
    meta_mask = torch.empty((2, 3), dtype=torch.bool, device="meta")
    with pytest.raises(LayoutValidationError, match="same device"):
        _layout(false_negative_mask=meta_mask)

    meta_weights = torch.empty((2, 3), device="meta")
    with pytest.raises(LayoutValidationError, match="mask device"):
        _layout(positive_weights=meta_weights)

    meta_scores = torch.empty((2, 3), device="meta")
    with pytest.raises(LayoutValidationError, match="mask device"):
        _layout(teacher_scores=meta_scores)
