"""Explicit tensor semantics for retrieval groups.

``GroupLayout`` is the sole authority for query, positive, negative, and mask
semantics. Objectives must not reconstruct those relationships from labels or
flattened positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from mira.core.exceptions import LayoutValidationError


@dataclass(frozen=True, slots=True)
class CrossDeviceLayoutMetadata:
    """Exact candidate roles and identity keys needed for safe remote negatives."""

    candidate_is_positive: tuple[bool, ...]
    candidate_keys: tuple[frozenset[str], ...]
    query_positive_keys: tuple[frozenset[str], ...]

    def __post_init__(self) -> None:
        """Reject metadata that cannot align to a non-empty retrieval layout."""
        if not self.candidate_is_positive or len(self.candidate_keys) != len(
            self.candidate_is_positive
        ):
            raise LayoutValidationError(
                "cross-device candidate roles and keys must be non-empty and aligned"
            )
        if not self.query_positive_keys:
            raise LayoutValidationError("cross-device query keys must not be empty")
        if any(not keys for keys in (*self.candidate_keys, *self.query_positive_keys)):
            raise LayoutValidationError("cross-device identity key sets must not be empty")


@dataclass(frozen=True, slots=True)
class GroupLayout:
    """Describe query-to-candidate relationships in flattened model output.

    All pairwise masks have shape ``[Q, C]``. ``query_indices`` and
    ``candidate_indices`` map those axes to rows in the flattened representation
    tensor. ``candidate_mask`` is the exact loss denominator: it always includes
    every positive and excludes every false negative. Positive weights default
    to one at positive positions and zero elsewhere; supplied weights are used
    verbatim and are never normalized.

    Args:
        query_indices: Integer tensor of shape ``[Q]``.
        candidate_indices: Integer tensor of shape ``[C]``.
        positive_mask: Boolean tensor of shape ``[Q, C]``.
        candidate_mask: Boolean denominator tensor of shape ``[Q, C]``.
        explicit_negative_mask: Boolean explicit-negative source mask.
        in_batch_negative_mask: Boolean in-batch-negative source mask.
        false_negative_mask: Boolean mask for pairs excluded from the denominator.
        task_ids: Exactly ``Q`` non-empty task identifiers.
        dataset_ids: Exactly ``Q`` non-empty dataset identifiers.
        positive_weights: Floating tensor of shape ``[Q, C]``. Empty means the
            stable default of weight one for every positive.
        relevance_scores: Optional finite relevance tensor of shape ``[Q, C]``.
        teacher_scores: Optional finite teacher tensor of shape ``[Q, C]``.

    Raises:
        LayoutValidationError: If dtype, device, shape, index, mask, identifier,
            or positive-weight invariants are violated.

    Example:
        ``GroupLayout(q, c, pos, denom, explicit, in_batch, false, ("t",), ("d",))``
        constructs a one-query layout without inferring any relationship from
        flattened row order.
    """

    query_indices: Tensor
    candidate_indices: Tensor
    positive_mask: Tensor
    candidate_mask: Tensor
    explicit_negative_mask: Tensor
    in_batch_negative_mask: Tensor
    false_negative_mask: Tensor
    task_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    positive_weights: Tensor = field(default_factory=lambda: torch.empty(0))
    relevance_scores: Tensor | None = None
    teacher_scores: Tensor | None = None

    def __post_init__(self) -> None:
        """Validate all tensor and retrieval-semantic invariants."""
        self._validate_indices("query_indices", self.query_indices)
        self._validate_indices("candidate_indices", self.candidate_indices)
        query_count = self.query_indices.numel()
        candidate_count = self.candidate_indices.numel()
        expected_shape = (query_count, candidate_count)

        masks = {
            "positive_mask": self.positive_mask,
            "candidate_mask": self.candidate_mask,
            "explicit_negative_mask": self.explicit_negative_mask,
            "in_batch_negative_mask": self.in_batch_negative_mask,
            "false_negative_mask": self.false_negative_mask,
        }
        for name, mask in masks.items():
            self._validate_pair_tensor(name, mask, expected_shape, torch.bool)

        tensors = (self.query_indices, self.candidate_indices, *masks.values())
        if len({tensor.device for tensor in tensors}) != 1:
            raise LayoutValidationError("all GroupLayout tensors must be on the same device")

        self._validate_identifier_axis("task_ids", self.task_ids, query_count)
        self._validate_identifier_axis("dataset_ids", self.dataset_ids, query_count)

        if torch.any(self.positive_mask & self.false_negative_mask).item():
            raise LayoutValidationError("a positive pair cannot be marked as a false negative")
        if torch.any(self.explicit_negative_mask & self.in_batch_negative_mask).item():
            raise LayoutValidationError(
                "explicit-negative and in-batch-negative source masks must be disjoint"
            )
        negative_sources = self.explicit_negative_mask | self.in_batch_negative_mask
        if torch.any(self.positive_mask & negative_sources).item():
            raise LayoutValidationError("positive and negative source masks must be disjoint")
        if torch.any(self.false_negative_mask & ~negative_sources).item():
            raise LayoutValidationError(
                "false_negative_mask must be a subset of explicit or in-batch negative sources"
            )

        expected_denominator = (self.positive_mask | negative_sources) & ~self.false_negative_mask
        if not torch.equal(self.candidate_mask, expected_denominator):
            raise LayoutValidationError(
                "candidate_mask must equal all positive/negative sources after false-negative "
                "exclusion"
            )
        if not torch.all(self.positive_mask.sum(dim=1) > 0).item():
            raise LayoutValidationError("every query must have at least one positive")

        if self.positive_weights.numel() == 0:
            weights = self.positive_mask.to(dtype=torch.float32)
            object.__setattr__(self, "positive_weights", weights)
        self._validate_positive_weights(expected_shape)
        self._validate_optional_scores("relevance_scores", self.relevance_scores, expected_shape)
        self._validate_optional_scores("teacher_scores", self.teacher_scores, expected_shape)

    @property
    def denominator_mask(self) -> Tensor:
        """Return the boolean ``[Q, C]`` loss-denominator mask."""
        return self.candidate_mask

    @property
    def valid_negative_mask(self) -> Tensor:
        """Return valid explicit and in-batch negatives with shape ``[Q, C]``."""
        return self.candidate_mask & ~self.positive_mask

    @property
    def num_queries(self) -> int:
        """Return ``Q``, the number of query rows."""
        return self.query_indices.numel()

    @property
    def num_candidates(self) -> int:
        """Return ``C``, the number of candidate rows."""
        return self.candidate_indices.numel()

    def positive_columns(self, query: int) -> Tensor:
        """Return positive candidate-axis indices for one query.

        Args:
            query: Query-axis index in ``[0, Q)``.

        Raises:
            IndexError: If ``query`` is outside the query axis.
        """
        self._validate_query_position(query)
        return torch.nonzero(self.positive_mask[query], as_tuple=False).flatten()

    def denominator_columns(self, query: int) -> Tensor:
        """Return denominator candidate-axis indices for one query.

        Args:
            query: Query-axis index in ``[0, Q)``.

        Raises:
            IndexError: If ``query`` is outside the query axis.
        """
        self._validate_query_position(query)
        return torch.nonzero(self.candidate_mask[query], as_tuple=False).flatten()

    def to(self, device: torch.device | str) -> GroupLayout:
        """Return an equivalent layout whose tensors live on ``device``."""
        return GroupLayout(
            query_indices=self.query_indices.to(device),
            candidate_indices=self.candidate_indices.to(device),
            positive_mask=self.positive_mask.to(device),
            candidate_mask=self.candidate_mask.to(device),
            explicit_negative_mask=self.explicit_negative_mask.to(device),
            in_batch_negative_mask=self.in_batch_negative_mask.to(device),
            false_negative_mask=self.false_negative_mask.to(device),
            task_ids=self.task_ids,
            dataset_ids=self.dataset_ids,
            positive_weights=self.positive_weights.to(device),
            relevance_scores=(
                None if self.relevance_scores is None else self.relevance_scores.to(device)
            ),
            teacher_scores=(
                None if self.teacher_scores is None else self.teacher_scores.to(device)
            ),
        )

    @staticmethod
    def _validate_indices(name: str, tensor: Tensor) -> None:
        if tensor.ndim != 1 or tensor.numel() == 0:
            raise LayoutValidationError(f"{name} must be a non-empty rank-1 tensor")
        if tensor.dtype not in {torch.int32, torch.int64}:
            raise LayoutValidationError(f"{name} must use int32 or int64 dtype")
        if torch.any(tensor < 0).item():
            raise LayoutValidationError(f"{name} cannot contain negative flattened indices")
        if torch.unique(tensor).numel() != tensor.numel():
            raise LayoutValidationError(f"{name} must contain unique flattened indices")

    @staticmethod
    def _validate_pair_tensor(
        name: str, tensor: Tensor, expected_shape: tuple[int, int], dtype: torch.dtype
    ) -> None:
        if tuple(tensor.shape) != expected_shape:
            raise LayoutValidationError(
                f"{name} has shape {tuple(tensor.shape)}; expected {expected_shape}"
            )
        if tensor.dtype != dtype:
            raise LayoutValidationError(f"{name} must use {dtype} dtype")

    @staticmethod
    def _validate_identifier_axis(name: str, values: tuple[str, ...], expected: int) -> None:
        if len(values) != expected:
            raise LayoutValidationError(f"{name} has {len(values)} entries; expected {expected}")
        if any(not value.strip() for value in values):
            raise LayoutValidationError(f"{name} entries must be non-empty")

    def _validate_positive_weights(self, expected_shape: tuple[int, int]) -> None:
        weights = self.positive_weights
        if tuple(weights.shape) != expected_shape:
            raise LayoutValidationError(
                f"positive_weights has shape {tuple(weights.shape)}; expected {expected_shape}"
            )
        if not weights.dtype.is_floating_point:
            raise LayoutValidationError("positive_weights must use a floating-point dtype")
        if weights.device != self.positive_mask.device:
            raise LayoutValidationError("positive_weights must be on the mask device")
        if not torch.isfinite(weights).all().item():
            raise LayoutValidationError("positive_weights must contain only finite values")
        if torch.any(weights[~self.positive_mask] != 0).item():
            raise LayoutValidationError("positive_weights must be zero outside positive pairs")
        selected = weights[self.positive_mask]
        if torch.any((selected <= 0) | (selected > 1)).item():
            raise LayoutValidationError("positive weights must satisfy 0 < weight <= 1")

    def _validate_optional_scores(
        self, name: str, tensor: Tensor | None, expected_shape: tuple[int, int]
    ) -> None:
        if tensor is None:
            return
        if tuple(tensor.shape) != expected_shape:
            raise LayoutValidationError(
                f"{name} has shape {tuple(tensor.shape)}; expected {expected_shape}"
            )
        if not tensor.dtype.is_floating_point:
            raise LayoutValidationError(f"{name} must use a floating-point dtype")
        if tensor.device != self.positive_mask.device:
            raise LayoutValidationError(f"{name} must be on the mask device")
        if not torch.isfinite(tensor[self.candidate_mask]).all().item():
            raise LayoutValidationError(f"{name} must be finite at denominator positions")

    def _validate_query_position(self, query: int) -> None:
        if query < 0 or query >= self.num_queries:
            raise IndexError(f"query index {query} is outside [0, {self.num_queries})")
