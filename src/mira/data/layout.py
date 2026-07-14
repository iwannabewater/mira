"""Construction of explicit :class:`mira.core.layout.GroupLayout` values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mira.core.exceptions import DataValidationError
from mira.core.layout import CrossDeviceLayoutMetadata, GroupLayout

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mira.data.schema import EmbeddingRecord, MultimodalItem


@dataclass(frozen=True, slots=True)
class LayoutBuildResult:
    """A layout and the exact flattened item order it indexes."""

    layout: GroupLayout
    flattened_items: tuple[MultimodalItem, ...]
    cross_device_metadata: CrossDeviceLayoutMetadata


def build_group_layout(  # noqa: PLR0912, PLR0913 - explicit mask controls are the contract
    records: Sequence[EmbeddingRecord],
    *,
    include_in_batch_negatives: bool = True,
    positive_weights: Mapping[str, Sequence[float]] | None = None,
    use_relevance_as_positive_weights: bool = False,
    apply_metadata_false_negative_mask: bool = True,
    device: torch.device | str | None = None,
) -> LayoutBuildResult:
    """Flatten records and construct every positive/negative/mask relation.

    Candidate columns contain each record's positives followed by its explicit
    negatives. Other records' positives become in-batch negatives when enabled;
    other records' explicit negatives never leak into a query's denominator.
    Candidate item identity and known group/product/hash equivalence generate
    explicit false-negative masks.
    """
    if not records:
        raise DataValidationError("cannot build GroupLayout from an empty record sequence")
    sample_ids = [record.sample_id for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise DataValidationError("batch records must have unique sample_id values")
    if positive_weights is not None and use_relevance_as_positive_weights:
        raise DataValidationError(
            "provide explicit positive_weights or use relevance labels, not both"
        )

    target_device = torch.device("cpu") if device is None else torch.device(device)
    query_count = len(records)
    candidates: list[MultimodalItem] = []
    candidate_owners: list[int] = []
    candidate_is_positive: list[bool] = []
    candidate_local_positions: list[int] = []
    for owner, record in enumerate(records):
        for position, positive_item in enumerate(record.positives):
            candidates.append(positive_item)
            candidate_owners.append(owner)
            candidate_is_positive.append(True)
            candidate_local_positions.append(position)
        for position, negative_item in enumerate(record.negatives):
            candidates.append(negative_item)
            candidate_owners.append(owner)
            candidate_is_positive.append(False)
            candidate_local_positions.append(position)

    candidate_count = len(candidates)
    shape = (query_count, candidate_count)
    positive_mask = torch.zeros(shape, dtype=torch.bool, device=target_device)
    explicit_negative_mask = torch.zeros_like(positive_mask)
    in_batch_negative_mask = torch.zeros_like(positive_mask)
    false_negative_mask = torch.zeros_like(positive_mask)

    for query, record in enumerate(records):
        own_positive_ids = {candidate.item_id for candidate in record.positives}
        for column, candidate_item in enumerate(candidates):
            owner = candidate_owners[column]
            is_positive = candidate_is_positive[column]
            if owner == query and is_positive:
                positive_mask[query, column] = True
            elif owner == query:
                explicit_negative_mask[query, column] = True
                if apply_metadata_false_negative_mask and _is_explicit_false_negative(
                    record,
                    candidate_local_position=candidate_local_positions[column],
                ):
                    false_negative_mask[query, column] = True
            elif include_in_batch_negatives and is_positive:
                in_batch_negative_mask[query, column] = True
                if apply_metadata_false_negative_mask and _is_false_negative(
                    query_record=record,
                    candidate_record=records[owner],
                    candidate_local_position=candidate_local_positions[column],
                    candidate_item_id=candidate_item.item_id,
                    own_positive_ids=own_positive_ids,
                ):
                    false_negative_mask[query, column] = True

    candidate_mask = (
        positive_mask | explicit_negative_mask | in_batch_negative_mask
    ) & ~false_negative_mask
    weights = _build_positive_weights(
        records,
        positive_mask,
        candidate_owners,
        candidate_local_positions,
        explicit_weights=positive_weights,
        use_relevance=use_relevance_as_positive_weights,
    )
    relevance_scores = _build_aligned_scores(
        records,
        candidate_owners,
        candidate_is_positive,
        candidate_local_positions,
        field_name="relevance_labels",
        shape=shape,
        device=target_device,
    )
    teacher_scores = _build_aligned_scores(
        records,
        candidate_owners,
        candidate_is_positive,
        candidate_local_positions,
        field_name="teacher_scores",
        shape=shape,
        device=target_device,
    )

    query_indices = torch.arange(query_count, dtype=torch.long, device=target_device)
    candidate_indices = torch.arange(
        query_count,
        query_count + candidate_count,
        dtype=torch.long,
        device=target_device,
    )
    layout = GroupLayout(
        query_indices=query_indices,
        candidate_indices=candidate_indices,
        positive_mask=positive_mask,
        candidate_mask=candidate_mask,
        explicit_negative_mask=explicit_negative_mask,
        in_batch_negative_mask=in_batch_negative_mask,
        false_negative_mask=false_negative_mask,
        task_ids=tuple(record.task_id for record in records),
        dataset_ids=tuple(record.metadata.dataset_name for record in records),
        positive_weights=weights,
        relevance_scores=relevance_scores,
        teacher_scores=teacher_scores,
    )
    flattened = tuple(record.query for record in records) + tuple(candidates)
    cross_device_metadata = CrossDeviceLayoutMetadata(
        candidate_is_positive=tuple(candidate_is_positive),
        candidate_keys=tuple(
            _candidate_identity_keys(
                records[owner],
                candidate_item_id=candidate.item_id,
                positive_position=(position if is_positive else None),
            )
            for candidate, owner, is_positive, position in zip(
                candidates,
                candidate_owners,
                candidate_is_positive,
                candidate_local_positions,
                strict=True,
            )
        ),
        query_positive_keys=tuple(_query_positive_keys(record) for record in records),
    )
    return LayoutBuildResult(
        layout=layout,
        flattened_items=flattened,
        cross_device_metadata=cross_device_metadata,
    )


class GroupLayoutBuilder:
    """Reusable configuration wrapper around :func:`build_group_layout`."""

    def __init__(
        self,
        *,
        include_in_batch_negatives: bool = True,
        use_relevance_as_positive_weights: bool = False,
        apply_metadata_false_negative_mask: bool = True,
    ) -> None:
        """Store stable in-batch and graded-weight construction options."""
        self.include_in_batch_negatives = include_in_batch_negatives
        self.use_relevance_as_positive_weights = use_relevance_as_positive_weights
        self.apply_metadata_false_negative_mask = apply_metadata_false_negative_mask

    def build(
        self,
        records: Sequence[EmbeddingRecord],
        *,
        positive_weights: Mapping[str, Sequence[float]] | None = None,
        device: torch.device | str | None = None,
    ) -> LayoutBuildResult:
        """Build a layout with this builder's stable semantics."""
        return build_group_layout(
            records,
            include_in_batch_negatives=self.include_in_batch_negatives,
            positive_weights=positive_weights,
            use_relevance_as_positive_weights=self.use_relevance_as_positive_weights,
            apply_metadata_false_negative_mask=self.apply_metadata_false_negative_mask,
            device=device,
        )


def _is_explicit_false_negative(
    record: EmbeddingRecord,
    *,
    candidate_local_position: int,
) -> bool:
    negative_groups = record.metadata.negative_group_ids
    if not negative_groups:
        return False
    candidate_group = negative_groups[candidate_local_position]
    positive_groups = set(record.metadata.positive_group_ids)
    if record.metadata.query_group_id is not None:
        positive_groups.add(record.metadata.query_group_id)
    return candidate_group in positive_groups


def _is_false_negative(
    *,
    query_record: EmbeddingRecord,
    candidate_record: EmbeddingRecord,
    candidate_local_position: int,
    candidate_item_id: str,
    own_positive_ids: set[str],
) -> bool:
    if candidate_item_id in own_positive_ids:
        return True
    query_groups = set(query_record.metadata.positive_group_ids)
    if query_record.metadata.query_group_id is not None:
        query_groups.add(query_record.metadata.query_group_id)
    candidate_groups = candidate_record.metadata.positive_group_ids
    if candidate_groups and candidate_groups[candidate_local_position] in query_groups:
        return True
    for field_name in ("canonical_product_id", "spu_id", "sku_id", "image_hash_group"):
        query_value = getattr(query_record.metadata, field_name)
        candidate_value = getattr(candidate_record.metadata, field_name)
        if query_value is not None and query_value == candidate_value:
            return True
    return False


def _query_positive_keys(record: EmbeddingRecord) -> frozenset[str]:
    keys = {f"item:{candidate.item_id}" for candidate in record.positives}
    keys.update(f"group:{value}" for value in record.metadata.positive_group_ids)
    if record.metadata.query_group_id is not None:
        keys.add(f"group:{record.metadata.query_group_id}")
    keys.update(_record_equivalence_keys(record))
    return frozenset(keys)


def _candidate_identity_keys(
    record: EmbeddingRecord,
    *,
    candidate_item_id: str,
    positive_position: int | None,
) -> frozenset[str]:
    keys = {f"item:{candidate_item_id}"}
    if positive_position is not None and record.metadata.positive_group_ids:
        keys.add(f"group:{record.metadata.positive_group_ids[positive_position]}")
    keys.update(_record_equivalence_keys(record))
    return frozenset(keys)


def _record_equivalence_keys(record: EmbeddingRecord) -> set[str]:
    keys: set[str] = set()
    for field_name in ("canonical_product_id", "spu_id", "sku_id", "image_hash_group"):
        value = getattr(record.metadata, field_name)
        if value is not None:
            keys.add(f"{field_name}:{value}")
    return keys


def _build_positive_weights(  # noqa: PLR0913
    records: Sequence[EmbeddingRecord],
    positive_mask: Tensor,
    owners: Sequence[int],
    local_positions: Sequence[int],
    *,
    explicit_weights: Mapping[str, Sequence[float]] | None,
    use_relevance: bool,
) -> Tensor:
    weights = torch.zeros_like(positive_mask, dtype=torch.float32)
    if explicit_weights is not None:
        unknown = set(explicit_weights).difference(record.sample_id for record in records)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise DataValidationError(f"positive_weights contains unknown sample IDs: {names}")
    for query, record in enumerate(records):
        if explicit_weights is not None and record.sample_id in explicit_weights:
            values = tuple(explicit_weights[record.sample_id])
        elif use_relevance:
            labels = record.metadata.relevance_labels
            if labels is None:
                raise DataValidationError(
                    f"record {record.sample_id!r} has no relevance_labels for positive weights"
                )
            values = labels[: len(record.positives)]
        else:
            values = (1.0,) * len(record.positives)
        if len(values) != len(record.positives):
            raise DataValidationError(
                f"record {record.sample_id!r} has {len(values)} positive weights; "
                f"expected {len(record.positives)}"
            )
        for column, (owner, local_position) in enumerate(zip(owners, local_positions, strict=True)):
            if owner == query and positive_mask[query, column]:
                weights[query, column] = float(values[local_position])
    return weights


def _build_aligned_scores(  # noqa: PLR0913
    records: Sequence[EmbeddingRecord],
    owners: Sequence[int],
    is_positive: Sequence[bool],
    local_positions: Sequence[int],
    *,
    field_name: str,
    shape: tuple[int, int],
    device: torch.device,
) -> Tensor | None:
    supplied = [getattr(record.metadata, field_name) is not None for record in records]
    if not any(supplied):
        return None
    if not all(supplied):
        raise DataValidationError(
            f"{field_name} must be supplied for every record in a batch or for none"
        )
    scores = torch.zeros(shape, dtype=torch.float32, device=device)
    for query, record in enumerate(records):
        values = getattr(record.metadata, field_name)
        if values is None:  # pragma: no cover - guarded above for static narrowing
            raise RuntimeError("unreachable missing aligned scores")
        positive_count = len(record.positives)
        candidate_axes = zip(owners, is_positive, local_positions, strict=True)
        for column, (owner, candidate_positive, local_position) in enumerate(candidate_axes):
            if owner != query:
                continue
            offset = 0 if candidate_positive else positive_count
            scores[query, column] = float(values[offset + local_position])
    return scores
