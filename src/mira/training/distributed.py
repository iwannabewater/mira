"""Cross-device negative construction and DDP-correct objective scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from mira.core.contracts import ObjectiveResult
from mira.core.exceptions import ContractError
from mira.core.layout import CrossDeviceLayoutMetadata, GroupLayout
from mira.distributed import CollectiveGuard, GatherMode, gather_variable, global_query_mean

if TYPE_CHECKING:
    from torch import Tensor

    from mira.objectives.base import EmbeddingObjective


@dataclass(frozen=True, slots=True)
class CrossDeviceResult:
    """Objective result plus the global candidate layout used to compute it."""

    objective: ObjectiveResult
    global_layout: GroupLayout
    candidate_sizes: tuple[int, ...]


class DistributedObjectiveRunner:
    """Add remote candidates and preserve a true global-query DDP mean."""

    def __init__(
        self,
        objective: EmbeddingObjective,
        *,
        gather_mode: GatherMode = "none",
        group: dist.ProcessGroup | None = None,
        guard: CollectiveGuard | None = None,
    ) -> None:
        """Bind an objective to explicit cross-rank gradient semantics."""
        self.objective = objective
        self.gather_mode = gather_mode
        self.group = group
        self.guard = guard

    def compute(
        self,
        query_embeddings: Tensor,
        candidate_embeddings: Tensor,
        layout: GroupLayout,
        metadata: CrossDeviceLayoutMetadata | None = None,
    ) -> CrossDeviceResult:
        """Compute against global candidates and return a DDP-scaled loss."""
        if self.gather_mode == "none":
            return self._compute_local(query_embeddings, candidate_embeddings, layout)
        if metadata is None:
            raise ContractError("cross-device objectives require candidate roles and identity keys")
        _validate_cross_device_metadata(layout, metadata)
        if layout.relevance_scores is not None or layout.teacher_scores is not None:
            raise ContractError(
                "cross-device relevance/teacher scores require an explicit remote score contract"
            )
        positive_columns = torch.tensor(
            [index for index, value in enumerate(metadata.candidate_is_positive) if value],
            dtype=torch.int64,
            device=candidate_embeddings.device,
        )
        gathered = gather_variable(
            candidate_embeddings.index_select(0, positive_columns),
            mode=self.gather_mode,
            group=self.group,
            guard=self.guard,
        )
        gathered_keys = _gather_candidate_keys(
            tuple(metadata.candidate_keys[index] for index in positive_columns.tolist()),
            group=self.group,
            guard=self.guard,
            device=candidate_embeddings.device,
        )
        explicit_columns = torch.nonzero(
            layout.explicit_negative_mask.any(dim=0), as_tuple=False
        ).flatten()
        explicit_embeddings = candidate_embeddings.index_select(0, explicit_columns)
        global_candidates = torch.cat((gathered.tensor, explicit_embeddings), dim=0)
        global_layout = _build_cross_device_layout(
            layout,
            candidate_sizes=gathered.sizes,
            local_rank=gathered.local_rank,
            local_positive_columns=positive_columns,
            local_explicit_columns=explicit_columns,
            gathered_candidate_keys=gathered_keys,
            query_positive_keys=metadata.query_positive_keys,
        )
        local_result = self.objective.compute(
            query_embeddings,
            global_candidates,
            global_layout,
        )
        mean = global_query_mean(
            local_result.loss * layout.num_queries,
            local_count=layout.num_queries,
            group=self.group,
            guard=self.guard,
        )
        metrics = dict(local_result.metrics)
        metrics["distributed/reporting_loss"] = mean.reporting_loss
        metrics["distributed/global_query_count"] = float(mean.global_count)
        diagnostics = dict(local_result.diagnostics)
        diagnostics["distributed"] = {
            "candidate_sizes": gathered.sizes,
            "gather_mode": self.gather_mode,
            "global_query_count": mean.global_count,
            "local_candidate_slice": (
                gathered.local_slice.start,
                gathered.local_slice.stop,
            ),
            "loss_semantics": "global_query_mean_with_ddp_gradient_correction",
        }
        result = ObjectiveResult(
            loss=mean.backward_loss,
            metrics=metrics,
            diagnostics=diagnostics,
        )
        return CrossDeviceResult(result, global_layout, gathered.sizes)

    def _compute_local(
        self,
        query_embeddings: Tensor,
        candidate_embeddings: Tensor,
        layout: GroupLayout,
    ) -> CrossDeviceResult:
        local_result = self.objective.compute(query_embeddings, candidate_embeddings, layout)
        mean = global_query_mean(
            local_result.loss * layout.num_queries,
            local_count=layout.num_queries,
            group=self.group,
            guard=self.guard,
        )
        metrics = dict(local_result.metrics)
        metrics["distributed/reporting_loss"] = mean.reporting_loss
        metrics["distributed/global_query_count"] = float(mean.global_count)
        diagnostics = dict(local_result.diagnostics)
        diagnostics["distributed"] = {
            "candidate_sizes": (layout.num_candidates,),
            "gather_mode": "none",
            "global_query_count": mean.global_count,
            "local_candidate_slice": (0, layout.num_candidates),
            "loss_semantics": "global_query_mean_with_ddp_gradient_correction",
        }
        result = ObjectiveResult(
            loss=mean.backward_loss,
            metrics=metrics,
            diagnostics=diagnostics,
        )
        return CrossDeviceResult(result, layout, (layout.num_candidates,))


def _build_cross_device_layout(  # noqa: PLR0913 - every alignment axis is explicit
    local: GroupLayout,
    *,
    candidate_sizes: tuple[int, ...],
    local_rank: int,
    local_positive_columns: Tensor,
    local_explicit_columns: Tensor,
    gathered_candidate_keys: tuple[frozenset[str], ...],
    query_positive_keys: tuple[frozenset[str], ...],
) -> GroupLayout:
    if local_rank < 0 or local_rank >= len(candidate_sizes):
        raise ValueError("local rank is outside candidate size metadata")
    if candidate_sizes[local_rank] != local_positive_columns.numel():
        raise ValueError(
            "local positive count disagrees with gathered size metadata: "
            f"{local_positive_columns.numel()} != {candidate_sizes[local_rank]}"
        )
    gathered_positive_count = sum(candidate_sizes)
    total_candidates = gathered_positive_count + local_explicit_columns.numel()
    local_offset = sum(candidate_sizes[:local_rank])
    shape = (local.num_queries, total_candidates)
    device = local.positive_mask.device

    positive = torch.zeros(shape, dtype=torch.bool, device=device)
    explicit = torch.zeros_like(positive)
    in_batch = torch.zeros_like(positive)
    false_negative = torch.zeros_like(positive)
    positive_weights = torch.zeros(shape, dtype=local.positive_weights.dtype, device=device)
    local_slice = slice(local_offset, local_offset + local_positive_columns.numel())
    positive[:, local_slice] = local.positive_mask.index_select(1, local_positive_columns)
    in_batch[:, local_slice] = local.in_batch_negative_mask.index_select(1, local_positive_columns)
    false_negative[:, local_slice] = local.false_negative_mask.index_select(
        1, local_positive_columns
    )
    positive_weights[:, local_slice] = local.positive_weights.index_select(
        1, local_positive_columns
    )
    remote = torch.ones(gathered_positive_count, dtype=torch.bool, device=device)
    remote[local_slice] = False
    for query, query_keys in enumerate(query_positive_keys):
        for column, candidate_keys in enumerate(gathered_candidate_keys):
            if remote[column]:
                in_batch[query, column] = True
                if query_keys.intersection(candidate_keys):
                    false_negative[query, column] = True
    explicit_slice = slice(gathered_positive_count, total_candidates)
    explicit[:, explicit_slice] = local.explicit_negative_mask.index_select(
        1, local_explicit_columns
    )
    false_negative[:, explicit_slice] = local.false_negative_mask.index_select(
        1, local_explicit_columns
    )
    candidate_mask = (positive | explicit | in_batch) & ~false_negative
    return GroupLayout(
        query_indices=torch.arange(local.num_queries, dtype=torch.int64, device=device),
        candidate_indices=torch.arange(total_candidates, dtype=torch.int64, device=device),
        positive_mask=positive,
        candidate_mask=candidate_mask,
        explicit_negative_mask=explicit,
        in_batch_negative_mask=in_batch,
        false_negative_mask=false_negative,
        task_ids=local.task_ids,
        dataset_ids=local.dataset_ids,
        positive_weights=positive_weights,
    )


def _validate_cross_device_metadata(
    layout: GroupLayout,
    metadata: CrossDeviceLayoutMetadata,
) -> None:
    if len(metadata.candidate_is_positive) != layout.num_candidates:
        raise ContractError("cross-device candidate metadata does not align with GroupLayout")
    if len(metadata.query_positive_keys) != layout.num_queries:
        raise ContractError("cross-device query metadata does not align with GroupLayout")
    declared = torch.tensor(
        metadata.candidate_is_positive,
        dtype=torch.bool,
        device=layout.positive_mask.device,
    )
    observed = layout.positive_mask.any(dim=0)
    if not torch.equal(declared, observed):
        raise ContractError("cross-device candidate roles disagree with positive_mask")


def _gather_candidate_keys(
    local_keys: tuple[frozenset[str], ...],
    *,
    group: dist.ProcessGroup | None,
    guard: CollectiveGuard | None,
    device: torch.device,
) -> tuple[frozenset[str], ...]:
    if guard is not None:
        guard.check("gather_cross_device_candidate_keys", device=device)
    world_size = dist.get_world_size(group)
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local_keys, group=group)
    flattened: list[frozenset[str]] = []
    for rank_keys in gathered:
        if not isinstance(rank_keys, tuple) or any(
            not isinstance(keys, frozenset) or any(not isinstance(key, str) for key in keys)
            for keys in rank_keys
        ):
            raise ContractError("cross-device candidate identity metadata is invalid")
        flattened.extend(rank_keys)
    return tuple(flattened)
