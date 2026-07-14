"""Loss scaling that preserves a true global-query mean under DDP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor

from mira.core.exceptions import DistributedError, NumericalError

_INVALID_ZERO_COUNT = 1
_INVALID_NONFINITE = 2

if TYPE_CHECKING:
    from mira.distributed.collectives import CollectiveGuard


@dataclass(frozen=True, slots=True)
class GlobalMeanResult:
    """Backward and reporting views of one global-query mean.

    ``backward_loss`` includes the world-size factor required because DDP
    averages parameter gradients. ``reporting_loss`` is the unscaled global
    mean, detached and identical on every rank.
    """

    backward_loss: Tensor
    reporting_loss: Tensor
    local_count: int
    global_count: int


def global_query_mean(
    local_losses: Tensor,
    *,
    valid_mask: Tensor | None = None,
    local_count: int | None = None,
    group: dist.ProcessGroup | None = None,
    guard: CollectiveGuard | None = None,
) -> GlobalMeanResult:
    """Create a DDP-correct mean when ranks own different query counts.

    Vector input is interpreted as per-query losses. A scalar input is
    interpreted as a pre-summed loss and requires ``local_count``. With DDP,
    the local sum is multiplied by ``world_size / global_count`` for backward;
    DDP's gradient averaging then yields the gradient of the global mean.

    Args:
        local_losses: Per-query losses or a scalar local sum.
        valid_mask: Boolean mask for vector input. Masked queries do not count.
        local_count: Number of queries represented by scalar input.
        group: Process group to use. ``None`` selects the default group.
        guard: Optional shared collective-order guard.

    Returns:
        Backward loss, report loss, and local/global query counts.

    Raises:
        DistributedError: If shapes, counts, or distributed state are invalid.
        NumericalError: If any rank contributes NaN or infinite loss.
    """
    if not local_losses.is_floating_point():
        raise DistributedError("global query mean requires floating-point losses")
    local_sum, count = _local_sum_and_count(local_losses, valid_mask, local_count)

    distributed = dist.is_available() and dist.is_initialized()
    if group is not None and not distributed:
        raise DistributedError("a process group was supplied before distributed initialization")
    world_size = dist.get_world_size(group) if distributed else 1
    if guard is not None and distributed:
        guard.check("global_query_mean", device=local_losses.device)

    finite = bool(torch.isfinite(local_sum.detach()).item())
    zero_count_has_loss = count == 0 and finite and bool(local_sum.detach().ne(0).item())
    invalid_code = _INVALID_NONFINITE if not finite else int(zero_count_has_loss)
    invalid = torch.tensor(
        (invalid_code,),
        dtype=torch.int64,
        device=local_losses.device,
    )
    if distributed:
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=group)
    if int(invalid.item()) == _INVALID_NONFINITE:
        raise NumericalError("at least one rank contributed a non-finite query loss")
    if int(invalid.item()) == _INVALID_ZERO_COUNT:
        raise DistributedError("a zero-query rank must contribute an exactly zero loss sum")

    statistics = torch.stack(
        (
            local_sum.detach().to(dtype=torch.float64),
            torch.tensor(float(count), dtype=torch.float64, device=local_losses.device),
        )
    )
    if distributed:
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM, group=group)

    global_count_float = float(statistics[1].item())
    global_count = int(global_count_float)
    if global_count <= 0 or float(global_count) != global_count_float:
        raise DistributedError(
            f"global query count must be a positive integer, got {global_count_float}"
        )

    backward_loss = local_sum * (world_size / global_count)
    reporting_loss = (statistics[0] / statistics[1]).to(dtype=local_losses.dtype).detach()
    return GlobalMeanResult(
        backward_loss=backward_loss,
        reporting_loss=reporting_loss,
        local_count=count,
        global_count=global_count,
    )


def _local_sum_and_count(
    local_losses: Tensor,
    valid_mask: Tensor | None,
    local_count: int | None,
) -> tuple[Tensor, int]:
    if local_losses.ndim == 0:
        if valid_mask is not None:
            raise DistributedError("valid_mask cannot be used with a scalar local loss sum")
        if local_count is None:
            raise DistributedError("scalar local loss sum requires local_count")
        if local_count < 0:
            raise DistributedError(f"local_count must be non-negative, got {local_count}")
        return local_losses, local_count

    if local_losses.ndim != 1:
        raise DistributedError(
            f"per-query losses must be rank 1, got shape {tuple(local_losses.shape)}"
        )
    if local_count is not None:
        raise DistributedError("local_count is inferred for vector losses and must not be supplied")
    if valid_mask is None:
        return local_losses.sum(), local_losses.numel()
    if valid_mask.shape != local_losses.shape:
        raise DistributedError(
            "valid_mask must match local_losses shape; "
            f"got {tuple(valid_mask.shape)} and {tuple(local_losses.shape)}"
        )
    if valid_mask.dtype is not torch.bool:
        raise DistributedError("valid_mask must have boolean dtype")
    return local_losses.masked_select(valid_mask).sum(), int(valid_mask.sum().item())
