"""Tensor-only collectives for variable local embedding batches."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import torch
import torch.distributed as dist
from torch import Tensor

from mira.core.exceptions import DistributedError

GatherMode: TypeAlias = Literal["none", "detached_remote", "autograd"]
"""Gradient semantics for a cross-rank gather."""

_GATHER_MODES = frozenset(("none", "detached_remote", "autograd"))
_GATHER_MODE_CODES = {"detached_remote": 1, "autograd": 2}


class _AllGatherWithGrad(torch.autograd.Function):
    """Equal-shape all-gather whose backward sums source-slice gradients."""

    @staticmethod
    def forward(
        ctx: Any,
        tensor: Tensor,
        group: dist.ProcessGroup | None,
    ) -> tuple[Tensor, ...]:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        outputs = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(outputs, tensor.contiguous(), group=group)
        return tuple(outputs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        gradients = torch.stack(grad_outputs).contiguous()
        dist.all_reduce(gradients, op=dist.ReduceOp.SUM, group=ctx.group)
        return gradients[ctx.rank], None


@dataclass(frozen=True, slots=True)
class GatheredTensor:
    """A concatenated tensor and the source-rank row boundaries.

    ``offsets`` contains ``world_size + 1`` prefix sums. The rows contributed
    by rank ``r`` are therefore ``tensor[offsets[r]:offsets[r + 1]]``.
    """

    tensor: Tensor
    sizes: tuple[int, ...]
    offsets: tuple[int, ...]
    local_rank: int

    @property
    def local_slice(self) -> slice:
        """Return the slice occupied by the current rank in ``tensor``."""
        return slice(self.offsets[self.local_rank], self.offsets[self.local_rank + 1])


class CollectiveGuard:
    """Detect collective-order divergence before an operation starts.

    Every rank must call :meth:`check` at the same logical boundary. The guard
    exchanges a monotonically increasing sequence number and a stable operation
    digest using tensors. It detects mismatched operation names or ordering. A
    process that never reaches the boundary still relies on the process-group
    timeout configured by the launcher.

    Args:
        group: Process group to guard. ``None`` selects the default group.
    """

    def __init__(self, group: dist.ProcessGroup | None = None) -> None:
        """Create a guard for ``group`` or the default process group."""
        self._group = group
        self._sequence = 0
        self._lock = threading.Lock()

    @property
    def sequence(self) -> int:
        """Return the next expected collective sequence number."""
        return self._sequence

    def check(self, operation: str, *, device: torch.device | None = None) -> None:
        """Verify that every rank reached the same named boundary.

        Args:
            operation: Stable name for the collective about to execute.
            device: Device accepted by the process-group backend. CPU is used
                by default except for NCCL/XCCL, where callers must supply the
                payload device.

        Raises:
            DistributedError: If distributed state is unavailable or ranks
                disagree on sequence or operation.
        """
        if not operation:
            raise DistributedError("collective operation name must not be empty")
        rank, world_size = _distributed_context(self._group)
        if world_size == 1:
            with self._lock:
                self._sequence += 1
            return

        collective_device = _resolve_collective_device(device, self._group)
        with self._lock:
            sequence = self._sequence
            self._sequence += 1

        digest = int.from_bytes(
            hashlib.blake2b(operation.encode("utf-8"), digest_size=8).digest(),
            byteorder="big",
            signed=False,
        ) & ((1 << 63) - 1)
        token = torch.tensor((sequence, digest), dtype=torch.int64, device=collective_device)
        tokens = [torch.empty_like(token) for _ in range(world_size)]
        dist.all_gather(tokens, token, group=self._group)
        observed = tuple((int(item[0].item()), int(item[1].item())) for item in tokens)
        if any(item != observed[0] for item in observed[1:]):
            raise DistributedError(
                "collective order mismatch before "
                f"{operation!r} on rank {rank}: observed rank tokens {observed}"
            )


def gather_variable(
    tensor: Tensor,
    *,
    mode: GatherMode = "none",
    group: dist.ProcessGroup | None = None,
    guard: CollectiveGuard | None = None,
) -> GatheredTensor:
    """Gather row-variable tensors without Python-object collectives.

    The function gathers shapes, pads the leading dimension, gathers equal-size
    tensors, removes padding, and concatenates rank contributions. Trailing
    dimensions must match because the result is concatenated along dimension 0.

    ``detached_remote`` keeps the local slice in the autograd graph and detaches
    every remote slice. ``autograd`` routes gradients from every rank back to
    the rank that produced each slice. ``none`` returns the local tensor and
    never enters a collective.

    Args:
        tensor: Tensor shaped ``[local_rows, ...]``.
        mode: Cross-rank gradient behavior.
        group: Process group to use. ``None`` selects the default group.
        guard: Optional shared guard for collective-order validation.

    Returns:
        Concatenated tensor with per-rank sizes and prefix offsets.

    Raises:
        DistributedError: If state, shapes, backend, or mode are invalid.
    """
    if tensor.ndim == 0:
        raise DistributedError("variable gather requires a tensor with at least one dimension")
    if mode not in _GATHER_MODES:
        raise DistributedError(
            f"unsupported gather mode {mode!r}; expected one of {sorted(_GATHER_MODES)}"
        )
    if mode == "none":
        size = int(tensor.shape[0])
        return GatheredTensor(tensor=tensor, sizes=(size,), offsets=(0, size), local_rank=0)

    rank, world_size = _distributed_context(group)
    if guard is not None:
        guard.check(f"gather_variable:{mode}", device=tensor.device)

    _validate_gather_metadata(tensor, mode=mode, world_size=world_size, group=group)
    shapes = _gather_shapes(tensor, world_size=world_size, group=group)
    trailing_shape = shapes[0][1:]
    mismatched = tuple(
        (source_rank, shape)
        for source_rank, shape in enumerate(shapes)
        if shape[1:] != trailing_shape
    )
    if mismatched:
        raise DistributedError(
            f"variable gather requires equal trailing dimensions; rank shapes are {tuple(shapes)}"
        )

    sizes = tuple(shape[0] for shape in shapes)
    max_rows = max(sizes)
    if max_rows == 0:
        raise DistributedError("variable gather received zero rows on every rank")
    padded = _pad_rows(tensor, max_rows)

    if mode == "detached_remote":
        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded.detach(), group=group)
        gathered[rank] = padded
    else:
        gathered_output = _AllGatherWithGrad.apply(padded, group)  # type: ignore[no-untyped-call]
        gathered = list(cast("tuple[Tensor, ...]", gathered_output))

    chunks = tuple(value[:size] for value, size in zip(gathered, sizes, strict=True))
    result = torch.cat(chunks, dim=0)
    offsets = _prefix_offsets(sizes)
    return GatheredTensor(tensor=result, sizes=sizes, offsets=offsets, local_rank=rank)


def _distributed_context(group: dist.ProcessGroup | None) -> tuple[int, int]:
    if not dist.is_available():
        raise DistributedError("torch.distributed is unavailable in this PyTorch build")
    if not dist.is_initialized():
        raise DistributedError(
            "cross-rank collective requested before torch.distributed.init_process_group"
        )
    try:
        return dist.get_rank(group), dist.get_world_size(group)
    except (RuntimeError, TypeError, ValueError) as error:
        raise DistributedError("invalid or non-member process group") from error


def _resolve_collective_device(
    device: torch.device | None,
    group: dist.ProcessGroup | None,
) -> torch.device:
    backend = str(dist.get_backend(group)).lower()
    accelerator_backend = "nccl" in backend or "xccl" in backend
    if device is None:
        if accelerator_backend:
            raise DistributedError(
                f"{backend} collective guard requires the payload accelerator device"
            )
        return torch.device("cpu")
    if accelerator_backend and device.type == "cpu":
        raise DistributedError(f"{backend} collectives cannot use a CPU guard tensor")
    return device


def _gather_shapes(
    tensor: Tensor,
    *,
    world_size: int,
    group: dist.ProcessGroup | None,
) -> tuple[tuple[int, ...], ...]:
    shape = torch.tensor(tensor.shape, dtype=torch.int64, device=tensor.device)
    ndim = torch.tensor((tensor.ndim,), dtype=torch.int64, device=tensor.device)
    gathered_ndim = [torch.empty_like(ndim) for _ in range(world_size)]
    dist.all_gather(gathered_ndim, ndim, group=group)
    dimensions = tuple(int(value.item()) for value in gathered_ndim)
    if any(value != tensor.ndim for value in dimensions):
        raise DistributedError(f"rank tensor dimensions differ: {dimensions}")

    gathered_shapes = [torch.empty_like(shape) for _ in range(world_size)]
    dist.all_gather(gathered_shapes, shape, group=group)
    return tuple(tuple(int(value) for value in item.tolist()) for item in gathered_shapes)


def _validate_gather_metadata(
    tensor: Tensor,
    *,
    mode: GatherMode,
    world_size: int,
    group: dist.ProcessGroup | None,
) -> None:
    descriptor = f"{tensor.dtype}|{tensor.device.type}"
    descriptor_digest = int.from_bytes(
        hashlib.blake2b(descriptor.encode("utf-8"), digest_size=8).digest(),
        byteorder="big",
        signed=False,
    ) & ((1 << 63) - 1)
    metadata = torch.tensor(
        (_GATHER_MODE_CODES[mode], descriptor_digest),
        dtype=torch.int64,
        device=tensor.device,
    )
    gathered = [torch.empty_like(metadata) for _ in range(world_size)]
    dist.all_gather(gathered, metadata, group=group)
    observed = tuple(tuple(int(value) for value in item.tolist()) for item in gathered)
    if any(item != observed[0] for item in observed[1:]):
        raise DistributedError(
            "gather mode, dtype, or device type differs across ranks; "
            f"observed metadata tokens {observed}"
        )


def _pad_rows(tensor: Tensor, rows: int) -> Tensor:
    missing = rows - int(tensor.shape[0])
    if missing < 0:
        raise DistributedError("internal gather padding size is smaller than the local tensor")
    if missing == 0:
        return tensor.contiguous()
    padding = tensor.new_zeros((missing, *tensor.shape[1:]))
    return torch.cat((tensor, padding), dim=0).contiguous()


def _prefix_offsets(sizes: tuple[int, ...]) -> tuple[int, ...]:
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    return tuple(offsets)
