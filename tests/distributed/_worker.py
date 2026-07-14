from __future__ import annotations

from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from mira.core import DistributedError
from mira.distributed import CollectiveGuard, gather_variable, global_query_mean


def _assert_close(actual: torch.Tensor, expected: torch.Tensor | float) -> None:
    expected_tensor = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
    if not torch.allclose(actual, expected_tensor, rtol=1e-6, atol=1e-7):
        raise AssertionError(f"{actual=} != {expected_tensor=}")


def main() -> None:  # noqa: PLR0915 - one ordered script is the collective parity oracle
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    rank = dist.get_rank()
    guard = CollectiveGuard()
    try:
        query_rows = 2 if rank == 0 else 1
        candidate_rows = 3 if rank == 0 else 2
        queries = torch.full((query_rows, 2), float(rank))
        candidates = torch.full((candidate_rows, 2), float(rank + 10))
        gathered_queries = gather_variable(queries, mode="detached_remote", guard=guard)
        gathered_candidates = gather_variable(candidates, mode="detached_remote", guard=guard)
        assert gathered_queries.sizes == (2, 1)
        assert gathered_queries.offsets == (0, 2, 3)
        assert gathered_candidates.sizes == (3, 2)
        assert gathered_candidates.offsets == (0, 3, 5)
        assert gathered_queries.local_slice == (slice(0, 2) if rank == 0 else slice(2, 3))

        partial_empty = torch.empty((0, 2)) if rank == 0 else torch.ones((1, 2))
        partial_result = gather_variable(partial_empty, mode="detached_remote", guard=guard)
        assert partial_result.sizes == (0, 1)
        assert partial_result.offsets == (0, 0, 1)
        assert partial_result.tensor.shape == (1, 2)

        ndim_mismatch_caught = False
        try:
            mismatched_ndim = torch.ones(1) if rank == 0 else torch.ones((1, 1))
            gather_variable(mismatched_ndim, mode="detached_remote", guard=guard)
        except DistributedError:
            ndim_mismatch_caught = True
        ndim_count = torch.tensor(int(ndim_mismatch_caught))
        dist.all_reduce(ndim_count)
        assert ndim_count.item() == 2

        shape_mismatch_caught = False
        try:
            mismatched_shape = torch.ones((1, 2 if rank == 0 else 3))
            gather_variable(mismatched_shape, mode="detached_remote", guard=guard)
        except DistributedError:
            shape_mismatch_caught = True
        shape_count = torch.tensor(int(shape_mismatch_caught))
        dist.all_reduce(shape_count)
        assert shape_count.item() == 2

        metadata_mismatch_caught = False
        try:
            gather_variable(
                torch.ones(1, dtype=torch.float32 if rank == 0 else torch.float64),
                mode="detached_remote" if rank == 0 else "autograd",
            )
        except DistributedError:
            metadata_mismatch_caught = True
        mismatch_count = torch.tensor(int(metadata_mismatch_caught))
        dist.all_reduce(mismatch_count)
        assert mismatch_count.item() == 2

        detached = torch.arange(rank + 1.0, requires_grad=True).reshape(rank + 1, 1)
        detached.retain_grad()
        detached_result = gather_variable(detached, mode="detached_remote", guard=guard)
        (detached_result.tensor * (rank + 1)).sum().backward()
        _assert_close(detached.grad, torch.full_like(detached, rank + 1.0))
        dist.barrier()

        autograd_value = torch.arange(rank + 1.0, requires_grad=True).reshape(rank + 1, 1)
        autograd_value.retain_grad()
        autograd_result = gather_variable(autograd_value, mode="autograd", guard=guard)
        (autograd_result.tensor * (rank + 1)).sum().backward()
        _assert_close(autograd_value.grad, torch.full_like(autograd_value, 3.0))

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.5)
        ddp = DistributedDataParallel(model)
        if rank == 0:
            x = torch.tensor([[1.0], [2.0]])
            y = torch.tensor([0.0, 1.0])
        else:
            x = torch.tensor([[3.0]])
            y = torch.tensor([2.0])
        losses = (ddp(x).squeeze(1) - y).square()
        mean = global_query_mean(losses, guard=guard)
        mean.backward_loss.backward()
        _assert_close(model.weight.grad, -2 / 3)
        _assert_close(mean.reporting_loss, 1 / 6)
        assert mean.global_count == 3

        caught = False
        try:
            guard.check("rank-zero-operation" if rank == 0 else "rank-one-operation")
        except DistributedError:
            caught = True
        caught_count = torch.tensor(int(caught))
        dist.all_reduce(caught_count)
        assert caught_count.item() == 2
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
