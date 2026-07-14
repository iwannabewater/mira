from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.distributed as dist

from mira.core import DistributedError, NumericalError
from mira.distributed import CollectiveGuard, gather_variable, global_query_mean

if TYPE_CHECKING:
    from pathlib import Path


def test_local_gather_and_global_mean_contracts() -> None:
    value = torch.tensor([[1.0], [2.0]], requires_grad=True)
    gathered = gather_variable(value, mode="none")
    assert gathered.tensor is value
    assert gathered.sizes == (2,)
    assert gathered.offsets == (0, 2)
    assert gathered.local_slice == slice(0, 2)

    result = global_query_mean(torch.tensor([1.0, 3.0], requires_grad=True))
    assert result.backward_loss.item() == pytest.approx(2.0)
    assert result.reporting_loss.item() == pytest.approx(2.0)
    assert result.local_count == result.global_count == 2

    masked = global_query_mean(torch.tensor([1.0, 3.0]), valid_mask=torch.tensor([True, False]))
    scalar = global_query_mean(torch.tensor(4.0), local_count=2)
    assert masked.reporting_loss.item() == pytest.approx(1.0)
    assert scalar.reporting_loss.item() == pytest.approx(2.0)


def test_collective_and_loss_validation_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(DistributedError, match="at least one dimension"):
        gather_variable(torch.tensor(1.0))
    with pytest.raises(DistributedError, match="unsupported gather mode"):
        gather_variable(torch.ones(1), mode="bad")  # type: ignore[arg-type]
    with pytest.raises(DistributedError, match=r"before torch\.distributed"):
        gather_variable(torch.ones(1), mode="detached_remote")
    with pytest.raises(DistributedError, match="must not be empty"):
        CollectiveGuard().check("")
    with pytest.raises(DistributedError, match="floating-point"):
        global_query_mean(torch.ones(1, dtype=torch.int64))
    with pytest.raises(DistributedError, match="requires local_count"):
        global_query_mean(torch.tensor(1.0))
    with pytest.raises(DistributedError, match="non-negative"):
        global_query_mean(torch.tensor(1.0), local_count=-1)
    with pytest.raises(DistributedError, match="zero-query rank"):
        global_query_mean(torch.tensor(1.0), local_count=0)
    with pytest.raises(DistributedError, match="rank 1"):
        global_query_mean(torch.ones((1, 1)))
    with pytest.raises(DistributedError, match="must not be supplied"):
        global_query_mean(torch.ones(1), local_count=1)
    with pytest.raises(DistributedError, match="match"):
        global_query_mean(torch.ones(2), valid_mask=torch.ones(1, dtype=torch.bool))
    with pytest.raises(DistributedError, match="boolean"):
        global_query_mean(torch.ones(2), valid_mask=torch.ones(2))
    with pytest.raises(NumericalError, match="non-finite"):
        global_query_mean(torch.tensor([float("nan")]))
    with pytest.raises(DistributedError, match="process group"):
        global_query_mean(torch.ones(1), group=object())  # type: ignore[arg-type]
    with pytest.raises(DistributedError, match="positive integer"):
        global_query_mean(torch.empty(0))
    with pytest.raises(DistributedError, match="cannot be used"):
        global_query_mean(
            torch.tensor(1.0),
            local_count=1,
            valid_mask=torch.ones((), dtype=torch.bool),
        )

    monkeypatch.setattr(dist, "is_available", lambda: False)
    with pytest.raises(DistributedError, match="unavailable"):
        gather_variable(torch.ones(1), mode="detached_remote")


def test_one_rank_gloo_collective_paths(tmp_path: Path) -> None:
    rendezvous = (tmp_path / "gloo-store").resolve()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        guard = CollectiveGuard()
        guard.check("one-rank")
        assert guard.sequence == 1

        detached_value = torch.ones((2, 2), requires_grad=True)
        detached = gather_variable(detached_value, mode="detached_remote", guard=guard)
        detached.tensor.sum().backward()
        assert torch.equal(detached_value.grad, torch.ones_like(detached_value))

        autograd_value = torch.ones((1, 2), requires_grad=True)
        gathered = gather_variable(autograd_value, mode="autograd", guard=guard)
        gathered.tensor.sum().backward()
        assert torch.equal(autograd_value.grad, torch.ones_like(autograd_value))

        mean = global_query_mean(torch.tensor([1.0, 2.0]), guard=guard)
        assert mean.reporting_loss.item() == pytest.approx(1.5)
        with pytest.raises(DistributedError, match="invalid or non-member"):
            gather_variable(
                torch.ones(1),
                mode="detached_remote",
                group=object(),  # type: ignore[arg-type]
            )
        with pytest.raises(DistributedError, match="zero rows"):
            gather_variable(torch.empty((0, 2)), mode="detached_remote")
    finally:
        dist.destroy_process_group()
