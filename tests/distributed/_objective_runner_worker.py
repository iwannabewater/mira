from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from mira.core import ContractError, CrossDeviceLayoutMetadata, GroupLayout
from mira.distributed import CollectiveGuard
from mira.objectives import MultiPositiveInfoNCE
from mira.training import DistributedObjectiveRunner

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class _RankCase:
    query_features: torch.Tensor
    candidate_features: torch.Tensor
    layout: GroupLayout
    metadata: CrossDeviceLayoutMetadata


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, rtol=1e-6, atol=1e-7):
        raise AssertionError(f"{actual=} != {expected=}")


def _encoder() -> nn.Linear:
    model = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor(
                [
                    [0.8, -0.2, 0.3],
                    [-0.1, 0.7, 0.5],
                ]
            )
        )
    return model


def _rank_case(rank: int) -> _RankCase:
    if rank == 0:
        query_features = torch.tensor(
            [
                [1.0, 0.2, -0.1],
                [0.1, 1.0, 0.3],
            ]
        )
        candidate_features = torch.tensor(
            [
                [0.9, 0.1, 0.0],
                [0.0, 0.8, 0.4],
                [-0.5, 0.2, 1.0],
            ]
        )
        positive = torch.tensor(
            [
                [True, False, False],
                [False, True, False],
            ]
        )
        explicit = torch.tensor(
            [
                [False, False, True],
                [False, False, True],
            ]
        )
        in_batch = torch.tensor(
            [
                [False, True, False],
                [True, False, False],
            ]
        )
        metadata = CrossDeviceLayoutMetadata(
            candidate_is_positive=(True, True, False),
            candidate_keys=(frozenset(("alpha",)), frozenset(("gamma",)), frozenset(("n0",))),
            query_positive_keys=(frozenset(("alpha",)), frozenset(("gamma",))),
        )
        task_ids = ("retrieval", "retrieval")
        dataset_ids = ("rank-zero", "rank-zero")
    elif rank == 1:
        query_features = torch.tensor([[0.7, -0.3, 0.5]])
        candidate_features = torch.tensor(
            [
                [0.6, 0.2, 0.1],
                [0.1, -0.7, 0.8],
            ]
        )
        positive = torch.tensor([[True, False]])
        explicit = torch.tensor([[False, True]])
        in_batch = torch.zeros_like(positive)
        metadata = CrossDeviceLayoutMetadata(
            candidate_is_positive=(True, False),
            candidate_keys=(frozenset(("alpha",)), frozenset(("n1",))),
            query_positive_keys=(frozenset(("alpha",)),),
        )
        task_ids = ("retrieval",)
        dataset_ids = ("rank-one",)
    else:
        raise ValueError(f"unsupported rank {rank}")

    false_negative = torch.zeros_like(positive)
    layout = GroupLayout(
        query_indices=torch.arange(query_features.shape[0]),
        candidate_indices=torch.arange(candidate_features.shape[0]),
        positive_mask=positive,
        candidate_mask=positive | explicit | in_batch,
        explicit_negative_mask=explicit,
        in_batch_negative_mask=in_batch,
        false_negative_mask=false_negative,
        task_ids=task_ids,
        dataset_ids=dataset_ids,
    )
    return _RankCase(query_features, candidate_features, layout, metadata)


def _expected_global_layout(rank: int) -> GroupLayout:
    if rank == 0:
        positive = torch.tensor(
            [
                [True, False, False, False],
                [False, True, False, False],
            ]
        )
        explicit = torch.tensor(
            [
                [False, False, False, True],
                [False, False, False, True],
            ]
        )
        in_batch = torch.tensor(
            [
                [False, True, True, False],
                [True, False, True, False],
            ]
        )
        false_negative = torch.tensor(
            [
                [False, False, True, False],
                [False, False, False, False],
            ]
        )
        task_ids = ("retrieval", "retrieval")
        dataset_ids = ("rank-zero", "rank-zero")
    elif rank == 1:
        positive = torch.tensor([[False, False, True, False]])
        explicit = torch.tensor([[False, False, False, True]])
        in_batch = torch.tensor([[True, True, False, False]])
        false_negative = torch.tensor([[True, False, False, False]])
        task_ids = ("retrieval",)
        dataset_ids = ("rank-one",)
    else:
        raise ValueError(f"unsupported rank {rank}")

    return GroupLayout(
        query_indices=torch.arange(positive.shape[0]),
        candidate_indices=torch.arange(positive.shape[1]),
        positive_mask=positive,
        candidate_mask=(positive | explicit | in_batch) & ~false_negative,
        explicit_negative_mask=explicit,
        in_batch_negative_mask=in_batch,
        false_negative_mask=false_negative,
        task_ids=task_ids,
        dataset_ids=dataset_ids,
    )


def _assert_layout(actual: GroupLayout, expected: GroupLayout) -> None:
    for field_name in (
        "query_indices",
        "candidate_indices",
        "positive_mask",
        "candidate_mask",
        "explicit_negative_mask",
        "in_batch_negative_mask",
        "false_negative_mask",
        "positive_weights",
    ):
        actual_value = getattr(actual, field_name)
        expected_value = getattr(expected, field_name)
        if not torch.equal(actual_value, expected_value):
            raise AssertionError(f"layout field {field_name!r} differs")
    assert actual.task_ids == expected.task_ids
    assert actual.dataset_ids == expected.dataset_ids


def _reference_loss_and_gradient() -> tuple[torch.Tensor, torch.Tensor]:
    model = _encoder()
    rank_zero = _rank_case(0)
    rank_one = _rank_case(1)
    all_positive_features = torch.cat(
        (rank_zero.candidate_features[:2], rank_one.candidate_features[:1]), dim=0
    )
    all_positive_embeddings = model(all_positive_features)
    rank_zero_candidates = torch.cat(
        (all_positive_embeddings, model(rank_zero.candidate_features[2:])), dim=0
    )
    rank_one_candidates = torch.cat(
        (all_positive_embeddings, model(rank_one.candidate_features[1:])), dim=0
    )
    objective = MultiPositiveInfoNCE(temperature=0.4)
    rank_zero_result = objective.compute(
        model(rank_zero.query_features),
        rank_zero_candidates,
        _expected_global_layout(0),
    )
    rank_one_result = objective.compute(
        model(rank_one.query_features),
        rank_one_candidates,
        _expected_global_layout(1),
    )
    loss = (rank_zero_result.loss * 2 + rank_one_result.loss) / 3
    loss.backward()
    assert model.weight.grad is not None
    return loss.detach(), model.weight.grad.detach()


def _assert_contract_error(call: Callable[[], object], pattern: str) -> None:
    try:
        call()
    except ContractError as error:
        if pattern not in str(error):
            raise AssertionError(f"{pattern!r} not found in {str(error)!r}") from error
    else:
        raise AssertionError(f"expected ContractError containing {pattern!r}")


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    rank = dist.get_rank()
    guard = CollectiveGuard()
    try:
        rank_case = _rank_case(rank)
        model = _encoder()
        ddp_model = DistributedDataParallel(model)
        all_local_features = torch.cat(
            (rank_case.query_features, rank_case.candidate_features), dim=0
        )
        all_local_embeddings = ddp_model(all_local_features)
        query_count = rank_case.query_features.shape[0]
        query_embeddings = all_local_embeddings[:query_count]
        candidate_embeddings = all_local_embeddings[query_count:]

        runner = DistributedObjectiveRunner(
            MultiPositiveInfoNCE(temperature=0.4),
            gather_mode="autograd",
            guard=guard,
        )
        result = runner.compute(
            query_embeddings,
            candidate_embeddings,
            rank_case.layout,
            rank_case.metadata,
        )

        assert result.candidate_sizes == (2, 1)
        assert result.global_layout.num_candidates == 4
        assert result.global_layout.explicit_negative_mask.any(dim=0).sum().item() == 1
        _assert_layout(result.global_layout, _expected_global_layout(rank))
        assert result.objective.metrics["distributed/global_query_count"] == 3.0

        result.objective.loss.backward()
        reference_loss, reference_gradient = _reference_loss_and_gradient()
        reporting_loss = result.objective.metrics["distributed/reporting_loss"]
        assert isinstance(reporting_loss, torch.Tensor)
        _assert_close(reporting_loss, reference_loss)
        assert model.weight.grad is not None
        _assert_close(model.weight.grad, reference_gradient)

        _assert_contract_error(
            lambda: runner.compute(
                query_embeddings,
                candidate_embeddings,
                rank_case.layout,
                None,
            ),
            "require candidate roles and identity keys",
        )
        for score_name in ("relevance_scores", "teacher_scores"):
            scored_layout = replace(
                rank_case.layout,
                **{
                    score_name: torch.zeros(
                        (rank_case.layout.num_queries, rank_case.layout.num_candidates)
                    )
                },
            )
            _assert_contract_error(
                lambda scored_layout=scored_layout: runner.compute(
                    query_embeddings,
                    candidate_embeddings,
                    scored_layout,
                    rank_case.metadata,
                ),
                "explicit remote score contract",
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
