from __future__ import annotations

import math

import pytest
import torch

from mira.core import ConfigurationError, NumericalError
from mira.evaluation import compute_retrieval_metrics


def test_metrics_support_multiple_positives_and_graded_relevance() -> None:
    ranked = torch.tensor(
        [
            [3.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    result = compute_retrieval_metrics(ranked, ks=(1, 2, 4))

    assert result.metrics["recall@1"] == pytest.approx(0.25)
    assert result.metrics["precision@1"] == pytest.approx(0.5)
    assert result.metrics["hit_rate@2"] == pytest.approx(1.0)
    assert result.metrics["mrr@2"] == pytest.approx(0.75)
    assert result.metrics["map"] == pytest.approx(2 / 3)
    assert result.metrics["mean_rank"] == pytest.approx(1.5)
    assert result.metrics["median_rank"] == pytest.approx(1.5)

    discounts = torch.log2(torch.arange(2, 4, dtype=torch.float64))
    query_0 = 7 / (7 + 1 / discounts[1])
    query_1 = (3 / discounts[1]) / (3 + 1 / discounts[1])
    assert result.metrics["ndcg@2"] == pytest.approx(float((query_0 + query_1) / 2))


def test_metrics_use_valid_rank_positions_and_precision_denominator() -> None:
    ranked = torch.tensor([[1.0, 0.0, 0.0]])
    valid = torch.tensor([[True, False, True]])

    result = compute_retrieval_metrics(
        ranked,
        ideal_relevance=torch.tensor([[0.0, 1.0, 0.0]]),
        valid_mask=valid,
        ks=(2,),
    )

    assert result.metrics["precision@2"] == pytest.approx(1.0)
    assert result.metrics["mean_rank"] == pytest.approx(1.0)


def test_metrics_reject_incomplete_rankings() -> None:
    with pytest.raises(ConfigurationError, match="complete rankings"):
        compute_retrieval_metrics(
            torch.tensor([[1.0, 0.0]]),
            ideal_relevance=torch.tensor([[1.0, 1.0]]),
            ks=(1,),
        )


def test_metrics_reject_queries_without_relevant_candidates() -> None:
    with pytest.raises(ConfigurationError, match="without relevant"):
        compute_retrieval_metrics(torch.zeros((1, 2)), ks=(1,))


def test_metric_outputs_are_finite() -> None:
    result = compute_retrieval_metrics(torch.eye(3), ks=(1, 3))
    assert all(math.isfinite(value) for value in result.metrics.values())


@pytest.mark.parametrize(
    ("ranked", "kwargs", "error", "message"),
    [
        (torch.ones(2), {"ks": (1,)}, ConfigurationError, "non-empty"),
        (
            torch.tensor([[float("nan")]]),
            {"ks": (1,)},
            NumericalError,
            "NaN",
        ),
        (torch.tensor([[-1.0]]), {"ks": (1,)}, ConfigurationError, "non-negative"),
        (torch.tensor([[1.0]]), {"ks": ()}, ConfigurationError, "at least one"),
        (torch.tensor([[1.0]]), {"ks": (0,)}, ConfigurationError, r"within \[1"),
        (torch.tensor([[1.0]]), {"ks": (True,)}, ConfigurationError, "integers"),
        (
            torch.tensor([[1.0, 0.0]]),
            {"ks": (1,), "ideal_relevance": torch.ones((1, 3))},
            ConfigurationError,
            "must match",
        ),
        (
            torch.tensor([[1.0, 0.0]]),
            {"ks": (1,), "valid_mask": torch.ones((1, 2))},
            ConfigurationError,
            "boolean",
        ),
        (
            torch.tensor([[1.0, 0.0]]),
            {"ks": (1,), "valid_mask": torch.zeros((1, 2), dtype=torch.bool)},
            ConfigurationError,
            "no valid",
        ),
    ],
)
def test_metric_contract_validation(
    ranked: torch.Tensor,
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        compute_retrieval_metrics(ranked, **kwargs)  # type: ignore[arg-type]


def test_boolean_relevance_and_overflow_guard() -> None:
    boolean = compute_retrieval_metrics(torch.eye(2, dtype=torch.bool), ks=(1, 2))
    assert boolean.metrics["recall@2"] == pytest.approx(1.0)
    integer = compute_retrieval_metrics(torch.eye(2, dtype=torch.int64), ks=(1, 2))
    assert integer.metrics["recall@2"] == pytest.approx(1.0)
    with pytest.raises(NumericalError, match="too large"):
        compute_retrieval_metrics(torch.tensor([[1_100.0]]), ks=(1,))


def test_metric_device_mismatch_is_explicit() -> None:
    with pytest.raises(ConfigurationError, match="ideal and ranked"):
        compute_retrieval_metrics(
            torch.tensor([[1.0]]),
            ideal_relevance=torch.empty((1, 1), device="meta"),
            ks=(1,),
        )
    with pytest.raises(ConfigurationError, match="valid_mask"):
        compute_retrieval_metrics(
            torch.tensor([[1.0]]),
            valid_mask=torch.empty((1, 1), dtype=torch.bool, device="meta"),
            ks=(1,),
        )
