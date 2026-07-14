from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mira.core import ConfigurationError, NumericalError, OptionalDependencyError
from mira.evaluation import ExactSearchEvaluator, FaissEvaluator, SearchResult


def test_exact_search_and_candidate_filter() -> None:
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor([[0.9, 0.0], [0.0, 0.8], [0.7, 0.7]])
    relevance = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[True, False, True], [False, True, True]])

    report = ExactSearchEvaluator(query_batch_size=1).evaluate(
        queries,
        candidates,
        relevance,
        ks=(1, 2),
        candidate_mask=mask,
    )

    assert report.search.indices.tolist() == [[0, 2, 1], [1, 2, 0]]
    assert report.search.valid_mask.tolist() == [
        [True, True, False],
        [True, True, False],
    ]
    assert report.metrics["recall@1"] == pytest.approx(1.0)
    assert report.metrics["map"] == pytest.approx(1.0)


def test_cosine_search_rejects_zero_norm() -> None:
    evaluator = ExactSearchEvaluator(metric="cosine")
    with pytest.raises(NumericalError, match="zero-norm"):
        evaluator.search(torch.zeros((1, 2)), torch.ones((1, 2)), k=1)


def test_cosine_search_rejects_finite_values_with_overflowing_l2_norm() -> None:
    overflow = torch.tensor([[1e38, 1e38]], dtype=torch.float32)

    with pytest.raises(NumericalError, match="non-finite embedding L2 norm"):
        ExactSearchEvaluator(metric="cosine").search(
            overflow,
            torch.tensor([[1.0, 0.0]]),
            k=1,
        )


def test_exact_search_rejects_empty_candidate_filter() -> None:
    evaluator = ExactSearchEvaluator()
    with pytest.raises(ConfigurationError, match="removes every"):
        evaluator.search(
            torch.ones((1, 2)),
            torch.ones((2, 2)),
            k=1,
            candidate_mask=torch.zeros((1, 2), dtype=torch.bool),
        )


@pytest.mark.optional
def test_faiss_flat_matches_exact_order_or_reports_missing_extra() -> None:
    queries = torch.tensor([[1.0, 0.2], [0.1, 1.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.4, 0.3]])
    if importlib.util.find_spec("faiss") is None:
        with pytest.raises(OptionalDependencyError, match="faiss"):
            FaissEvaluator().search(queries, candidates, k=3)
        return

    if sys.platform == "darwin":
        with pytest.raises(OptionalDependencyError, match="libomp"):
            FaissEvaluator().search(queries, candidates, k=3)
        return

    exact = ExactSearchEvaluator().search(queries, candidates, k=3)
    faiss = FaissEvaluator().search(queries, candidates, k=3)
    assert torch.equal(faiss.indices, exact.indices)
    assert torch.allclose(faiss.scores, exact.scores)


@pytest.mark.parametrize(
    ("indices", "scores", "valid", "message"),
    [
        (torch.tensor([0]), torch.tensor([1.0]), torch.tensor([True]), "non-empty"),
        (torch.tensor([[0]]), torch.tensor([[1.0, 0.0]]), torch.tensor([[True]]), "share"),
        (torch.tensor([[0.0]]), torch.tensor([[1.0]]), torch.tensor([[True]]), "int32"),
        (torch.tensor([[0]]), torch.tensor([[1]]), torch.tensor([[True]]), "floating"),
        (torch.tensor([[-1]]), torch.tensor([[1.0]]), torch.tensor([[True]]), "negative"),
        (torch.tensor([[0]]), torch.tensor([[torch.nan]]), torch.tensor([[True]]), "NaN"),
        (
            torch.tensor([[0, 1]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[False, True]]),
            "precede",
        ),
        (
            torch.tensor([[0, 1]]),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[True, True]]),
            "highest",
        ),
    ],
)
def test_search_result_rejects_invalid_contracts(
    indices: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((ConfigurationError, NumericalError), match=message):
        SearchResult(indices=indices, scores=scores, valid_mask=valid)


@pytest.mark.parametrize(
    ("queries", "candidates", "message"),
    [
        (torch.ones(2), torch.ones((2, 2)), r"\[N, D\]"),
        (torch.empty((0, 2)), torch.ones((2, 2)), "non-empty"),
        (torch.ones((1, 2)), torch.ones((1, 3)), "dimensions differ"),
        (torch.ones((1, 2), dtype=torch.int64), torch.ones((1, 2)), "floating point"),
        (torch.tensor([[float("nan"), 0.0]]), torch.ones((1, 2)), "NaN"),
    ],
)
def test_exact_search_rejects_invalid_embedding_contracts(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((ConfigurationError, NumericalError), match=message):
        ExactSearchEvaluator().search(queries, candidates, k=1)


def test_exact_evaluator_option_and_search_validation() -> None:
    with pytest.raises(ConfigurationError, match="unsupported"):
        ExactSearchEvaluator(metric="bad")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="positive"):
        ExactSearchEvaluator(query_batch_size=0)
    with pytest.raises(ConfigurationError, match="within"):
        ExactSearchEvaluator().search(torch.ones((1, 2)), torch.ones((2, 2)), k=3)
    with pytest.raises(ConfigurationError, match="boolean"):
        ExactSearchEvaluator().search(
            torch.ones((1, 2)),
            torch.ones((2, 2)),
            k=1,
            candidate_mask=torch.ones((1, 2)),
        )


def test_euclidean_search_orders_nearest_candidate_first() -> None:
    result = ExactSearchEvaluator(metric="euclidean").search(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[2.0, 0.0], [0.5, 0.0]]),
        k=2,
    )
    assert result.indices.tolist() == [[1, 0]]


def test_valid_cosine_search_normalizes_both_sides() -> None:
    result = ExactSearchEvaluator(metric="cosine").search(
        torch.tensor([[2.0, 0.0]]),
        torch.tensor([[1.0, 1.0], [3.0, 0.0]]),
        k=2,
    )
    assert result.indices.tolist() == [[1, 0]]


def test_evaluate_validates_relevance_and_complete_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries = torch.ones((1, 2))
    candidates = torch.ones((2, 2))
    evaluator = ExactSearchEvaluator()
    with pytest.raises(ConfigurationError, match="relevance must have"):
        evaluator.evaluate(queries, candidates, torch.ones((1, 3)), ks=(1,))
    with pytest.raises(ConfigurationError, match="share a device"):
        evaluator.evaluate(
            queries,
            candidates,
            torch.empty((1, 2), device="meta"),
            ks=(1,),
        )
    with pytest.raises(ConfigurationError, match="share a device"):
        evaluator.search(queries, torch.empty((2, 2), device="meta"), k=1)

    def wrong_width(*_args: object, **_kwargs: object) -> SearchResult:
        return SearchResult(
            indices=torch.tensor([[0]]),
            scores=torch.tensor([[1.0]]),
            valid_mask=torch.tensor([[True]]),
        )

    monkeypatch.setattr(evaluator, "search", wrong_width)
    with pytest.raises(ConfigurationError, match="full evaluation"):
        evaluator.evaluate(queries, candidates, torch.tensor([[1.0, 0.0]]), ks=(1,))


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        (torch.tensor([[0, 2]]), "out-of-range"),
        (torch.tensor([[0, 0]]), "repeats"),
    ],
)
def test_evaluate_rejects_invalid_complete_candidate_indices(
    monkeypatch: pytest.MonkeyPatch,
    indices: torch.Tensor,
    message: str,
) -> None:
    evaluator = ExactSearchEvaluator()

    def invalid_search(*_args: object, **_kwargs: object) -> SearchResult:
        return SearchResult(
            indices=indices,
            scores=torch.tensor([[1.0, 0.0]]),
            valid_mask=torch.tensor([[True, True]]),
        )

    monkeypatch.setattr(evaluator, "search", invalid_search)
    with pytest.raises(ConfigurationError, match=message):
        evaluator.evaluate(
            torch.ones((1, 2)),
            torch.ones((2, 2)),
            torch.tensor([[1.0, 0.0]]),
            ks=(1,),
        )


class _FakeFaissIndex:
    def __init__(self, _dimension: int, *, euclidean: bool) -> None:
        self.euclidean = euclidean
        self.candidates: np.ndarray | None = None

    def add(self, candidates: np.ndarray) -> None:
        self.candidates = candidates

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.candidates is not None
        if self.euclidean:
            scores = ((queries[:, None, :] - self.candidates[None, :, :]) ** 2).sum(axis=2)
            order = np.argsort(scores, axis=1, kind="stable")[:, :k]
        else:
            scores = queries @ self.candidates.T
            order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(scores, order, axis=1), order.astype(np.int64)


def test_faiss_paths_with_lazy_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_faiss = SimpleNamespace(
        IndexFlatIP=lambda dimension: _FakeFaissIndex(dimension, euclidean=False),
        IndexFlatL2=lambda dimension: _FakeFaissIndex(dimension, euclidean=True),
    )
    monkeypatch.setattr(
        "mira.evaluation.evaluators.importlib.import_module", lambda _name: fake_faiss
    )
    queries = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.0]])

    inner = FaissEvaluator().search(queries, candidates, k=2)
    filtered_l2 = FaissEvaluator(metric="euclidean").search(
        queries,
        candidates,
        k=2,
        candidate_mask=torch.tensor([[False, True, True]]),
    )

    assert inner.indices.tolist() == [[0, 2]]
    assert filtered_l2.indices.tolist() == [[2, 1]]
    assert filtered_l2.valid_mask.all()
    with pytest.raises(ConfigurationError, match="unsupported"):
        FaissEvaluator(metric="bad")  # type: ignore[arg-type]
