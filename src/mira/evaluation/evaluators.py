"""Exact, FAISS, and external retrieval search interfaces."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, runtime_checkable

import torch
from torch import Tensor

from mira.core.exceptions import ConfigurationError, NumericalError, OptionalDependencyError
from mira.evaluation.metrics import compute_retrieval_metrics

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SearchMetric: TypeAlias = Literal["inner_product", "cosine", "euclidean"]
_SEARCH_METRICS = frozenset(("inner_product", "cosine", "euclidean"))
_EMBEDDING_RANK = 2


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Candidate indices, descending scores, and validity for each query."""

    indices: Tensor
    scores: Tensor
    valid_mask: Tensor

    def __post_init__(self) -> None:
        """Reject malformed, non-finite, or unordered search results."""
        if self.indices.ndim != _EMBEDDING_RANK or self.indices.numel() == 0:
            raise ConfigurationError("search indices must be a non-empty [Q, K] tensor")
        if self.scores.shape != self.indices.shape or self.valid_mask.shape != self.indices.shape:
            raise ConfigurationError("search indices, scores, and valid_mask must share [Q, K]")
        if self.indices.dtype not in {torch.int32, torch.int64}:
            raise ConfigurationError("search indices must use int32 or int64 dtype")
        if not self.scores.is_floating_point() or self.valid_mask.dtype is not torch.bool:
            raise ConfigurationError("search scores must be floating point and validity boolean")
        if bool((self.indices < 0).any().item()):
            raise ConfigurationError("search indices cannot be negative")
        if not bool(torch.isfinite(self.scores[self.valid_mask]).all().item()):
            raise NumericalError("valid search scores contain NaN or infinite values")
        if bool((self.valid_mask[:, 1:] & ~self.valid_mask[:, :-1]).any().item()):
            raise ConfigurationError("valid search results must precede padded invalid entries")
        adjacent_valid = self.valid_mask[:, 1:] & self.valid_mask[:, :-1]
        if bool(((self.scores[:, 1:] > self.scores[:, :-1]) & adjacent_valid).any().item()):
            raise ConfigurationError("search scores must be ordered from highest to lowest")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Retrieval aggregates, query-level evidence, and ranked candidates."""

    metrics: Mapping[str, float]
    per_query: Mapping[str, Tensor]
    search: SearchResult


@runtime_checkable
class ExternalANNEvaluator(Protocol):
    """Protocol implemented by a service that owns its candidate index.

    Returned indices address the stable candidate ordering declared by the
    external index manifest. The protocol deliberately does not accept local
    candidate embeddings, which would defeat an external or remote index.
    """

    def search(
        self,
        query_embeddings: Tensor,
        *,
        k: int,
        filters: Sequence[Mapping[str, object]] | None = None,
    ) -> SearchResult:
        """Search an owned index with optional one-to-one query filters."""
        del query_embeddings, k, filters
        raise NotImplementedError


class _RetrievalEvaluator:
    """Shared evaluation logic for search implementations."""

    def search(
        self,
        query_embeddings: Tensor,
        candidate_embeddings: Tensor,
        *,
        k: int,
        candidate_mask: Tensor | None = None,
    ) -> SearchResult:
        raise NotImplementedError

    def evaluate(
        self,
        query_embeddings: Tensor,
        candidate_embeddings: Tensor,
        relevance: Tensor,
        *,
        ks: Sequence[int],
        candidate_mask: Tensor | None = None,
    ) -> EvaluationReport:
        """Search the complete pool and compute all retrieval metrics.

        Args:
            query_embeddings: Query vectors shaped ``[Q, D]``.
            candidate_embeddings: Candidate vectors shaped ``[C, D]``.
            relevance: Non-negative binary or graded relevance shaped ``[Q, C]``.
            ks: Metric cutoffs.
            candidate_mask: Optional per-query candidate filter shaped ``[Q, C]``.

        Returns:
            Metrics and complete per-query rankings.
        """
        query_count, candidate_count = _validate_embeddings(query_embeddings, candidate_embeddings)
        if relevance.shape != (query_count, candidate_count):
            raise ConfigurationError(
                "relevance must have shape [query_count, candidate_count]; "
                f"got {tuple(relevance.shape)}, expected {(query_count, candidate_count)}"
            )
        if relevance.device != query_embeddings.device:
            raise ConfigurationError("relevance and embeddings must share a device")
        mask = _validate_candidate_mask(candidate_mask, query_count, candidate_count)
        effective_relevance = (
            relevance
            if mask is None
            else torch.where(
                mask,
                relevance,
                torch.zeros((), dtype=relevance.dtype, device=relevance.device),
            )
        )
        search = self.search(
            query_embeddings,
            candidate_embeddings,
            k=candidate_count,
            candidate_mask=mask,
        )
        _validate_complete_search(search, query_count, candidate_count, relevance.device)
        ranked_relevance = effective_relevance.gather(1, search.indices)
        metric_result = compute_retrieval_metrics(
            ranked_relevance,
            ks=ks,
            ideal_relevance=effective_relevance,
            valid_mask=search.valid_mask,
        )
        return EvaluationReport(
            metrics=metric_result.metrics,
            per_query=metric_result.per_query,
            search=search,
        )


class ExactSearchEvaluator(_RetrievalEvaluator):
    """Exact batched retrieval over every candidate.

    Args:
        metric: Similarity or distance used for ranking.
        query_batch_size: Maximum query rows materialized in one score matrix.
    """

    def __init__(
        self,
        metric: SearchMetric = "inner_product",
        *,
        query_batch_size: int = 256,
    ) -> None:
        """Configure exact search metric and query chunk size."""
        if metric not in _SEARCH_METRICS:
            raise ConfigurationError(
                f"unsupported search metric {metric!r}; expected one of {sorted(_SEARCH_METRICS)}"
            )
        if query_batch_size <= 0:
            raise ConfigurationError("query_batch_size must be positive")
        self.metric = metric
        self.query_batch_size = query_batch_size

    def search(
        self,
        query_embeddings: Tensor,
        candidate_embeddings: Tensor,
        *,
        k: int,
        candidate_mask: Tensor | None = None,
    ) -> SearchResult:
        """Return exact top-k results with deterministic stable tie ordering."""
        query_count, candidate_count = _validate_embeddings(query_embeddings, candidate_embeddings)
        _validate_k(k, candidate_count)
        mask = _validate_candidate_mask(candidate_mask, query_count, candidate_count)
        queries, candidates = _prepare_vectors(query_embeddings, candidate_embeddings, self.metric)

        index_batches: list[Tensor] = []
        score_batches: list[Tensor] = []
        valid_batches: list[Tensor] = []
        for start in range(0, query_count, self.query_batch_size):
            stop = min(start + self.query_batch_size, query_count)
            scores = _pairwise_scores(queries[start:stop], candidates, self.metric)
            batch_mask = None if mask is None else mask[start:stop]
            if batch_mask is not None:
                scores = scores.masked_fill(~batch_mask, -torch.inf)
            order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
            ordered_scores = scores.gather(1, order)
            ordered_valid = (
                torch.ones_like(order, dtype=torch.bool)
                if batch_mask is None
                else batch_mask.gather(1, order)
            )
            index_batches.append(order)
            score_batches.append(ordered_scores)
            valid_batches.append(ordered_valid)
        return SearchResult(
            indices=torch.cat(index_batches, dim=0),
            scores=torch.cat(score_batches, dim=0),
            valid_mask=torch.cat(valid_batches, dim=0),
        )


class FaissEvaluator(_RetrievalEvaluator):
    """Flat FAISS retrieval with a lazy optional dependency import.

    The evaluator intentionally uses exact flat indices. Approximate index
    lifecycle and persistence belong to external implementations of
    :class:`ExternalANNEvaluator`.

    Args:
        metric: Similarity or distance used for ranking.
    """

    def __init__(self, metric: SearchMetric = "inner_product") -> None:
        """Configure the metric used by the lazy FAISS flat index."""
        if metric not in _SEARCH_METRICS:
            raise ConfigurationError(
                f"unsupported search metric {metric!r}; expected one of {sorted(_SEARCH_METRICS)}"
            )
        self.metric = metric

    def search(
        self,
        query_embeddings: Tensor,
        candidate_embeddings: Tensor,
        *,
        k: int,
        candidate_mask: Tensor | None = None,
    ) -> SearchResult:
        """Return flat-index top-k results, post-filtered when requested."""
        faiss = _load_faiss()
        query_count, candidate_count = _validate_embeddings(query_embeddings, candidate_embeddings)
        _validate_k(k, candidate_count)
        mask = _validate_candidate_mask(candidate_mask, query_count, candidate_count)
        queries, candidates = _prepare_vectors(query_embeddings, candidate_embeddings, self.metric)
        query_array = queries.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        candidate_array = (
            candidates.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        )
        dimension = int(candidates.shape[1])
        if self.metric == "euclidean":
            index = faiss.IndexFlatL2(dimension)
        else:
            index = faiss.IndexFlatIP(dimension)
        index.add(candidate_array)
        requested = candidate_count if mask is not None else k
        raw_scores, raw_indices = index.search(query_array, requested)
        indices = torch.from_numpy(raw_indices).to(device=query_embeddings.device, dtype=torch.long)
        scores = torch.from_numpy(raw_scores).to(device=query_embeddings.device)
        if self.metric == "euclidean":
            scores = -scores
        valid = torch.ones_like(indices, dtype=torch.bool)
        if mask is not None:
            valid = mask.gather(1, indices)
            scores = scores.masked_fill(~valid, -torch.inf)
            order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
            indices = indices.gather(1, order)
            scores = scores.gather(1, order)
            valid = valid.gather(1, order)
        return SearchResult(indices=indices[:, :k], scores=scores[:, :k], valid_mask=valid[:, :k])


def _load_faiss() -> Any:
    try:
        module = importlib.import_module("faiss")
    except ImportError as error:
        raise OptionalDependencyError(
            "FAISS evaluation requires the 'faiss' extra; install with `uv sync --extra faiss`"
        ) from error
    _validate_faiss_openmp_runtime(module)
    return module


def _validate_faiss_openmp_runtime(module: Any) -> None:
    """Reject known macOS wheels that would abort on duplicate OpenMP runtimes."""
    if sys.platform != "darwin":
        return
    module_file = getattr(module, "__file__", None)
    torch_file = getattr(torch, "__file__", None)
    if not isinstance(module_file, str) or not isinstance(torch_file, str):
        return
    faiss_openmp = Path(module_file).parent / ".dylibs" / "libomp.dylib"
    torch_openmp = Path(torch_file).parent / "lib" / "libomp.dylib"
    if (
        faiss_openmp.is_file()
        and torch_openmp.is_file()
        and not faiss_openmp.samefile(torch_openmp)
    ):
        raise OptionalDependencyError(
            "the installed macOS PyTorch and faiss-cpu wheels bundle different libomp.dylib "
            "runtimes and abort when loaded together; use a compatible shared-OpenMP build "
            "or run FAISS evaluation on Linux"
        )


def _validate_embeddings(queries: Tensor, candidates: Tensor) -> tuple[int, int]:
    if queries.ndim != _EMBEDDING_RANK or candidates.ndim != _EMBEDDING_RANK:
        raise ConfigurationError("query and candidate embeddings must be [N, D] tensors")
    if queries.shape[0] == 0 or candidates.shape[0] == 0:
        raise ConfigurationError("query and candidate embeddings must be non-empty")
    if queries.shape[1] != candidates.shape[1]:
        raise ConfigurationError(
            "query and candidate embedding dimensions differ: "
            f"{queries.shape[1]} != {candidates.shape[1]}"
        )
    if queries.device != candidates.device:
        raise ConfigurationError(
            "query and candidate embeddings must share a device: "
            f"{queries.device} != {candidates.device}"
        )
    if not queries.is_floating_point() or not candidates.is_floating_point():
        raise ConfigurationError("query and candidate embeddings must be floating point")
    if not bool(torch.isfinite(queries).all().item()) or not bool(
        torch.isfinite(candidates).all().item()
    ):
        raise NumericalError("query or candidate embeddings contain NaN or infinite values")
    return int(queries.shape[0]), int(candidates.shape[0])


def _validate_complete_search(
    result: SearchResult,
    query_count: int,
    candidate_count: int,
    relevance_device: torch.device,
) -> None:
    expected = (query_count, candidate_count)
    if result.indices.shape != expected:
        raise ConfigurationError(
            f"full evaluation requires search shape {expected}, got {tuple(result.indices.shape)}"
        )
    if result.indices.device != relevance_device:
        raise ConfigurationError("search indices and relevance must share a device")
    if bool((result.indices >= candidate_count).any().item()):
        raise ConfigurationError("search result contains an out-of-range candidate index")
    for row in range(query_count):
        selected = result.indices[row][result.valid_mask[row]]
        if torch.unique(selected).numel() != selected.numel():
            raise ConfigurationError(f"search result repeats a valid candidate for query {row}")


def _validate_candidate_mask(
    mask: Tensor | None,
    query_count: int,
    candidate_count: int,
) -> Tensor | None:
    if mask is None:
        return None
    if mask.dtype is not torch.bool or mask.shape != (query_count, candidate_count):
        raise ConfigurationError(
            "candidate_mask must be boolean with shape [Q, C]; "
            f"got dtype={mask.dtype}, shape={tuple(mask.shape)}"
        )
    if bool((mask.sum(dim=1) == 0).any().item()):
        rows = torch.nonzero(mask.sum(dim=1) == 0, as_tuple=False).flatten().tolist()
        raise ConfigurationError(f"candidate filter removes every candidate for queries {rows}")
    return mask


def _validate_k(k: int, candidate_count: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= candidate_count:
        raise ConfigurationError(f"k must be an integer within [1, {candidate_count}], got {k!r}")


def _prepare_vectors(
    queries: Tensor,
    candidates: Tensor,
    metric: SearchMetric,
) -> tuple[Tensor, Tensor]:
    if metric != "cosine":
        return queries, candidates
    query_norms = torch.linalg.vector_norm(queries, dim=1)
    candidate_norms = torch.linalg.vector_norm(candidates, dim=1)
    if not bool(torch.isfinite(query_norms).all().item()) or not bool(
        torch.isfinite(candidate_norms).all().item()
    ):
        raise NumericalError("cosine search received a non-finite embedding L2 norm")
    if bool((query_norms == 0).any().item()) or bool((candidate_norms == 0).any().item()):
        raise NumericalError("cosine search cannot normalize zero-norm embeddings")
    return queries / query_norms.unsqueeze(1), candidates / candidate_norms.unsqueeze(1)


def _pairwise_scores(queries: Tensor, candidates: Tensor, metric: SearchMetric) -> Tensor:
    if metric == "euclidean":
        return -torch.cdist(queries, candidates, p=2).square()
    return queries @ candidates.transpose(0, 1)
