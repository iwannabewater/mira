"""Deterministic exact-reference hard-negative mining pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from mira.core.exceptions import ContractError
from mira.mining.manifests import (
    MiningManifest,
    checksum_mined_rows,
    checksum_positive_mapping,
)

if TYPE_CHECKING:
    from mira.core.encoded import EncodedCollection
    from mira.mining.filters import (
        MetadataFalseNegativeFilter,
        PositiveAwareSimilarityFilter,
    )


@dataclass(frozen=True, slots=True)
class MinedNegative:
    """One retained negative with rank, score, and difficulty stratum."""

    query_id: str
    candidate_id: str
    score: float
    rank: int
    difficulty: str


@dataclass(frozen=True, slots=True)
class MiningResult:
    """Mined rows and the complete run manifest."""

    negatives: tuple[MinedNegative, ...]
    manifest: MiningManifest


@dataclass(frozen=True, slots=True)
class MiningRequest:
    """Aligned inputs and run settings for one offline mining operation."""

    queries: EncodedCollection
    corpus: EncodedCollection
    positives: Mapping[str, tuple[str, ...]]
    top_k: int
    retrieve_k: int
    manifest_values: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _QueryMiningContext:
    request: MiningRequest
    query_index: int
    query_id: str
    retrieved_score_row: Tensor
    retrieved_index_row: Tensor
    corpus_positions: dict[str, int]
    known_positives: frozenset[str]
    best_positive: float


class MiningPipeline:
    """Exact-search reference miner with explicit false-negative rules."""

    def __init__(
        self,
        *,
        metadata_filter: MetadataFalseNegativeFilter | None = None,
        similarity_filter: PositiveAwareSimilarityFilter | None = None,
        query_block_size: int = 256,
        corpus_block_size: int = 8192,
    ) -> None:
        """Configure filters and bounded-memory exact-search block sizes."""
        for name, value in (
            ("query_block_size", query_block_size),
            ("corpus_block_size", corpus_block_size),
        ):
            if type(value) is not int or value <= 0:
                raise ContractError(f"{name} must be a positive integer")
        self.metadata_filter = metadata_filter
        self.similarity_filter = similarity_filter
        self.query_block_size = query_block_size
        self.corpus_block_size = corpus_block_size

    def run(self, request: MiningRequest) -> MiningResult:
        """Retrieve, filter, stratify, and return versioned negatives.

        Equal scores preserve corpus input order.
        """
        queries = request.queries
        corpus = request.corpus
        retrieve_k = request.retrieve_k
        _validate_request(request)
        corpus_positions = {item_id: index for index, item_id in enumerate(corpus.item_ids)}
        retrieved_scores, retrieved_indices = _chunked_exact_topk(
            queries.embeddings,
            corpus.embeddings,
            retrieve_k,
            query_block_size=self.query_block_size,
            corpus_block_size=self.corpus_block_size,
        )
        rows: list[MinedNegative] = []
        for query_index, query_id in enumerate(queries.item_ids):
            known_positives = frozenset(request.positives.get(query_id, ()))
            missing_positives = known_positives.difference(corpus_positions)
            if missing_positives:
                raise ContractError(
                    f"query {query_id!r} references positives absent from corpus: "
                    f"{missing_positives}"
                )
            context = _QueryMiningContext(
                request=request,
                query_index=query_index,
                query_id=query_id,
                retrieved_score_row=retrieved_scores[query_index],
                retrieved_index_row=retrieved_indices[query_index],
                corpus_positions=corpus_positions,
                known_positives=known_positives,
                best_positive=_best_positive_score(
                    queries.embeddings[query_index],
                    corpus.embeddings,
                    set(known_positives),
                    corpus_positions,
                ),
            )
            accepted = self._mine_query(context)
            rows.extend(_stratify(query_id, accepted))

        defaults = self._manifest_defaults(request)
        defaults["mined_rows_checksum"] = checksum_mined_rows(rows)
        defaults["positive_mapping_checksum"] = checksum_positive_mapping(request.positives)
        overridden = sorted(set(defaults).intersection(request.manifest_values))
        if overridden:
            raise ContractError(
                f"manifest_values cannot override mining-derived fields: {tuple(overridden)}"
            )
        defaults.update(request.manifest_values)
        manifest = MiningManifest.create(**defaults)
        return MiningResult(tuple(rows), manifest)

    def _mine_query(self, context: _QueryMiningContext) -> list[tuple[str, float]]:
        request = context.request
        accepted: list[tuple[str, float]] = []
        for score, candidate_index in zip(
            context.retrieved_score_row,
            context.retrieved_index_row,
            strict=True,
        ):
            position = int(candidate_index.item())
            candidate_id = request.corpus.item_ids[position]
            candidate_score = float(score.item())
            if self._keep_candidate(
                context,
                position,
                candidate_id,
                candidate_score,
            ):
                accepted.append((candidate_id, candidate_score))
            if len(accepted) == request.top_k:
                break
        if len(accepted) < request.top_k:
            raise ContractError(
                f"query {context.query_id!r} retained {len(accepted)} negatives; "
                f"required {request.top_k}"
            )
        return accepted

    def _keep_candidate(
        self,
        context: _QueryMiningContext,
        candidate_index: int,
        candidate_id: str,
        candidate_score: float,
    ) -> bool:
        request = context.request
        if candidate_id in context.known_positives:
            return False
        if self.metadata_filter is not None and not self.metadata_filter.keep(
            request.queries.metadata[context.query_index],
            request.corpus.metadata[candidate_index],
        ):
            return False
        if self.similarity_filter is None:
            return True
        return self.similarity_filter.keep(candidate_score, context.best_positive)

    def _manifest_defaults(self, request: MiningRequest) -> dict[str, Any]:
        filter_values: list[dict[str, Any]] = []
        if self.metadata_filter is not None:
            filter_values.append({"type": "metadata", "keys": self.metadata_filter.keys})
        if self.similarity_filter is not None:
            filter_values.append(
                {"type": "positive_aware_similarity", "alpha": self.similarity_filter.alpha}
            )
        return {
            "corpus_checksum": request.corpus.checksum(),
            "embedding_dimension": int(request.corpus.embeddings.shape[1]),
            "filters": tuple(filter_values),
            "index_parameters": {
                "corpus_block_size": self.corpus_block_size,
                "query_block_size": self.query_block_size,
                "retrieve_k": request.retrieve_k,
            },
            "index_type": "torch_exact_inner_product",
            "query_checksum": request.queries.checksum(),
            "top_k": request.top_k,
        }


def _validate_request(request: MiningRequest) -> None:
    queries = request.queries
    corpus = request.corpus
    if queries.embeddings.shape[1] != corpus.embeddings.shape[1]:
        raise ContractError("query and corpus embedding dimensions must match")
    if type(request.top_k) is not int or type(request.retrieve_k) is not int:
        raise ContractError("top_k and retrieve_k must be integers")
    if (
        request.top_k <= 0
        or request.retrieve_k < request.top_k
        or request.retrieve_k > len(corpus.item_ids)
    ):
        raise ContractError("require 0 < top_k <= retrieve_k <= corpus size")
    if not isinstance(request.positives, Mapping):
        raise ContractError("positive mapping must be a mapping")
    if not isinstance(request.manifest_values, dict):
        raise ContractError("manifest_values must be a mapping")
    expected_queries = set(queries.item_ids)
    observed_queries = set(request.positives)
    if observed_queries != expected_queries:
        missing = tuple(sorted(expected_queries - observed_queries))
        unknown = tuple(sorted(observed_queries - expected_queries))
        raise ContractError(
            "positive mapping must exactly cover encoded query IDs; "
            f"missing={missing}, unknown={unknown}"
        )
    for query_id in queries.item_ids:
        positive_ids_value: Any = request.positives[query_id]
        if isinstance(positive_ids_value, (str, bytes)) or not isinstance(
            positive_ids_value, tuple
        ):
            raise ContractError(
                f"positive mapping query {query_id!r} must contain a tuple of candidate IDs"
            )
        positive_ids = cast("tuple[str, ...]", positive_ids_value)
        if not positive_ids:
            raise ContractError(
                f"positive mapping query {query_id!r} must contain at least one candidate"
            )
        if any(
            not isinstance(candidate_id, str) or not candidate_id for candidate_id in positive_ids
        ):
            raise ContractError(
                f"positive mapping query {query_id!r} must contain non-empty string candidate IDs"
            )
        if len(set(positive_ids)) != len(positive_ids):
            raise ContractError(
                f"positive mapping query {query_id!r} must contain unique candidate IDs"
            )


def _best_positive_score(
    query_embedding: Tensor,
    corpus_embeddings: Tensor,
    positive_ids: set[str],
    positions: dict[str, int],
) -> float:
    indices = torch.tensor(
        [positions[item_id] for item_id in sorted(positive_ids)],
        device=corpus_embeddings.device,
    )
    positives = corpus_embeddings.index_select(0, indices)
    scores = positives.float() @ query_embedding.to(
        device=corpus_embeddings.device,
        dtype=torch.float32,
    )
    return float(scores.max().item())


def _chunked_exact_topk(
    queries: Tensor,
    corpus: Tensor,
    k: int,
    *,
    query_block_size: int,
    corpus_block_size: int,
) -> tuple[Tensor, Tensor]:
    """Return stable exact top-k with memory bounded by two block sizes."""
    score_rows: list[Tensor] = []
    index_rows: list[Tensor] = []
    for query_start in range(0, queries.shape[0], query_block_size):
        query_block = queries[query_start : query_start + query_block_size].float()
        best_scores = torch.empty(
            (query_block.shape[0], 0),
            dtype=torch.float32,
            device=query_block.device,
        )
        best_indices = torch.empty(
            (query_block.shape[0], 0),
            dtype=torch.long,
            device=query_block.device,
        )
        for corpus_start in range(0, corpus.shape[0], corpus_block_size):
            corpus_block = corpus[corpus_start : corpus_start + corpus_block_size].to(
                device=query_block.device,
                dtype=torch.float32,
            )
            block_scores = query_block @ corpus_block.transpose(0, 1)
            block_indices = torch.arange(
                corpus_start,
                corpus_start + corpus_block.shape[0],
                dtype=torch.long,
                device=query_block.device,
            ).expand(query_block.shape[0], -1)
            candidate_scores = torch.cat((best_scores, block_scores), dim=1)
            candidate_indices = torch.cat((best_indices, block_indices), dim=1)
            keep = min(k, candidate_scores.shape[1])
            order = torch.argsort(
                candidate_scores,
                dim=1,
                descending=True,
                stable=True,
            )[:, :keep]
            best_scores = candidate_scores.gather(1, order)
            best_indices = candidate_indices.gather(1, order)
        score_rows.append(best_scores)
        index_rows.append(best_indices)
    return torch.cat(score_rows, dim=0), torch.cat(index_rows, dim=0)


def _stratify(query_id: str, accepted: list[tuple[str, float]]) -> list[MinedNegative]:
    count = len(accepted)
    rows: list[MinedNegative] = []
    for offset, (candidate_id, score) in enumerate(accepted):
        fraction = offset / count
        difficulty = "hard" if fraction < 1 / 3 else "medium" if fraction < 2 / 3 else "easy"
        rows.append(MinedNegative(query_id, candidate_id, score, offset + 1, difficulty))
    return rows
