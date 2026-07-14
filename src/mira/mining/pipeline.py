"""Deterministic exact-reference hard-negative mining pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from mira.core.exceptions import ContractError
from mira.core.serialization import canonicalize_json_value, freeze_json_value
from mira.mining.manifests import (
    MiningManifest,
    checksum_mined_rows,
    checksum_positive_mapping,
)

if TYPE_CHECKING:
    from mira.mining.filters import (
        MetadataFalseNegativeFilter,
        PositiveAwareSimilarityFilter,
    )

_MATRIX_RANK = 2


@dataclass(frozen=True, slots=True)
class EncodedCollection:
    """IDs, embeddings, and optional metadata aligned on one item axis."""

    item_ids: tuple[str, ...]
    embeddings: Tensor
    metadata: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate alignment, uniqueness, dtype, and finite values."""
        count = len(self.item_ids)
        if (
            count == 0
            or any(not isinstance(item_id, str) or not item_id for item_id in self.item_ids)
            or len(set(self.item_ids)) != count
        ):
            raise ContractError("encoded item IDs must be non-empty strings and unique")
        if self.embeddings.ndim != _MATRIX_RANK or self.embeddings.shape[0] != count:
            raise ContractError("encoded embeddings must have shape [len(item_ids), dimension]")
        if (
            not self.embeddings.is_floating_point()
            or not torch.isfinite(self.embeddings).all().item()
        ):
            raise ContractError("encoded embeddings must be finite floating-point values")
        if len(self.metadata) != count:
            raise ContractError("encoded metadata must align one-to-one with item IDs")
        if any(not isinstance(item, Mapping) for item in self.metadata):
            raise ContractError("encoded metadata entries must be JSON objects")
        try:
            frozen_metadata = freeze_json_value(
                self.metadata,
                context="encoded metadata",
            )
        except (TypeError, ValueError) as error:
            raise ContractError(str(error)) from error
        object.__setattr__(
            self,
            "metadata",
            cast("tuple[Mapping[str, Any], ...]", frozen_metadata),
        )
        if not isinstance(self.provenance, Mapping) or any(
            not isinstance(name, str) or not name or not isinstance(value, str) or not value
            for name, value in self.provenance.items()
        ):
            raise ContractError("encoded provenance must contain non-empty string keys and values")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def checksum(self) -> str:
        """Hash identifiers, metadata, shape, dtype, and raw tensor bytes."""
        digest = hashlib.sha256()
        header = {
            "dtype": str(self.embeddings.dtype),
            "item_ids": self.item_ids,
            "metadata": canonicalize_json_value(self.metadata, context="encoded metadata"),
            "provenance": dict(self.provenance),
            "shape": tuple(self.embeddings.shape),
        }
        digest.update(
            json.dumps(
                header,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        tensor = self.embeddings.detach().cpu().contiguous()
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


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
    score_row: Tensor
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
    ) -> None:
        """Configure filters applied after known-positive removal."""
        self.metadata_filter = metadata_filter
        self.similarity_filter = similarity_filter

    def run(self, request: MiningRequest) -> MiningResult:
        """Retrieve, filter, stratify, and return versioned negatives.

        Equal scores preserve corpus input order.
        """
        queries = request.queries
        corpus = request.corpus
        retrieve_k = request.retrieve_k
        _validate_request(request)
        corpus_positions = {item_id: index for index, item_id in enumerate(corpus.item_ids)}
        scores = queries.embeddings.float() @ corpus.embeddings.float().transpose(0, 1)
        retrieved_indices = torch.argsort(scores, dim=1, descending=True, stable=True)[
            :, :retrieve_k
        ]
        retrieved_scores = scores.gather(1, retrieved_indices)
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
                score_row=scores[query_index],
                retrieved_score_row=retrieved_scores[query_index],
                retrieved_index_row=retrieved_indices[query_index],
                corpus_positions=corpus_positions,
                known_positives=known_positives,
                best_positive=_best_positive_score(
                    scores[query_index], set(known_positives), corpus_positions
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
            "index_parameters": {"retrieve_k": request.retrieve_k},
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
    scores: Tensor,
    positive_ids: set[str],
    positions: dict[str, int],
) -> float:
    indices = torch.tensor([positions[item_id] for item_id in sorted(positive_ids)])
    return float(scores.index_select(0, indices).max().item())


def _stratify(query_id: str, accepted: list[tuple[str, float]]) -> list[MinedNegative]:
    count = len(accepted)
    rows: list[MinedNegative] = []
    for offset, (candidate_id, score) in enumerate(accepted):
        fraction = offset / count
        difficulty = "hard" if fraction < 1 / 3 else "medium" if fraction < 2 / 3 else "easy"
        rows.append(MinedNegative(query_id, candidate_id, score, offset + 1, difficulty))
    return rows
