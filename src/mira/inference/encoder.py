"""Unified query, document, token, and hybrid embedding inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, cast, runtime_checkable

import torch
from torch import Tensor
from torch.nn import functional

from mira.core.contracts import (
    EmbeddingOutput,
    RepresentationRequest,
    RepresentationType,
)
from mira.core.exceptions import ConfigurationError, InvalidEmbeddingError

InferenceRole: TypeAlias = Literal["query", "document"]
"""Semantic role used to select instructions and backend formatting."""

InferenceInput: TypeAlias = object
"""Raw backend input, such as text, a multimodal item, or an input mapping."""

_GLOBAL_EMBEDDING_RANK = 2
_TOKEN_EMBEDDING_RANK = 3


@runtime_checkable
class EncoderBackend(Protocol):
    """Raw-input inference boundary implemented by model integrations."""

    def encode_batch(
        self,
        items: Sequence[InferenceInput],
        *,
        role: InferenceRole,
        instruction: str | None,
        request: RepresentationRequest,
    ) -> EmbeddingOutput:
        """Encode one non-empty batch without changing item order."""
        del items, role, instruction, request  # pragma: no cover - Protocol fallback
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class TokenEmbeddingBatch:
    """Padded token embeddings and their exact validity mask."""

    embeddings: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        """Validate token shape, mask alignment, values, and valid norms."""
        if self.embeddings.ndim != _TOKEN_EMBEDDING_RANK or any(
            size == 0 for size in self.embeddings.shape
        ):
            raise InvalidEmbeddingError("token inference embeddings must be non-empty [N, L, D]")
        if not self.embeddings.is_floating_point():
            raise InvalidEmbeddingError("token inference embeddings must be floating point")
        if self.mask.dtype is not torch.bool or self.mask.shape != self.embeddings.shape[:2]:
            raise InvalidEmbeddingError("token inference mask must be boolean with shape [N, L]")
        if self.mask.device != self.embeddings.device:
            raise InvalidEmbeddingError("token inference mask and embeddings must share a device")
        valid = self.embeddings[self.mask]
        if valid.numel() == 0:
            raise InvalidEmbeddingError("token inference mask must select at least one vector")
        if not bool(torch.isfinite(valid).all().item()):
            raise InvalidEmbeddingError("valid token inference embeddings must be finite")
        norms = torch.linalg.vector_norm(valid, dim=-1)
        if not bool(torch.isfinite(norms).all().item()):
            raise InvalidEmbeddingError("valid token inference embeddings have non-finite L2 norms")
        if bool((norms == 0).any().item()):
            raise InvalidEmbeddingError("valid token inference embeddings cannot have zero norm")


@dataclass(frozen=True, slots=True)
class HybridEmbeddingBatch:
    """Aligned global and token representations for one input batch."""

    global_embeddings: Tensor
    token_embeddings: Tensor
    token_mask: Tensor

    def __post_init__(self) -> None:
        """Validate aligned global and token batch contracts."""
        token_batch = TokenEmbeddingBatch(self.token_embeddings, self.token_mask)
        if self.global_embeddings.ndim != _GLOBAL_EMBEDDING_RANK or any(
            size == 0 for size in self.global_embeddings.shape
        ):
            raise InvalidEmbeddingError("hybrid global embeddings must be non-empty [N, D]")
        if self.global_embeddings.shape[0] != token_batch.embeddings.shape[0]:
            raise InvalidEmbeddingError("hybrid global and token batch axes must align")
        if self.global_embeddings.shape[1] != token_batch.embeddings.shape[2]:
            raise InvalidEmbeddingError("hybrid global and token dimensions must align")
        if self.global_embeddings.device != token_batch.embeddings.device:
            raise InvalidEmbeddingError("hybrid global and token embeddings must share a device")
        if not self.global_embeddings.is_floating_point() or not bool(
            torch.isfinite(self.global_embeddings).all().item()
        ):
            raise InvalidEmbeddingError("hybrid global embeddings must be finite floating point")
        norms = torch.linalg.vector_norm(self.global_embeddings, dim=-1)
        if not bool(torch.isfinite(norms).all().item()):
            raise InvalidEmbeddingError("hybrid global embeddings have non-finite L2 norms")
        if bool((norms == 0).any().item()):
            raise InvalidEmbeddingError("hybrid global embeddings cannot have zero norm")


class Encoder:
    """Batch raw inputs through a backend and enforce output semantics.

    Output is always batch-major, including single-item calls. A requested MRL
    dimension is applied after backend inference and always followed by L2
    normalization. Token batches with different sequence lengths are padded
    across backend batches while preserving the exact token mask.

    Args:
        backend: Integration implementing :class:`EncoderBackend`.
        query_instruction: Default instruction for query inputs.
        document_instruction: Default instruction for document inputs.
        default_batch_size: Positive batch size used when a call omits one.
    """

    def __init__(
        self,
        backend: EncoderBackend,
        *,
        query_instruction: str | None = None,
        document_instruction: str | None = None,
        default_batch_size: int = 32,
    ) -> None:
        """Create an encoder with role-specific default instructions."""
        if default_batch_size <= 0:
            raise ConfigurationError("default_batch_size must be positive")
        self._backend = backend
        self._query_instruction = _validate_instruction("query_instruction", query_instruction)
        self._document_instruction = _validate_instruction(
            "document_instruction", document_instruction
        )
        self._default_batch_size = default_batch_size

    def encode_query(
        self,
        items: InferenceInput | Sequence[InferenceInput],
        *,
        batch_size: int | None = None,
        dimension: int | None = None,
        normalize: bool = True,
        instruction: str | None = None,
    ) -> Tensor:
        """Encode query inputs as a ``[N, D]`` global embedding tensor."""
        result = self.encode(
            items,
            role="query",
            representation=RepresentationType.GLOBAL,
            batch_size=batch_size,
            dimension=dimension,
            normalize=normalize,
            instruction=instruction,
        )
        return cast("Tensor", result)

    def encode_document(
        self,
        items: InferenceInput | Sequence[InferenceInput],
        *,
        batch_size: int | None = None,
        dimension: int | None = None,
        normalize: bool = True,
        instruction: str | None = None,
    ) -> Tensor:
        """Encode document inputs as a ``[N, D]`` global embedding tensor."""
        result = self.encode(
            items,
            role="document",
            representation=RepresentationType.GLOBAL,
            batch_size=batch_size,
            dimension=dimension,
            normalize=normalize,
            instruction=instruction,
        )
        return cast("Tensor", result)

    def encode_tokens(  # noqa: PLR0913 - mirrors the unified public encode contract
        self,
        items: InferenceInput | Sequence[InferenceInput],
        *,
        role: InferenceRole = "document",
        batch_size: int | None = None,
        dimension: int | None = None,
        normalize: bool = True,
        instruction: str | None = None,
    ) -> TokenEmbeddingBatch:
        """Encode inputs as padded token embeddings and a boolean mask."""
        result = self.encode(
            items,
            role=role,
            representation=RepresentationType.TOKEN,
            batch_size=batch_size,
            dimension=dimension,
            normalize=normalize,
            instruction=instruction,
        )
        return cast("TokenEmbeddingBatch", result)

    def encode(  # noqa: PLR0913 - explicit options are the public inference contract
        self,
        items: InferenceInput | Sequence[InferenceInput],
        *,
        role: InferenceRole = "document",
        representation: RepresentationType = RepresentationType.GLOBAL,
        batch_size: int | None = None,
        dimension: int | None = None,
        normalize: bool = True,
        instruction: str | None = None,
    ) -> Tensor | TokenEmbeddingBatch | HybridEmbeddingBatch:
        """Encode inputs using one explicit role and representation contract.

        Args:
            items: One input or a non-empty sequence. Strings and mappings are
                treated as single inputs rather than iterated containers.
            role: Query or document semantics.
            representation: Global, token, or hybrid output.
            batch_size: Positive override for backend batch size.
            dimension: Optional MRL prefix dimension. Truncated vectors are
                always normalized, even when ``normalize`` is false.
            normalize: Normalize full-dimension outputs when true.
            instruction: Per-call instruction override.

        Returns:
            A batch-major tensor or typed token/hybrid batch.

        Raises:
            ConfigurationError: If call options or input batching are invalid.
            InvalidEmbeddingError: If a backend omits or misaligns output.
        """
        if role not in {"query", "document"}:
            raise ConfigurationError(f"unsupported inference role {role!r}")
        if not isinstance(representation, RepresentationType):
            raise ConfigurationError(
                "representation must be a RepresentationType value, not an inferred string"
            )
        actual_batch_size = self._default_batch_size if batch_size is None else batch_size
        if actual_batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")
        if dimension is not None and dimension <= 0:
            raise ConfigurationError("MRL dimension must be positive")
        values = _coerce_items(items)
        selected_instruction = _validate_instruction("instruction", instruction)
        if selected_instruction is None:
            selected_instruction = (
                self._query_instruction if role == "query" else self._document_instruction
            )

        outputs: list[EmbeddingOutput] = []
        backend_request = RepresentationRequest(
            representation=representation,
            dimension=None,
            normalize=False,
        )
        for start in range(0, len(values), actual_batch_size):
            batch = values[start : start + actual_batch_size]
            output = self._backend.encode_batch(
                batch,
                role=role,
                instruction=selected_instruction,
                request=backend_request,
            )
            _validate_batch_rows(output, expected=len(batch), representation=representation)
            outputs.append(output)
        return _combine_outputs(
            outputs,
            representation=representation,
            dimension=dimension,
            normalize=normalize,
        )


def _coerce_items(
    items: InferenceInput | Sequence[InferenceInput],
) -> list[InferenceInput]:
    values: list[InferenceInput]
    if isinstance(items, (str, bytes, bytearray, Mapping)):
        values = [items]
    elif isinstance(items, Sequence):
        values = list(items)
    else:
        values = [items]
    if not values:
        raise ConfigurationError("inference inputs must not be empty")
    return values


def _validate_instruction(name: str, value: str | None) -> str | None:
    if value is not None and not value.strip():
        raise ConfigurationError(f"{name} must be non-empty when supplied")
    return value


def _validate_batch_rows(
    output: EmbeddingOutput,
    *,
    expected: int,
    representation: RepresentationType,
) -> None:
    if representation in {RepresentationType.GLOBAL, RepresentationType.HYBRID}:
        if output.global_embeddings is None:
            raise InvalidEmbeddingError("backend omitted requested global embeddings")
        if output.global_embeddings.shape[0] != expected:
            raise InvalidEmbeddingError(
                "backend changed global batch order or size: "
                f"returned {output.global_embeddings.shape[0]}, expected {expected}"
            )
    if representation in {RepresentationType.TOKEN, RepresentationType.HYBRID}:
        if output.token_embeddings is None or output.token_mask is None:
            raise InvalidEmbeddingError("backend omitted requested token embeddings or token mask")
        if output.token_embeddings.shape[0] != expected:
            raise InvalidEmbeddingError(
                "backend changed token batch order or size: "
                f"returned {output.token_embeddings.shape[0]}, expected {expected}"
            )


def _combine_outputs(
    outputs: Sequence[EmbeddingOutput],
    *,
    representation: RepresentationType,
    dimension: int | None,
    normalize: bool,
) -> Tensor | TokenEmbeddingBatch | HybridEmbeddingBatch:
    global_embeddings: Tensor | None = None
    token_batch: TokenEmbeddingBatch | None = None
    if representation in {RepresentationType.GLOBAL, RepresentationType.HYBRID}:
        globals_present = [output.global_embeddings for output in outputs]
        global_embeddings = _transform_vectors(
            torch.cat([value for value in globals_present if value is not None], dim=0),
            dimension=dimension,
            normalize=normalize,
        )
    if representation in {RepresentationType.TOKEN, RepresentationType.HYBRID}:
        token_batch = _combine_token_outputs(
            outputs,
            dimension=dimension,
            normalize=normalize,
        )
    if representation is RepresentationType.GLOBAL:
        if global_embeddings is None:
            raise InvalidEmbeddingError("internal global output assembly failed")
        return global_embeddings
    if representation is RepresentationType.TOKEN:
        if token_batch is None:
            raise InvalidEmbeddingError("internal token output assembly failed")
        return token_batch
    if global_embeddings is None or token_batch is None:
        raise InvalidEmbeddingError("internal hybrid output assembly failed")
    return HybridEmbeddingBatch(
        global_embeddings=global_embeddings,
        token_embeddings=token_batch.embeddings,
        token_mask=token_batch.mask,
    )


def _combine_token_outputs(
    outputs: Sequence[EmbeddingOutput],
    *,
    dimension: int | None,
    normalize: bool,
) -> TokenEmbeddingBatch:
    token_values = [output.token_embeddings for output in outputs]
    masks = [output.token_mask for output in outputs]
    present_tokens = [value for value in token_values if value is not None]
    present_masks = [value for value in masks if value is not None]
    max_length = max(int(value.shape[1]) for value in present_tokens)
    padded_tokens: list[Tensor] = []
    padded_masks: list[Tensor] = []
    for tokens, mask in zip(present_tokens, present_masks, strict=True):
        transformed = _transform_vectors(
            tokens,
            dimension=dimension,
            normalize=normalize,
            valid_mask=mask,
        )
        missing = max_length - int(tokens.shape[1])
        if missing:
            transformed = functional.pad(transformed, (0, 0, 0, missing))
            padded_mask = functional.pad(mask, (0, missing), value=False)
        else:
            padded_mask = mask
        padded_tokens.append(transformed)
        padded_masks.append(padded_mask)
    return TokenEmbeddingBatch(
        embeddings=torch.cat(padded_tokens, dim=0),
        mask=torch.cat(padded_masks, dim=0),
    )


def _transform_vectors(
    vectors: Tensor,
    *,
    dimension: int | None,
    normalize: bool,
    valid_mask: Tensor | None = None,
) -> Tensor:
    if dimension is not None:
        if dimension > vectors.shape[-1]:
            raise InvalidEmbeddingError(
                f"MRL dimension {dimension} exceeds embedding dimension {vectors.shape[-1]}"
            )
        vectors = vectors[..., :dimension]
    should_normalize = normalize or dimension is not None
    if should_normalize:
        norms = torch.linalg.vector_norm(vectors, dim=-1)
        selected_norms = norms if valid_mask is None else norms[valid_mask]
        if not bool(torch.isfinite(selected_norms).all().item()):
            raise InvalidEmbeddingError(
                "cannot normalize an inference embedding with non-finite L2 norm"
            )
        if bool((selected_norms == 0).any().item()):
            raise InvalidEmbeddingError("cannot normalize a zero-norm inference embedding")
        divisors = norms
        if valid_mask is not None:
            divisors = torch.where(valid_mask, norms, torch.ones_like(norms))
        vectors = vectors / divisors.unsqueeze(-1)
    if valid_mask is not None:
        vectors = torch.where(valid_mask.unsqueeze(-1), vectors, torch.zeros_like(vectors))
    return vectors
