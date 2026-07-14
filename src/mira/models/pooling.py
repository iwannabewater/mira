"""Validated pooling and normalization for embedding representations."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from mira.core.exceptions import InvalidEmbeddingError

_MATRIX_RANK = 2
_SEQUENCE_RANK = 3


def normalize_embeddings(embeddings: Tensor, *, epsilon: float = 1e-12) -> Tensor:
    """L2-normalize the final dimension and reject zero or non-finite rows."""
    if not embeddings.is_floating_point():
        raise InvalidEmbeddingError("embedding normalization requires a floating-point tensor")
    if embeddings.ndim < _MATRIX_RANK:
        raise InvalidEmbeddingError(
            f"embeddings must have at least two dimensions, got {embeddings.ndim}"
        )
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise InvalidEmbeddingError(f"normalization epsilon must be positive, got {epsilon}")
    if not bool(torch.isfinite(embeddings).all().item()):
        raise InvalidEmbeddingError("embeddings contain NaN or infinite values")
    norms = torch.linalg.vector_norm(embeddings, dim=-1)
    if not bool(torch.isfinite(norms).all().item()):
        raise InvalidEmbeddingError("embedding L2 norms are non-finite")
    invalid_rows = norms <= epsilon
    if bool(invalid_rows.any().item()):
        indices = invalid_rows.nonzero(as_tuple=False).tolist()
        raise InvalidEmbeddingError(f"cannot normalize zero-norm embedding rows: {indices}")
    return torch.div(embeddings, norms.unsqueeze(-1))


def last_token_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Select the last valid token for left, right, or irregular padding."""
    _validate_pooling_inputs(hidden_states, attention_mask)
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    invalid_position = torch.full_like(positions, -1)
    last_positions = torch.where(attention_mask.bool(), positions, invalid_position).amax(dim=1)
    batch_positions = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_positions, last_positions]


def masked_mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool valid tokens using accumulation in the input dtype."""
    _validate_pooling_inputs(hidden_states, attention_mask)
    weights = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    totals = (hidden_states * weights).sum(dim=1)
    counts = weights.sum(dim=1)
    return totals / counts


def truncate_and_normalize(embeddings: Tensor, dimension: int) -> Tensor:
    """Truncate MRL embeddings and re-normalize the retained dimensions."""
    if embeddings.ndim != _MATRIX_RANK:
        raise InvalidEmbeddingError(
            f"MRL truncation requires [batch, dimension], got {tuple(embeddings.shape)}"
        )
    full_dimension = int(embeddings.shape[1])
    if dimension <= 0 or dimension > full_dimension:
        raise InvalidEmbeddingError(
            f"MRL dimension must be in [1, {full_dimension}], got {dimension}"
        )
    return normalize_embeddings(embeddings[:, :dimension])


def _validate_pooling_inputs(hidden_states: Tensor, attention_mask: Tensor) -> None:
    if hidden_states.ndim != _SEQUENCE_RANK:
        raise InvalidEmbeddingError(
            f"hidden states must have shape [batch, sequence, dimension], got "
            f"{tuple(hidden_states.shape)}"
        )
    if attention_mask.ndim != _MATRIX_RANK or attention_mask.shape != hidden_states.shape[:2]:
        raise InvalidEmbeddingError(
            "attention mask must match the hidden-state batch and sequence dimensions"
        )
    if attention_mask.device != hidden_states.device:
        raise InvalidEmbeddingError("attention mask and hidden states must use the same device")
    if attention_mask.dtype is not torch.bool:
        is_binary = bool(((attention_mask == 0) | (attention_mask == 1)).all().item())
        if not is_binary:
            raise InvalidEmbeddingError("attention mask values must be binary")
    valid_counts = attention_mask.bool().sum(dim=1)
    if bool((valid_counts == 0).any().item()):
        rows = (valid_counts == 0).nonzero(as_tuple=False).flatten().tolist()
        raise InvalidEmbeddingError(f"attention mask has empty rows: {rows}")
