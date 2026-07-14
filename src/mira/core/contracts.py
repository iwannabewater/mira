"""Backend-independent tensor and objective contracts.

Record-level models are canonically defined by :mod:`mira.data.schema`. This
module exposes lazy compatibility aliases for those names while keeping batch
and objective contracts independent of any training backend.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from mira.core.exceptions import ContractError, InvalidEmbeddingError, ObjectiveError
from mira.core.layout import GroupLayout

if TYPE_CHECKING:
    from mira.data.schema import (
        CandidateItem,
        EmbeddingRecord,
        MediaRef,
        Message,
        MultimodalItem,
        ObjectAnnotations,
        RetrievalMetadata,
    )

_DATA_EXPORTS = frozenset(
    {
        "CandidateItem",
        "EmbeddingRecord",
        "MediaRef",
        "Message",
        "MultimodalItem",
        "ObjectAnnotations",
        "RetrievalMetadata",
    }
)


class RepresentationType(StrEnum):
    """Supported representation families returned by a model adapter."""

    GLOBAL = "global"
    TOKEN = "token"  # noqa: S105 - representation name
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class BatchMetadata:
    """Metadata aligned one-to-one with the query axis of a batch.

    Args:
        sample_ids: Exactly ``Q`` source record identifiers.
        task_ids: Exactly ``Q`` task identifiers; these must equal the layout.
        dataset_ids: Exactly ``Q`` dataset identifiers; these must equal the layout.
        extras: Backend-independent batch annotations.

    Raises:
        ContractError: If axes differ in length or contain empty identifiers.
    """

    sample_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate query-axis alignment and identifiers."""
        for name, values in (
            ("sample_ids", self.sample_ids),
            ("task_ids", self.task_ids),
            ("dataset_ids", self.dataset_ids),
        ):
            if type(values) is not tuple:
                raise ContractError(f"{name} must be a tuple of strings")
            if any(type(value) is not str or not value.strip() for value in values):
                raise ContractError(f"{name} entries must be non-empty strings")
        expected = len(self.sample_ids)
        if expected == 0:
            raise ContractError("BatchMetadata must describe at least one query")
        for name, values in (
            ("task_ids", self.task_ids),
            ("dataset_ids", self.dataset_ids),
        ):
            if len(values) != expected:
                raise ContractError(f"{name} has {len(values)} entries; expected {expected}")
        if not isinstance(self.extras, Mapping):
            raise ContractError("BatchMetadata.extras must be a mapping")
        if any(type(name) is not str for name in self.extras):
            raise ContractError("BatchMetadata.extras keys must be strings")
        object.__setattr__(self, "extras", dict(self.extras))


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Model inputs paired with their sole source of retrieval semantics.

    ``model_inputs`` values are tensors accepted by the selected adapter. The
    flattened representation produced from them is addressed by
    ``layout.query_indices`` and ``layout.candidate_indices``.

    Raises:
        ContractError: If model inputs are empty/non-tensor or metadata is not
            aligned with the ``Q`` query axis.
    """

    model_inputs: Mapping[str, Tensor]
    layout: GroupLayout
    metadata: BatchMetadata

    def __post_init__(self) -> None:
        """Validate input and metadata alignment without guessing tensor roles."""
        if not isinstance(self.model_inputs, Mapping):
            raise ContractError("EmbeddingBatch.model_inputs must be a mapping")
        if not isinstance(self.layout, GroupLayout):
            raise ContractError("EmbeddingBatch.layout must be a GroupLayout")
        if not isinstance(self.metadata, BatchMetadata):
            raise ContractError("EmbeddingBatch.metadata must be BatchMetadata")
        if not self.model_inputs:
            raise ContractError("EmbeddingBatch.model_inputs must not be empty")
        invalid_keys = [
            name
            for name, value in self.model_inputs.items()
            if type(name) is not str or not name.strip() or not isinstance(value, Tensor)
        ]
        if invalid_keys:
            joined = ", ".join(repr(key) for key in invalid_keys)
            raise ContractError(f"model_inputs must map non-empty names to tensors: {joined}")
        if len(self.metadata.sample_ids) != self.layout.num_queries:
            raise ContractError(
                "BatchMetadata query count does not match GroupLayout: "
                f"{len(self.metadata.sample_ids)} != {self.layout.num_queries}"
            )
        if self.metadata.task_ids != self.layout.task_ids:
            raise ContractError("BatchMetadata.task_ids must equal GroupLayout.task_ids")
        if self.metadata.dataset_ids != self.layout.dataset_ids:
            raise ContractError("BatchMetadata.dataset_ids must equal GroupLayout.dataset_ids")
        object.__setattr__(self, "model_inputs", dict(self.model_inputs))


@dataclass(frozen=True, slots=True)
class RepresentationRequest:
    """Declare representation semantics requested from a model adapter.

    Args:
        representation: Global ``[N, D]``, token ``[N, L, D]``, or hybrid output.
        dimension: Optional positive MRL export dimension.
        hidden_layers: Unique hidden-state layer indices to return.
        normalize: Whether returned embeddings must be L2-normalized.

    Raises:
        ContractError: If dimensions or hidden-layer indices are invalid.
    """

    representation: RepresentationType = RepresentationType.GLOBAL
    dimension: int | None = None
    hidden_layers: tuple[int, ...] = ()
    normalize: bool = True

    def __post_init__(self) -> None:
        """Validate request values before model execution."""
        if type(self.representation) is not RepresentationType:
            raise ContractError("representation must be a RepresentationType")
        if self.dimension is not None:
            if type(self.dimension) is not int:
                raise ContractError("representation dimension must be an exact integer")
            if self.dimension <= 0:
                raise ContractError("representation dimension must be positive")
        if type(self.hidden_layers) is not tuple or any(
            type(layer) is not int for layer in self.hidden_layers
        ):
            raise ContractError("hidden_layers must be a tuple of exact integers")
        if len(set(self.hidden_layers)) != len(self.hidden_layers):
            raise ContractError("hidden_layers must contain unique indices")
        if type(self.normalize) is not bool:
            raise ContractError("normalize must be an exact boolean")


@dataclass(frozen=True, slots=True)
class EmbeddingOutput:
    """Typed representations returned by a model adapter.

    Shapes are ``global_embeddings=[N,D]``, ``token_embeddings=[N,L,D]``, and
    ``token_mask=[N,L]``. ``normalized`` explicitly records normalization
    semantics, so objectives never infer them from observed norms.

    Raises:
        InvalidEmbeddingError: If shapes disagree, values are non-finite, valid
            vectors have zero norm, or ``normalized=True`` disagrees with norms.
    """

    global_embeddings: Tensor | None = None
    token_embeddings: Tensor | None = None
    token_mask: Tensor | None = None
    layer_embeddings: Mapping[int, Tensor] | None = None
    logits: Tensor | None = None
    normalized: bool = False
    auxiliary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate all present output tensors and cross-output axes."""
        self._validate_runtime_types()
        if (
            self.global_embeddings is None
            and self.token_embeddings is None
            and self.layer_embeddings is None
            and self.logits is None
        ):
            raise InvalidEmbeddingError("EmbeddingOutput must contain at least one output tensor")

        batch_size: int | None = None
        if self.global_embeddings is not None:
            self._validate_rank("global_embeddings", self.global_embeddings, 2)
            batch_size = self.global_embeddings.shape[0]
            self._validate_vectors(
                "global_embeddings", self.global_embeddings, normalized=self.normalized
            )

        if self.token_embeddings is not None:
            self._validate_rank("token_embeddings", self.token_embeddings, 3)
            if self.token_mask is None:
                raise InvalidEmbeddingError("token_embeddings requires token_mask")
            self._validate_token_mask(self.token_embeddings, self.token_mask)
            batch_size = self._merge_batch_size(
                batch_size, self.token_embeddings.shape[0], "token_embeddings"
            )
            valid_vectors = self.token_embeddings[self.token_mask]
            if valid_vectors.numel() == 0:
                raise InvalidEmbeddingError("token_mask must select at least one token")
            self._validate_vectors(
                "valid token_embeddings", valid_vectors, normalized=self.normalized
            )
        elif self.token_mask is not None:
            raise InvalidEmbeddingError("token_mask cannot be supplied without token_embeddings")

        batch_size = self._validate_layers(batch_size)

        if self.logits is not None:
            if self.logits.ndim == 0:
                raise InvalidEmbeddingError("logits must have a batch axis")
            if not self.logits.dtype.is_floating_point:
                raise InvalidEmbeddingError("logits must use a floating-point dtype")
            self._validate_finite("logits", self.logits)
            self._merge_batch_size(batch_size, self.logits.shape[0], "logits")
        object.__setattr__(self, "auxiliary", dict(self.auxiliary))

    def _validate_runtime_types(self) -> None:
        """Reject Python coercion before inspecting tensor or mapping attributes."""
        if type(self.normalized) is not bool:
            raise InvalidEmbeddingError("normalized must be an exact boolean")
        for name, value in (
            ("global_embeddings", self.global_embeddings),
            ("token_embeddings", self.token_embeddings),
            ("token_mask", self.token_mask),
            ("logits", self.logits),
        ):
            if value is not None and not isinstance(value, Tensor):
                raise InvalidEmbeddingError(f"{name} must be a Tensor when supplied")
        if self.layer_embeddings is not None and not isinstance(self.layer_embeddings, Mapping):
            raise InvalidEmbeddingError("layer_embeddings must be a mapping when supplied")
        if not isinstance(self.auxiliary, Mapping):
            raise InvalidEmbeddingError("auxiliary must be a mapping")
        if any(type(name) is not str for name in self.auxiliary):
            raise InvalidEmbeddingError("auxiliary keys must be strings")

    @staticmethod
    def _validate_rank(name: str, tensor: Tensor, rank: int) -> None:
        if tensor.ndim != rank or any(size == 0 for size in tensor.shape):
            raise InvalidEmbeddingError(f"{name} must be a non-empty rank-{rank} tensor")
        if not tensor.dtype.is_floating_point:
            raise InvalidEmbeddingError(f"{name} must use a floating-point dtype")

    def _validate_layers(self, batch_size: int | None) -> int | None:
        if self.layer_embeddings is None:
            return batch_size
        if not self.layer_embeddings:
            raise InvalidEmbeddingError("layer_embeddings cannot be an empty mapping")
        for layer, embeddings in self.layer_embeddings.items():
            if type(layer) is not int:
                raise InvalidEmbeddingError(
                    "layer_embeddings keys must be integers of exact built-in type"
                )
            if not isinstance(embeddings, Tensor):
                raise InvalidEmbeddingError("layer_embeddings values must be tensors")
            if embeddings.ndim not in {2, 3}:
                raise InvalidEmbeddingError(
                    f"layer {layer} embeddings must have shape [N,D] or [N,L,D]"
                )
            if not embeddings.dtype.is_floating_point:
                raise InvalidEmbeddingError(
                    f"layer {layer} embeddings must use a floating-point dtype"
                )
            self._validate_finite(f"layer_embeddings[{layer}]", embeddings)
            batch_size = self._merge_batch_size(
                batch_size, embeddings.shape[0], f"layer_embeddings[{layer}]"
            )
        object.__setattr__(self, "layer_embeddings", dict(self.layer_embeddings))
        return batch_size

    @classmethod
    def _validate_vectors(cls, name: str, tensor: Tensor, *, normalized: bool) -> None:
        cls._validate_finite(name, tensor)
        norms = torch.linalg.vector_norm(tensor, dim=-1)
        if not torch.isfinite(norms).all().item():
            raise InvalidEmbeddingError(f"{name} contains a non-finite L2 norm")
        if torch.any(norms == 0).item():
            raise InvalidEmbeddingError(f"{name} contains a zero-norm vector")
        tolerance = max(1e-4, float(torch.finfo(tensor.dtype).eps))
        if normalized and not torch.allclose(
            norms,
            torch.ones_like(norms),
            rtol=tolerance,
            atol=tolerance,
        ):
            raise InvalidEmbeddingError(f"{name} is marked normalized but has non-unit norms")

    @staticmethod
    def _validate_finite(name: str, tensor: Tensor) -> None:
        if not torch.isfinite(tensor).all().item():
            raise InvalidEmbeddingError(f"{name} contains NaN or infinite values")

    @staticmethod
    def _validate_token_mask(embeddings: Tensor, mask: Tensor) -> None:
        if mask.dtype != torch.bool:
            raise InvalidEmbeddingError("token_mask must use boolean dtype")
        if tuple(mask.shape) != tuple(embeddings.shape[:2]):
            raise InvalidEmbeddingError(
                f"token_mask has shape {tuple(mask.shape)}; expected {tuple(embeddings.shape[:2])}"
            )
        if mask.device != embeddings.device:
            raise InvalidEmbeddingError("token_mask must be on the token embedding device")

    @staticmethod
    def _merge_batch_size(current: int | None, value: int, name: str) -> int:
        if current is not None and current != value:
            raise InvalidEmbeddingError(
                f"{name} batch size {value} does not match existing batch size {current}"
            )
        return value


@dataclass(frozen=True, slots=True)
class ObjectiveContext:
    """Runtime facts supplied to an objective independently of trainer internals."""

    global_step: int
    epoch: float
    training: bool = True
    rank: int = 0
    world_size: int = 1
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate non-negative progress and distributed coordinates."""
        for name, value in (
            ("global_step", self.global_step),
            ("rank", self.rank),
            ("world_size", self.world_size),
        ):
            if type(value) is not int:
                raise ContractError(f"{name} must be an exact integer")
        if type(self.training) is not bool:
            raise ContractError("training must be an exact boolean")
        if type(self.epoch) not in {int, float}:
            raise ContractError("epoch must be a built-in real number")
        try:
            epoch = float(self.epoch)
        except OverflowError as error:
            raise ContractError("epoch must be finite and non-negative") from error
        if self.global_step < 0:
            raise ContractError("global_step cannot be negative")
        if epoch < 0 or not math.isfinite(epoch):
            raise ContractError("epoch must be finite and non-negative")
        if self.world_size <= 0:
            raise ContractError("world_size must be positive")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ContractError("rank must satisfy 0 <= rank < world_size")
        if not isinstance(self.extras, Mapping):
            raise ContractError("ObjectiveContext.extras must be a mapping")
        if any(type(name) is not str for name in self.extras):
            raise ContractError("ObjectiveContext.extras keys must be strings")
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "extras", dict(self.extras))


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    """Scalar loss plus detached reporting values and diagnostics.

    Raises:
        ObjectiveError: If loss is non-scalar/non-finite or a metric is not a
            finite real Python number or real scalar tensor.
    """

    loss: Tensor
    metrics: Mapping[str, Tensor | float] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate loss and scalar metric contracts."""
        if not isinstance(self.loss, Tensor):
            raise ObjectiveError("objective loss must be a Tensor")
        if self.loss.ndim != 0:
            raise ObjectiveError(
                f"objective loss must be scalar; got shape {tuple(self.loss.shape)}"
            )
        if not self.loss.dtype.is_floating_point or not torch.isfinite(self.loss).item():
            raise ObjectiveError("objective loss must be a finite floating-point scalar")
        if not isinstance(self.diagnostics, Mapping):
            raise ObjectiveError("objective diagnostics must be a mapping")
        metrics = self._validated_metrics()
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def _validated_metrics(self) -> dict[str, Tensor | float]:
        """Return detached scalar metrics after enforcing their runtime union."""
        if not isinstance(self.metrics, Mapping):
            raise ObjectiveError("objective metrics must be a mapping")
        metrics: dict[str, Tensor | float] = {}
        for name, value in self.metrics.items():
            if type(name) is not str or not name.strip():
                raise ObjectiveError("metric names must be non-empty strings")
            if isinstance(value, Tensor):
                if (
                    value.ndim != 0
                    or value.dtype == torch.bool
                    or value.is_complex()
                    or not torch.isfinite(value).item()
                ):
                    raise ObjectiveError(f"metric {name!r} must be a finite real scalar tensor")
                metrics[name] = value.detach()
            else:
                if type(value) not in {int, float}:
                    raise ObjectiveError(f"metric {name!r} must be a finite real number")
                try:
                    normalized = float(value)
                except OverflowError as error:
                    raise ObjectiveError(f"metric {name!r} must be a finite real number") from error
                if not math.isfinite(normalized):
                    raise ObjectiveError(f"metric {name!r} must be a finite real number")
                metrics[name] = normalized
        return metrics


def __getattr__(name: str) -> Any:
    """Resolve canonical record contracts lazily from :mod:`mira.data.schema`."""
    if name not in _DATA_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module("mira.data.schema"), name)


__all__ = [
    "BatchMetadata",
    "CandidateItem",
    "EmbeddingBatch",
    "EmbeddingOutput",
    "EmbeddingRecord",
    "MediaRef",
    "Message",
    "MultimodalItem",
    "ObjectAnnotations",
    "ObjectiveContext",
    "ObjectiveResult",
    "RepresentationRequest",
    "RepresentationType",
    "RetrievalMetadata",
]
