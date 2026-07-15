"""Backend-neutral encoded vector collections shared across workflows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import torch
from torch import Tensor

from mira.core.exceptions import ContractError
from mira.core.serialization import canonicalize_json_value, freeze_json_value

_MATRIX_RANK = 2


@dataclass(frozen=True, slots=True)
class EncodedCollection:
    """IDs, embeddings, metadata, and provenance aligned on one item axis."""

    item_ids: tuple[str, ...]
    embeddings: Tensor
    metadata: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate alignment, uniqueness, dtype, values, and JSON metadata."""
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
