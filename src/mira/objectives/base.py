"""Backend-independent objective interface and embedding extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from mira.core.exceptions import InvalidEmbeddingError

if TYPE_CHECKING:
    from torch import Tensor

    from mira.core.contracts import (
        EmbeddingBatch,
        EmbeddingOutput,
        ObjectiveContext,
        ObjectiveResult,
    )
    from mira.core.layout import GroupLayout


class EmbeddingObjective(nn.Module, ABC):
    """Base class for objectives consuming only explicit ``GroupLayout`` semantics."""

    def validate(
        self,
        model_capabilities: Any,
        batch_spec: Any,
        distributed_spec: Any,
    ) -> None:
        """Validate static compatibility before training.

        Phase-1 objectives have no backend-specific capability requirements;
        subclasses may override this hook as their contracts grow.
        """
        del model_capabilities, batch_spec, distributed_spec

    @abstractmethod
    def forward(
        self,
        output: EmbeddingOutput,
        batch: EmbeddingBatch,
        context: ObjectiveContext,
    ) -> ObjectiveResult:
        """Compute a scalar objective result from model and batch contracts."""

    @abstractmethod
    def compute(
        self,
        query_embeddings: Tensor,
        candidate_embeddings: Tensor,
        layout: GroupLayout,
    ) -> ObjectiveResult:
        """Compute directly from already separated ``[Q,D]`` and ``[C,D]`` tensors."""

    @staticmethod
    def extract_global_embeddings(
        output: EmbeddingOutput, batch: EmbeddingBatch
    ) -> tuple[Tensor, Tensor, GroupLayout]:
        """Index global embeddings exactly as declared by the batch layout."""
        embeddings = output.global_embeddings
        if embeddings is None:
            raise InvalidEmbeddingError("objective requires global_embeddings")
        layout = batch.layout
        if embeddings.device != layout.query_indices.device:
            raise InvalidEmbeddingError(
                "embedding and GroupLayout tensors must be on the same device"
            )
        largest_index = torch.cat((layout.query_indices, layout.candidate_indices)).max()
        if largest_index.item() >= embeddings.shape[0]:
            raise InvalidEmbeddingError(
                f"GroupLayout flattened index {largest_index.item()} exceeds "
                f"embedding row count {embeddings.shape[0]}"
            )
        return (
            embeddings.index_select(0, layout.query_indices.to(dtype=torch.long)),
            embeddings.index_select(0, layout.candidate_indices.to(dtype=torch.long)),
            layout,
        )
