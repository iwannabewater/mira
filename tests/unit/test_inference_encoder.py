from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from mira.core import (
    ConfigurationError,
    EmbeddingOutput,
    InvalidEmbeddingError,
    RepresentationRequest,
    RepresentationType,
)
from mira.inference import Encoder, HybridEmbeddingBatch, TokenEmbeddingBatch

if TYPE_CHECKING:
    from collections.abc import Sequence


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, RepresentationRequest]] = []

    def encode_batch(
        self,
        items: Sequence[object],
        *,
        role: str,
        instruction: str | None,
        request: RepresentationRequest,
    ) -> EmbeddingOutput:
        self.calls.append((role, instruction, request))
        lengths = [max(1, len(str(item))) for item in items]
        token_length = max(lengths)
        global_embeddings = torch.tensor([[float(length), 1.0, 2.0, 3.0] for length in lengths])
        token_embeddings = torch.zeros((len(items), token_length, 4))
        token_mask = torch.zeros((len(items), token_length), dtype=torch.bool)
        for row, length in enumerate(lengths):
            token_embeddings[row, :length] = torch.tensor([1.0, 2.0, 3.0, 4.0])
            token_mask[row, :length] = True
        return EmbeddingOutput(
            global_embeddings=global_embeddings,
            token_embeddings=token_embeddings,
            token_mask=token_mask,
        )


class GlobalOnlyBackend:
    def encode_batch(
        self,
        items: Sequence[object],
        *,
        role: str,
        instruction: str | None,
        request: RepresentationRequest,
    ) -> EmbeddingOutput:
        del role, instruction, request
        return EmbeddingOutput(global_embeddings=torch.ones((len(items), 4)))


class PrefixZeroBackend:
    def encode_batch(
        self,
        items: Sequence[object],
        *,
        role: str,
        instruction: str | None,
        request: RepresentationRequest,
    ) -> EmbeddingOutput:
        del role, instruction, request
        return EmbeddingOutput(
            global_embeddings=torch.tensor([[0.0, 0.0, 1.0, 1.0]]).repeat(len(items), 1)
        )


class WrongRowsBackend:
    def encode_batch(
        self,
        items: Sequence[object],
        *,
        role: str,
        instruction: str | None,
        request: RepresentationRequest,
    ) -> EmbeddingOutput:
        del items, role, instruction, request
        return EmbeddingOutput(global_embeddings=torch.ones((1, 4)))


class TokenOnlyBackend:
    def encode_batch(
        self,
        items: Sequence[object],
        *,
        role: str,
        instruction: str | None,
        request: RepresentationRequest,
    ) -> EmbeddingOutput:
        del role, instruction, request
        return EmbeddingOutput(
            token_embeddings=torch.ones((len(items), 1, 4)),
            token_mask=torch.ones((len(items), 1), dtype=torch.bool),
        )


class WrongTokenRowsBackend:
    def encode_batch(
        self,
        items: Sequence[object],
        *,
        role: str,
        instruction: str | None,
        request: RepresentationRequest,
    ) -> EmbeddingOutput:
        del items, role, instruction, request
        return EmbeddingOutput(
            token_embeddings=torch.ones((1, 1, 4)),
            token_mask=torch.ones((1, 1), dtype=torch.bool),
        )


class UncheckedOverflowBackend:
    def encode_batch(
        self,
        items: Sequence[object],
        *,
        role: str,
        instruction: str | None,
        request: RepresentationRequest,
    ) -> EmbeddingOutput:
        del role, instruction, request
        return cast(
            "EmbeddingOutput",
            SimpleNamespace(
                global_embeddings=torch.tensor([[1e38, 1e38]], dtype=torch.float32).repeat(
                    len(items), 1
                )
            ),
        )


def test_encoder_batches_preserves_order_and_applies_role_instruction() -> None:
    backend = FakeBackend()
    encoder = Encoder(backend, query_instruction="retrieve this", default_batch_size=2)

    embeddings = encoder.encode_query(["a", "bbb", "cc"])

    assert embeddings.shape == (3, 4)
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(3))
    assert [call[:2] for call in backend.calls] == [
        ("query", "retrieve this"),
        ("query", "retrieve this"),
    ]


def test_mrl_truncation_always_renormalizes() -> None:
    encoder = Encoder(FakeBackend())

    embeddings = encoder.encode_document("abc", dimension=2, normalize=False)

    assert embeddings.shape == (1, 2)
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(1))


def test_token_batches_pad_across_backend_batches() -> None:
    encoder = Encoder(FakeBackend(), default_batch_size=1)

    result = encoder.encode_tokens(["a", "abcd"])

    assert isinstance(result, TokenEmbeddingBatch)
    assert result.embeddings.shape == (2, 4, 4)
    assert result.mask.tolist() == [
        [True, False, False, False],
        [True, True, True, True],
    ]
    assert torch.equal(result.embeddings[0, 1:], torch.zeros((3, 4)))


def test_hybrid_output_has_aligned_batch_axes() -> None:
    encoder = Encoder(FakeBackend())

    result = encoder.encode(["a", "bb"], representation=RepresentationType.HYBRID)

    assert isinstance(result, HybridEmbeddingBatch)
    assert result.global_embeddings.shape[0] == result.token_embeddings.shape[0] == 2


def test_encoder_rejects_mrl_dimension_above_hidden_size() -> None:
    with pytest.raises(InvalidEmbeddingError, match="exceeds"):
        Encoder(FakeBackend()).encode_query("a", dimension=5)


def test_encoder_explicit_call_validation() -> None:
    encoder = Encoder(FakeBackend())
    with pytest.raises(ConfigurationError, match="default_batch_size"):
        Encoder(FakeBackend(), default_batch_size=0)
    with pytest.raises(ConfigurationError, match="role"):
        encoder.encode("a", role="invalid")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="RepresentationType"):
        encoder.encode("a", representation="global")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="batch_size"):
        encoder.encode("a", batch_size=0)
    with pytest.raises(ConfigurationError, match="dimension"):
        encoder.encode("a", dimension=0)
    with pytest.raises(ConfigurationError, match="must not be empty"):
        encoder.encode([])
    with pytest.raises(ConfigurationError, match="instruction"):
        encoder.encode("a", instruction="  ")


def test_encoder_treats_mapping_as_one_item_and_can_preserve_raw_norm() -> None:
    backend = FakeBackend()
    encoder = Encoder(backend, document_instruction="index")

    result = encoder.encode_document({"text": "a"}, normalize=False)

    assert result.shape == (1, 4)
    assert not torch.allclose(torch.linalg.vector_norm(result, dim=1), torch.ones(1))
    assert backend.calls[0][:2] == ("document", "index")

    scalar = encoder.encode_document(42, instruction="override")
    assert scalar.shape == (1, 4)
    assert backend.calls[-1][:2] == ("document", "override")


def test_encoder_rejects_backend_contract_violations() -> None:
    with pytest.raises(InvalidEmbeddingError, match="omitted requested global"):
        Encoder(TokenOnlyBackend()).encode_query("a")
    with pytest.raises(InvalidEmbeddingError, match="omitted requested token"):
        Encoder(GlobalOnlyBackend()).encode_tokens("a")
    with pytest.raises(InvalidEmbeddingError, match="batch order or size"):
        Encoder(WrongRowsBackend()).encode_query(["a", "b"], batch_size=1_000)
    with pytest.raises(InvalidEmbeddingError, match="token batch order or size"):
        Encoder(WrongTokenRowsBackend()).encode_tokens(["a", "b"], batch_size=1_000)


def test_mrl_rejects_zero_norm_prefix() -> None:
    with pytest.raises(InvalidEmbeddingError, match="zero-norm"):
        Encoder(PrefixZeroBackend()).encode_query("a", dimension=2)


def test_inference_rejects_finite_values_with_overflowing_l2_norm() -> None:
    overflow = torch.tensor([[[1e38, 1e38]]], dtype=torch.float32)
    mask = torch.ones((1, 1), dtype=torch.bool)

    with pytest.raises(InvalidEmbeddingError, match="non-finite L2 norm"):
        TokenEmbeddingBatch(overflow, mask)
    with pytest.raises(InvalidEmbeddingError, match="non-finite L2 norm"):
        HybridEmbeddingBatch(overflow[:, 0], torch.ones_like(overflow), mask)
    with pytest.raises(InvalidEmbeddingError, match="non-finite L2 norm"):
        Encoder(UncheckedOverflowBackend()).encode_query("a")


def test_public_token_and_hybrid_contracts_reject_misalignment() -> None:
    with pytest.raises(InvalidEmbeddingError, match=r"\[N, L, D\]"):
        TokenEmbeddingBatch(torch.ones((1, 4)), torch.ones(1, dtype=torch.bool))
    with pytest.raises(InvalidEmbeddingError, match="boolean"):
        TokenEmbeddingBatch(torch.ones((1, 1, 4)), torch.ones((1, 1)))
    with pytest.raises(InvalidEmbeddingError, match="batch axes"):
        HybridEmbeddingBatch(
            global_embeddings=torch.ones((2, 4)),
            token_embeddings=torch.ones((1, 1, 4)),
            token_mask=torch.ones((1, 1), dtype=torch.bool),
        )
    with pytest.raises(InvalidEmbeddingError, match="dimensions"):
        HybridEmbeddingBatch(
            global_embeddings=torch.ones((1, 3)),
            token_embeddings=torch.ones((1, 1, 4)),
            token_mask=torch.ones((1, 1), dtype=torch.bool),
        )


@pytest.mark.parametrize(
    ("embeddings", "mask", "message"),
    [
        (
            torch.ones((1, 1, 2), dtype=torch.int64),
            torch.ones((1, 1), dtype=torch.bool),
            "floating",
        ),
        (torch.ones((1, 1, 2)), torch.zeros((1, 1), dtype=torch.bool), "select"),
        (torch.tensor([[[float("nan"), 1.0]]]), torch.ones((1, 1), dtype=torch.bool), "finite"),
        (torch.zeros((1, 1, 2)), torch.ones((1, 1), dtype=torch.bool), "zero norm"),
    ],
)
def test_token_contract_rejects_invalid_valid_vectors(
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingError, match=message):
        TokenEmbeddingBatch(embeddings, mask)


def test_inference_contracts_reject_device_and_global_value_errors() -> None:
    with pytest.raises(InvalidEmbeddingError, match="share a device"):
        TokenEmbeddingBatch(
            torch.empty((1, 1, 2), device="meta"),
            torch.ones((1, 1), dtype=torch.bool),
        )
    token_embeddings = torch.ones((1, 1, 2))
    token_mask = torch.ones((1, 1), dtype=torch.bool)
    with pytest.raises(InvalidEmbeddingError, match=r"\[N, D\]"):
        HybridEmbeddingBatch(torch.ones(2), token_embeddings, token_mask)
    with pytest.raises(InvalidEmbeddingError, match="finite floating"):
        HybridEmbeddingBatch(torch.ones((1, 2), dtype=torch.int64), token_embeddings, token_mask)
    with pytest.raises(InvalidEmbeddingError, match="zero norm"):
        HybridEmbeddingBatch(torch.zeros((1, 2)), token_embeddings, token_mask)
    with pytest.raises(InvalidEmbeddingError, match="share a device"):
        HybridEmbeddingBatch(torch.empty((1, 2), device="meta"), token_embeddings, token_mask)
