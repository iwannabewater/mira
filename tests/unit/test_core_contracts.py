from __future__ import annotations

import pytest
import torch

from mira import core
from mira.core import (
    BatchMetadata,
    ContractError,
    EmbeddingBatch,
    EmbeddingOutput,
    GroupLayout,
    InvalidEmbeddingError,
    ObjectiveContext,
    ObjectiveError,
    ObjectiveResult,
    RepresentationRequest,
    contracts,
)
from mira.data.schema import Message as DataMessage


def _single_layout() -> GroupLayout:
    return GroupLayout(
        query_indices=torch.tensor([0]),
        candidate_indices=torch.tensor([1, 2]),
        positive_mask=torch.tensor([[True, False]]),
        candidate_mask=torch.tensor([[True, True]]),
        explicit_negative_mask=torch.tensor([[False, True]]),
        in_batch_negative_mask=torch.tensor([[False, False]]),
        false_negative_mask=torch.tensor([[False, False]]),
        task_ids=("task",),
        dataset_ids=("dataset",),
    )


def test_embedding_batch_requires_metadata_layout_alignment() -> None:
    metadata = BatchMetadata(sample_ids=("sample",), task_ids=("task",), dataset_ids=("dataset",))
    batch = EmbeddingBatch(
        model_inputs={"input_ids": torch.tensor([[1, 2]])},
        layout=_single_layout(),
        metadata=metadata,
    )
    assert batch.metadata.sample_ids == ("sample",)

    wrong = BatchMetadata(sample_ids=("sample",), task_ids=("other",), dataset_ids=("dataset",))
    with pytest.raises(ContractError, match="task_ids"):
        EmbeddingBatch(
            model_inputs={"input_ids": torch.tensor([[1]])},
            layout=_single_layout(),
            metadata=wrong,
        )


def test_batch_metadata_and_model_inputs_fail_explicitly() -> None:
    with pytest.raises(ContractError, match="at least one"):
        BatchMetadata(sample_ids=(), task_ids=(), dataset_ids=())
    metadata = BatchMetadata(sample_ids=("sample",), task_ids=("task",), dataset_ids=("dataset",))
    with pytest.raises(ContractError, match="must not be empty"):
        EmbeddingBatch(model_inputs={}, layout=_single_layout(), metadata=metadata)

    with pytest.raises(ContractError, match="task_ids has"):
        BatchMetadata(sample_ids=("a", "b"), task_ids=("task",), dataset_ids=("d", "d"))
    with pytest.raises(ContractError, match="non-empty"):
        BatchMetadata(sample_ids=(" ",), task_ids=("task",), dataset_ids=("d",))
    with pytest.raises(ContractError, match="map non-empty"):
        EmbeddingBatch(model_inputs={"": torch.ones(1)}, layout=_single_layout(), metadata=metadata)
    with pytest.raises(ContractError, match="map non-empty"):
        EmbeddingBatch(
            model_inputs={"bad": object()},  # type: ignore[dict-item]
            layout=_single_layout(),
            metadata=metadata,
        )


def test_embedding_batch_rejects_query_count_and_dataset_mismatch() -> None:
    two = BatchMetadata(
        sample_ids=("a", "b"),
        task_ids=("task", "task"),
        dataset_ids=("dataset", "dataset"),
    )
    with pytest.raises(ContractError, match="query count"):
        EmbeddingBatch(model_inputs={"x": torch.ones(1)}, layout=_single_layout(), metadata=two)

    wrong_dataset = BatchMetadata(sample_ids=("a",), task_ids=("task",), dataset_ids=("other",))
    with pytest.raises(ContractError, match="dataset_ids"):
        EmbeddingBatch(
            model_inputs={"x": torch.ones(1)},
            layout=_single_layout(),
            metadata=wrong_dataset,
        )


def test_global_embedding_output_has_explicit_normalization_contract() -> None:
    output = EmbeddingOutput(
        global_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]), normalized=True
    )
    assert output.global_embeddings is not None
    assert output.global_embeddings.shape == (2, 2)

    with pytest.raises(InvalidEmbeddingError, match="non-unit"):
        EmbeddingOutput(global_embeddings=torch.tensor([[2.0, 0.0]]), normalized=True)
    with pytest.raises(InvalidEmbeddingError, match="zero-norm"):
        EmbeddingOutput(global_embeddings=torch.zeros((1, 2)))
    with pytest.raises(InvalidEmbeddingError, match="NaN"):
        EmbeddingOutput(global_embeddings=torch.tensor([[float("nan"), 1.0]]))


def test_embedding_output_rejects_finite_values_with_overflowing_l2_norm() -> None:
    overflow = torch.tensor([[1e38, 1e38]], dtype=torch.float32)

    with pytest.raises(InvalidEmbeddingError, match="non-finite L2 norm"):
        EmbeddingOutput(global_embeddings=overflow)
    with pytest.raises(InvalidEmbeddingError, match="non-finite L2 norm"):
        EmbeddingOutput(
            token_embeddings=overflow.unsqueeze(1),
            token_mask=torch.ones((1, 1), dtype=torch.bool),
        )


def test_token_and_hybrid_output_shapes_are_cross_checked() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[True, False]])
    output = EmbeddingOutput(
        global_embeddings=torch.tensor([[1.0, 0.0]]),
        token_embeddings=tokens,
        token_mask=mask,
        normalized=True,
    )
    assert output.token_mask is mask

    with pytest.raises(InvalidEmbeddingError, match="requires token_mask"):
        EmbeddingOutput(token_embeddings=tokens)
    with pytest.raises(InvalidEmbeddingError, match="shape"):
        EmbeddingOutput(token_embeddings=tokens, token_mask=torch.tensor([[True]]))
    with pytest.raises(InvalidEmbeddingError, match="batch size"):
        EmbeddingOutput(
            global_embeddings=torch.ones((2, 2)),
            token_embeddings=tokens,
            token_mask=mask,
        )
    with pytest.raises(InvalidEmbeddingError, match="cannot be supplied"):
        EmbeddingOutput(global_embeddings=torch.ones((1, 2)), token_mask=mask)
    with pytest.raises(InvalidEmbeddingError, match="boolean"):
        EmbeddingOutput(token_embeddings=tokens, token_mask=torch.ones((1, 2)))
    with pytest.raises(InvalidEmbeddingError, match="at least one token"):
        EmbeddingOutput(
            token_embeddings=tokens,
            token_mask=torch.zeros((1, 2), dtype=torch.bool),
        )


def test_layer_logits_and_empty_output_contracts() -> None:
    output = EmbeddingOutput(
        layer_embeddings={1: torch.ones((2, 3)), 4: torch.ones((2, 1, 3))},
        logits=torch.ones((2, 5)),
    )
    assert output.layer_embeddings is not None
    with pytest.raises(InvalidEmbeddingError, match="at least one"):
        EmbeddingOutput()
    with pytest.raises(InvalidEmbeddingError, match="cannot be an empty"):
        EmbeddingOutput(layer_embeddings={})
    with pytest.raises(InvalidEmbeddingError, match="keys must be integers"):
        EmbeddingOutput(layer_embeddings={"one": torch.ones((1, 2))})  # type: ignore[dict-item]
    with pytest.raises(InvalidEmbeddingError, match=r"\[N,D\]"):
        EmbeddingOutput(layer_embeddings={1: torch.ones((2, 1, 1, 1))})
    with pytest.raises(InvalidEmbeddingError, match="NaN"):
        EmbeddingOutput(layer_embeddings={1: torch.tensor([[float("nan")]])})
    with pytest.raises(InvalidEmbeddingError, match="floating-point"):
        EmbeddingOutput(layer_embeddings={1: torch.ones((1, 2), dtype=torch.int64)})
    with pytest.raises(InvalidEmbeddingError, match="batch size"):
        EmbeddingOutput(
            global_embeddings=torch.ones((2, 2)), layer_embeddings={1: torch.ones((1, 2))}
        )
    with pytest.raises(InvalidEmbeddingError, match="batch axis"):
        EmbeddingOutput(logits=torch.tensor(1.0))
    with pytest.raises(InvalidEmbeddingError, match="NaN"):
        EmbeddingOutput(logits=torch.tensor([[float("nan")]]))
    with pytest.raises(InvalidEmbeddingError, match="floating-point"):
        EmbeddingOutput(logits=torch.ones((1, 2), dtype=torch.int64))
    with pytest.raises(InvalidEmbeddingError, match="batch size"):
        EmbeddingOutput(global_embeddings=torch.ones((2, 2)), logits=torch.ones((1, 3)))


def test_embedding_output_rejects_empty_and_integer_embedding_axes() -> None:
    with pytest.raises(InvalidEmbeddingError, match="non-empty rank-2"):
        EmbeddingOutput(global_embeddings=torch.empty((0, 2)))
    with pytest.raises(InvalidEmbeddingError, match="floating-point"):
        EmbeddingOutput(global_embeddings=torch.ones((1, 2), dtype=torch.int64))


def test_request_context_and_objective_result_validate_boundaries() -> None:
    assert RepresentationRequest(dimension=512).dimension == 512
    with pytest.raises(ContractError, match="positive"):
        RepresentationRequest(dimension=0)
    with pytest.raises(ContractError, match="unique"):
        RepresentationRequest(hidden_layers=(1, 1))

    context = ObjectiveContext(global_step=4, epoch=0.5, rank=1, world_size=2)
    assert context.rank == 1
    with pytest.raises(ContractError, match="rank"):
        ObjectiveContext(global_step=0, epoch=0, rank=2, world_size=2)
    with pytest.raises(ContractError, match="global_step"):
        ObjectiveContext(global_step=-1, epoch=0)
    with pytest.raises(ContractError, match="epoch"):
        ObjectiveContext(global_step=0, epoch=float("inf"))
    with pytest.raises(ContractError, match="world_size"):
        ObjectiveContext(global_step=0, epoch=0, world_size=0)

    result = ObjectiveResult(loss=torch.tensor(1.0), metrics={"temperature": 0.07})
    assert result.loss.item() == pytest.approx(1.0)
    with pytest.raises(ObjectiveError, match="scalar"):
        ObjectiveResult(loss=torch.ones(2))
    with pytest.raises(ObjectiveError, match="finite"):
        ObjectiveResult(loss=torch.tensor(float("inf")))
    with pytest.raises(ObjectiveError, match="floating-point"):
        ObjectiveResult(loss=torch.tensor(1))
    with pytest.raises(ObjectiveError, match="metric names"):
        ObjectiveResult(loss=torch.tensor(1.0), metrics={" ": 1.0})
    with pytest.raises(ObjectiveError, match="scalar tensor"):
        ObjectiveResult(loss=torch.tensor(1.0), metrics={"bad": torch.ones(2)})
    with pytest.raises(ObjectiveError, match="finite"):
        ObjectiveResult(loss=torch.tensor(1.0), metrics={"bad": float("nan")})


def test_objective_result_detaches_reporting_tensors() -> None:
    metric = torch.tensor(2.0, requires_grad=True)

    result = ObjectiveResult(
        loss=torch.tensor(1.0, requires_grad=True),
        metrics={"tracked": metric},
    )

    detached = result.metrics["tracked"]
    assert isinstance(detached, torch.Tensor)
    assert not detached.requires_grad
    assert detached is not metric


def test_record_contracts_are_lazy_aliases_to_canonical_data_schema() -> None:
    assert core.Message is DataMessage
    assert contracts.Message is DataMessage
    with pytest.raises(AttributeError):
        core.__getattr__("NotAContract")
    with pytest.raises(AttributeError):
        contracts.__getattr__("NotAContract")
