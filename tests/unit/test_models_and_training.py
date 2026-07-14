"""Phase-1 model, training-step, distributed-runner, and checkpoint tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor, nn

from mira.core.contracts import (
    BatchMetadata,
    EmbeddingBatch,
    ObjectiveContext,
    RepresentationRequest,
)
from mira.core.exceptions import CheckpointError, ContractError, InvalidEmbeddingError
from mira.core.layout import GroupLayout
from mira.models.adapters import Qwen3VLEmbeddingAdapter, QwenEmbeddingSemantics
from mira.models.base import TuningSpec
from mira.models.pooling import (
    last_token_pool,
    masked_mean_pool,
    normalize_embeddings,
    truncate_and_normalize,
)
from mira.objectives import MultiPositiveInfoNCE
from mira.training import (
    CheckpointComponents,
    CheckpointManager,
    CheckpointRestoreTargets,
    CheckpointSnapshot,
    DistributedObjectiveRunner,
    TrainingState,
    TrainingStep,
)

if TYPE_CHECKING:
    from pathlib import Path


class _TinyCausalModel(nn.Module):
    def __init__(self, dimension: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Embedding(32, dimension)

    def forward(self, input_ids: Tensor, **_: object) -> SimpleNamespace:
        hidden = self.backbone(input_ids).cumsum(dim=1)
        return SimpleNamespace(hidden_states=(hidden * 0.5, hidden))


def _layout() -> GroupLayout:
    positive = torch.tensor([[True, False]])
    explicit = torch.tensor([[False, True]])
    in_batch = torch.zeros_like(positive)
    false_negative = torch.zeros_like(positive)
    return GroupLayout(
        query_indices=torch.tensor([0]),
        candidate_indices=torch.tensor([1, 2]),
        positive_mask=positive,
        candidate_mask=positive | explicit,
        explicit_negative_mask=explicit,
        in_batch_negative_mask=in_batch,
        false_negative_mask=false_negative,
        task_ids=("retrieval",),
        dataset_ids=("tiny",),
    )


def _batch(fingerprint: str = "template-v1") -> EmbeddingBatch:
    return EmbeddingBatch(
        model_inputs={
            "input_ids": torch.tensor([[1, 9], [1, 9], [2, 9]]),
            "attention_mask": torch.ones(3, 2, dtype=torch.long),
        },
        layout=_layout(),
        metadata=BatchMetadata(
            sample_ids=("sample",),
            task_ids=("retrieval",),
            dataset_ids=("tiny",),
            extras={
                "instruction": "Represent the user's input.",
                "template_fingerprint": fingerprint,
            },
        ),
    )


def _adapter() -> Qwen3VLEmbeddingAdapter:
    return Qwen3VLEmbeddingAdapter(
        _TinyCausalModel(),
        semantics=QwenEmbeddingSemantics(
            model_id="local/tiny-qwen",
            eos_token_id=9,
            template_fingerprint="template-v1",
            embedding_dimension=4,
        ),
    )


def test_pooling_supports_irregular_masks_and_mrl_renormalization() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
        ]
    )
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]])
    assert torch.equal(last_token_pool(hidden, mask), torch.tensor([[3.0, 0.0], [0.0, 2.0]]))
    assert torch.equal(masked_mean_pool(hidden, mask), torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    truncated = truncate_and_normalize(torch.tensor([[3.0, 4.0, 12.0]]), 2)
    assert torch.allclose(truncated, torch.tensor([[0.6, 0.8]]))


def test_pooling_rejects_zero_vectors_and_empty_rows() -> None:
    with pytest.raises(InvalidEmbeddingError, match="zero-norm"):
        normalize_embeddings(torch.zeros(1, 3))
    with pytest.raises(InvalidEmbeddingError, match="empty rows"):
        last_token_pool(torch.ones(1, 2, 3), torch.zeros(1, 2))


def test_qwen_adapter_enforces_prompt_eos_and_dimension_semantics() -> None:
    adapter = _adapter()
    batch = _batch()
    inputs = adapter.prepare_inputs(batch, "train")
    output = adapter.forward_representations(
        inputs,
        RepresentationRequest(dimension=2, hidden_layers=(0,)),
    )
    assert output.global_embeddings is not None
    assert output.global_embeddings.shape == (3, 2)
    assert torch.allclose(output.global_embeddings.norm(dim=1), torch.ones(3))
    assert output.layer_embeddings is not None
    assert output.normalized is True

    with pytest.raises(ContractError, match="fingerprint"):
        adapter.prepare_inputs(_batch("wrong"), "train")
    invalid_inputs = dict(inputs)
    invalid_inputs["input_ids"] = torch.tensor([[1, 8], [1, 9], [2, 9]])
    with pytest.raises(ContractError, match="EOS"):
        adapter.forward_representations(invalid_inputs, RepresentationRequest())
    with pytest.raises(InvalidEmbeddingError, match=r"dimension 5.*exceeds.*dimension 4"):
        adapter.forward_representations(
            inputs,
            RepresentationRequest(dimension=5, normalize=False),
        )


def test_qwen_parameter_groups_are_explicit_and_freeze_unclaimed_parameters() -> None:
    adapter = _adapter()
    groups = adapter.get_parameter_groups(TuningSpec({"backbone": 3e-4}))
    assert [group.name for group in groups] == ["backbone"]
    assert all(parameter.requires_grad for parameter in groups[0].parameters)
    with pytest.raises(ContractError, match="matched no"):
        adapter.get_parameter_groups(TuningSpec({"missing": 1e-4}))


def test_training_step_runs_forward_loss_backward_and_optimizer() -> None:
    torch.manual_seed(7)
    adapter = _adapter()
    objective = MultiPositiveInfoNCE(temperature=0.2)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    before = adapter.model.backbone.weight.detach().clone()
    step = TrainingStep(adapter, objective, optimizer, RepresentationRequest())
    result = step.run(_batch(), ObjectiveContext(global_step=0, epoch=0.0))
    assert result.loss > 0
    assert result.gradient_norm > 0
    assert not torch.equal(before, adapter.model.backbone.weight)


def test_distributed_runner_none_preserves_local_layout_and_mean() -> None:
    layout = _layout()
    queries = torch.tensor([[1.0, 0.0]], requires_grad=True)
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    runner = DistributedObjectiveRunner(MultiPositiveInfoNCE(temperature=0.5), gather_mode="none")
    result = runner.compute(queries, candidates, layout)
    assert result.candidate_sizes == (2,)
    assert result.global_layout.num_candidates == 2
    result.objective.loss.backward()
    assert queries.grad is not None
    assert candidates.grad is not None


def _optimization_step(model: nn.Linear, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    prediction = model(torch.tensor([[1.0, 2.0]]))
    loss = (prediction - torch.tensor([[0.25]])).square().mean()
    loss.backward()
    optimizer.step()


def test_full_checkpoint_resume_matches_continuous_training(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    _optimization_step(model, optimizer)
    manager = CheckpointManager()
    checkpoint = tmp_path / "step-1"
    manager.save(
        checkpoint,
        snapshot=CheckpointSnapshot(
            model=model,
            training_state=TrainingState(global_step=1),
            resolved_config={"seed": 11},
            components=CheckpointComponents(optimizer=optimizer),
        ),
    )
    _optimization_step(model, optimizer)
    expected = model.weight.detach().clone()

    resumed_model = nn.Linear(2, 1)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.05, momentum=0.9)
    loaded = manager.load(
        checkpoint,
        model=resumed_model,
        mode="resume_full",
        targets=CheckpointRestoreTargets(optimizer=resumed_optimizer),
    )
    assert loaded.training_state == TrainingState(global_step=1)
    _optimization_step(resumed_model, resumed_optimizer)
    assert torch.equal(resumed_model.weight, expected)


def test_checkpoint_detects_tampering(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    checkpoint = tmp_path / "checkpoint"
    manager = CheckpointManager()
    manager.save(
        checkpoint,
        snapshot=CheckpointSnapshot(
            model=model,
            training_state=TrainingState(global_step=0),
            resolved_config={},
        ),
    )
    with (checkpoint / "state.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(CheckpointError, match="checksum mismatch"):
        manager.verify(checkpoint)
