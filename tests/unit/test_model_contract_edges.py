"""Weight-free edge coverage for model adapter and pooling contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import Tensor, nn

import mira.models.adapters.qwen3_vl_embedding as qwen_adapter_module
from mira.core.contracts import (
    BatchMetadata,
    EmbeddingBatch,
    RepresentationRequest,
    RepresentationType,
)
from mira.core.exceptions import CapabilityError, ContractError, InvalidEmbeddingError
from mira.core.layout import GroupLayout
from mira.models.adapters import Qwen3VLEmbeddingAdapter, Qwen3VLEmbeddingSemantics
from mira.models.base import ExportConfig, ExportManifest, ParameterGroupSpec, TuningSpec
from mira.models.pooling import (
    last_token_pool,
    masked_mean_pool,
    normalize_embeddings,
    truncate_and_normalize,
)

if TYPE_CHECKING:
    from pathlib import Path


class _ForwardModel(nn.Module):
    def __init__(self, dimension: int = 4, *, output_kind: str = "attribute") -> None:
        super().__init__()
        self.dimension = dimension
        self.output_kind = output_kind
        self.weight = nn.Parameter(torch.ones(1))
        self.received: dict[str, Any] = {}

    def forward(self, input_ids: Tensor, **kwargs: Any) -> object:
        self.received = dict(kwargs)
        hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, self.dimension)
        hidden_states: object = (hidden * self.weight, hidden * (self.weight + 1))
        if self.output_kind == "mapping":
            return {"hidden_states": hidden_states}
        if self.output_kind == "missing":
            return SimpleNamespace()
        if self.output_kind == "empty":
            return SimpleNamespace(hidden_states=())
        if self.output_kind == "wrong-rank":
            return SimpleNamespace(hidden_states=(hidden[:, -1],))
        if self.output_kind == "non-tensor":
            return SimpleNamespace(hidden_states=("not-a-tensor",))
        return SimpleNamespace(hidden_states=hidden_states)


class _SavingModel(_ForwardModel):
    def __init__(self, *, produce_files: bool = True) -> None:
        super().__init__()
        self.produce_files = produce_files
        self.safe_serialization: bool | None = None
        self.save_calls = 0

    def save_pretrained(self, destination: Path, *, safe_serialization: bool) -> None:
        self.save_calls += 1
        self.safe_serialization = safe_serialization
        if self.produce_files:
            (destination / "z-model.bin").write_bytes(b"weights")
            (destination / "a-config.json").write_text("{}\n", encoding="utf-8")
            (destination / "ignored-directory").mkdir()


def _layout() -> GroupLayout:
    return GroupLayout(
        query_indices=torch.tensor([0]),
        candidate_indices=torch.tensor([1]),
        positive_mask=torch.tensor([[True]]),
        candidate_mask=torch.tensor([[True]]),
        explicit_negative_mask=torch.tensor([[False]]),
        in_batch_negative_mask=torch.tensor([[False]]),
        false_negative_mask=torch.tensor([[False]]),
        task_ids=("retrieval",),
        dataset_ids=("tiny",),
    )


def _batch(
    *,
    instruction: str = "Represent the user's input.",
    fingerprint: str = "template-v1",
    model_inputs: dict[str, Tensor] | None = None,
) -> EmbeddingBatch:
    inputs = model_inputs or {
        "input_ids": torch.tensor([[1, 9], [2, 9]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
    }
    return EmbeddingBatch(
        model_inputs=inputs,
        layout=_layout(),
        metadata=BatchMetadata(
            sample_ids=("sample",),
            task_ids=("retrieval",),
            dataset_ids=("tiny",),
            extras={
                "instruction": instruction,
                "template_fingerprint": fingerprint,
            },
        ),
    )


def _adapter(
    model: nn.Module | None = None,
    *,
    model_id: str = "local/tiny-qwen",
    dimension: int | None = 4,
) -> Qwen3VLEmbeddingAdapter:
    return Qwen3VLEmbeddingAdapter(
        model or _ForwardModel(),
        semantics=Qwen3VLEmbeddingSemantics(
            model_id=model_id,
            terminal_token_id=9,
            template_fingerprint="template-v1",
            embedding_dimension=dimension,
        ),
    )


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (
            Qwen3VLEmbeddingSemantics(" ", 9, "template", embedding_dimension=4),
            "model_id",
        ),
        (
            Qwen3VLEmbeddingSemantics("local/model", -1, "template", embedding_dimension=4),
            "terminal_token_id",
        ),
        (
            Qwen3VLEmbeddingSemantics("local/model", 9, " ", embedding_dimension=4),
            "fingerprint",
        ),
        (
            Qwen3VLEmbeddingSemantics(
                "local/model",
                9,
                "template",
                instruction=" ",
                embedding_dimension=4,
            ),
            "instruction",
        ),
        (
            Qwen3VLEmbeddingSemantics("local/model", 9, "template"),
            "embedding_dimension",
        ),
        (
            Qwen3VLEmbeddingSemantics(
                "Qwen/Qwen3-VL-Embedding-2B",
                9,
                "template",
                embedding_dimension=1024,
            ),
            "official dimension",
        ),
        (
            Qwen3VLEmbeddingSemantics(
                "Qwen/Qwen3-VL-Embedding-2B",
                9,
                "template",
                embedding_dimension=0,
            ),
            "positive integer",
        ),
        (
            Qwen3VLEmbeddingSemantics(
                "local/model",
                9,
                "template",
                embedding_dimension=False,  # type: ignore[arg-type]
            ),
            "positive integer",
        ),
    ],
)
def test_qwen_semantics_reject_invalid_contracts(
    spec: Qwen3VLEmbeddingSemantics,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        Qwen3VLEmbeddingAdapter(_ForwardModel(), semantics=spec)


def test_qwen_known_model_resolves_its_official_dimension() -> None:
    adapter = _adapter(model_id="Qwen/Qwen3-VL-Embedding-2B", dimension=None)
    assert adapter.embedding_dimension == 2048
    assert adapter.capabilities.supports_global_embedding is True
    assert adapter.capabilities.supports_audio is False


def test_prepare_inputs_rejects_mode_instruction_and_invalid_tokens() -> None:
    adapter = _adapter()
    with pytest.raises(ContractError, match="unsupported adapter mode"):
        adapter.prepare_inputs(_batch(), "invalid")  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="instruction"):
        adapter.prepare_inputs(_batch(instruction="different"), "eval")
    with pytest.raises(ContractError, match="tensor field 'attention_mask'"):
        adapter.prepare_inputs(
            _batch(model_inputs={"input_ids": torch.tensor([[1, 9], [2, 9]])}),
            "inference",
        )


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"attention_mask": torch.ones(2, 2)}, "input_ids"),
        (
            {
                "input_ids": torch.tensor([[1.0, 9.0], [2.0, 9.0]]),
                "attention_mask": torch.ones(2, 2),
            },
            "integer",
        ),
        (
            {
                "input_ids": torch.tensor([1, 9]),
                "attention_mask": torch.ones(2, dtype=torch.long),
            },
            "integer",
        ),
        (
            {
                "input_ids": torch.tensor([[1, 9], [2, 9]]),
                "attention_mask": torch.ones(2, 1, dtype=torch.long),
            },
            "shape",
        ),
    ],
)
def test_qwen_forward_rejects_invalid_token_inputs(
    inputs: dict[str, Tensor],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        _adapter().forward_representations(inputs, RepresentationRequest())


def test_qwen_token_validation_rejects_cross_device_inputs() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 9], [2, 9]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long, device="meta"),
    }
    with pytest.raises(ContractError, match="same device"):
        _adapter().forward_representations(inputs, RepresentationRequest())


def test_qwen_rejects_unsupported_representation_and_empty_eos_row() -> None:
    adapter = _adapter()
    with pytest.raises(CapabilityError, match="token and hybrid"):
        adapter.forward_representations(
            _batch().model_inputs,
            RepresentationRequest(representation=RepresentationType.TOKEN),
        )
    empty_row = {
        "input_ids": torch.tensor([[1, 9], [2, 9]]),
        "attention_mask": torch.tensor([[1, 1], [0, 0]]),
    }
    with pytest.raises(ContractError, match="empty input row"):
        adapter.forward_representations(empty_row, RepresentationRequest())


def test_qwen_forward_filters_labels_and_supports_unnormalized_outputs() -> None:
    model = _ForwardModel(output_kind="mapping")
    adapter = _adapter(model)
    inputs = dict(_batch().model_inputs)
    inputs["labels"] = torch.tensor([[1, 9], [2, 9]])
    output = adapter.forward_representations(
        inputs,
        RepresentationRequest(dimension=2, hidden_layers=(0,), normalize=False),
    )
    assert output.global_embeddings is not None
    assert output.global_embeddings.shape == (2, 2)
    assert output.normalized is False
    assert output.layer_embeddings is not None
    assert output.layer_embeddings[0].shape == (2, 2)
    assert "labels" not in model.received
    assert model.received["output_hidden_states"] is True
    assert model.received["return_dict"] is True
    assert model.received["use_cache"] is False


def test_qwen_forward_supports_full_dimension_without_normalization() -> None:
    output = _adapter().forward_representations(
        _batch().model_inputs,
        RepresentationRequest(hidden_layers=(-1,), normalize=False),
    )
    assert output.global_embeddings is not None
    assert output.global_embeddings.shape == (2, 4)
    assert output.layer_embeddings is not None
    assert output.layer_embeddings[-1].shape == (2, 4)


def test_qwen_global_forward_does_not_retain_all_hidden_layers() -> None:
    model = _ForwardModel()
    output = _adapter(model).forward_representations(
        _batch().model_inputs,
        RepresentationRequest(normalize=True),
    )
    assert output.global_embeddings is not None
    assert model.received["output_hidden_states"] is False


@pytest.mark.parametrize("output_kind", ["missing", "empty", "wrong-rank", "non-tensor"])
def test_qwen_rejects_missing_or_invalid_hidden_states(output_kind: str) -> None:
    adapter = _adapter(_ForwardModel(output_kind=output_kind))
    with pytest.raises(InvalidEmbeddingError, match="hidden states"):
        adapter.forward_representations(_batch().model_inputs, RepresentationRequest())


def test_qwen_rejects_wrong_output_dimension_and_unavailable_layer() -> None:
    with pytest.raises(InvalidEmbeddingError, match="does not match declared"):
        _adapter(_ForwardModel(dimension=3)).forward_representations(
            _batch().model_inputs,
            RepresentationRequest(),
        )
    with pytest.raises(InvalidEmbeddingError, match="hidden layer 7 is unavailable"):
        _adapter().forward_representations(
            _batch().model_inputs,
            RepresentationRequest(hidden_layers=(7,)),
        )


class _ComponentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Linear(2, 2)
        self.language_model = nn.Linear(2, 2)
        self.container = nn.Module()
        self.container.custom = nn.Linear(2, 2)
        self.unclaimed = nn.Linear(2, 2)
        self.unclaimed.bias.requires_grad = False


class _OverlappingComponentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = nn.Module()
        self.vision.language_model = nn.Linear(2, 2)


def test_qwen_parameter_groups_match_markers_and_freeze_unclaimed_parameters() -> None:
    model = _ComponentModel()
    groups = _adapter(model).get_parameter_groups(
        TuningSpec(
            {"vision": 1e-4, "language_model": 2e-4, "custom": 3e-4},
            weight_decay=0.01,
        )
    )
    assert [group.name for group in groups] == ["vision", "language_model", "custom"]
    assert [group.learning_rate for group in groups] == [1e-4, 2e-4, 3e-4]
    assert all(group.weight_decay == 0.01 for group in groups)
    assert all(parameter.requires_grad for group in groups for parameter in group.parameters)
    assert all(not parameter.requires_grad for parameter in model.unclaimed.parameters())


def test_qwen_parameter_groups_reject_overlap_and_allow_freeze_all() -> None:
    with pytest.raises(ContractError, match="overlaps"):
        _adapter(_OverlappingComponentModel()).get_parameter_groups(
            TuningSpec({"vision": 1e-4, "language_model": 1e-4})
        )

    model = _ComponentModel()
    assert _adapter(model).get_parameter_groups(TuningSpec({})) == []
    assert all(not parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": " ", "parameters": (nn.Parameter(torch.ones(1)),), "learning_rate": 1.0}, "name"),
        ({"name": "group", "parameters": (), "learning_rate": 1.0}, "contain parameters"),
        (
            {
                "name": "group",
                "parameters": (nn.Parameter(torch.ones(1)),),
                "learning_rate": 0.0,
            },
            "positive",
        ),
        (
            {
                "name": "group",
                "parameters": (nn.Parameter(torch.ones(1)),),
                "learning_rate": 1.0,
                "weight_decay": -0.1,
            },
            "non-negative",
        ),
        (
            {
                "name": "group",
                "parameters": (nn.Parameter(torch.ones(1)),),
                "learning_rate": float("nan"),
            },
            "finite positive",
        ),
        (
            {
                "name": "group",
                "parameters": (nn.Parameter(torch.ones(1)),),
                "learning_rate": True,
            },
            "finite positive",
        ),
        (
            {
                "name": "group",
                "parameters": (nn.Parameter(torch.ones(1)),),
                "learning_rate": "0.1",
            },
            "finite positive",
        ),
        (
            {
                "name": "group",
                "parameters": (nn.Parameter(torch.ones(1)),),
                "learning_rate": 10**10_000,
            },
            "finite positive",
        ),
        (
            {
                "name": "group",
                "parameters": (nn.Parameter(torch.ones(1)),),
                "learning_rate": 1e-4,
                "weight_decay": float("nan"),
            },
            "finite non-negative",
        ),
        (
            {
                "name": "group",
                "parameters": (nn.Parameter(torch.ones(1)),),
                "learning_rate": 1e-4,
                "weight_decay": False,
            },
            "finite non-negative",
        ),
        (
            {
                "name": "group",
                "parameters": (nn.Parameter(torch.ones(1)),),
                "learning_rate": 1e-4,
                "weight_decay": "0.1",
            },
            "finite non-negative",
        ),
    ],
)
def test_parameter_group_spec_rejects_invalid_values(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ParameterGroupSpec(**kwargs)


def test_parameter_group_spec_accepts_valid_values() -> None:
    parameter = nn.Parameter(torch.ones(1))
    spec = ParameterGroupSpec("group", (parameter,), 1e-4, 0.0)
    assert spec.parameters == (parameter,)


@pytest.mark.parametrize(
    ("rates", "weight_decay", "error_type", "message"),
    [
        ([], 0.0, TypeError, "must be a mapping"),
        ({" ": 1e-4}, 0.0, ValueError, "component names"),
        ({1: 1e-4}, 0.0, ValueError, "component names"),
        ({"vision": 0.0}, 0.0, ValueError, "finite positive"),
        ({"vision": float("nan")}, 0.0, ValueError, "finite positive"),
        ({"vision": False}, 0.0, ValueError, "finite positive"),
        ({"vision": "0.1"}, 0.0, ValueError, "finite positive"),
        ({}, float("nan"), ValueError, "finite non-negative"),
        ({}, True, ValueError, "finite non-negative"),
        ({}, "0.1", ValueError, "finite non-negative"),
    ],
)
def test_tuning_spec_rejects_invalid_runtime_values(
    rates: Any,
    weight_decay: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        TuningSpec(rates, weight_decay=weight_decay)


def test_tuning_spec_keeps_empty_mapping_valid_and_detached() -> None:
    source: dict[str, float] = {}
    tuning = TuningSpec(source)
    source["late"] = 1e-4
    assert dict(tuning.component_learning_rates) == {}
    assert tuning.weight_decay == 0.0
    with pytest.raises(TypeError):
        tuning.component_learning_rates["late"] = 1e-4  # type: ignore[index]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"safe_serialization": 1}, ValueError, "safe_serialization must be boolean"),
        ({"merge_adapter": 0}, ValueError, "merge_adapter must be boolean"),
        ({"metadata": []}, TypeError, "metadata must be a mapping"),
        ({"metadata": {"score": float("nan")}}, ValueError, "non-finite"),
        ({"metadata": {"opaque": object()}}, ValueError, "unsupported value"),
        ({"metadata": {1: "coerced"}}, ValueError, "non-string object key"),
    ],
)
def test_export_config_rejects_noncanonical_values(
    kwargs: dict[str, Any],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        ExportConfig(**kwargs)


def test_export_config_rejects_cycles_and_detaches_valid_metadata() -> None:
    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cycle"):
        ExportConfig(metadata={"cycle": cycle})

    source: dict[str, Any] = {"nested": {"values": (1, 2)}}
    config = ExportConfig(metadata=source)
    source["nested"]["values"] = (9,)
    assert config.metadata["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        config.metadata["new"] = "value"  # type: ignore[index]


def _internal_export_manifest(**overrides: Any) -> ExportManifest:
    values: dict[str, Any] = {
        "model_id": "local/model",
        "pooling": "last_eos",
        "instruction": "Represent the input.",
        "normalization": "l2",
        "embedding_dimension": 4,
        "template_fingerprint": "template-v1",
        "files": ("weights.bin",),
        "metadata": {"revision": "abc123"},
    }
    values.update(overrides)
    return ExportManifest(**values)


@pytest.mark.parametrize(
    "field",
    ["model_id", "pooling", "instruction", "normalization", "template_fingerprint"],
)
def test_internal_export_manifest_rejects_blank_required_strings(field: str) -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        _internal_export_manifest(**{field: " "})


@pytest.mark.parametrize("dimension", [True, 0, -1, 4.0])
def test_internal_export_manifest_rejects_invalid_dimension(dimension: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _internal_export_manifest(embedding_dimension=dimension)


@pytest.mark.parametrize(
    "files",
    [
        (),
        ("",),
        (1,),
        ("weights.bin", "weights.bin"),
        ("/absolute.bin",),
        ("../escape.bin",),
        ("nested//noncanonical.bin",),
        (r"windows\path.bin",),
    ],
)
def test_internal_export_manifest_rejects_invalid_files(files: Any) -> None:
    with pytest.raises(ValueError, match="files"):
        _internal_export_manifest(files=files)


def test_internal_export_manifest_normalizes_and_freezes_collections() -> None:
    source: dict[str, Any] = {"nested": {"values": [1, 2]}}
    manifest = _internal_export_manifest(files=["nested/weights.bin"], metadata=source)
    source["nested"]["values"].append(3)
    assert manifest.files == ("nested/weights.bin",)
    assert manifest.metadata["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        manifest.metadata["new"] = "value"  # type: ignore[index]


def test_qwen_export_rejects_unsupported_and_unsafe_destinations(tmp_path: Path) -> None:
    adapter = _adapter()
    with pytest.raises(CapabilityError, match="merging"):
        adapter.export(tmp_path / "merged", ExportConfig(merge_adapter=True))

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "existing").write_text("occupied", encoding="utf-8")
    with pytest.raises(ContractError, match="not empty"):
        adapter.export(nonempty, ExportConfig())

    with pytest.raises(CapabilityError, match="save_pretrained"):
        adapter.export(tmp_path / "unsupported", ExportConfig())


@pytest.mark.parametrize(
    ("config_factory", "message"),
    [
        (
            lambda: ExportConfig(safe_serialization=1),  # type: ignore[arg-type]
            "safe_serialization must be boolean",
        ),
        (
            lambda: ExportConfig(merge_adapter=0),  # type: ignore[arg-type]
            "merge_adapter must be boolean",
        ),
        (lambda: ExportConfig(metadata={"score": float("nan")}), "non-finite"),
        (lambda: ExportConfig(metadata={"opaque": object()}), "unsupported value"),
        (
            lambda: ExportConfig(metadata={1: "coerced"}),  # type: ignore[dict-item]
            "non-string object key",
        ),
    ],
)
def test_invalid_export_config_fails_before_model_or_filesystem_side_effects(
    tmp_path: Path,
    config_factory: Any,
    message: str,
) -> None:
    model = _SavingModel()
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match=message):
        _adapter(model).export(destination, config_factory())
    assert model.save_calls == 0
    assert not destination.exists()


def test_qwen_export_rejects_empty_save_and_writes_portable_manifest(tmp_path: Path) -> None:
    empty = _adapter(_SavingModel(produce_files=False))
    with pytest.raises(ContractError, match="produced no export files"):
        empty.export(tmp_path / "empty", ExportConfig())

    model = _SavingModel()
    adapter = _adapter(model)
    destination = tmp_path / "portable"
    manifest = adapter.export(
        destination,
        ExportConfig(safe_serialization=False, metadata={"revision": "abc123"}),
    )
    assert model.safe_serialization is False
    assert manifest.files == ("a-config.json", "z-model.bin")
    assert manifest.metadata == {"revision": "abc123"}
    payload = json.loads((destination / "mira-export.json").read_text(encoding="utf-8"))
    assert payload["files"] == ["a-config.json", "z-model.bin"]
    assert payload["embedding_dimension"] == 4
    assert payload["metadata"] == {"revision": "abc123"}


def test_qwen_export_forces_strict_json_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    real_dumps = json.dumps

    def strict_dumps(value: Any, **kwargs: Any) -> str:
        observed.update(kwargs)
        return real_dumps(value, **kwargs)

    monkeypatch.setattr(
        qwen_adapter_module,
        "json",
        SimpleNamespace(dumps=strict_dumps),
    )
    _adapter(_SavingModel()).export(tmp_path / "strict", ExportConfig())
    assert observed["allow_nan"] is False


@pytest.mark.parametrize(
    ("embeddings", "epsilon", "message"),
    [
        (torch.tensor([[1, 2]]), 1e-12, "floating-point"),
        (torch.tensor([1.0, 2.0]), 1e-12, "at least two dimensions"),
        (torch.tensor([[1.0, 2.0]]), 0.0, "epsilon must be positive"),
        (torch.tensor([[1.0, 2.0]]), float("nan"), "epsilon must be positive"),
        (torch.tensor([[float("nan"), 1.0]]), 1e-12, "NaN or infinite"),
        (torch.tensor([[1e38, 1e38]]), 1e-12, "L2 norms are non-finite"),
    ],
)
def test_normalize_embeddings_rejects_invalid_inputs(
    embeddings: Tensor,
    epsilon: float,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingError, match=message):
        normalize_embeddings(embeddings, epsilon=epsilon)


@pytest.mark.parametrize(
    ("hidden", "mask", "message"),
    [
        (torch.ones(2, 3), torch.ones(2, 3), "hidden states"),
        (torch.ones(2, 3, 4), torch.ones(3), "attention mask"),
        (torch.ones(2, 3, 4), torch.ones(2, 2), "attention mask"),
        (torch.ones(2, 3, 4), torch.tensor([[1, 2, 0], [1, 1, 1]]), "binary"),
    ],
)
def test_pooling_rejects_shape_and_mask_contract_violations(
    hidden: Tensor,
    mask: Tensor,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingError, match=message):
        masked_mean_pool(hidden, mask)


def test_pooling_rejects_cross_device_mask() -> None:
    with pytest.raises(InvalidEmbeddingError, match="same device"):
        last_token_pool(
            torch.ones(2, 3, 4),
            torch.ones(2, 3, dtype=torch.bool, device="meta"),
        )


def test_pooling_accepts_boolean_mask() -> None:
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]])
    mask = torch.tensor([[True, False]])
    assert torch.equal(masked_mean_pool(hidden, mask), torch.tensor([[1.0, 3.0]]))


@pytest.mark.parametrize(
    ("embeddings", "dimension", "message"),
    [
        (torch.ones(1, 2, 3), 2, r"requires \[batch, dimension\]"),
        (torch.ones(1, 3), 0, r"must be in \[1, 3\]"),
        (torch.ones(1, 3), 4, r"must be in \[1, 3\]"),
    ],
)
def test_truncate_and_normalize_rejects_invalid_dimension(
    embeddings: Tensor,
    dimension: int,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingError, match=message):
        truncate_and_normalize(embeddings, dimension)
