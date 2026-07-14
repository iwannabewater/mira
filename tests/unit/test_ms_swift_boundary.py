"""Behavior tests for the lazy, audited MS-SWIFT boundary."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Lock, RLock, Thread
from types import SimpleNamespace
from typing import Any, Self

import pytest
import torch
from torch import nn

from mira.cli._media import resolve_item_media
from mira.core.exceptions import ConfigurationValidationError, ContractError
from mira.data.schema import MediaRef, Message, MultimodalItem
from mira.integrations.ms_swift import (
    SwiftBackendAdapter,
    SwiftBackendAssets,
    backend,
    compatibility,
)
from mira.integrations.ms_swift.arguments import SwiftLoadSpec, SwiftTemplateSpec
from mira.integrations.ms_swift.backend import (
    _attention_name,
    _resolve_eos_token_id,
    _torch_dtype,
    load_qwen3_vl_embedding,
)
from mira.integrations.ms_swift.compatibility import (
    CompatibilityReport,
    DependencyCheck,
    require_compatible,
)
from mira.integrations.ms_swift.peft_bridge import LoraSpec, apply_lora
from mira.integrations.ms_swift.template_bridge import SwiftTemplateBridge, template_fingerprint
from mira.integrations.ms_swift.trainer_bridge import SwiftTrainerSpec, create_swift_trainer
from mira.models.base import ExportConfig, ExportManifest
from mira.models.qwen_audit import AUDITED_QWEN_TARGETS
from mira.training import (
    CheckpointLoadResult,
    CheckpointRestoreTargets,
    CheckpointSnapshot,
    TrainingState,
)


class _ObservableRLock:
    def __init__(self) -> None:
        self._lock = RLock()
        self._counter_lock = Lock()
        self._attempts = 0
        self.second_attempted = Event()

    def __enter__(self) -> Self:
        with self._counter_lock:
            self._attempts += 1
            if self._attempts == 2:
                self.second_attempted.set()
        self._lock.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self._lock.release()


def _load_spec(
    model_id: str = "Qwen/Qwen3-VL-Embedding-2B",
    **kwargs: Any,
) -> SwiftLoadSpec:
    target = AUDITED_QWEN_TARGETS[model_id]
    return SwiftLoadSpec(
        model_id=model_id,
        revision=kwargs.pop("revision", target.revision),
        embedding_dimension=kwargs.pop("embedding_dimension", target.embedding_dimension),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: SwiftLoadSpec(" "), "model_id"),
        (lambda: SwiftLoadSpec(1), "model_id"),  # type: ignore[arg-type]
        (lambda: _load_spec(revision="main"), "immutable revision"),
        (lambda: _load_spec(embedding_dimension=1), "embedding_dimension"),
        (lambda: _load_spec(model_type="other"), "model_type"),  # type: ignore[arg-type]
        (lambda: _load_spec(image_max_tokens=0), "positive integer"),
        (lambda: _load_spec(image_max_tokens=True), "positive integer"),
        (lambda: _load_spec(use_hf=False), "use_hf"),
        (lambda: _load_spec(load_model=False), "load_model"),
        (lambda: _load_spec(attention="magic"), "attention"),  # type: ignore[arg-type]
        (lambda: SwiftTemplateSpec(instruction=" "), "instruction"),
        (lambda: SwiftTemplateSpec(template_type="other"), "template_type"),
        (lambda: SwiftTemplateSpec(max_length=0), "max_length"),
        (lambda: SwiftTemplateSpec(max_length=True), "max_length"),
        (lambda: SwiftTemplateSpec(padding_side="middle"), "padding_side"),  # type: ignore[arg-type]
        (
            lambda: SwiftTemplateSpec(truncation_strategy="truncate"),  # type: ignore[arg-type]
            "truncation_strategy",
        ),
    ],
)
def test_swift_specs_reject_unsupported_or_ambiguous_values(
    factory: Any,
    match: str,
) -> None:
    with pytest.raises(ConfigurationValidationError, match=match):
        factory()


def test_compatibility_report_covers_versions_modules_and_failure_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "ms-swift": "4.4.1",
        "transformers": "4.57.1.dev0",
        "qwen-vl-utils": "0.0.14",
    }
    monkeypatch.setattr(compatibility, "_installed_version", versions.get)
    monkeypatch.setattr(
        compatibility.importlib.util,
        "find_spec",
        lambda name: object() if name in {"decord", "swift"} else None,
    )
    report = compatibility.check_compatibility()
    assert report.compatible
    assert not report.failures
    assert compatibility._release_tuple("5.1") == (5, 1, 0)
    assert compatibility._release_tuple("invalid") is None
    assert compatibility._format_version((1, 2, 3)) == "1.2.3"

    monkeypatch.setattr(
        compatibility,
        "check_compatibility",
        lambda: CompatibilityReport(
            (
                DependencyCheck(
                    "ms-swift",
                    "4.3.0",
                    compatible=False,
                    requirement="==4.4.1",
                    reason="wrong version",
                ),
                DependencyCheck(
                    "swift",
                    None,
                    compatible=False,
                    requirement="importable module",
                    reason="missing",
                ),
            )
        ),
    )
    with pytest.raises(ConfigurationValidationError, match=r"ms-swift.*swift"):
        require_compatible()


class _FakeTemplate:
    def __init__(self) -> None:
        self.task_type = "embedding"
        self.mode = ""
        self.media_bytes: dict[str, list[bytes]] = {}
        self.media_paths: dict[str, list[Path]] = {}

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        for field in ("images", "videos", "audios"):
            if field in inputs:
                paths = [Path(value) for value in inputs[field]]
                self.media_paths[field] = paths
                self.media_bytes[field] = [path.read_bytes() for path in paths]
        return {"input_ids": [1, 9], "labels": [1, 9], "sample": inputs}

    def data_collator(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {"input_ids": torch.tensor([[1, 9]] * len(batch)), "labels": torch.ones(1)}


def test_template_bridge_preserves_media_order_and_removes_training_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    template = _FakeTemplate()
    calls: dict[str, Any] = {}

    def get_template(processor: object, **kwargs: Any) -> _FakeTemplate:
        calls.update({"processor": processor, **kwargs})
        return template

    monkeypatch.setattr(
        "mira.integrations.ms_swift.template_bridge.import_module",
        lambda _name: SimpleNamespace(get_template=get_template),
    )
    spec = SwiftTemplateSpec()
    bridge = SwiftTemplateBridge(object(), spec)
    (tmp_path / "a.jpg").write_bytes(b"image")
    (tmp_path / "v.mp4").write_bytes(b"video")
    item = resolve_item_media(
        MultimodalItem(
            item_id="mixed",
            messages=(Message(role="user", content="look <image> then <video>"),),
            images=(MediaRef(uri="a.jpg"),),
            videos=(MediaRef(uri="v.mp4"),),
        ),
        tmp_path,
    )
    encoded = bridge.encode_item(item)
    assert "labels" not in encoded
    assert template.media_bytes == {"images": [b"image"], "videos": [b"video"]}
    assert all(not path.exists() for paths in template.media_paths.values() for path in paths)
    assert encoded["sample"]["images"] != [str(tmp_path / "a.jpg")]
    assert encoded["sample"]["videos"] != [str(tmp_path / "v.mp4")]
    assert template.task_type == "causal_lm"
    assert template.mode == "transformers"
    assert calls["default_system"] == spec.instruction
    assert len(bridge.fingerprint) == 64
    assert bridge.fingerprint == template_fingerprint(spec)
    assert "labels" not in bridge.collate_items([encoded])
    with pytest.raises(ContractError, match="empty"):
        bridge.collate_items([])


def test_template_bridge_rejects_unresolved_or_changed_media(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    template = _FakeTemplate()
    monkeypatch.setattr(
        "mira.integrations.ms_swift.template_bridge.import_module",
        lambda _name: SimpleNamespace(get_template=lambda *_args, **_kwargs: template),
    )
    bridge = SwiftTemplateBridge(object(), SwiftTemplateSpec())
    unresolved = MultimodalItem(
        item_id="unresolved",
        messages=(Message(role="user", content="inspect <image>"),),
        images=(MediaRef(uri="image.jpg"),),
    )
    with pytest.raises(ContractError, match="resolved to an absolute path"):
        bridge.encode_item(unresolved)

    source = tmp_path / "image.jpg"
    source.write_bytes(b"accepted")
    resolved = resolve_item_media(unresolved, tmp_path)
    source.write_bytes(b"changed")
    with pytest.raises(ContractError, match="checksum changed"):
        bridge.encode_item(resolved)
    assert template.media_paths == {}


def test_template_bridge_rejects_symlink_replacement_after_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    template = _FakeTemplate()
    monkeypatch.setattr(
        "mira.integrations.ms_swift.template_bridge.import_module",
        lambda _name: SimpleNamespace(get_template=lambda *_args, **_kwargs: template),
    )
    bridge = SwiftTemplateBridge(object(), SwiftTemplateSpec())
    source = tmp_path / "image.jpg"
    source.write_bytes(b"accepted")
    item = resolve_item_media(
        MultimodalItem(
            item_id="image",
            messages=(Message(role="user", content="inspect <image>"),),
            images=(MediaRef(uri="image.jpg"),),
        ),
        tmp_path,
    )
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    source.unlink()
    source.symlink_to(outside)

    with pytest.raises(ContractError, match="unavailable or changed"):
        bridge.encode_item(item)
    assert template.media_paths == {}


@pytest.mark.parametrize("bad_template", [object(), SimpleNamespace(encode=lambda _: {})])
def test_template_bridge_rejects_incompatible_upstream_objects(
    monkeypatch: pytest.MonkeyPatch,
    bad_template: object,
) -> None:
    monkeypatch.setattr(
        "mira.integrations.ms_swift.template_bridge.import_module",
        lambda _name: SimpleNamespace(get_template=lambda *_args, **_kwargs: bad_template),
    )
    with pytest.raises(ContractError, match="incompatible"):
        SwiftTemplateBridge(object(), SwiftTemplateSpec())


def test_template_bridge_rejects_missing_payload_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    template = _FakeTemplate()
    monkeypatch.setattr(
        "mira.integrations.ms_swift.template_bridge.import_module",
        lambda _name: SimpleNamespace(get_template=lambda *_args, **_kwargs: template),
    )
    bridge = SwiftTemplateBridge(object(), SwiftTemplateSpec())
    item = MultimodalItem(item_id="text", messages=(Message(role="user", content="text"),))
    template.encode = lambda _value: {}  # type: ignore[method-assign]
    with pytest.raises(ContractError, match="no input_ids"):
        bridge.encode_item(item)
    template.data_collator = lambda _value: []  # type: ignore[method-assign]
    with pytest.raises(ContractError, match="no input_ids"):
        bridge.collate_items([{"input_ids": [1]}])


def test_backend_loads_with_explicit_arguments_and_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IMAGE_MAX_TOKEN_NUM", raising=False)
    calls: dict[str, Any] = {}
    model = nn.Linear(2, 2)
    processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=9))

    def get_model_processor(model_id: str, **kwargs: Any) -> tuple[nn.Module, object]:
        calls.update(
            {
                "image_max_tokens": backend.os.environ.get("IMAGE_MAX_TOKEN_NUM"),
                "model_id": model_id,
                **kwargs,
            }
        )
        return model, processor

    monkeypatch.setattr(backend, "require_compatible", lambda: None)
    monkeypatch.setattr(
        backend,
        "import_module",
        lambda _name: SimpleNamespace(get_model_processor=get_model_processor),
    )
    monkeypatch.setattr(
        backend,
        "SwiftTemplateBridge",
        lambda _processor, _spec: SimpleNamespace(fingerprint="f" * 64),
    )
    assets = load_qwen3_vl_embedding(
        _load_spec(
            torch_dtype="bfloat16",
            attention="flash_attention_2",
            image_max_tokens=512,
        ),
        SwiftTemplateSpec(),
    )
    assert assets.adapter.model is model
    assert calls["task_type"] == "embedding"
    assert calls["torch_dtype"] is torch.bfloat16
    assert calls["attn_impl"] == "flash_attn"
    assert calls["load_model"] is True
    assert calls["image_max_tokens"] == str(512)
    assert "IMAGE_MAX_TOKEN_NUM" not in backend.os.environ


def test_backend_restores_existing_image_token_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_limit = "previous"
    monkeypatch.setenv("IMAGE_MAX_TOKEN_NUM", previous_limit)
    model = nn.Linear(2, 2)
    processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=9))
    monkeypatch.setattr(backend, "require_compatible", lambda: None)
    monkeypatch.setattr(
        backend,
        "import_module",
        lambda _name: SimpleNamespace(
            get_model_processor=lambda *_args, **_kwargs: (model, processor)
        ),
    )
    monkeypatch.setattr(
        backend,
        "SwiftTemplateBridge",
        lambda _processor, _spec: SimpleNamespace(fingerprint="f" * 64),
    )

    load_qwen3_vl_embedding(
        _load_spec(image_max_tokens=256),
        SwiftTemplateSpec(),
    )

    assert backend.os.environ["IMAGE_MAX_TOKEN_NUM"] == previous_limit


@pytest.mark.parametrize(
    ("first_caller", "second_caller"),
    [
        ("direct", "direct"),
        ("facade", "facade"),
        ("facade", "direct"),
    ],
)
def test_backend_serializes_process_global_image_token_environment(
    monkeypatch: pytest.MonkeyPatch,
    first_caller: str,
    second_caller: str,
) -> None:
    environment_name = "IMAGE_MAX_TOKEN_NUM"
    monkeypatch.setenv(environment_name, "baseline")
    process_lock = _ObservableRLock()
    monkeypatch.setattr(backend, "_IMAGE_LIMIT_ENVIRONMENT_LOCK", process_lock)
    monkeypatch.setattr(backend, "require_compatible", lambda: None)
    monkeypatch.setattr(
        backend,
        "SwiftTemplateBridge",
        lambda _processor, _spec: SimpleNamespace(fingerprint="f" * 64),
    )

    model_ids = ("Qwen/Qwen3-VL-Embedding-2B", "Qwen/Qwen3-VL-Embedding-8B")
    limits = {model_ids[0]: 111, model_ids[1]: 222}
    first_entered = Event()
    release_first = Event()
    second_entered = Event()
    observed: dict[str, tuple[str | None, str | None]] = {}
    failures: list[Exception] = []

    def get_model_processor(model_id: str, **_kwargs: object) -> tuple[nn.Module, object]:
        entry_value = backend.os.environ.get(environment_name)
        if model_id == model_ids[0]:
            first_entered.set()
            if not release_first.wait(5):
                raise RuntimeError("test did not release the first loader")
        else:
            second_entered.set()
        observed[model_id] = (entry_value, backend.os.environ.get(environment_name))
        processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=9))
        return nn.Linear(2, 2), processor

    monkeypatch.setattr(
        backend,
        "import_module",
        lambda _name: SimpleNamespace(get_model_processor=get_model_processor),
    )

    def load(model_id: str, caller: str) -> None:
        try:
            load_spec = _load_spec(model_id, image_max_tokens=limits[model_id])
            template_spec = SwiftTemplateSpec()
            if caller == "facade":
                SwiftBackendAdapter(load_spec, template_spec).build_model()
            else:
                load_qwen3_vl_embedding(load_spec, template_spec)
        except (AssertionError, ContractError, RuntimeError) as error:
            failures.append(error)

    first = Thread(target=load, args=(model_ids[0], first_caller))
    second = Thread(target=load, args=(model_ids[1], second_caller))
    first.start()
    assert first_entered.wait(2)
    second.start()
    second_attempted, second_entered_while_first_active = (
        process_lock.second_attempted.wait(2),
        second_entered.is_set(),
    )
    release_first.set()
    first.join(2)
    second.join(2)

    assert second_attempted
    assert not second_entered_while_first_active
    assert not first.is_alive()
    assert not second.is_alive()
    assert not failures
    assert observed == {
        model_ids[0]: ("111", "111"),
        model_ids[1]: ("222", "222"),
    }
    assert backend.os.environ[environment_name] == "baseline"


def test_backend_process_environment_lock_is_reentrant() -> None:
    process_lock = backend._IMAGE_LIMIT_ENVIRONMENT_LOCK
    first_acquired = process_lock.acquire(blocking=False)
    second_acquired = False
    try:
        assert first_acquired
        second_acquired = process_lock.acquire(blocking=False)
        assert second_acquired
    finally:
        if second_acquired:
            process_lock.release()
        if first_acquired:
            process_lock.release()


def test_backend_releases_environment_lock_and_restores_value_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_name = "IMAGE_MAX_TOKEN_NUM"
    monkeypatch.setenv(environment_name, "baseline")
    monkeypatch.setattr(backend, "require_compatible", lambda: None)
    monkeypatch.setattr(
        backend,
        "SwiftTemplateBridge",
        lambda _processor, _spec: SimpleNamespace(fingerprint="f" * 64),
    )
    attempts = 0

    def get_model_processor(*_args: object, **_kwargs: object) -> tuple[nn.Module, object]:
        nonlocal attempts
        attempts += 1
        assert backend.os.environ[environment_name] == str(attempts * 128)
        if attempts == 1:
            raise RuntimeError("upstream load failed")
        processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=9))
        return nn.Linear(2, 2), processor

    monkeypatch.setattr(
        backend,
        "import_module",
        lambda _name: SimpleNamespace(get_model_processor=get_model_processor),
    )

    with pytest.raises(RuntimeError, match="upstream load failed"):
        load_qwen3_vl_embedding(
            _load_spec(image_max_tokens=128),
            SwiftTemplateSpec(),
        )
    assert backend.os.environ[environment_name] == "baseline"

    load_qwen3_vl_embedding(
        _load_spec("Qwen/Qwen3-VL-Embedding-8B", image_max_tokens=256),
        SwiftTemplateSpec(),
    )
    assert attempts == 2
    assert backend.os.environ[environment_name] == "baseline"


def test_backend_helpers_and_invalid_upstream_results(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _torch_dtype("auto") is None
    assert _torch_dtype("float16") is torch.float16
    assert _attention_name("sdpa") == "sdpa"
    assert _attention_name("auto") is None
    with pytest.raises(ContractError, match="dtype"):
        _torch_dtype("int8")
    with pytest.raises(ContractError, match="attention"):
        _attention_name("magic")
    with pytest.raises(ContractError, match="eos_token_id"):
        _resolve_eos_token_id(SimpleNamespace(eos_token_id=-1))

    monkeypatch.setattr(backend, "require_compatible", lambda: None)
    monkeypatch.setattr(backend, "import_module", lambda _name: SimpleNamespace())
    with pytest.raises(ContractError, match="get_model_processor"):
        load_qwen3_vl_embedding(
            _load_spec(),
            SwiftTemplateSpec(),
        )
    monkeypatch.setattr(
        backend,
        "import_module",
        lambda _name: SimpleNamespace(get_model_processor=lambda *_args, **_kwargs: (None, None)),
    )
    with pytest.raises(ContractError, match="no torch model"):
        load_qwen3_vl_embedding(_load_spec(), SwiftTemplateSpec())
    monkeypatch.setattr(
        backend,
        "import_module",
        lambda _name: SimpleNamespace(
            get_model_processor=lambda *_args, **_kwargs: (nn.Linear(2, 2), None)
        ),
    )
    with pytest.raises(ContractError, match="no processor"):
        load_qwen3_vl_embedding(
            _load_spec(),
            SwiftTemplateSpec(),
        )


class _LoraModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(2, 2)


def test_peft_bridge_validates_topology_and_wraps_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _LoraModel()
    captured: dict[str, Any] = {}

    class FakeLoraConfig:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "mira.integrations.ms_swift.peft_bridge.import_module",
        lambda _name: SimpleNamespace(
            LoraConfig=FakeLoraConfig,
            get_peft_model=lambda value, _config: value,
        ),
    )
    assert apply_lora(model, LoraSpec(4, 8.0, 0.1, ("q_proj",))) is model
    assert captured["target_modules"] == ["q_proj"]
    assert captured["task_type"] == "FEATURE_EXTRACTION"
    assert apply_lora(model, LoraSpec(4, 8.0, 0.0, "all-linear")) is model


@pytest.mark.parametrize(
    "spec",
    [
        LoraSpec,
    ],
)
def test_lora_spec_type_is_constructible(spec: type[LoraSpec]) -> None:
    assert spec(1, 1.0, 0.0, "all-linear").rank == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0, "alpha": 1.0, "dropout": 0.0, "target_modules": "all-linear"},
        {"rank": 1, "alpha": 0.0, "dropout": 0.0, "target_modules": "all-linear"},
        {"rank": 1, "alpha": 1.0, "dropout": 1.0, "target_modules": "all-linear"},
        {"rank": 1, "alpha": 1.0, "dropout": 0.0, "target_modules": "q_proj"},
        {"rank": 1, "alpha": 1.0, "dropout": 0.0, "target_modules": ()},
    ],
)
def test_lora_spec_rejects_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationValidationError):
        LoraSpec(**kwargs)


def test_peft_bridge_rejects_missing_targets_or_invalid_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _LoraModel()
    with pytest.raises(ContractError, match="absent"):
        apply_lora(model, LoraSpec(1, 1.0, 0.0, ("k_proj",)))
    monkeypatch.setattr(
        "mira.integrations.ms_swift.peft_bridge.import_module",
        lambda _name: SimpleNamespace(),
    )
    with pytest.raises(ContractError, match="construction API"):
        apply_lora(model, LoraSpec(1, 1.0, 0.0, "all-linear"))


def test_trainer_bridge_forwards_mira_owned_loss_and_collator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Trainer:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def train(self) -> None:
            return None

    monkeypatch.setattr(
        "mira.integrations.ms_swift.trainer_bridge.import_module",
        lambda _name: SimpleNamespace(Trainer=Trainer),
    )

    def loss(*_args: object) -> torch.Tensor:
        return torch.tensor(0.0)

    def collator(batch: list[Any]) -> dict[str, Any]:
        return {"batch": batch}

    model = nn.Linear(1, 1)
    trainer = create_swift_trainer(
        SwiftTrainerSpec(model, object(), [1], collator, loss, eval_dataset=[2])
    )
    assert isinstance(trainer, Trainer)
    assert captured["model"] is model
    assert captured["compute_loss_func"] is loss
    assert captured["data_collator"] is collator


def test_trainer_bridge_rejects_missing_or_incompatible_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1)
    spec = SwiftTrainerSpec(model, object(), [], lambda _batch: {}, lambda *_args: None)
    monkeypatch.setattr(
        "mira.integrations.ms_swift.trainer_bridge.import_module",
        lambda _name: SimpleNamespace(),
    )
    with pytest.raises(ContractError, match="does not expose"):
        create_swift_trainer(spec)
    monkeypatch.setattr(
        "mira.integrations.ms_swift.trainer_bridge.import_module",
        lambda _name: SimpleNamespace(Trainer=lambda **_kwargs: object()),
    )
    with pytest.raises(ContractError, match="incompatible"):
        create_swift_trainer(spec)


class _FacadeAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(2, 2)
        self.export_calls: list[tuple[Path, ExportConfig]] = []

    def export(self, output_dir: Path, export_config: ExportConfig) -> ExportManifest:
        self.export_calls.append((output_dir, export_config))
        return ExportManifest(
            model_id="fake/model",
            pooling="last_eos",
            instruction="Represent the user's input.",
            normalization="l2_after_optional_mrl_truncation",
            embedding_dimension=2,
            template_fingerprint="f" * 64,
            files=("model.safetensors",),
        )


class _FakeCheckpointManager:
    def __init__(self) -> None:
        self.saved: list[tuple[Path, CheckpointSnapshot]] = []
        self.loaded: list[dict[str, Any]] = []

    def save(
        self,
        output_dir: Path,
        *,
        snapshot: CheckpointSnapshot,
    ) -> dict[str, Any]:
        self.saved.append((output_dir, snapshot))
        return {"saved": True}

    def load(self, checkpoint_dir: Path, **kwargs: Any) -> CheckpointLoadResult:
        self.loaded.append({"checkpoint_dir": checkpoint_dir, **kwargs})
        return CheckpointLoadResult(
            mode=kwargs["mode"],
            training_state=TrainingState(global_step=3),
            restored_components=("model",),
            manifest={"loaded": True},
        )


def _fake_facade_assets(adapter: _FacadeAdapter) -> SwiftBackendAssets:
    return SwiftBackendAssets(
        adapter=adapter,  # type: ignore[arg-type]
        processor=object(),
        template=SimpleNamespace(fingerprint="f" * 64),  # type: ignore[arg-type]
    )


def test_backend_facade_caches_one_asset_bundle_and_preserves_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FacadeAdapter()
    assets = _fake_facade_assets(adapter)
    load_spec = _load_spec()
    template_spec = SwiftTemplateSpec(instruction="Represent this.")
    calls: list[tuple[SwiftLoadSpec, SwiftTemplateSpec]] = []

    def load(
        received_load_spec: SwiftLoadSpec,
        received_template_spec: SwiftTemplateSpec,
    ) -> SwiftBackendAssets:
        calls.append((received_load_spec, received_template_spec))
        return assets

    monkeypatch.setattr(backend, "load_qwen3_vl_embedding", load)
    facade = SwiftBackendAdapter(load_spec, template_spec)

    assert facade.build_processor() is assets.processor
    assert facade.build_template() is assets.template
    assert facade.build_model() is adapter
    assert calls == [(load_spec, template_spec)]


def test_backend_facade_does_not_cache_a_failed_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FacadeAdapter()
    assets = _fake_facade_assets(adapter)
    attempts = 0

    def load(*_args: object) -> SwiftBackendAssets:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ContractError("upstream load failed")
        return assets

    monkeypatch.setattr(backend, "load_qwen3_vl_embedding", load)
    facade = SwiftBackendAdapter(
        _load_spec(),
        SwiftTemplateSpec(),
    )

    with pytest.raises(ContractError, match="upstream load failed"):
        facade.build_model()
    assert facade.build_model() is adapter
    assert attempts == 2


def test_backend_facade_observes_assets_filled_while_waiting_for_cache_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FacadeAdapter()
    assets = _fake_facade_assets(adapter)
    facade = SwiftBackendAdapter(
        _load_spec(),
        SwiftTemplateSpec(),
    )

    class FillCacheOnEnter:
        def __enter__(self) -> None:
            facade._assets = assets

        def __exit__(self, *_args: object) -> None:
            return None

    facade._assets_lock = FillCacheOnEnter()  # type: ignore[assignment]
    monkeypatch.setattr(
        backend,
        "load_qwen3_vl_embedding",
        lambda *_args: pytest.fail("already-filled cache must not load again"),
    )

    assert facade.build_model() is adapter


def test_backend_facade_forwards_trainer_and_peft_without_model_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FacadeAdapter()
    assets = _fake_facade_assets(adapter)
    monkeypatch.setattr(backend, "load_qwen3_vl_embedding", lambda *_args: assets)
    trainer_calls: list[SwiftTrainerSpec] = []
    trainer = SimpleNamespace(train=lambda: None)
    monkeypatch.setattr(
        backend,
        "create_swift_trainer",
        lambda spec: trainer_calls.append(spec) or trainer,
    )
    peft_calls: list[tuple[nn.Module, LoraSpec]] = []
    wrapped = nn.Sequential(adapter.model, nn.Identity())

    def wrap(model: nn.Module, spec: LoraSpec) -> nn.Module:
        peft_calls.append((model, spec))
        return wrapped

    monkeypatch.setattr(backend, "apply_lora", wrap)
    facade = SwiftBackendAdapter(
        _load_spec(),
        SwiftTemplateSpec(),
    )
    foreign_spec = SwiftTrainerSpec(
        nn.Linear(1, 1),
        object(),
        [],
        lambda _batch: {},
        lambda *_args: None,
    )
    with pytest.raises(ContractError, match="model owned"):
        facade.build_trainer(foreign_spec)
    assert not trainer_calls

    lora_spec = LoraSpec(2, 4.0, 0.0, "all-linear")
    assert facade.build_peft_model(lora_spec) is adapter
    assert peft_calls[0][0] is wrapped[0]
    assert peft_calls[0][1] is lora_spec
    assert adapter.model is wrapped
    with pytest.raises(ContractError, match="already been applied"):
        facade.build_peft_model(lora_spec)

    owned_spec = SwiftTrainerSpec(
        adapter.model,
        object(),
        [1],
        lambda batch: {"batch": batch},
        lambda *_args: torch.tensor(0.0),
        eval_dataset=[2],
    )
    assert facade.build_trainer(owned_spec) is trainer
    assert trainer_calls == [owned_spec]


def test_backend_facade_rejects_peft_after_trainer_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FacadeAdapter()
    assets = _fake_facade_assets(adapter)
    monkeypatch.setattr(backend, "load_qwen3_vl_embedding", lambda *_args: assets)
    monkeypatch.setattr(
        backend,
        "create_swift_trainer",
        lambda _spec: SimpleNamespace(train=lambda: None),
    )
    apply_calls: list[object] = []
    monkeypatch.setattr(
        backend,
        "apply_lora",
        lambda *_args: apply_calls.append(object()),
    )
    facade = SwiftBackendAdapter(
        _load_spec(),
        SwiftTemplateSpec(),
    )
    trainer_spec = SwiftTrainerSpec(
        adapter.model,
        object(),
        [],
        lambda _batch: {},
        lambda *_args: None,
    )

    facade.build_trainer(trainer_spec)
    with pytest.raises(ContractError, match="before building"):
        facade.build_peft_model(LoraSpec(2, 4.0, 0.0, "all-linear"))
    assert not apply_calls


def test_backend_facade_binds_checkpoint_resume_and_export_to_cached_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _FacadeAdapter()
    assets = _fake_facade_assets(adapter)
    monkeypatch.setattr(backend, "load_qwen3_vl_embedding", lambda *_args: assets)
    manager = _FakeCheckpointManager()
    facade = SwiftBackendAdapter(
        _load_spec(),
        SwiftTemplateSpec(),
        checkpoint_manager=manager,  # type: ignore[arg-type]
    )
    snapshot = CheckpointSnapshot(
        model=adapter,
        training_state=TrainingState(global_step=3),
        resolved_config={"schema_version": 1},
    )
    checkpoint_dir = tmp_path / "checkpoint"
    assert facade.save_checkpoint(checkpoint_dir, snapshot=snapshot) == {"saved": True}
    assert manager.saved == [(checkpoint_dir, snapshot)]

    foreign_snapshot = CheckpointSnapshot(
        model=nn.Linear(1, 1),
        training_state=TrainingState(global_step=0),
        resolved_config={},
    )
    with pytest.raises(ContractError, match="model owned"):
        facade.save_checkpoint(tmp_path / "foreign", snapshot=foreign_snapshot)
    assert manager.saved == [(checkpoint_dir, snapshot)]

    targets = CheckpointRestoreTargets()
    device = torch.device("cpu")
    loaded = facade.resume_checkpoint(
        checkpoint_dir,
        mode="resume_model_only",
        targets=targets,
        map_location=device,
        expected_manifest_sha256="a" * 64,
    )
    assert loaded.manifest == {"loaded": True}
    assert manager.loaded == [
        {
            "checkpoint_dir": checkpoint_dir,
            "model": adapter,
            "mode": "resume_model_only",
            "targets": targets,
            "map_location": device,
            "expected_manifest_sha256": "a" * 64,
        }
    ]

    export_config = ExportConfig(metadata={"config_checksum": "b" * 64})
    export_dir = tmp_path / "export"
    manifest = facade.export_model(export_dir, export_config=export_config)
    assert manifest.model_id == "fake/model"
    assert adapter.export_calls == [(export_dir, export_config)]
