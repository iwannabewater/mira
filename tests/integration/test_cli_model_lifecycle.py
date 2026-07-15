"""Weight-free model lifecycle tests using a differentiable tiny Qwen boundary fake."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from torch import nn
from typer.testing import CliRunner

from mira.cli import _model as model_cli
from mira.cli import doctor as doctor_cli
from mira.cli import encode as encode_cli
from mira.cli import export as export_cli
from mira.cli import train as train_cli
from mira.cli._common import file_sha256, load_encoded_collection
from mira.cli.app import app
from mira.config import resolve_config
from mira.core import ExportManifest
from mira.core.exceptions import CheckpointError, ConfigurationValidationError, ContractError
from mira.inference import (
    InferenceRuntimeDescriptor,
    load_inference_manifest,
    write_inference_manifest,
)
from mira.inference import sources as inference_sources
from mira.integrations.ms_swift import SwiftTemplateSpec
from mira.integrations.ms_swift.template_bridge import template_fingerprint
from mira.models.adapters import Qwen3VLEmbeddingAdapter, Qwen3VLEmbeddingSemantics

_ROOT = Path(__file__).parents[2]
_CONFIG = _ROOT / "configs/experiments/qwen3_vl_embedding_2b_single_gpu_lora.yaml"
_DIMENSION = 2048
_TERMINAL_TOKEN_ID = 2


def _write_local_export(
    path: Path,
    *,
    pooling: str = "last_token",
    template: str = "qwen3_vl_embedding",
) -> tuple[Any, ExportManifest]:
    resolved = resolve_config(_CONFIG)
    path.mkdir()
    artifacts = {
        "config.json": b'{"hidden_size":2048}',
        "mira-export.json": b'{"format":"mira"}',
        "model.safetensors": b"safe-model",
        "processor.json": b'{"processor":"tiny"}',
    }
    for relative, content in artifacts.items():
        (path / relative).write_bytes(content)
    manifest = ExportManifest(
        model_name_or_path=resolved.config.model.name_or_path,
        model_revision=resolved.config.model.revision,
        pooling=pooling,
        normalized=True,
        embedding_dimension=resolved.config.model.hidden_size or _DIMENSION,
        representation_type="global",
        visual_preprocessing={
            "attention": resolved.config.runtime.attention,
            "image_max_tokens": resolved.config.data.media.image_max_tokens,
        },
        template=template,
        tokenizer="TinyTokenizer",
        processor="TinyProcessor",
        artifact_checksums={relative: file_sha256(path / relative) for relative in artifacts},
        instruction=SwiftTemplateSpec().instruction,
        model_checksum="0" * 64,
        licenses=resolved.config.model.licenses,
    )
    write_inference_manifest(manifest, path / "inference-manifest.json")
    return resolved.config, manifest


def test_v01_export_labels_load_only_as_an_exact_legacy_pair(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy-export"
    config, legacy = _write_local_export(
        legacy_path,
        pooling="last_eos",
        template="qwen3_vl_emb",
    )
    model_cli._validate_exported_model(
        legacy_path,
        config,
        expected_manifest_checksum=legacy.checksum,
    )

    mixed_path = tmp_path / "mixed-export"
    config, mixed = _write_local_export(
        mixed_path,
        pooling="last_token",
        template="qwen3_vl_emb",
    )
    with pytest.raises(ContractError, match="semantics differ"):
        model_cli._validate_exported_model(
            mixed_path,
            config,
            expected_manifest_checksum=mixed.checksum,
        )


def _rewrite_export_manifest(
    path: Path,
    manifest: ExportManifest,
    *,
    artifact_checksums: dict[str, str],
) -> ExportManifest:
    rewritten = replace(manifest, artifact_checksums=artifact_checksums)
    write_inference_manifest(rewritten, path / "inference-manifest.json")
    return rewritten


class _TinyQwenModel(nn.Module):
    """Small differentiable model with names matching configured component groups."""

    def __init__(self) -> None:
        super().__init__()
        self.merger = nn.Linear(2, _DIMENSION, bias=False)
        self.language_model = nn.Linear(2, _DIMENSION, bias=False)
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True

    def forward(self, input_ids: torch.Tensor, **_: Any) -> Any:
        token = input_ids[:, 0].float()
        features = torch.stack((token, token.square() / 100.0), dim=1)
        embedding = self.merger(features) + self.language_model(features.flip(1))
        hidden = embedding.unsqueeze(1).expand(-1, input_ids.shape[1], -1)
        return SimpleNamespace(hidden_states=(hidden,))

    def merge_and_unload(self, *, safe_merge: bool) -> _TinyQwenModel:
        if not safe_merge:
            raise AssertionError("test merge must be safe")
        return self

    def save_pretrained(self, destination: Path, *, safe_serialization: bool) -> None:
        if not safe_serialization:
            raise AssertionError("test export must use safe serialization")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model.safetensors").write_bytes(b"tiny-safe-model")
        (destination / "config.json").write_text("{}\n", encoding="utf-8")


class _TinyTemplate:
    fingerprint = template_fingerprint(SwiftTemplateSpec())

    def encode_item(self, item: Any) -> dict[str, Any]:
        token = sum(item.item_id.encode("utf-8")) % 89 + 3
        return {"input_ids": [token, _TERMINAL_TOKEN_ID]}

    def collate_items(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([item["input_ids"] for item in items]),
            "attention_mask": torch.ones((len(items), 2), dtype=torch.int64),
        }


class _TinyTokenizer:
    eos_token_id = 3

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|endoftext|>"  # noqa: S105 - tokenizer terminal token
        return _TERMINAL_TOKEN_ID


class _TinyProcessor:
    def __init__(self) -> None:
        self.tokenizer = _TinyTokenizer()

    def save_pretrained(self, destination: Path) -> None:
        (destination / "processor.json").write_text("{}\n", encoding="utf-8")


def _assets() -> Any:
    adapter = Qwen3VLEmbeddingAdapter(
        _TinyQwenModel(),
        semantics=Qwen3VLEmbeddingSemantics(
            model_id="Qwen/Qwen3-VL-Embedding-2B",
            terminal_token_id=_TERMINAL_TOKEN_ID,
            template_fingerprint=_TinyTemplate.fingerprint,
            embedding_dimension=_DIMENSION,
        ),
    )
    return SimpleNamespace(
        adapter=adapter,
        processor=_TinyProcessor(),
        template=_TinyTemplate(),
    )


def _item(item_id: str) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "messages": [{"role": "user", "content": f"text for {item_id}"}],
    }


def _record(index: int, *, task_id: str = "multimodal_retrieval") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sample_id": f"sample-{index}",
        "task_id": task_id,
        "query": _item(f"query-{index}"),
        "positives": [_item(f"positive-{index}")],
        "negatives": [_item(f"negative-{index}-{offset}") for offset in range(4)],
        "metadata": {
            "modality_pair": "text_to_text",
            "dataset_name": "retrieval-train",
            "canonical_product_id": f"product-{index}",
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
        encoding="utf-8",
    )


def _write_config(
    path: Path,
    *,
    train_path: Path,
    validation_path: Path,
    epochs: int = 2,
) -> None:
    document = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    document["data"]["train"][0]["path"] = str(train_path)
    document["data"]["validation"]["path"] = str(validation_path)
    document["runtime"]["epochs"] = epochs
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            '{"item_id":"first","item_id":"shadow",'
            '"messages":[{"role":"user","content":"text"}]}\n',
            "duplicate JSON object key 'item_id'",
        ),
        (
            '{"item_id":"unsafe","messages":[{"role":"user","content":"text"}],'
            '"metadata":{"score":NaN}}\n',
            "nonstandard JSON constant 'NaN'",
        ),
        (
            '{"item_id":"unsafe","messages":[{"role":"user","content":"text"}],'
            '"metadata":{"score":1e999}}\n',
            "non-finite JSON number '1e999'",
        ),
    ],
)
def test_encode_item_reader_rejects_ambiguous_or_nonfinite_json(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "items.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ContractError, match=message):
        encode_cli._read_items(path)


def _install_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_cli, "load_qwen3_vl_embedding", lambda *_: _assets())
    monkeypatch.setattr(train_cli, "apply_configured_lora", lambda *_: None)


def test_tiny_train_resume_export_reload_and_encode_lifecycle(  # noqa: PLR0915 - lifecycle proof
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_data = tmp_path / "train.jsonl"
    validation_data = tmp_path / "validation.jsonl"
    _write_jsonl(train_data, [_record(1), _record(2)])
    _write_jsonl(validation_data, [_record(3)])
    config = tmp_path / "config.yaml"
    _write_config(
        config,
        train_path=train_data,
        validation_path=validation_data,
    )
    _install_fake_backend(monkeypatch)
    runner = CliRunner()
    output_dir = tmp_path / "checkpoints"

    first = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
            "--max-steps",
            "2",
            "--run-steps",
            "1",
        ],
    )
    assert first.exit_code == 0, first.output
    checkpoint_one = output_dir / "checkpoint-step-1"
    assert (checkpoint_one / "manifest.json").is_file()
    first_manifest = json.loads((checkpoint_one / "manifest.json").read_text())
    assert first_manifest["manifests"]["dataset_licenses"] == {"retrieval-train": ["NOASSERTION"]}
    validation_provenance = first_manifest["manifests"]["validation_dataset"]
    assert validation_provenance["evaluation_status"] == "NOT RUN"
    assert len(validation_provenance["checksum"]) == 64

    original_validation = validation_data.read_text(encoding="utf-8")
    _write_jsonl(validation_data, [_record(30)])
    drifted = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
            "--max-steps",
            "2",
            "--resume-from",
            str(checkpoint_one),
        ],
    )
    assert isinstance(drifted.exception, ConfigurationValidationError)
    assert "validation dataset checksum mismatch" in str(drifted.exception)
    validation_data.write_text(original_validation, encoding="utf-8")

    second = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
            "--max-steps",
            "2",
            "--resume-from",
            str(checkpoint_one),
            "--expected-manifest-sha256",
            file_sha256(checkpoint_one / "manifest.json"),
        ],
    )
    assert second.exit_code == 0, second.output
    checkpoint_two = output_dir / "checkpoint-step-2"
    assert (checkpoint_two / "state.pt").is_file()

    checkpoint_two_manifest_sha256 = file_sha256(checkpoint_two / "manifest.json")

    def load_for_export(*_args: Any, **kwargs: Any) -> Any:
        assert kwargs["expected_checkpoint_manifest_sha256"] == checkpoint_two_manifest_sha256
        return _assets()

    monkeypatch.setattr(export_cli, "load_configured_qwen_assets", load_for_export)
    exported = tmp_path / "exported"
    export_result = runner.invoke(
        app,
        [
            "export",
            "--checkpoint",
            str(checkpoint_two),
            "--config",
            str(config),
            "--output-dir",
            str(exported),
            "--expected-checkpoint-manifest-sha256",
            checkpoint_two_manifest_sha256,
        ],
    )
    assert export_result.exit_code == 0, export_result.output
    inference_manifest = load_inference_manifest(exported / "inference-manifest.json")
    assert inference_manifest.model_revision == resolve_config(config).config.model.revision
    assert set(inference_manifest.artifact_checksums) == {
        "config.json",
        "mira-export.json",
        "model.safetensors",
        "processor.json",
    }

    resolved = resolve_config(config)
    model_cli._validate_exported_model(
        exported,
        resolved.config,
        expected_manifest_checksum=inference_manifest.checksum,
    )
    loaded_snapshot_paths: list[Path] = []

    def load_verified_export(*_args: Any, **kwargs: Any) -> Any:
        source_value = kwargs.get("model_source")
        if source_value is not None:
            source = Path(source_value)
            assert source != exported
            assert (source / "model.safetensors").read_bytes() == b"tiny-safe-model"
            assert {
                artifact.relative_to(source).as_posix()
                for artifact in source.rglob("*")
                if artifact.is_file()
            } == {
                "config.json",
                "inference-manifest.json",
                "mira-export.json",
                "model.safetensors",
                "processor.json",
            }
            loaded_snapshot_paths.append(source)
        return _assets()

    monkeypatch.setattr(model_cli, "load_qwen3_vl_embedding", load_verified_export)
    monkeypatch.setattr(model_cli, "apply_configured_lora", lambda *_: None)
    with pytest.raises(CheckpointError, match="trusted digest"):
        model_cli.load_configured_qwen_assets(
            str(checkpoint_two),
            resolved.config,
            config_checksum=resolved.checksum,
            expected_checkpoint_manifest_sha256="0" * 64,
        )
    native_assets = model_cli.load_configured_qwen_assets(
        str(checkpoint_two),
        resolved.config,
        config_checksum=resolved.checksum,
        expected_checkpoint_manifest_sha256=checkpoint_two_manifest_sha256,
    )
    assert isinstance(native_assets.adapter, Qwen3VLEmbeddingAdapter)
    with pytest.raises(ContractError, match="trusted checksum"):
        model_cli.load_configured_qwen_assets(
            str(exported),
            resolved.config,
            config_checksum=resolved.checksum,
            expected_export_manifest_checksum="0" * 64,
        )
    exported_assets = model_cli.load_configured_qwen_assets(
        str(exported),
        resolved.config,
        config_checksum=resolved.checksum,
        expected_export_manifest_checksum=inference_manifest.checksum,
    )
    assert isinstance(exported_assets.processor, _TinyProcessor)
    assert len(loaded_snapshot_paths) == 1
    assert not loaded_snapshot_paths[0].exists()
    with pytest.raises(ContractError, match="remote model source"):
        model_cli.load_configured_qwen_assets(
            resolved.config.model.name_or_path,
            resolved.config,
            config_checksum=resolved.checksum,
            expected_checkpoint_manifest_sha256=checkpoint_two_manifest_sha256,
        )

    encode_input = tmp_path / "encode.jsonl"
    _write_jsonl(encode_input, [_item("encoded-one"), _item("encoded-two")])

    def load_for_encode(*_args: Any, **kwargs: Any) -> Any:
        assert kwargs["expected_export_manifest_checksum"] == inference_manifest.checksum
        return fake_encoder

    class FakeEncoder:
        runtime = InferenceRuntimeDescriptor(
            backend="qwen3-vl-embedding-transformers",
            model_id="Qwen/Qwen3-VL-Embedding-2B",
            revision=resolved.config.model.revision,
            template_fingerprint=_TinyTemplate.fingerprint,
            template_profile="mira_v1",
            device="cpu",
            dtype="float32",
        )

        @staticmethod
        def _encode(items: list[Any], dimension: int | None) -> torch.Tensor:
            rows = []
            for item in items:
                token = sum(item.item_id.encode("utf-8")) % 89 + 3
                row = torch.arange(1, _DIMENSION + 1, dtype=torch.float32) * token
                rows.append(row)
            values = torch.stack(rows)
            if dimension is not None:
                values = values[:, :dimension]
            return torch.nn.functional.normalize(values, dim=1)

        def encode_query(
            self,
            items: list[Any],
            *,
            batch_size: int,
            dimension: int | None,
            normalize: bool,
        ) -> torch.Tensor:
            assert batch_size > 0
            assert normalize
            return self._encode(items, dimension)

        def encode_document(
            self,
            items: list[Any],
            *,
            batch_size: int,
            dimension: int | None,
            normalize: bool,
        ) -> torch.Tensor:
            assert batch_size > 0
            assert normalize
            return self._encode(items, dimension)

    fake_encoder = FakeEncoder()

    monkeypatch.setattr(encode_cli, "load_encoder", load_for_encode)
    encoded_path = tmp_path / "encoded.pt"
    encode_result = runner.invoke(
        app,
        [
            "encode",
            "--checkpoint",
            str(exported),
            "--config",
            str(config),
            "--input",
            str(encode_input),
            "--output",
            str(encoded_path),
            "--role",
            "document",
            "--dimension",
            "16",
            "--expected-export-manifest-checksum",
            inference_manifest.checksum,
        ],
    )
    assert encode_result.exit_code == 0, encode_result.output
    encoded = load_encoded_collection(encoded_path)
    assert encoded.embeddings.shape == (2, 16)
    assert encoded.provenance["model_revision"] == resolved.config.model.revision


def test_export_never_replaces_destination_created_during_serialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "state.pt").write_bytes(b"state")
    output = tmp_path / "exported"
    created_inode: list[int] = []
    assets = _assets()
    original_save = assets.adapter.model.save_pretrained

    def create_competing_destination(
        destination: Path,
        *,
        safe_serialization: bool,
    ) -> None:
        output.mkdir()
        created_inode.append(output.stat().st_ino)
        original_save(destination, safe_serialization=safe_serialization)

    assets.adapter.model.save_pretrained = create_competing_destination
    monkeypatch.setattr(export_cli, "load_configured_qwen_assets", lambda *_args, **_kwargs: assets)

    with pytest.raises(ContractError, match="output directory already exists"):
        export_cli.export_command(
            checkpoint=str(checkpoint),
            config=_CONFIG,
            output_dir=output,
        )

    assert output.is_dir()
    assert output.stat().st_ino == created_inode[0]
    assert not list(output.iterdir())
    assert not list(tmp_path.glob(".exported.tmp-*"))


def test_validate_data_and_doctor_success_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_data = tmp_path / "train.jsonl"
    validation_data = tmp_path / "validation.jsonl"
    _write_jsonl(train_data, [_record(10), _record(11)])
    _write_jsonl(validation_data, [_record(12)])
    config = tmp_path / "config.yaml"
    _write_config(
        config,
        train_path=train_data,
        validation_path=validation_data,
        epochs=1,
    )
    runner = CliRunner()

    validation = runner.invoke(
        app,
        ["validate-data", "--config", str(config), "--skip-media"],
    )
    assert validation.exit_code == 0, validation.output
    validation_report = json.loads(validation.output)
    assert validation_report["records"] == 3

    monkeypatch.setattr(doctor_cli, "check_compatibility", lambda: SimpleNamespace(checks=()))
    monkeypatch.setattr(doctor_cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(doctor_cli.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(doctor_cli.dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(doctor_cli.importlib.util, "find_spec", lambda _name: object())
    doctor = runner.invoke(
        app,
        [
            "doctor",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "doctor-output"),
            "--device",
            "cpu",
        ],
    )
    assert doctor.exit_code == 0, doctor.output
    doctor_report = json.loads(doctor.output)
    assert doctor_report["status"] == "pass"
    assert doctor_report["workflow"] == "training"

    monkeypatch.setattr(
        doctor_cli,
        "check_inference_compatibility",
        lambda: SimpleNamespace(checks=()),
    )
    inference_doctor = runner.invoke(
        app,
        [
            "doctor",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "inference-doctor-output"),
            "--workflow",
            "inference",
            "--device",
            "cpu",
        ],
    )
    assert inference_doctor.exit_code == 0, inference_doctor.output
    inference_report = json.loads(inference_doctor.output)
    assert inference_report["status"] == "pass"
    assert inference_report["workflow"] == "inference"


def test_training_source_attribution_rejects_mislabeled_records(tmp_path: Path) -> None:
    train_data = tmp_path / "train.jsonl"
    validation_data = tmp_path / "validation.jsonl"
    record = _record(20)
    record["metadata"]["dataset_name"] = "wrong-source"
    _write_jsonl(train_data, [record])
    _write_jsonl(validation_data, [_record(21)])
    config = tmp_path / "config.yaml"
    _write_config(
        config,
        train_path=train_data,
        validation_path=validation_data,
        epochs=1,
    )

    with pytest.raises(ConfigurationValidationError, match=r"metadata\.dataset_name"):
        train_cli._load_training_records(resolve_config(config).config)


def test_training_epoch_budget_counts_each_deterministic_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_data = tmp_path / "train.jsonl"
    validation_data = tmp_path / "validation.jsonl"
    pairs = (("0", "2"), ("1", "0"), ("2", "0"), ("2", "1"), ("0", "1"))
    records = []
    for index, (canonical, image_group) in enumerate(pairs):
        record = _record(index + 100)
        record["metadata"]["canonical_product_id"] = canonical
        record["metadata"]["image_hash_group"] = image_group
        records.append(record)
    _write_jsonl(train_data, records)
    _write_jsonl(validation_data, [_record(200)])
    config = tmp_path / "config.yaml"
    _write_config(config, train_path=train_data, validation_path=validation_data, epochs=2)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["project"]["seed"] = 11
    document["runtime"]["train_batch_size_per_device"] = 3
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    backend_called = False

    def fail_if_loaded(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal backend_called
        backend_called = True
        raise AssertionError("model backend must not load for an invalid step budget")

    monkeypatch.setattr(train_cli, "load_qwen3_vl_embedding", fail_if_loaded)
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-steps",
            "6",
        ],
    )

    assert isinstance(result.exception, ConfigurationValidationError)
    assert "epoch budget of 5 steps" in str(result.exception)
    assert not backend_called

    sampler = train_cli.DeterministicNoDuplicateBatchSampler(
        train_cli.read_jsonl(train_data),
        batch_size=3,
        seed=11,
        duplicate_keys=("canonical_product_id", "image_hash_group"),
        homogeneous_keys=("task_id", "dataset_name", "modality_pair"),
    )
    initial_state = sampler.state_dict()
    assert train_cli._configured_step_budget(sampler, 1.5) == 4
    assert sampler.state_dict() == initial_state


def test_fp16_training_fails_before_model_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_data = tmp_path / "train.jsonl"
    validation_data = tmp_path / "validation.jsonl"
    _write_jsonl(train_data, [_record(300)])
    _write_jsonl(validation_data, [_record(301)])
    config = tmp_path / "config.yaml"
    _write_config(config, train_path=train_data, validation_path=validation_data, epochs=1)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["runtime"]["precision"] = "fp16"
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    backend_called = False

    def fail_if_loaded(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal backend_called
        backend_called = True
        raise AssertionError("model backend must not load for unsupported FP16 training")

    monkeypatch.setattr(train_cli, "load_qwen3_vl_embedding", fail_if_loaded)
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-steps",
            "1",
        ],
    )

    assert isinstance(result.exception, ConfigurationValidationError)
    assert "FP16 loss scaling" in str(result.exception)
    assert not backend_called


def test_local_export_snapshot_rejects_final_file_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = tmp_path / "exported"
    config, manifest = _write_local_export(exported)
    source = exported / "model.safetensors"
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside-content")
    original_open = os.open
    swapped = False

    def racing_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "model.safetensors" and dir_fd is not None and not swapped:
            swapped = True
            source.unlink()
            source.symlink_to(outside)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(inference_sources.os, "open", racing_open)
    with pytest.raises(ContractError, match="symlinked"):
        model_cli._validate_exported_model(
            exported,
            config,
            expected_manifest_checksum=manifest.checksum,
        )
    assert swapped


def test_local_export_snapshot_rejects_symlinked_path_components(tmp_path: Path) -> None:
    exported = tmp_path / "exported"
    config, manifest = _write_local_export(exported)
    linked = tmp_path / "linked-export"
    linked.symlink_to(exported, target_is_directory=True)

    with pytest.raises(ContractError, match="symlinked"):
        model_cli._validate_exported_model(
            linked,
            config,
            expected_manifest_checksum=manifest.checksum,
        )


def test_local_export_requires_the_correct_trusted_checksum_scope(tmp_path: Path) -> None:
    exported = tmp_path / "exported"
    config, _manifest = _write_local_export(exported)

    with pytest.raises(ContractError, match="require --expected-export"):
        model_cli.load_configured_qwen_assets(
            str(exported),
            config,
            config_checksum="0" * 64,
        )
    with pytest.raises(ContractError, match="only valid for a native MIRA checkpoint"):
        model_cli.load_configured_qwen_assets(
            str(exported),
            config,
            config_checksum="0" * 64,
            expected_checkpoint_manifest_sha256="0" * 64,
        )


@pytest.mark.parametrize("kind", ["unlisted", "fifo"])
def test_local_export_snapshot_rejects_unmanifested_or_special_artifacts(
    tmp_path: Path,
    kind: str,
) -> None:
    exported = tmp_path / "exported"
    config, manifest = _write_local_export(exported)
    unexpected = exported / "unexpected.json"
    if kind == "fifo":
        os.mkfifo(unexpected)
        expected_message = "non-regular artifact"
    else:
        unexpected.write_text("{}\n", encoding="utf-8")
        expected_message = "file set differs"

    with pytest.raises(ContractError, match=expected_message):
        model_cli._validate_exported_model(
            exported,
            config,
            expected_manifest_checksum=manifest.checksum,
        )


def test_local_export_snapshot_rejects_artifact_checksum_drift(tmp_path: Path) -> None:
    exported = tmp_path / "exported"
    config, manifest = _write_local_export(exported)
    (exported / "model.safetensors").write_bytes(b"tampered-after-manifest")

    with pytest.raises(ContractError, match="artifact checksum mismatch"):
        model_cli._validate_exported_model(
            exported,
            config,
            expected_manifest_checksum=manifest.checksum,
        )


@pytest.mark.parametrize(
    ("relative", "content", "message"),
    [
        ("modeling.py", b"raise RuntimeError('unsafe')\n", "unsafe artifact"),
        (
            "nested/config.json",
            b'{"nested":[{"auto_map":{"AutoModel":"modeling.Custom"}}]}',
            "requests remote code",
        ),
        ("nested/tokenizer_config.json", b'{"tokenizer":"first","tokenizer":"second"}', "invalid"),
    ],
)
def test_local_export_snapshot_rejects_executable_or_ambiguous_metadata(
    tmp_path: Path,
    relative: str,
    content: bytes,
    message: str,
) -> None:
    exported = tmp_path / "exported"
    config, manifest = _write_local_export(exported)
    artifact = exported / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(content)
    checksums = dict(manifest.artifact_checksums)
    checksums[relative] = file_sha256(artifact)
    manifest = _rewrite_export_manifest(
        exported,
        manifest,
        artifact_checksums=checksums,
    )

    with pytest.raises(ContractError, match=message):
        model_cli._validate_exported_model(
            exported,
            config,
            expected_manifest_checksum=manifest.checksum,
        )


def test_local_export_snapshot_supports_manifested_nested_artifacts(tmp_path: Path) -> None:
    exported = tmp_path / "exported"
    config, manifest = _write_local_export(exported)
    source = exported / "processor.json"
    nested = exported / "processor" / "processor.json"
    nested.parent.mkdir()
    source.replace(nested)
    checksums = dict(manifest.artifact_checksums)
    del checksums["processor.json"]
    checksums["processor/processor.json"] = file_sha256(nested)
    manifest = _rewrite_export_manifest(
        exported,
        manifest,
        artifact_checksums=checksums,
    )

    model_cli._validate_exported_model(
        exported,
        config,
        expected_manifest_checksum=manifest.checksum,
    )


def test_local_export_snapshot_rejects_manifest_self_reference(tmp_path: Path) -> None:
    exported = tmp_path / "exported"
    config, manifest = _write_local_export(exported)
    checksums = dict(manifest.artifact_checksums)
    checksums["inference-manifest.json"] = "0" * 64
    manifest = _rewrite_export_manifest(
        exported,
        manifest,
        artifact_checksums=checksums,
    )

    with pytest.raises(ContractError, match="must not include their own manifest"):
        model_cli._validate_exported_model(
            exported,
            config,
            expected_manifest_checksum=manifest.checksum,
        )
