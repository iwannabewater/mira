"""Construction and trust-boundary tests for the public inference factory."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from mira.config import TuningMethod, resolve_config
from mira.core.exceptions import ConfigurationValidationError, ContractError
from mira.inference import factory

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mira.inference.encoder import EncoderBackend
    from mira.integrations.qwen3_vl_embedding import Qwen3VLEmbeddingLoadSpec

_ROOT = Path(__file__).parents[2]
_CONFIG = _ROOT / "configs/experiments/qwen3_vl_embedding_2b_single_gpu_lora.yaml"


def _assets(spec: Qwen3VLEmbeddingLoadSpec) -> Any:
    return SimpleNamespace(
        adapter=SimpleNamespace(model=object()),
        backend=cast("EncoderBackend", object()),
        preprocessor=SimpleNamespace(spec=spec),
    )


def test_public_factory_remote_spec_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Backend:
        pass

    backend = Backend()

    def load(spec: Qwen3VLEmbeddingLoadSpec, *, model_source: str | None = None) -> Any:
        captured.update(spec=spec, model_source=model_source)
        return SimpleNamespace(
            adapter=SimpleNamespace(model=object()),
            backend=backend,
            preprocessor=SimpleNamespace(spec=spec),
        )

    monkeypatch.setattr(factory, "load_qwen3_vl_embedding_transformers", load)
    resolved = resolve_config(_CONFIG)
    encoder = factory.load_encoder(resolved.config.model.name_or_path, _CONFIG)

    spec = captured["spec"]
    assert encoder.backend is backend
    assert spec.model_id == "Qwen/Qwen3-VL-Embedding-2B"
    assert spec.revision == "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
    assert spec.template_profile == "official"
    assert spec.max_length == 8192
    assert captured["model_source"] is None


def test_public_factory_reconstructs_native_checkpoint_before_loading_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = resolve_config(_CONFIG)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "state.pt").write_bytes(b"state")
    calls: list[str] = []
    captured: dict[str, Any] = {}

    class Manager:
        def verify(self, path: Path, *, expected_manifest_sha256: str | None) -> dict[str, object]:
            assert path == checkpoint
            assert expected_manifest_sha256 == "a" * 64
            calls.append("verify")
            return {
                "manifests": {
                    "config_checksum": resolved.checksum,
                    "model_name_or_path": resolved.config.model.name_or_path,
                    "model_revision": resolved.config.model.revision,
                }
            }

        def load(self, path: Path, **kwargs: Any) -> None:
            assert path == checkpoint
            assert kwargs["mode"] == "resume_model_only"
            assert kwargs["expected_manifest_sha256"] == "a" * 64
            calls.append("load")

    def load(spec: Qwen3VLEmbeddingLoadSpec, *, model_source: str | None = None) -> Any:
        captured.update(spec=spec, model_source=model_source)
        calls.append("construct")
        return _assets(spec)

    monkeypatch.setattr(factory, "CheckpointManager", Manager)
    monkeypatch.setattr(factory, "load_qwen3_vl_embedding_transformers", load)
    monkeypatch.setattr(factory, "_apply_configured_lora", lambda *_: calls.append("peft"))

    factory.load_encoder(
        str(checkpoint),
        _CONFIG,
        expected_checkpoint_manifest_sha256="a" * 64,
    )

    assert calls == ["verify", "construct", "peft", "load"]
    assert captured["spec"].template_profile == "mira_v1"
    assert captured["spec"].max_length == 32768
    assert captured["model_source"] is None


def test_public_factory_loads_only_a_verified_private_export_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = tmp_path / "exported"
    exported.mkdir()
    snapshot = tmp_path / "private-snapshot"
    snapshot.mkdir()
    captured: dict[str, Any] = {}

    @contextmanager
    def verified(
        path: Path,
        _config: Any,
        *,
        expected_manifest_checksum: str,
    ) -> Iterator[Path]:
        assert path == exported
        assert expected_manifest_checksum == "b" * 64
        yield snapshot

    def load(spec: Qwen3VLEmbeddingLoadSpec, *, model_source: str | None = None) -> Any:
        captured.update(spec=spec, model_source=model_source)
        return _assets(spec)

    monkeypatch.setattr(factory, "verified_export_snapshot", verified)
    monkeypatch.setattr(factory, "load_qwen3_vl_embedding_transformers", load)

    factory.load_encoder(
        str(exported),
        _CONFIG,
        expected_export_manifest_checksum="b" * 64,
    )

    assert captured["model_source"] == str(snapshot)
    assert captured["spec"].template_profile == "mira_v1"


@pytest.mark.parametrize(
    ("checkpoint", "kwargs", "message"),
    [
        (" Qwen/Qwen3-VL-Embedding-2B", {}, "canonical"),
        ("Qwen/Qwen3-VL-Embedding-2B", {"batch_size": True}, "positive integer"),
        (
            "Qwen/Qwen3-VL-Embedding-2B",
            {"backend": "qwen3-vl-transformers"},
            "unsupported inference backend",
        ),
        (
            "Qwen/Qwen3-VL-Embedding-2B",
            {"expected_export_manifest_checksum": "0" * 64},
            "cannot be supplied for a remote model",
        ),
        ("generic/model", {}, "must equal config.model.name_or_path"),
    ],
)
def test_public_factory_rejects_ambiguous_source_contracts(
    checkpoint: str,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises((ConfigurationValidationError, ContractError), match=message):
        factory.load_encoder(checkpoint, _CONFIG, **kwargs)


def test_public_factory_requires_local_export_checksum(tmp_path: Path) -> None:
    exported = tmp_path / "exported"
    exported.mkdir()
    with pytest.raises(ContractError, match="trusted manifest checksum"):
        factory.load_encoder(str(exported), _CONFIG)


def test_qlora_inference_never_degrades_to_plain_lora() -> None:
    config = resolve_config(_CONFIG).config
    tuning = config.model.tuning.model_copy(update={"method": TuningMethod.QLORA})
    model = config.model.model_copy(update={"tuning": tuning})
    qlora_config = config.model_copy(update={"model": model})
    assets = SimpleNamespace(adapter=SimpleNamespace(model=object()))

    with pytest.raises(ConfigurationValidationError, match="quantized"):
        factory._apply_configured_lora(assets, qlora_config)
