from __future__ import annotations

import json

import pytest

from mira.core import (
    ChecksumMismatchError,
    ExportManifest,
    ManifestEnvelope,
    ManifestError,
    ManifestKind,
)
from mira.inference import (
    load_inference_manifest,
    validate_inference_manifest,
    write_inference_manifest,
)


def test_export_manifest_round_trip_preserves_inference_semantics(tmp_path) -> None:
    manifest = ExportManifest(
        model_name_or_path="model",
        model_revision="revision",
        adapter_name_or_path="adapter",
        adapter_checksum="b" * 64,
        pooling="last-token",
        normalized=True,
        embedding_dimension=256,
        representation_type="global",
        visual_preprocessing={"max_pixels": 1024},
        template="embedding-v1",
        tokenizer="tokenizer",
        processor="processor",
        artifact_checksums={"model.safetensors": "a" * 64},
        instruction="retrieve",
        licenses=("Apache-2.0",),
    )
    path = tmp_path / "export-manifest.json"

    write_inference_manifest(manifest, path)

    loaded = load_inference_manifest(path)
    assert loaded == manifest
    assert loaded.checksum == manifest.checksum


def test_export_manifest_detects_tampering(tmp_path) -> None:
    manifest = ExportManifest(
        model_name_or_path="model",
        model_revision="revision",
        pooling="mean",
        normalized=True,
        embedding_dimension=4,
        representation_type="global",
        visual_preprocessing={},
        template="template",
        tokenizer="tokenizer",
        processor="processor",
        artifact_checksums={"model.safetensors": "a" * 64},
        licenses=("Apache-2.0",),
    )
    path = tmp_path / "export-manifest.json"
    write_inference_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["payload"]["pooling"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ChecksumMismatchError):
        load_inference_manifest(path)


def test_inference_manifest_rejects_wrong_type_and_kind(tmp_path) -> None:
    with pytest.raises(ManifestError, match="ExportManifest"):
        validate_inference_manifest(object())  # type: ignore[arg-type]
    path = tmp_path / "run.json"
    ManifestEnvelope(kind=ManifestKind.RUN, payload={}).write(path)
    with pytest.raises(ManifestError, match="expected an export"):
        load_inference_manifest(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_name_or_path", 1, "must be a string"),
        ("instruction", 1, "string or null"),
        ("normalized", 1, "boolean"),
        ("embedding_dimension", True, "integer"),
        ("visual_preprocessing", [], "mapping"),
        ("licenses", [1], "string list"),
    ],
)
def test_inference_manifest_strict_payload_types(
    tmp_path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest = ExportManifest(
        model_name_or_path="model",
        model_revision="revision",
        pooling="mean",
        normalized=True,
        embedding_dimension=4,
        representation_type="global",
        visual_preprocessing={},
        template="template",
        tokenizer="tokenizer",
        processor="processor",
        artifact_checksums={"model.safetensors": "a" * 64},
        licenses=("Apache-2.0",),
    )
    payload = dict(manifest.to_envelope().payload)
    payload[field] = value  # type: ignore[assignment]
    path = tmp_path / f"bad-{field}.json"
    ManifestEnvelope(kind=ManifestKind.EXPORT, payload=payload).write(path)  # type: ignore[arg-type]

    with pytest.raises(ManifestError, match=message):
        load_inference_manifest(path)


def test_inference_manifest_rejects_extra_payload_field(tmp_path) -> None:
    manifest = ExportManifest(
        model_name_or_path="model",
        model_revision="revision",
        pooling="mean",
        normalized=True,
        embedding_dimension=4,
        representation_type="global",
        visual_preprocessing={},
        template="template",
        tokenizer="tokenizer",
        processor="processor",
        artifact_checksums={"model.safetensors": "a" * 64},
        licenses=("Apache-2.0",),
    )
    payload = dict(manifest.to_envelope().payload)
    payload["extra"] = "value"
    path = tmp_path / "extra.json"
    ManifestEnvelope(kind=ManifestKind.EXPORT, payload=payload).write(path)

    with pytest.raises(ManifestError, match="fields differ"):
        load_inference_manifest(path)
