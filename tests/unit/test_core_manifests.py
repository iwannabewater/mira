from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import pytest

from mira.core import (
    ChecksumMismatchError,
    ExportManifest,
    ManifestEnvelope,
    ManifestError,
    ManifestKind,
    VerificationStatus,
    file_sha256,
    sha256_digest,
)
from mira.core.serialization import strict_json_loads


def test_manifest_round_trip_and_tamper_detection(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = ManifestEnvelope(
        kind=ManifestKind.RUN,
        payload={"config_checksum": "a" * 64, "smoke": VerificationStatus.NOT_RUN.value},
        created_at="2026-07-14T12:00:00+00:00",
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)

    restored = ManifestEnvelope.read(path)
    assert restored == manifest
    assert restored.checksum == sha256_digest(restored.body())

    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["smoke"] = VerificationStatus.PASSED.value
    with pytest.raises(ChecksumMismatchError, match="mismatch"):
        ManifestEnvelope.from_dict(document)


def test_manifest_payload_is_deeply_immutable_and_serializes_plain_copies() -> None:
    source = {
        "nested": {"items": [{"value": 1}]},
        "numbers": [1],
    }
    manifest = ManifestEnvelope(
        kind=ManifestKind.RUN,
        payload=source,
        created_at="2026-07-14T12:00:00+00:00",
    )
    expected = manifest.to_dict()
    checksum = manifest.checksum

    source["nested"]["items"][0]["value"] = 2
    source["numbers"].append(2)
    assert manifest.to_dict() == expected

    assert isinstance(manifest.payload, Mapping)
    nested = manifest.payload["nested"]
    assert isinstance(nested, Mapping)
    with pytest.raises(TypeError):
        nested["replacement"] = True  # type: ignore[index]
    items = nested["items"]
    assert isinstance(items, tuple)
    with pytest.raises(AttributeError):
        items.append({"value": 2})  # type: ignore[attr-defined]
    first_item = items[0]
    assert isinstance(first_item, Mapping)
    with pytest.raises(TypeError):
        first_item["value"] = 2  # type: ignore[index]

    body = manifest.body()
    body_payload = body["payload"]
    assert isinstance(body_payload, dict)
    assert isinstance(body_payload["nested"], dict)
    assert isinstance(body_payload["nested"]["items"], list)
    body_payload["nested"]["items"][0]["value"] = 3
    assert manifest.to_dict() == expected
    assert manifest.checksum == checksum


def test_manifest_rejects_extra_fields_nonfinite_values_and_naive_time() -> None:
    manifest = ManifestEnvelope(
        kind=ManifestKind.DATASET,
        payload={"count": 1},
        created_at="2026-07-14T12:00:00+00:00",
    )
    document = manifest.to_dict()
    document["extra"] = True
    with pytest.raises(ManifestError, match="fields differ"):
        ManifestEnvelope.from_dict(document)
    with pytest.raises(ManifestError, match="non-finite"):
        ManifestEnvelope(kind=ManifestKind.RUN, payload={"loss": float("nan")})
    with pytest.raises(ManifestError, match="timezone"):
        ManifestEnvelope(
            kind=ManifestKind.RUN,
            payload={},
            created_at="2026-07-14T12:00:00",
        )


def test_export_manifest_carries_inference_semantics_and_checksum(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = ExportManifest(
        model_name_or_path="Qwen/Qwen3-VL-Embedding-8B",
        model_revision="2c4565515e0f265c6511776e7193b22c0968ddc7",
        pooling="native",
        normalized=True,
        embedding_dimension=4096,
        representation_type="global",
        visual_preprocessing={"image_max_tokens": 768, "image_mean": [0.5]},
        template="qwen3_vl_emb",
        tokenizer="tokenizer.json",
        processor="processor_config.json",
        artifact_checksums={"model.safetensors": "b" * 64},
        instruction="Represent the user's input.",
        model_checksum="a" * 64,
        licenses=("Apache-2.0",),
        created_at="2026-07-14T12:00:00+00:00",
    )
    path = tmp_path / "export-manifest.json"
    manifest.write(path)

    restored = ManifestEnvelope.read(path)
    assert restored.kind is ManifestKind.EXPORT
    assert restored.checksum == manifest.checksum
    assert restored.payload["embedding_dimension"] == 4096


def test_export_manifest_collection_mappings_are_deeply_immutable() -> None:
    visual_preprocessing = {"resize": {"shortest_edge": [224]}}
    artifact_checksums = {"model.safetensors": "b" * 64}
    manifest = ExportManifest(
        model_name_or_path="model",
        model_revision="revision",
        pooling="native",
        normalized=True,
        embedding_dimension=16,
        representation_type="global",
        visual_preprocessing=visual_preprocessing,
        template="template",
        tokenizer="tokenizer",
        processor="processor",
        artifact_checksums=artifact_checksums,
        licenses=("Apache-2.0",),
        created_at="2026-07-14T12:00:00+00:00",
    )
    expected_body = manifest.to_envelope().body()
    checksum = manifest.checksum

    visual_preprocessing["resize"]["shortest_edge"].append(448)
    artifact_checksums["model.safetensors"] = "c" * 64
    assert manifest.to_envelope().body() == expected_body

    resize = manifest.visual_preprocessing["resize"]
    assert isinstance(resize, Mapping)
    with pytest.raises(TypeError):
        resize["mode"] = "bicubic"  # type: ignore[index]
    shortest_edge = resize["shortest_edge"]
    assert isinstance(shortest_edge, tuple)
    with pytest.raises(AttributeError):
        shortest_edge.append(448)  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        manifest.artifact_checksums["model.safetensors"] = "d" * 64  # type: ignore[index]

    envelope_body = manifest.to_envelope().body()
    envelope_payload = envelope_body["payload"]
    assert isinstance(envelope_payload, dict)
    assert isinstance(envelope_payload["visual_preprocessing"], dict)
    assert isinstance(envelope_payload["visual_preprocessing"]["resize"], dict)
    assert isinstance(envelope_payload["visual_preprocessing"]["resize"]["shortest_edge"], list)
    assert manifest.checksum == checksum


def test_file_sha256_streams_content_and_checks_chunk_size(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "weights.bin"
    path.write_bytes(b"mira-weights")
    assert file_sha256(path, chunk_size=2) == hashlib.sha256(b"mira-weights").hexdigest()
    with pytest.raises(ManifestError, match="positive"):
        file_sha256(path, chunk_size=0)
    with pytest.raises(ManifestError, match="cannot checksum"):
        file_sha256(tmp_path / "missing.bin")


def test_manifest_envelope_rejects_schema_checksum_and_read_errors(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ManifestError, match="schema_version"):
        ManifestEnvelope(kind=ManifestKind.RUN, payload={}, schema_version=2)
    with pytest.raises(ManifestError, match="64-character"):
        ManifestEnvelope(kind=ManifestKind.RUN, payload={}, checksum="bad")

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(ManifestError, match="cannot read"):
        ManifestEnvelope.read(invalid_json)
    array_json = tmp_path / "array.json"
    array_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ManifestError, match="JSON object"):
        ManifestEnvelope.read(array_json)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"value":1,"value":2}', "duplicate JSON object key 'value'"),
        ('{"value":NaN}', "nonstandard JSON constant 'NaN'"),
        ('{"value":1e999}', "non-finite JSON number '1e999'"),
    ],
)
def test_strict_json_decoder_rejects_ambiguous_or_nonfinite_documents(
    content: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        strict_json_loads(content, context="test document")


def test_manifest_reader_rejects_duplicate_and_nonstandard_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "manifest.json"
    manifest = ManifestEnvelope(
        kind=ManifestKind.RUN,
        payload={"count": 1},
        created_at="2026-07-14T12:00:00+00:00",
    )
    manifest.write(path)
    original = path.read_text(encoding="utf-8")
    path.write_text(
        original.replace('"kind": "run"', '"kind": "run",\n  "kind": "dataset"', 1),
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="duplicate JSON object key 'kind'"):
        ManifestEnvelope.read(path)

    path.write_text(original.replace('"count": 1', '"count": NaN', 1), encoding="utf-8")
    with pytest.raises(ManifestError, match="nonstandard JSON constant 'NaN'"):
        ManifestEnvelope.read(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": True}, "schema_version"),
        ({"kind": "run"}, "ManifestKind"),
        ({"payload": []}, "payload must be a mapping"),
        ({"created_at": 1}, "timestamp must be a string"),
        ({"checksum": 1}, "checksum must be a string"),
    ],
)
def test_manifest_envelope_constructor_rejects_coerced_field_types(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {"kind": ManifestKind.RUN, "payload": {}}
    values.update(changes)
    with pytest.raises(ManifestError, match=message):
        ManifestEnvelope(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "must be an integer"),
        ("kind", 1, "kind must be a string"),
        ("created_at", 1, "created_at must be a string"),
        ("payload", [], "payload must be a mapping"),
        ("checksum", 1, "checksum must be a string"),
        ("kind", "unknown", "unknown manifest kind"),
    ],
)
def test_manifest_from_dict_rejects_malformed_field_types(
    field: str, value: object, message: str
) -> None:
    manifest = ManifestEnvelope(
        kind=ManifestKind.RUN, payload={}, created_at="2026-07-14T12:00:00+00:00"
    )
    document = manifest.to_dict()
    document[field] = value  # type: ignore[assignment]
    with pytest.raises(ManifestError, match=message):
        ManifestEnvelope.from_dict(document)


@pytest.mark.parametrize(
    "changes",
    [
        {"pooling": ""},
        {"pooling": 1},
        {"normalized": 1},
        {"embedding_dimension": True},
        {"embedding_dimension": 0},
        {"visual_preprocessing": []},
        {"schema_version": 2},
        {"schema_version": True},
        {"instruction": 1},
        {"model_checksum": "BAD"},
        {"adapter_checksum": "a" * 64},
        {"adapter_name_or_path": "adapter"},
        {"adapter_name_or_path": "   "},
        {"artifact_checksums": {}},
        {"artifact_checksums": {"../model.safetensors": "b" * 64}},
        {"artifact_checksums": {"model.safetensors": 1}},
        {"licenses": ()},
        {"licenses": ("",)},
        {"created_at": "not-a-time"},
    ],
)
def test_export_manifest_rejects_incomplete_or_corrupt_semantics(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "model_name_or_path": "model",
        "model_revision": "revision",
        "pooling": "native",
        "normalized": True,
        "embedding_dimension": 16,
        "representation_type": "global",
        "visual_preprocessing": {},
        "template": "template",
        "tokenizer": "tokenizer",
        "processor": "processor",
        "artifact_checksums": {"model.safetensors": "b" * 64},
        "licenses": ("Apache-2.0",),
        "created_at": "2026-07-14T12:00:00+00:00",
    }
    values.update(changes)
    with pytest.raises(ManifestError):
        ExportManifest(**values)  # type: ignore[arg-type]


def test_manifest_payload_rejects_non_json_values_and_keys() -> None:
    with pytest.raises(ManifestError, match="non-string mapping key"):
        ManifestEnvelope(kind=ManifestKind.RUN, payload={1: "bad"})  # type: ignore[dict-item]
    with pytest.raises(ManifestError, match="unsupported manifest value"):
        ManifestEnvelope(kind=ManifestKind.RUN, payload={"bad": object()})  # type: ignore[dict-item]
