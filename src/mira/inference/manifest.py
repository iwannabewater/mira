"""Typed helpers for checksum-protected inference export manifests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from mira.core.exceptions import ManifestError
from mira.core.manifests import ExportManifest, ManifestEnvelope, ManifestKind

if TYPE_CHECKING:
    from pathlib import Path

REQUIRED_INFERENCE_SEMANTICS = frozenset(
    {
        "adapter_name_or_path",
        "artifact_checksums",
        "embedding_dimension",
        "instruction",
        "model_name_or_path",
        "model_revision",
        "normalized",
        "pooling",
        "processor",
        "representation_type",
        "template",
        "tokenizer",
        "visual_preprocessing",
    }
)
"""Export payload fields needed to reproduce inference behavior."""

_EXPORT_PAYLOAD_FIELDS = REQUIRED_INFERENCE_SEMANTICS | {
    "adapter_checksum",
    "licenses",
    "model_checksum",
}


def validate_inference_manifest(manifest: ExportManifest) -> ExportManifest:
    """Validate and return a typed export manifest.

    ``ExportManifest`` validates its fields when constructed. Building its
    envelope here also verifies that all nested preprocessing values are valid
    canonical JSON and that no inference semantics are omitted.
    """
    if not isinstance(manifest, ExportManifest):
        raise ManifestError("inference export must use mira.core.ExportManifest")
    envelope = manifest.to_envelope()
    missing = REQUIRED_INFERENCE_SEMANTICS - envelope.payload.keys()
    if missing:
        raise ManifestError(f"export manifest is missing inference semantics: {sorted(missing)}")
    return manifest


def write_inference_manifest(manifest: ExportManifest, path: Path) -> None:
    """Atomically persist a validated, checksum-protected export manifest."""
    validate_inference_manifest(manifest).write(path)


def load_inference_manifest(path: Path) -> ExportManifest:
    """Read, checksum-verify, and strictly parse an export manifest."""
    envelope = ManifestEnvelope.read(path)
    if envelope.kind is not ManifestKind.EXPORT:
        raise ManifestError(
            f"expected an export manifest, found manifest kind {envelope.kind.value!r}"
        )
    payload = envelope.payload
    actual_fields = set(payload)
    if actual_fields != _EXPORT_PAYLOAD_FIELDS:
        raise ManifestError(
            "export payload fields differ; "
            f"missing={sorted(_EXPORT_PAYLOAD_FIELDS - actual_fields)}, "
            f"extra={sorted(actual_fields - _EXPORT_PAYLOAD_FIELDS)}"
        )
    return ExportManifest(
        model_name_or_path=_required_string(payload, "model_name_or_path"),
        model_revision=_required_string(payload, "model_revision"),
        pooling=_required_string(payload, "pooling"),
        normalized=_required_bool(payload, "normalized"),
        embedding_dimension=_required_int(payload, "embedding_dimension"),
        representation_type=_required_string(payload, "representation_type"),
        visual_preprocessing=_required_mapping(payload, "visual_preprocessing"),
        template=_required_string(payload, "template"),
        tokenizer=_required_string(payload, "tokenizer"),
        processor=_required_string(payload, "processor"),
        artifact_checksums=_required_checksum_mapping(payload, "artifact_checksums"),
        instruction=_optional_string(payload, "instruction"),
        model_checksum=_optional_string(payload, "model_checksum"),
        adapter_name_or_path=_optional_string(payload, "adapter_name_or_path"),
        adapter_checksum=_optional_string(payload, "adapter_checksum"),
        licenses=_required_strings(payload, "licenses"),
        created_at=envelope.created_at,
        schema_version=envelope.schema_version,
    )


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ManifestError(f"export payload field {key!r} must be a string")
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload[key]
    if value is not None and not isinstance(value, str):
        raise ManifestError(f"export payload field {key!r} must be a string or null")
    return value


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise ManifestError(f"export payload field {key!r} must be boolean")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"export payload field {key!r} must be an integer")
    return int(value)


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload[key]
    if not isinstance(value, Mapping):
        raise ManifestError(f"export payload field {key!r} must be a mapping")
    return cast("Mapping[str, Any]", value)


def _required_checksum_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, str]:
    value = payload[key]
    if not isinstance(value, Mapping) or any(
        not isinstance(path, str) or not isinstance(checksum, str)
        for path, checksum in value.items()
    ):
        raise ManifestError(f"export payload field {key!r} must be a string mapping")
    return cast("Mapping[str, str]", value)


def _required_strings(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload[key]
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ManifestError(f"export payload field {key!r} must be a string list")
    return tuple(cast("list[str] | tuple[str, ...]", value))
