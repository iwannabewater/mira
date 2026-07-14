"""Versioned, canonical, SHA-256-verified manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any, Final, TypeAlias, cast

from mira.core.exceptions import ChecksumMismatchError, ManifestError
from mira.core.serialization import strict_json_loads

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
_FrozenJSONValue: TypeAlias = (
    JSONPrimitive | tuple["_FrozenJSONValue", ...] | Mapping[str, "_FrozenJSONValue"]
)
MANIFEST_SCHEMA_VERSION: Final = 1
_SHA256_LENGTH: Final = 64


class ManifestKind(StrEnum):
    """Stable manifest categories emitted by MIRA workflows."""

    RUN = "run"
    DATASET = "dataset"
    CHECKPOINT = "checkpoint"
    MINING = "mining"
    EXPORT = "export"


class VerificationStatus(StrEnum):
    """Truthful status for checks that may be unavailable."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_RUN = "NOT RUN"


def canonical_json_bytes(value: Mapping[str, JSONValue]) -> bytes:
    """Serialize a JSON mapping deterministically for hashing.

    Raises:
        ManifestError: If the value includes non-JSON or non-finite content.
    """
    _validate_json_value(dict(value), path="$", require_mapping=True)
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ManifestError(f"manifest content is not canonical JSON: {error}") from error
    return text.encode("utf-8")


def sha256_digest(value: Mapping[str, JSONValue]) -> str:
    """Return the lowercase SHA-256 digest of canonical JSON content."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it fully into memory.

    Raises:
        ManifestError: If ``chunk_size`` is invalid or the file cannot be read.
    """
    if chunk_size <= 0:
        raise ManifestError("checksum chunk_size must be positive")
    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
    except OSError as error:
        raise ManifestError(f"cannot checksum {source}: {error}") from error
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ManifestEnvelope:
    """A versioned manifest payload protected by a canonical checksum.

    The checksum covers ``schema_version``, ``kind``, ``created_at``, and
    ``payload``; it never covers itself.

    Raises:
        ManifestError: If fields are malformed or the schema is unsupported.
        ChecksumMismatchError: If a supplied checksum does not match the body.
    """

    kind: ManifestKind
    payload: Mapping[str, JSONValue]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    schema_version: int = MANIFEST_SCHEMA_VERSION
    checksum: str = ""

    def __post_init__(self) -> None:
        """Validate and freeze the envelope, computing an omitted checksum."""
        if type(self.schema_version) is not int or self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestError(
                f"unsupported manifest schema_version {self.schema_version}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )
        if not isinstance(self.kind, ManifestKind):
            raise ManifestError("manifest kind must be a ManifestKind")
        if not isinstance(self.payload, Mapping):
            raise ManifestError("manifest payload must be a mapping")
        if not isinstance(self.checksum, str):
            raise ManifestError("manifest checksum must be a string")
        _validate_timestamp(self.created_at)
        frozen_payload = _freeze_json_mapping(self.payload)
        object.__setattr__(self, "payload", frozen_payload)
        actual = sha256_digest(self.body())
        if self.checksum:
            _validate_sha256(self.checksum, "manifest checksum")
            if self.checksum != actual:
                raise ChecksumMismatchError(
                    f"manifest checksum mismatch: expected {self.checksum}, computed {actual}"
                )
        else:
            object.__setattr__(self, "checksum", actual)

    def body(self) -> dict[str, JSONValue]:
        """Return the exact checksum-covered manifest body."""
        return {
            "created_at": self.created_at,
            "kind": self.kind.value,
            "payload": _copy_json_mapping(self.payload),
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the persisted envelope including ``checksum``."""
        document = self.body()
        document["checksum"] = self.checksum
        return document

    def write(self, path: str | Path) -> None:
        """Atomically write canonical UTF-8 JSON to ``path``.

        Raises:
            ManifestError: If the destination cannot be created or replaced.
        """
        destination = Path(path)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(
                self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(f"{serialized}\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        except OSError as error:
            if "temporary" in locals():
                temporary.unlink(missing_ok=True)
            raise ManifestError(f"cannot write manifest {destination}: {error}") from error

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> ManifestEnvelope:
        """Parse and verify a manifest mapping with no ignored fields.

        Raises:
            ManifestError: If fields are missing, extra, or have invalid types.
            ChecksumMismatchError: If content was modified after creation.
        """
        required = {"schema_version", "kind", "created_at", "payload", "checksum"}
        actual = set(document)
        if actual != required:
            missing = sorted(required - actual)
            extra = sorted(actual - required)
            raise ManifestError(f"manifest fields differ; missing={missing}, extra={extra}")
        schema_version = document["schema_version"]
        kind = document["kind"]
        created_at = document["created_at"]
        payload = document["payload"]
        checksum = document["checksum"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ManifestError("manifest schema_version must be an integer")
        if not isinstance(kind, str):
            raise ManifestError("manifest kind must be a string")
        if not isinstance(created_at, str):
            raise ManifestError("manifest created_at must be a string")
        if not isinstance(payload, Mapping):
            raise ManifestError("manifest payload must be a mapping")
        if not isinstance(checksum, str):
            raise ManifestError("manifest checksum must be a string")
        try:
            parsed_kind = ManifestKind(kind)
        except ValueError as error:
            raise ManifestError(f"unknown manifest kind {kind!r}") from error
        return cls(
            kind=parsed_kind,
            payload=_copy_json_mapping(cast("Mapping[str, Any]", payload)),
            created_at=created_at,
            schema_version=schema_version,
            checksum=checksum,
        )

    @classmethod
    def read(cls, path: str | Path) -> ManifestEnvelope:
        """Read, strictly decode, and checksum-verify a JSON manifest."""
        source = Path(path)
        try:
            decoded = strict_json_loads(
                source.read_text(encoding="utf-8"),
                context=f"manifest {source}",
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise ManifestError(f"cannot read manifest {source}: {error}") from error
        if not isinstance(decoded, dict):
            raise ManifestError(f"manifest {source} must contain a JSON object")
        return cls.from_dict(cast("dict[str, Any]", decoded))


@dataclass(frozen=True, slots=True)
class ExportManifest:
    """Inference semantics that must travel with exported model weights."""

    model_name_or_path: str
    model_revision: str
    pooling: str
    normalized: bool
    embedding_dimension: int
    representation_type: str
    visual_preprocessing: Mapping[str, JSONValue]
    template: str
    tokenizer: str
    processor: str
    artifact_checksums: Mapping[str, str]
    instruction: str | None = None
    model_checksum: str | None = None
    adapter_name_or_path: str | None = None
    adapter_checksum: str | None = None
    licenses: tuple[str, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject exports that omit or corrupt required inference semantics."""
        _validate_export_required_strings(self)
        if type(self.normalized) is not bool:
            raise ManifestError("export normalized must be boolean")
        if type(self.embedding_dimension) is not int or self.embedding_dimension <= 0:
            raise ManifestError("export embedding_dimension must be positive")
        if type(self.schema_version) is not int or self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestError(f"unsupported export schema_version {self.schema_version}")
        _validate_timestamp(self.created_at)
        _validate_export_optional_fields(self)
        _validate_export_collections(self)
        object.__setattr__(
            self, "visual_preprocessing", _freeze_json_mapping(self.visual_preprocessing)
        )
        object.__setattr__(
            self,
            "artifact_checksums",
            MappingProxyType(dict(self.artifact_checksums)),
        )
        object.__setattr__(self, "licenses", tuple(self.licenses))

    @property
    def checksum(self) -> str:
        """Return the checksum of the complete export envelope."""
        return self.to_envelope().checksum

    def to_envelope(self) -> ManifestEnvelope:
        """Return the versioned, checksummed generic envelope."""
        payload: dict[str, JSONValue] = {
            "adapter_checksum": self.adapter_checksum,
            "adapter_name_or_path": self.adapter_name_or_path,
            "artifact_checksums": dict(self.artifact_checksums),
            "embedding_dimension": self.embedding_dimension,
            "instruction": self.instruction,
            "licenses": list(self.licenses),
            "model_checksum": self.model_checksum,
            "model_name_or_path": self.model_name_or_path,
            "model_revision": self.model_revision,
            "normalized": self.normalized,
            "pooling": self.pooling,
            "processor": self.processor,
            "representation_type": self.representation_type,
            "template": self.template,
            "tokenizer": self.tokenizer,
            "visual_preprocessing": _copy_json_mapping(self.visual_preprocessing),
        }
        return ManifestEnvelope(
            kind=ManifestKind.EXPORT,
            payload=payload,
            created_at=self.created_at,
            schema_version=self.schema_version,
        )

    def write(self, path: str | Path) -> None:
        """Write a checksum-protected export manifest atomically."""
        self.to_envelope().write(path)


def _validate_export_required_strings(manifest: ExportManifest) -> None:
    required_strings: dict[str, Any] = {
        "model_name_or_path": manifest.model_name_or_path,
        "model_revision": manifest.model_revision,
        "pooling": manifest.pooling,
        "representation_type": manifest.representation_type,
        "template": manifest.template,
        "tokenizer": manifest.tokenizer,
        "processor": manifest.processor,
    }
    invalid = sorted(
        name
        for name, value in required_strings.items()
        if not isinstance(value, str) or not value.strip()
    )
    if invalid:
        raise ManifestError(
            f"export manifest fields must be non-empty strings: {', '.join(invalid)}"
        )


def _validate_export_optional_fields(manifest: ExportManifest) -> None:
    for name, value in (
        ("instruction", manifest.instruction),
        ("model_checksum", manifest.model_checksum),
        ("adapter_name_or_path", manifest.adapter_name_or_path),
        ("adapter_checksum", manifest.adapter_checksum),
    ):
        if value is not None and not isinstance(value, str):
            raise ManifestError(f"export {name} must be a string or null")
    if manifest.adapter_name_or_path is not None and not manifest.adapter_name_or_path.strip():
        raise ManifestError("adapter_name_or_path must be non-empty when supplied")
    for name, checksum in (
        ("model_checksum", manifest.model_checksum),
        ("adapter_checksum", manifest.adapter_checksum),
    ):
        if checksum is not None:
            _validate_sha256(checksum, name)
    if (manifest.adapter_checksum is None) != (manifest.adapter_name_or_path is None):
        raise ManifestError("adapter_name_or_path and adapter_checksum must be supplied together")


def _validate_export_collections(manifest: ExportManifest) -> None:
    if not isinstance(manifest.licenses, (tuple, list)) or any(
        not isinstance(license_name, str) or not license_name.strip()
        for license_name in manifest.licenses
    ):
        raise ManifestError("export licenses must be non-empty strings")
    if not manifest.licenses:
        raise ManifestError("export licenses must not be empty")
    if not isinstance(manifest.visual_preprocessing, Mapping):
        raise ManifestError("export visual_preprocessing must be a mapping")
    if not isinstance(manifest.artifact_checksums, Mapping) or not manifest.artifact_checksums:
        raise ManifestError("export artifact_checksums must not be empty")
    for path, checksum in manifest.artifact_checksums.items():
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ManifestError(f"invalid export artifact path {path!r}")
        if not isinstance(checksum, str):
            raise ManifestError(f"artifact checksum for {path} must be a string")
        _validate_sha256(checksum, f"artifact checksum for {path}")


def _validate_timestamp(value: str) -> None:
    if not isinstance(value, str):
        raise ManifestError("manifest timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ManifestError(f"invalid ISO-8601 manifest timestamp {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ManifestError("manifest timestamp must include a timezone")


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ManifestError(f"{name} must be a lowercase 64-character SHA-256 digest")


def _copy_json_mapping(value: Mapping[str, Any]) -> dict[str, JSONValue]:
    copied = _copy_json_value(dict(value), path="$")
    if not isinstance(copied, dict):
        raise ManifestError("manifest payload must be a mapping")
    return copied


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, JSONValue]:
    copied = _copy_json_mapping(value)
    frozen = _freeze_json_value(copied)
    if not isinstance(frozen, Mapping):  # pragma: no cover - copied is always a dict
        raise ManifestError("manifest payload must be a mapping")
    return cast("Mapping[str, JSONValue]", frozen)


def _freeze_json_value(value: JSONValue) -> _FrozenJSONValue:
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
    return value


def _copy_json_value(value: Any, *, path: str) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return cast("JSONPrimitive", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestError(f"non-finite number at {path}")
        return value
    if isinstance(value, list | tuple):
        return [_copy_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ManifestError(f"non-string mapping key at {path}")
            result[key] = _copy_json_value(item, path=f"{path}.{key}")
        return result
    raise ManifestError(f"unsupported manifest value at {path}: {type(value).__name__}")


def _validate_json_value(value: Any, *, path: str, require_mapping: bool = False) -> None:
    copied = _copy_json_value(value, path=path)
    if require_mapping and not isinstance(copied, dict):
        raise ManifestError(f"{path} must be a mapping")
