"""Shared, dependency-light CLI helpers."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from mira.core.exceptions import ContractError
from mira.core.serialization import canonicalize_json_value
from mira.inference.artifacts import load_encoded_artifact
from mira.mining import EncodedCollection

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mira.config import MiraConfig

_ENCODED_PROVENANCE_SCHEMA = "mira.encoded.v1"
_ENCODED_PROVENANCE_SCHEMA_V2 = "mira.encoded.v2"


def emit_json(value: Any) -> None:
    """Write deterministic JSON to standard output."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    print(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))


def file_sha256(path: Path) -> str:
    """Hash a file or a directory tree in stable relative-path order."""
    source = path.resolve()
    digest = hashlib.sha256()
    if source.is_file():
        _update_file_digest(digest, source)
        return digest.hexdigest()
    if not source.is_dir():
        raise ContractError(f"checksum source does not exist: {source}")
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise ContractError(f"checksum directory contains no files: {source}")
    digest.update(b"MIRA-DIRECTORY-SHA256-V1\0")
    for item in files:
        relative = item.relative_to(source).as_posix().encode()
        file_digest = _file_digest(item)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        digest.update(file_digest)
    return digest.hexdigest()


def load_encoded_collection(
    path: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_role: Literal["query", "document"] | None = None,
) -> EncodedCollection:
    """Load a v2 artifact directory or a legacy v1 safe tensor bundle."""
    if path.is_dir():
        return load_encoded_artifact(
            path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_role=expected_role,
        ).collection
    if expected_manifest_sha256 is not None or expected_role is not None:
        raise ContractError(
            "trusted manifest and role checks require a v2 encoded artifact directory; "
            "re-encode this legacy v1 collection"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ContractError(f"cannot read encoded collection: {path}") from error
    if not isinstance(payload, dict):
        raise ContractError("encoded collection root must be a mapping")
    item_ids = payload.get("item_ids")
    embeddings = payload.get("embeddings")
    metadata = payload.get("metadata")
    provenance = payload.get("provenance", {})
    if not isinstance(item_ids, (tuple, list)) or any(
        not isinstance(item, str) or not item for item in item_ids
    ):
        raise ContractError("encoded collection item_ids must be a non-empty string sequence")
    if not isinstance(embeddings, torch.Tensor):
        raise ContractError("encoded collection embeddings must be a tensor")
    if metadata is None:
        metadata = tuple({} for _ in item_ids)
    if not isinstance(metadata, (tuple, list)) or any(
        not isinstance(item, dict) for item in metadata
    ):
        raise ContractError("encoded collection metadata must be a mapping sequence")
    if not isinstance(provenance, dict) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in provenance.items()
    ):
        raise ContractError("encoded collection provenance must be a string mapping")
    return EncodedCollection(tuple(item_ids), embeddings, tuple(metadata), provenance)


def save_encoded_collection(collection: EncodedCollection, path: Path) -> None:
    """Persist a tensor bundle without arbitrary Python objects."""

    def writer(temporary: Path) -> None:
        torch.save(
            {
                "item_ids": list(collection.item_ids),
                "embeddings": collection.embeddings.detach().cpu(),
                "metadata": canonicalize_json_value(
                    collection.metadata,
                    context="encoded metadata",
                ),
                "provenance": dict(collection.provenance),
            },
            temporary,
        )

    try:
        write_new_artifacts(((path, writer),))
    except FileExistsError as error:
        raise ContractError(f"encoded collection output already exists: {path}") from error


def build_encoded_provenance(  # noqa: PLR0913 - complete semantic identity is explicit
    checkpoint: str,
    config: MiraConfig,
    *,
    config_checksum: str,
    embedding_dimension: int,
    template_checksum: str,
    schema: str = _ENCODED_PROVENANCE_SCHEMA,
    backend: str | None = None,
    template_profile: str | None = None,
) -> dict[str, str]:
    """Build the exact semantic and model identity carried by encoded tensors."""
    if schema not in {_ENCODED_PROVENANCE_SCHEMA, _ENCODED_PROVENANCE_SCHEMA_V2}:
        raise ContractError(f"unsupported encoded provenance schema {schema!r}")
    if schema == _ENCODED_PROVENANCE_SCHEMA_V2 and (type(backend) is not str or not backend):
        raise ContractError("v2 encoded provenance requires an explicit inference backend")
    if schema == _ENCODED_PROVENANCE_SCHEMA_V2 and template_profile not in {
        "official",
        "mira_v1",
    }:
        raise ContractError("v2 encoded provenance requires an explicit template profile")
    provenance = {
        "schema": schema,
        "config_checksum": config_checksum,
        "model_name_or_path": config.model.name_or_path,
        "model_revision": config.model.revision,
        "model_source_identity": model_source_identity(checkpoint, config),
        "template_checksum": template_checksum,
        "normalization_rule": "l2_normalized_inner_product",
        "embedding_dimension": str(embedding_dimension),
    }
    if backend is not None:
        provenance["backend"] = backend
    if template_profile is not None:
        provenance["template_profile"] = template_profile
    return provenance


def model_source_identity(checkpoint: str, config: MiraConfig) -> str:
    """Identify a local artifact by content or an audited remote by immutable revision."""
    path = Path(checkpoint).expanduser()
    if path.exists():
        return f"sha256:{file_sha256(path)}"
    if checkpoint != config.model.name_or_path:
        raise ContractError("remote model identity must equal config.model.name_or_path")
    return f"huggingface:{config.model.name_or_path}@{config.model.revision}"


def model_source_checksum(checkpoint: str, config: MiraConfig) -> str:
    """Return a file checksum or a deterministic digest of an immutable remote identity."""
    path = Path(checkpoint).expanduser()
    if path.exists():
        return file_sha256(path)
    identity = model_source_identity(checkpoint, config)
    return hashlib.sha256(identity.encode()).hexdigest()


def validate_encoded_provenance(
    collection: EncodedCollection,
    checkpoint: str,
    config: MiraConfig,
    *,
    config_checksum: str,
    template_checksum: str,
) -> None:
    """Reject encoded tensors produced by a different model or semantic contract."""
    observed = dict(collection.provenance)
    expected = build_encoded_provenance(
        checkpoint,
        config,
        config_checksum=config_checksum,
        embedding_dimension=int(collection.embeddings.shape[1]),
        template_checksum=template_checksum,
        schema=observed.get("schema", ""),
        backend=observed.get("backend"),
        template_profile=observed.get("template_profile"),
    )
    if observed != expected:
        raise ContractError(
            "encoded collection provenance differs from the requested evaluation/mining "
            f"contract: observed={observed}, expected={expected}"
        )
    if observed["normalization_rule"] == "l2_normalized_inner_product":
        norms = torch.linalg.vector_norm(collection.embeddings, dim=1)
        if not bool(torch.isfinite(norms).all().item()):
            raise ContractError(
                "encoded collection claims l2_normalized_inner_product but has a non-finite "
                "embedding L2 norm"
            )
        tolerance = max(1e-4, float(torch.finfo(collection.embeddings.dtype).eps))
        if not torch.allclose(
            norms,
            torch.ones_like(norms),
            rtol=tolerance,
            atol=tolerance,
        ):
            raise ContractError(
                "encoded collection claims l2_normalized_inner_product but embeddings are "
                "not L2-normalized"
            )


def require_matching_encoded_provenance(
    first: EncodedCollection,
    second: EncodedCollection,
) -> None:
    """Require two encoded axes to originate from the same encoding contract."""
    if dict(first.provenance) != dict(second.provenance):
        raise ContractError("encoded collections have different provenance")


def encoded_template_checksum(
    collection: EncodedCollection,
    config: MiraConfig,
) -> str:
    """Resolve the exact legacy or Transformers preprocessing fingerprint."""
    schema = collection.provenance.get("schema")
    if schema == _ENCODED_PROVENANCE_SCHEMA:
        from mira.integrations.ms_swift import SwiftTemplateSpec  # noqa: PLC0415
        from mira.integrations.ms_swift.template_bridge import (  # noqa: PLC0415
            template_fingerprint,
        )

        return template_fingerprint(SwiftTemplateSpec())
    if schema == _ENCODED_PROVENANCE_SCHEMA_V2:
        from mira.integrations.qwen3_vl_embedding import (  # noqa: PLC0415
            Qwen3VLEmbeddingLoadSpec,
            preprocessing_fingerprint,
        )

        if config.model.hidden_size is None:
            raise ContractError("Qwen3-VL-Embedding provenance requires model.hidden_size")
        profile = collection.provenance.get("template_profile")
        if profile not in {"official", "mira_v1"}:
            raise ContractError(f"unsupported encoded template profile {profile!r}")
        return preprocessing_fingerprint(
            Qwen3VLEmbeddingLoadSpec(
                model_id=config.model.name_or_path,
                revision=config.model.revision,
                embedding_dimension=config.model.hidden_size,
                attention=config.runtime.attention,
                image_max_tokens=config.data.media.image_max_tokens,
                instruction="Represent the user's input.",
                template_profile=cast(
                    "Literal['official', 'mira_v1']",
                    profile,
                ),
                max_length=32768 if profile == "mira_v1" else 8192,
            )
        )
    raise ContractError(f"unsupported encoded provenance schema {schema!r}")


def write_json_file(value: Any, path: Path) -> None:
    """Write a new JSON artifact and refuse to overwrite evidence."""

    def writer(temporary: Path) -> None:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")

    write_new_artifacts(((path, writer),))


def write_new_artifacts(
    artifacts: Sequence[tuple[Path, Callable[[Path], None]]],
) -> None:
    """Write one or more new files through temporary siblings with rollback."""
    if not artifacts:
        raise ValueError("artifacts must not be empty")
    destinations = tuple(path.resolve() for path, _ in artifacts)
    if len(set(destinations)) != len(destinations):
        raise ValueError("artifact destinations must be unique")
    if any(path.exists() for path in destinations):
        existing = next(path for path in destinations if path.exists())
        raise FileExistsError(existing)
    prepared: list[tuple[Path, Path]] = []
    temporaries: list[Path] = []
    committed: list[Path] = []
    try:
        for (destination, writer), resolved in zip(artifacts, destinations, strict=True):
            resolved.parent.mkdir(parents=True, exist_ok=True)
            temporary = resolved.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
            temporaries.append(temporary)
            writer(temporary)
            _require_artifact_file(temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            prepared.append((temporary, resolved))
        for temporary, destination in prepared:
            os.link(temporary, destination)
            committed.append(destination)
    except Exception:
        for destination in committed:
            destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)


def _require_artifact_file(path: Path) -> None:
    if not path.is_file():
        raise OSError(f"artifact writer produced no file: {path}")


def _update_file_digest(digest: Any, path: Path) -> None:
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)


def _file_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    _update_file_digest(digest, path)
    return digest.digest()
