"""Shared, dependency-light CLI helpers."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from mira.core.exceptions import ContractError
from mira.core.serialization import canonicalize_json_value
from mira.mining import EncodedCollection

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mira.config import MiraConfig

_ENCODED_PROVENANCE_SCHEMA = "mira.encoded.v1"


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


def load_encoded_collection(path: Path) -> EncodedCollection:
    """Load a safe tensor bundle with IDs, embeddings, and metadata."""
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


def build_encoded_provenance(
    checkpoint: str,
    config: MiraConfig,
    *,
    config_checksum: str,
    embedding_dimension: int,
    template_checksum: str,
) -> dict[str, str]:
    """Build the exact semantic and model identity carried by encoded tensors."""
    return {
        "schema": _ENCODED_PROVENANCE_SCHEMA,
        "config_checksum": config_checksum,
        "model_name_or_path": config.model.name_or_path,
        "model_revision": config.model.revision,
        "model_source_identity": model_source_identity(checkpoint, config),
        "template_checksum": template_checksum,
        "normalization_rule": "l2_normalized_inner_product",
        "embedding_dimension": str(embedding_dimension),
    }


def model_source_identity(checkpoint: str, config: MiraConfig) -> str:
    """Identify a local artifact by content or an audited remote by immutable revision."""
    path = Path(checkpoint).expanduser()
    if path.exists():
        return f"sha256:{file_sha256(path)}"
    if checkpoint != config.model.name_or_path:
        raise ContractError("remote model identity must equal config.model.name_or_path")
    return f"huggingface:{config.model.name_or_path}@{config.model.revision}"


def validate_encoded_provenance(
    collection: EncodedCollection,
    checkpoint: str,
    config: MiraConfig,
    *,
    config_checksum: str,
    template_checksum: str,
) -> None:
    """Reject encoded tensors produced by a different model or semantic contract."""
    expected = build_encoded_provenance(
        checkpoint,
        config,
        config_checksum=config_checksum,
        embedding_dimension=int(collection.embeddings.shape[1]),
        template_checksum=template_checksum,
    )
    observed = dict(collection.provenance)
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
