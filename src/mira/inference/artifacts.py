"""Streaming, checksummed ``mira.encoded.v2`` embedding artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from mira.core.encoded import EncodedCollection
from mira.core.exceptions import ContractError
from mira.core.filesystem import atomic_rename_directory_no_replace, open_bound_directory
from mira.core.serialization import canonicalize_json_value, strict_json_loads

if TYPE_CHECKING:
    from types import TracebackType

ENCODED_ARTIFACT_SCHEMA = "mira.encoded.v2"
DEFAULT_MAX_MATERIALIZED_BYTES = 8 * 1024 * 1024 * 1024
_MANIFEST_NAME = "manifest.json"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_ARTIFACT_FILES = 8193
_MAX_ARTIFACT_SHARDS = 4096
_EMBEDDING_KEY = "embeddings"
_EMBEDDING_RANK = 2
_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True, slots=True)
class EncodedArtifactSummary:
    """Published artifact identity returned by the streaming writer."""

    path: Path
    manifest_sha256: str
    item_count: int
    embedding_dimension: int
    dtype: str
    shards: int


@dataclass(frozen=True, slots=True)
class EncodedArtifact:
    """Materialized vectors plus the complete verified artifact semantics."""

    collection: EncodedCollection
    path: Path
    role: Literal["query", "document"]
    normalized: bool
    manifest_sha256: str

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Return item IDs in embedding row order."""
        return self.collection.item_ids

    @property
    def embeddings(self) -> Tensor:
        """Return the materialized embedding matrix."""
        return self.collection.embeddings

    @property
    def metadata(self) -> tuple[Mapping[str, Any], ...]:
        """Return immutable row-aligned metadata."""
        return self.collection.metadata

    @property
    def provenance(self) -> Mapping[str, str]:
        """Return immutable model and preprocessing provenance."""
        return self.collection.provenance

    def checksum(self) -> str:
        """Return the materialized collection checksum used by retrieval workflows."""
        return self.collection.checksum()


@dataclass(frozen=True, slots=True)
class EncodedArtifactShard:
    """One independently verified shard from a streaming artifact reader."""

    index: int
    collection: EncodedCollection
    role: Literal["query", "document"]
    normalized: bool
    manifest_sha256: str

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Return item IDs for this shard."""
        return self.collection.item_ids

    @property
    def embeddings(self) -> Tensor:
        """Return this shard's verified embedding matrix."""
        return self.collection.embeddings

    @property
    def metadata(self) -> tuple[Mapping[str, Any], ...]:
        """Return this shard's row-aligned metadata."""
        return self.collection.metadata

    @property
    def provenance(self) -> Mapping[str, str]:
        """Return the artifact-level provenance copied onto this shard."""
        return self.collection.provenance


@dataclass(frozen=True, slots=True)
class _ShardRecord:
    tensor: str
    records: str
    rows: int
    tensor_sha256: str
    records_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "records_sha256": self.records_sha256,
            "rows": self.rows,
            "tensor": self.tensor,
            "tensor_sha256": self.tensor_sha256,
        }


class EncodedArtifactReader:
    """Descriptor-bound context manager that verifies and streams v2 shards."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_manifest_sha256: str | None = None,
        expected_role: Literal["query", "document"] | None = None,
    ) -> None:
        """Store the unopened path and optional trusted semantic expectations."""
        self.path = Path(path)
        self.expected_manifest_sha256 = expected_manifest_sha256
        self.expected_role = expected_role
        self._root_fd: int | None = None
        self._temporary: TemporaryDirectory[str] | None = None
        self._snapshot: Path | None = None
        self._manifest: dict[str, Any] | None = None
        self._manifest_sha256: str | None = None
        self._expected_names: set[str] | None = None
        self._iterating = False

    def __enter__(self) -> Self:
        """Bind the directory and verify its manifest and exact file set."""
        if self._root_fd is not None:
            raise ContractError("encoded artifact reader is already open")
        try:
            self._root_fd, _absolute = open_bound_directory(self.path)
        except OSError as error:
            raise ContractError("encoded artifact is missing, symlinked, or unreadable") from error
        self._temporary = TemporaryDirectory(prefix="mira-encoded-v2-")
        self._snapshot = Path(self._temporary.name)
        try:
            names = _list_regular_files(self._root_fd)
            if _MANIFEST_NAME not in names:
                raise ContractError(  # noqa: TRY301 - cleanup belongs to this boundary
                    "encoded artifact has no manifest.json"
                )
            manifest_path = self._snapshot / _MANIFEST_NAME
            manifest_digest = _snapshot_file(
                self._root_fd,
                _MANIFEST_NAME,
                manifest_path,
                max_bytes=_MAX_MANIFEST_BYTES,
            )
            if (
                self.expected_manifest_sha256 is not None
                and manifest_digest != self.expected_manifest_sha256
            ):
                raise ContractError(  # noqa: TRY301 - cleanup belongs to this boundary
                    "encoded artifact manifest checksum differs from the trusted checksum"
                )
            manifest = _parse_manifest(manifest_path)
            if self.expected_role is not None and manifest["role"] != self.expected_role:
                raise ContractError(  # noqa: TRY301 - cleanup belongs to this boundary
                    "encoded artifact role differs from the requested role: "
                    f"{manifest['role']!r} != {self.expected_role!r}"
                )
            expected_names = {_MANIFEST_NAME}
            for shard in manifest["shards"]:
                expected_names.update((shard.tensor, shard.records))
            if names != expected_names:
                raise ContractError(  # noqa: TRY301 - cleanup belongs to this boundary
                    "encoded artifact file set differs from its manifest: "
                    f"actual={sorted(names)}, expected={sorted(expected_names)}"
                )
            self._manifest = manifest
            self._manifest_sha256 = manifest_digest
            self._expected_names = expected_names
            return self  # noqa: TRY300 - cleanup belongs to the paired exception path
        except Exception:
            self.close()
            raise

    def __exit__(
        self,
        _error_type: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close the reader on normal exit or any consumer failure."""
        self.close()

    @property
    def role(self) -> Literal["query", "document"]:
        """Return the retrieval-axis role declared by the manifest."""
        manifest = self._require_open()
        return cast("Literal['query', 'document']", manifest["role"])

    @property
    def normalized(self) -> bool:
        """Return whether every vector is declared and verified L2-normalized."""
        return cast("bool", self._require_open()["normalized"])

    @property
    def manifest_sha256(self) -> str:
        """Return the SHA-256 of the exact verified manifest bytes."""
        self._require_open()
        if self._manifest_sha256 is None:
            raise ContractError("encoded artifact reader has no manifest digest")
        return self._manifest_sha256

    @property
    def item_count(self) -> int:
        """Return the manifest's total row count."""
        return cast("int", self._require_open()["item_count"])

    @property
    def embedding_dimension(self) -> int:
        """Return the manifest's vector width."""
        return cast("int", self._require_open()["embedding_dimension"])

    @property
    def dtype(self) -> str:
        """Return the manifest's torch dtype name."""
        return cast("str", self._require_open()["dtype"])

    @property
    def provenance(self) -> Mapping[str, str]:
        """Return model and preprocessing provenance from the manifest."""
        return cast("Mapping[str, str]", self._require_open()["provenance"])

    def iter_shards(self) -> Iterator[EncodedArtifactShard]:
        """Yield one verified CPU shard at a time without materializing prior shards."""
        manifest = self._require_open()
        if self._iterating:
            raise ContractError("encoded artifact reader already has an active shard iterator")
        self._iterating = True
        seen_ids: set[str] = set()
        total_rows = 0
        try:
            for index, shard in enumerate(manifest["shards"]):
                collection = self._load_shard_collection(shard, manifest, seen_ids)
                seen_ids.update(collection.item_ids)
                total_rows += shard.rows
                yield EncodedArtifactShard(
                    index=index,
                    collection=collection,
                    role=self.role,
                    normalized=self.normalized,
                    manifest_sha256=self.manifest_sha256,
                )
            if total_rows != manifest["item_count"]:
                raise ContractError("encoded artifact total item count differs from its manifest")
            if self._expected_names != _list_regular_files(self._root_descriptor()):
                raise ContractError("encoded artifact file set changed while it was read")
        finally:
            self._iterating = False

    def close(self) -> None:
        """Release the bound directory and private snapshot directory."""
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self._snapshot = None
        self._manifest = None
        self._manifest_sha256 = None
        self._expected_names = None
        self._iterating = False

    def _snapshot_paths(self, shard: _ShardRecord) -> tuple[Path, Path]:
        if self._snapshot is None:
            raise ContractError("encoded artifact reader is not open")
        return self._snapshot / shard.tensor, self._snapshot / shard.records

    def _load_shard_collection(
        self,
        shard: _ShardRecord,
        manifest: dict[str, Any],
        seen_ids: set[str],
    ) -> EncodedCollection:
        """Snapshot and fully validate one shard against artifact-wide semantics."""
        tensor_path, records_path = self._snapshot_paths(shard)
        try:
            tensor_digest = _snapshot_file(self._root_descriptor(), shard.tensor, tensor_path)
            records_digest = _snapshot_file(self._root_descriptor(), shard.records, records_path)
            if tensor_digest != shard.tensor_sha256:
                raise ContractError(f"encoded artifact tensor checksum mismatch: {shard.tensor}")
            if records_digest != shard.records_sha256:
                raise ContractError(f"encoded artifact records checksum mismatch: {shard.records}")
            tensor = _load_safetensor(tensor_path)
            shard_ids, shard_metadata = _load_records(records_path)
        finally:
            tensor_path.unlink(missing_ok=True)
            records_path.unlink(missing_ok=True)
        if (
            tensor.ndim != _EMBEDDING_RANK
            or tensor.shape[0] != shard.rows
            or len(shard_ids) != shard.rows
        ):
            raise ContractError(
                f"encoded artifact shard shape or row count mismatch: {shard.tensor}"
            )
        if tensor.shape[1] != manifest["embedding_dimension"]:
            raise ContractError(
                f"encoded artifact shard embedding dimension differs from manifest: {shard.tensor}"
            )
        if str(tensor.dtype).removeprefix("torch.") != manifest["dtype"]:
            raise ContractError(
                f"encoded artifact shard dtype differs from manifest: {shard.tensor}"
            )
        duplicate_ids = seen_ids.intersection(shard_ids)
        if len(set(shard_ids)) != len(shard_ids) or duplicate_ids:
            raise ContractError("encoded artifact item IDs must be globally unique")
        collection = EncodedCollection(
            tuple(shard_ids),
            tensor,
            tuple(shard_metadata),
            manifest["provenance"],
        )
        if manifest["normalized"]:
            _require_normalized(collection.embeddings)
        return collection

    def _root_descriptor(self) -> int:
        if self._root_fd is None:
            raise ContractError("encoded artifact reader is not open")
        return self._root_fd

    def _require_open(self) -> dict[str, Any]:
        if self._root_fd is None or self._manifest is None:
            raise ContractError("encoded artifact reader is not open")
        return self._manifest


class EncodedArtifactWriter:
    """Append embedding batches and atomically publish one immutable artifact."""

    def __init__(
        self,
        destination: str | Path,
        *,
        provenance: Mapping[str, str],
        role: Literal["query", "document"],
        normalized: bool,
        shard_size: int = 4096,
    ) -> None:
        """Prepare an unpublished sibling directory with explicit semantics."""
        self.destination = Path(destination).resolve()
        if self.destination.exists():
            raise ContractError(f"encoded artifact already exists: {self.destination}")
        if type(shard_size) is not int or shard_size <= 0:
            raise ContractError("encoded artifact shard_size must be a positive integer")
        if role not in {"query", "document"}:
            raise ContractError(f"unsupported encoded artifact role {role!r}")
        if type(normalized) is not bool:
            raise ContractError("encoded artifact normalized must be boolean")
        if not isinstance(provenance, Mapping) or any(
            type(name) is not str or not name or type(value) is not str or not value
            for name, value in provenance.items()
        ):
            raise ContractError(
                "encoded artifact provenance must contain non-empty string keys and values"
            )
        self.provenance = dict(provenance)
        self.role = role
        self.normalized = normalized
        self.shard_size = shard_size
        self._temporary = self.destination.with_name(
            f".{self.destination.name}.tmp-{uuid.uuid4().hex}"
        )
        self._temporary.parent.mkdir(parents=True, exist_ok=True)
        self._temporary.mkdir()
        self._pending_ids: list[str] = []
        self._pending_metadata: list[Mapping[str, Any]] = []
        self._pending_tensors: list[Tensor] = []
        self._seen_ids: set[str] = set()
        self._shards: list[_ShardRecord] = []
        self._count = 0
        self._dimension: int | None = None
        self._dtype: torch.dtype | None = None
        self._closed = False
        self._failed = False

    def __enter__(self) -> Self:
        """Return this active writer to a context manager."""
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Publish on success and remove every temporary file on failure."""
        if error_type is None:
            self.close()
        else:
            self.abort()

    def append(
        self,
        item_ids: Sequence[str],
        embeddings: Tensor,
        metadata: Sequence[Mapping[str, Any]],
    ) -> None:
        """Validate and buffer one ordered batch without retaining earlier shards."""
        self._require_open()
        if not item_ids:
            raise ContractError("encoded artifact batch must not be empty")
        if any(type(item_id) is not str or not item_id for item_id in item_ids):
            raise ContractError("encoded artifact item IDs must be non-empty strings")
        if len(set(item_ids)) != len(item_ids) or self._seen_ids.intersection(item_ids):
            raise ContractError("encoded artifact item IDs must be globally unique")
        if len(metadata) != len(item_ids) or any(
            not isinstance(value, Mapping) for value in metadata
        ):
            raise ContractError("encoded artifact metadata must align with item IDs")
        tensor = embeddings.detach().cpu().contiguous()
        if (
            tensor.ndim != _EMBEDDING_RANK
            or tensor.shape[0] != len(item_ids)
            or tensor.shape[1] == 0
        ):
            raise ContractError("encoded artifact embeddings must have shape [N, D]")
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all().item()):
            raise ContractError("encoded artifact embeddings must be finite floating point")
        if self.normalized:
            _require_normalized(tensor)
        dimension = int(tensor.shape[1])
        if self._dimension is None:
            self._dimension = dimension
            self._dtype = tensor.dtype
        elif self._dimension != dimension or self._dtype != tensor.dtype:
            raise ContractError(
                "encoded artifact batches must use one embedding dimension and dtype"
            )
        try:
            frozen_metadata = canonicalize_json_value(
                tuple(metadata),
                context="encoded artifact metadata",
            )
        except (TypeError, ValueError) as error:
            raise ContractError(str(error)) from error
        self._pending_ids.extend(item_ids)
        if not isinstance(frozen_metadata, list) or any(
            not isinstance(value, dict) for value in frozen_metadata
        ):
            raise ContractError("encoded artifact metadata canonicalization failed")
        self._pending_metadata.extend(cast("list[Mapping[str, Any]]", frozen_metadata))
        self._pending_tensors.append(tensor)
        self._seen_ids.update(item_ids)
        while len(self._pending_ids) >= self.shard_size:
            self._flush(self.shard_size)

    def close(self) -> EncodedArtifactSummary:
        """Flush, fsync, and publish the artifact without replacing a destination."""
        self._require_open()
        try:
            if self._pending_ids:
                self._flush(len(self._pending_ids))
            if self._count == 0 or self._dimension is None or self._dtype is None:
                raise ContractError(  # noqa: TRY301 - rollback belongs to this boundary
                    "cannot publish an empty encoded artifact"
                )
            manifest = {
                "dtype": str(self._dtype).removeprefix("torch."),
                "embedding_dimension": self._dimension,
                "item_count": self._count,
                "normalized": self.normalized,
                "provenance": dict(sorted(self.provenance.items())),
                "role": self.role,
                "schema": ENCODED_ARTIFACT_SCHEMA,
                "shards": [shard.to_json() for shard in self._shards],
            }
            manifest_path = self._temporary / _MANIFEST_NAME
            _write_json(manifest_path, manifest)
            manifest_sha256 = _file_sha256(manifest_path)
            _fsync_directory(self._temporary)
            try:
                atomic_rename_directory_no_replace(self._temporary, self.destination)
            except FileExistsError as error:
                raise ContractError(
                    f"encoded artifact already exists: {self.destination}"
                ) from error
            except OSError as error:
                raise ContractError(
                    f"cannot atomically publish encoded artifact: {self.destination}"
                ) from error
            self._closed = True
            return EncodedArtifactSummary(
                path=self.destination,
                manifest_sha256=manifest_sha256,
                item_count=self._count,
                embedding_dimension=self._dimension,
                dtype=str(self._dtype).removeprefix("torch."),
                shards=len(self._shards),
            )
        except Exception:
            self._failed = True
            shutil.rmtree(self._temporary, ignore_errors=True)
            raise

    def abort(self) -> None:
        """Remove every unpublished shard after a caller or writer failure."""
        if self._closed:
            return
        self._failed = True
        shutil.rmtree(self._temporary, ignore_errors=True)

    def _flush(self, rows: int) -> None:
        embeddings = torch.cat(self._pending_tensors, dim=0)
        shard_tensor = embeddings[:rows].contiguous()
        remaining = embeddings[rows:].contiguous()
        shard_ids = self._pending_ids[:rows]
        shard_metadata = self._pending_metadata[:rows]
        self._pending_ids = self._pending_ids[rows:]
        self._pending_metadata = self._pending_metadata[rows:]
        self._pending_tensors = [remaining] if remaining.shape[0] else []
        index = len(self._shards)
        if index >= _MAX_ARTIFACT_SHARDS:
            raise ContractError(f"encoded artifact cannot exceed {_MAX_ARTIFACT_SHARDS} shards")
        tensor_name = f"embeddings-{index:06d}.safetensors"
        records_name = f"records-{index:06d}.jsonl"
        tensor_path = self._temporary / tensor_name
        records_path = self._temporary / records_name
        save_file({_EMBEDDING_KEY: shard_tensor}, str(tensor_path))
        _fsync_file(tensor_path)
        _write_records(records_path, shard_ids, shard_metadata)
        self._shards.append(
            _ShardRecord(
                tensor=tensor_name,
                records=records_name,
                rows=rows,
                tensor_sha256=_file_sha256(tensor_path),
                records_sha256=_file_sha256(records_path),
            )
        )
        self._count += rows

    def _require_open(self) -> None:
        if self._closed or self._failed:
            raise ContractError("encoded artifact writer is already closed")


def write_encoded_artifact(
    collection: EncodedCollection,
    destination: str | Path,
    *,
    role: Literal["query", "document"],
    normalized: bool,
    shard_size: int = 4096,
) -> EncodedArtifactSummary:
    """Write an in-memory collection through the same streaming publication path."""
    writer = EncodedArtifactWriter(
        destination,
        provenance=collection.provenance,
        role=role,
        normalized=normalized,
        shard_size=shard_size,
    )
    try:
        writer.append(collection.item_ids, collection.embeddings, collection.metadata)
        return writer.close()
    except Exception:
        writer.abort()
        raise


def load_encoded_artifact(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_role: Literal["query", "document"] | None = None,
    max_materialized_bytes: int = DEFAULT_MAX_MATERIALIZED_BYTES,
) -> EncodedArtifact:
    """Fully verify, budget, and materialize a v2 directory."""
    if type(max_materialized_bytes) is not int or max_materialized_bytes <= 0:
        raise ContractError("max_materialized_bytes must be a positive integer")
    ids: list[str] = []
    metadata: list[Mapping[str, Any]] = []
    verified_dtype: torch.dtype | None = None
    offset = 0
    with EncodedArtifactReader(
        path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_role=expected_role,
    ) as reader:
        # Do not trust manifest-wide dimensions for a large allocation. The
        # first pass validates every checksum, tensor header, row count, ID,
        # record, dtype, dimension, and normalization claim one shard at a time.
        for shard in reader.iter_shards():
            if verified_dtype is None:
                verified_dtype = shard.embeddings.dtype
        if verified_dtype is None:
            raise ContractError("encoded artifact yielded no verified tensor shards")
        embeddings = _allocate_materialized_embeddings(
            item_count=reader.item_count,
            embedding_dimension=reader.embedding_dimension,
            dtype=verified_dtype,
            max_materialized_bytes=max_materialized_bytes,
        )
        # Re-read through the same descriptor-bound, checksum-verifying path.
        # This keeps peak overhead to one shard while the final matrix is filled.
        for shard in reader.iter_shards():
            rows = int(shard.embeddings.shape[0])
            embeddings[offset : offset + rows].copy_(shard.embeddings)
            offset += rows
            ids.extend(shard.item_ids)
            metadata.extend(shard.metadata)
        if offset != reader.item_count:
            raise ContractError("encoded artifact yielded an incomplete tensor collection")
        collection = EncodedCollection(
            tuple(ids),
            embeddings,
            tuple(metadata),
            reader.provenance,
        )
        return EncodedArtifact(
            collection=collection,
            path=Path(path),
            role=reader.role,
            normalized=reader.normalized,
            manifest_sha256=reader.manifest_sha256,
        )


def _allocate_materialized_embeddings(
    *,
    item_count: int,
    embedding_dimension: int,
    dtype: torch.dtype,
    max_materialized_bytes: int,
) -> Tensor:
    """Allocate only after a verified size fits the caller's explicit budget."""
    element_bytes = torch.finfo(dtype).bits // 8
    required_bytes = item_count * embedding_dimension * element_bytes
    if required_bytes > max_materialized_bytes:
        raise ContractError(
            "encoded artifact materialization requires "
            f"{required_bytes} bytes, exceeding max_materialized_bytes="
            f"{max_materialized_bytes}; use EncodedArtifactReader.iter_shards() "
            "or explicitly raise the trusted budget"
        )
    try:
        return torch.empty((item_count, embedding_dimension), dtype=dtype)
    except (MemoryError, OverflowError, RuntimeError) as error:
        raise ContractError(
            "cannot allocate the verified encoded artifact within available memory; "
            "use EncodedArtifactReader.iter_shards()"
        ) from error


def _parse_manifest(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ContractError("encoded artifact manifest is oversized")
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"), context="encoded manifest")
    except (OSError, UnicodeError, ValueError) as error:
        raise ContractError("encoded artifact manifest is invalid") from error
    if not isinstance(value, dict):
        raise ContractError("encoded artifact manifest root must be a mapping")
    expected = {
        "dtype",
        "embedding_dimension",
        "item_count",
        "normalized",
        "provenance",
        "role",
        "schema",
        "shards",
    }
    if set(value) != expected:
        raise ContractError("encoded artifact manifest fields differ from schema")
    if value["schema"] != ENCODED_ARTIFACT_SCHEMA:
        raise ContractError(f"unsupported encoded artifact schema {value['schema']!r}")
    if value["role"] not in {"query", "document"} or type(value["normalized"]) is not bool:
        raise ContractError("encoded artifact role or normalization semantics are invalid")
    for name in ("embedding_dimension", "item_count"):
        if type(value[name]) is not int or value[name] <= 0:
            raise ContractError(f"encoded artifact {name} must be a positive integer")
    if type(value["dtype"]) is not str or not value["dtype"]:
        raise ContractError("encoded artifact dtype must be a non-empty string")
    provenance = value["provenance"]
    if not isinstance(provenance, dict) or any(
        type(name) is not str or not name or type(item) is not str or not item
        for name, item in provenance.items()
    ):
        raise ContractError("encoded artifact provenance must be a string mapping")
    shards = value["shards"]
    if not isinstance(shards, list) or not shards:
        raise ContractError("encoded artifact shards must be a non-empty list")
    parsed_shards = [_parse_shard(shard, index) for index, shard in enumerate(shards)]
    if sum(shard.rows for shard in parsed_shards) != value["item_count"]:
        raise ContractError("encoded artifact shard rows do not sum to item_count")
    value["shards"] = parsed_shards
    return value


def _parse_shard(value: Any, index: int) -> _ShardRecord:
    if not isinstance(value, dict) or set(value) != {
        "records",
        "records_sha256",
        "rows",
        "tensor",
        "tensor_sha256",
    }:
        raise ContractError(f"encoded artifact shard {index} fields differ from schema")
    expected_names = {
        "tensor": f"embeddings-{index:06d}.safetensors",
        "records": f"records-{index:06d}.jsonl",
    }
    for field, expected in expected_names.items():
        if value[field] != expected:
            raise ContractError(f"encoded artifact shard {index} has invalid {field} name")
    if type(value["rows"]) is not int or value["rows"] <= 0:
        raise ContractError(f"encoded artifact shard {index} rows must be positive")
    for field in ("tensor_sha256", "records_sha256"):
        digest = value[field]
        if (
            type(digest) is not str
            or len(digest) != _SHA256_HEX_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ContractError(f"encoded artifact shard {index} has invalid {field}")
    return _ShardRecord(
        tensor=value["tensor"],
        records=value["records"],
        rows=value["rows"],
        tensor_sha256=value["tensor_sha256"],
        records_sha256=value["records_sha256"],
    )


def _load_safetensor(path: Path) -> Tensor:
    tensor: Tensor | None = None
    try:
        with safe_open(path, framework="pt", device="cpu") as values:
            keys = tuple(values.keys())
            if keys == (_EMBEDDING_KEY,):
                tensor = cast("Tensor", values.get_tensor(_EMBEDDING_KEY))
    except Exception as error:
        raise ContractError(f"cannot read encoded safetensors shard: {path.name}") from error
    if keys != (_EMBEDDING_KEY,):
        raise ContractError(f"encoded artifact tensor keys must equal {(_EMBEDDING_KEY,)!r}")
    if tensor is None:
        raise ContractError(f"cannot read encoded safetensors shard: {path.name}")
    return tensor


def _load_records(path: Path) -> tuple[list[str], list[Mapping[str, Any]]]:
    ids: list[str] = []
    metadata: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ContractError(f"cannot read encoded records shard: {path.name}") from error
    if not lines or any(not line for line in lines):
        raise ContractError(f"encoded records shard is empty or contains blank lines: {path.name}")
    for line_number, line in enumerate(lines, start=1):
        try:
            value = strict_json_loads(
                line,
                context=f"encoded record {path.name}:{line_number}",
            )
        except ValueError as error:
            raise ContractError(str(error)) from error
        if not isinstance(value, dict) or set(value) != {"item_id", "metadata"}:
            raise ContractError(f"encoded record fields are invalid at {path.name}:{line_number}")
        item_id = value["item_id"]
        item_metadata = value["metadata"]
        if type(item_id) is not str or not item_id or not isinstance(item_metadata, dict):
            raise ContractError(f"encoded record values are invalid at {path.name}:{line_number}")
        ids.append(item_id)
        metadata.append(item_metadata)
    return ids, metadata


def _write_records(
    path: Path,
    item_ids: Sequence[str],
    metadata: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for item_id, item_metadata in zip(item_ids, metadata, strict=True):
            json.dump(
                {"item_id": item_id, "metadata": item_metadata},
                output,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, allow_nan=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _list_regular_files(root_fd: int) -> set[str]:
    try:
        names = os.listdir(root_fd)
    except OSError as error:
        raise ContractError("cannot list encoded artifact files") from error
    if len(names) > _MAX_ARTIFACT_FILES:
        raise ContractError("encoded artifact contains too many files")
    result: set[str] = set()
    for name in names:
        if name in {"", ".", ".."} or "/" in name:
            raise ContractError("encoded artifact contains an invalid file name")
        try:
            value = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except OSError as error:
            raise ContractError(f"cannot inspect encoded artifact file: {name}") from error
        if not stat.S_ISREG(value.st_mode):
            raise ContractError(f"encoded artifact contains a non-regular file: {name}")
        result.add(name)
    return result


def _snapshot_file(
    root_fd: int,
    name: str,
    destination: Path,
    *,
    max_bytes: int | None = None,
) -> str:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, dir_fd=root_fd)
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(opened, named):
            raise ContractError(  # noqa: TRY301 - cleanup belongs to this boundary
                f"encoded artifact file is not bound and regular: {name}"
            )
        if max_bytes is not None and opened.st_size > max_bytes:
            raise ContractError(  # noqa: TRY301 - cleanup belongs to this boundary
                f"encoded artifact file is oversized: {name}"
            )
        digest = hashlib.sha256()
        copied = 0
        with os.fdopen(descriptor, "rb") as source, destination.open("xb") as output:
            descriptor = None
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                copied += len(chunk)
                if max_bytes is not None and copied > max_bytes:
                    destination.unlink(missing_ok=True)
                    raise ContractError(  # noqa: TRY301 - cleanup belongs to this boundary
                        f"encoded artifact file is oversized: {name}"
                    )
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
            final = os.fstat(source.fileno())
        if not os.path.samestat(opened, final) or (
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ):
            destination.unlink(missing_ok=True)
            raise ContractError(  # noqa: TRY301 - cleanup belongs to this boundary
                f"encoded artifact file changed while read: {name}"
            )
        return digest.hexdigest()
    except ContractError:
        raise
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise ContractError(f"cannot snapshot encoded artifact file: {name}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _require_normalized(embeddings: Tensor) -> None:
    norms = torch.linalg.vector_norm(embeddings.float(), dim=1)
    tolerance = max(1e-4, float(torch.finfo(embeddings.dtype).eps))
    if not bool(torch.isfinite(norms).all().item()) or not torch.allclose(
        norms,
        torch.ones_like(norms),
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ContractError("encoded artifact claims normalized vectors but norms differ from 1")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_stream:
        for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as input_stream:
        os.fsync(input_stream.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
