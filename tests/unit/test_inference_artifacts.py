"""Adversarial evidence for streaming encoded-artifact publication and loading."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from safetensors.torch import save_file

import mira.inference.artifacts as artifact_module
from mira.core.exceptions import ContractError
from mira.inference import (
    ENCODED_ARTIFACT_SCHEMA,
    EncodedArtifactReader,
    EncodedArtifactWriter,
    load_encoded_artifact,
    write_encoded_artifact,
)
from mira.mining import EncodedCollection

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _collection(*, count: int = 5, dimension: int = 4) -> EncodedCollection:
    values = torch.arange(1, count * dimension + 1, dtype=torch.float32).reshape(count, dimension)
    embeddings = torch.nn.functional.normalize(values, dim=1)
    return EncodedCollection(
        tuple(f"item-{index}" for index in range(count)),
        embeddings,
        tuple({"index": index, "nested": {"valid": True}} for index in range(count)),
        {"model": "Qwen/Qwen3-VL-Embedding-2B", "revision": "a" * 40},
    )


def _manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    (path / "manifest.json").write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_streaming_artifact_round_trip_crosses_batch_and_shard_boundaries(
    tmp_path: Path,
) -> None:
    collection = _collection()
    destination = tmp_path / "encoded"
    metadata = [dict(value) for value in collection.metadata]
    writer = EncodedArtifactWriter(
        destination,
        provenance=collection.provenance,
        role="document",
        normalized=True,
        shard_size=3,
    )
    writer.append(collection.item_ids[:2], collection.embeddings[:2], metadata[:2])
    metadata[0]["index"] = 999
    writer.append(collection.item_ids[2:], collection.embeddings[2:], metadata[2:])
    summary = writer.close()

    assert summary.path == destination
    assert summary.item_count == 5
    assert summary.embedding_dimension == 4
    assert summary.dtype == "float32"
    assert summary.shards == 2
    assert summary.manifest_sha256 == _sha256(destination / "manifest.json")
    manifest = _manifest(destination)
    assert manifest["schema"] == ENCODED_ARTIFACT_SCHEMA
    assert manifest["role"] == "document"
    assert manifest["normalized"] is True
    assert [shard["rows"] for shard in manifest["shards"]] == [3, 2]

    loaded = load_encoded_artifact(
        destination,
        expected_manifest_sha256=summary.manifest_sha256,
    )
    assert loaded.item_ids == collection.item_ids
    assert torch.equal(loaded.embeddings, collection.embeddings)
    assert loaded.metadata == collection.metadata
    assert loaded.provenance == collection.provenance
    assert loaded.path == destination
    assert loaded.role == "document"
    assert loaded.normalized is True
    assert loaded.manifest_sha256 == summary.manifest_sha256


def test_reader_streams_verified_shards_and_rejects_concurrent_iteration(
    tmp_path: Path,
) -> None:
    collection = _collection(count=5)
    summary = write_encoded_artifact(
        collection,
        tmp_path / "encoded",
        role="document",
        normalized=True,
        shard_size=2,
    )
    with EncodedArtifactReader(
        summary.path,
        expected_manifest_sha256=summary.manifest_sha256,
        expected_role="document",
    ) as reader:
        assert reader.item_count == 5
        assert reader.embedding_dimension == 4
        assert reader.dtype == "float32"
        assert reader.provenance == collection.provenance
        iterator = reader.iter_shards()
        first = next(iterator)
        assert first.index == 0
        assert first.item_ids == collection.item_ids[:2]
        assert first.role == "document"
        assert first.normalized is True
        assert first.manifest_sha256 == summary.manifest_sha256
        competing = reader.iter_shards()
        with pytest.raises(ContractError, match="active shard iterator"):
            next(competing)
        remaining = tuple(iterator)
        assert (
            tuple(item_id for shard in remaining for item_id in shard.item_ids)
            == (collection.item_ids[2:])
        )
    with pytest.raises(ContractError, match="not open"):
        _ = reader.item_count


def test_reader_rejects_duplicate_ids_across_shards(tmp_path: Path) -> None:
    summary = write_encoded_artifact(
        _collection(count=3),
        tmp_path / "encoded",
        role="query",
        normalized=True,
        shard_size=2,
    )
    manifest = _manifest(summary.path)
    second = manifest["shards"][1]
    records_path = summary.path / second["records"]
    record = json.loads(records_path.read_text(encoding="utf-8"))
    record["item_id"] = "item-0"
    records_path.write_text(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    second["records_sha256"] = _sha256(records_path)
    _write_manifest(summary.path, manifest)
    with pytest.raises(ContractError, match="globally unique"):
        load_encoded_artifact(summary.path)


def test_in_memory_writer_uses_same_artifact_path(tmp_path: Path) -> None:
    collection = _collection(count=2)
    summary = write_encoded_artifact(
        collection,
        tmp_path / "encoded",
        role="query",
        normalized=True,
        shard_size=1,
    )
    assert summary.shards == 2
    assert load_encoded_artifact(summary.path).checksum() == collection.checksum()


@pytest.mark.parametrize(
    ("item_ids", "embeddings", "metadata", "message"),
    [
        ((), torch.ones(0, 2), (), "must not be empty"),
        (("same", "same"), torch.ones(2, 2), ({}, {}), "globally unique"),
        (("one",), torch.ones(2, 2), ({},), r"shape \[N, D\]"),
        (("one",), torch.ones(1, 2, 1), ({},), r"shape \[N, D\]"),
        (("one",), torch.tensor([[float("nan"), 1.0]]), ({},), "finite"),
        (("one",), torch.ones(1, 2), (), "align"),
        (("one",), torch.ones(1, 2), ({"bad": object()},), "unsupported"),
    ],
)
def test_writer_rejects_invalid_batches_and_rolls_back(
    tmp_path: Path,
    item_ids: tuple[str, ...],
    embeddings: torch.Tensor,
    metadata: tuple[dict[str, object], ...],
    message: str,
) -> None:
    destination = tmp_path / "encoded"
    with (
        pytest.raises(ContractError, match=message),
        EncodedArtifactWriter(
            destination,
            provenance={"source": "test"},
            role="query",
            normalized=False,
        ) as writer,
    ):
        writer.append(item_ids, embeddings, metadata)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_writer_rejects_cross_batch_identity_dimension_and_dtype_drift(
    tmp_path: Path,
) -> None:
    writer = EncodedArtifactWriter(
        tmp_path / "encoded",
        provenance={"source": "test"},
        role="query",
        normalized=False,
    )
    writer.append(("one",), torch.ones(1, 2), ({},))
    with pytest.raises(ContractError, match="globally unique"):
        writer.append(("one",), torch.ones(1, 2), ({},))
    with pytest.raises(ContractError, match="dimension and dtype"):
        writer.append(("two",), torch.ones(1, 3), ({},))
    with pytest.raises(ContractError, match="dimension and dtype"):
        writer.append(("two",), torch.ones(1, 2, dtype=torch.float64), ({},))
    writer.abort()


def test_writer_rejects_false_normalization_claim_and_empty_publish(tmp_path: Path) -> None:
    writer = EncodedArtifactWriter(
        tmp_path / "unnormalized",
        provenance={"source": "test"},
        role="query",
        normalized=True,
    )
    with pytest.raises(ContractError, match="norms differ"):
        writer.append(("one",), torch.ones(1, 2), ({},))
    writer.abort()

    empty = EncodedArtifactWriter(
        tmp_path / "empty",
        provenance={"source": "test"},
        role="query",
        normalized=False,
    )
    with pytest.raises(ContractError, match="empty"):
        empty.close()
    assert list(tmp_path.iterdir()) == []


def test_atomic_publish_never_replaces_a_racing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "encoded"
    writer = EncodedArtifactWriter(
        destination,
        provenance={"source": "test"},
        role="query",
        normalized=False,
    )
    writer.append(("one",), torch.ones(1, 2), ({},))
    destination.mkdir()
    marker = destination / "owner.txt"
    marker.write_text("user", encoding="utf-8")
    with pytest.raises(ContractError, match="already exists"):
        writer.close()
    assert marker.read_text(encoding="utf-8") == "user"
    assert [path.name for path in tmp_path.iterdir()] == ["encoded"]


def test_loader_rejects_manifest_checksum_tampering_and_extra_files(tmp_path: Path) -> None:
    summary = write_encoded_artifact(
        _collection(count=2),
        tmp_path / "encoded",
        role="query",
        normalized=True,
    )
    manifest = _manifest(summary.path)
    manifest["role"] = "document"
    _write_manifest(summary.path, manifest)
    with pytest.raises(ContractError, match="trusted checksum"):
        load_encoded_artifact(
            summary.path,
            expected_manifest_sha256=summary.manifest_sha256,
        )

    extra = summary.path / "extra.txt"
    extra.write_text("unexpected", encoding="utf-8")
    with pytest.raises(ContractError, match="file set differs"):
        load_encoded_artifact(summary.path)


def test_loader_enforces_query_and_document_roles(tmp_path: Path) -> None:
    summary = write_encoded_artifact(
        _collection(count=2),
        tmp_path / "encoded",
        role="query",
        normalized=True,
    )
    load_encoded_artifact(summary.path, expected_role="query")
    with pytest.raises(ContractError, match="role differs"):
        load_encoded_artifact(summary.path, expected_role="document")


@pytest.mark.parametrize("member", ["tensor", "records"])
def test_loader_rejects_tampered_shard_content(tmp_path: Path, member: str) -> None:
    summary = write_encoded_artifact(
        _collection(count=2),
        tmp_path / "encoded",
        role="query",
        normalized=True,
    )
    manifest = _manifest(summary.path)
    path = summary.path / manifest["shards"][0][member]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ContractError, match=f"{member} checksum mismatch"):
        load_encoded_artifact(summary.path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema="mira.encoded.v999"), "unsupported"),
        (lambda value: value.update(item_count=0), "positive integer"),
        (lambda value: value.update(normalized="yes"), "normalization semantics"),
        (lambda value: value.update(dtype="float64"), "dtype differs"),
        (lambda value: value.update(embedding_dimension=999), "dimension differs"),
        (lambda value: value.update(unexpected=True), "fields differ"),
    ],
)
def test_loader_rejects_invalid_manifest_semantics(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    summary = write_encoded_artifact(
        _collection(count=2),
        tmp_path / "encoded",
        role="query",
        normalized=True,
    )
    manifest = _manifest(summary.path)
    mutate(manifest)
    _write_manifest(summary.path, manifest)
    with pytest.raises(ContractError, match=message):
        load_encoded_artifact(summary.path)


def test_loader_rejects_unexpected_tensor_keys_even_with_matching_checksum(
    tmp_path: Path,
) -> None:
    summary = write_encoded_artifact(
        _collection(count=2),
        tmp_path / "encoded",
        role="query",
        normalized=True,
    )
    manifest = _manifest(summary.path)
    shard = manifest["shards"][0]
    tensor_path = summary.path / shard["tensor"]
    save_file({"wrong": torch.ones(2, 4)}, tensor_path)
    shard["tensor_sha256"] = _sha256(tensor_path)
    _write_manifest(summary.path, manifest)
    with pytest.raises(ContractError, match="tensor keys"):
        load_encoded_artifact(summary.path)


def test_loader_rejects_symlinked_root_and_non_regular_member(tmp_path: Path) -> None:
    summary = write_encoded_artifact(
        _collection(count=2),
        tmp_path / "encoded",
        role="query",
        normalized=True,
    )
    symlink = tmp_path / "linked"
    symlink.symlink_to(summary.path, target_is_directory=True)
    with pytest.raises(ContractError, match="symlinked"):
        load_encoded_artifact(symlink)

    (summary.path / "directory").mkdir()
    with pytest.raises(ContractError, match="non-regular"):
        load_encoded_artifact(summary.path)


def test_loader_opens_members_nonblocking_and_bounds_manifest_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = write_encoded_artifact(
        _collection(count=2),
        tmp_path / "encoded",
        role="query",
        normalized=True,
    )
    original_open = artifact_module.os.open
    observed_flags: list[int] = []

    def recording_open(
        name: str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if name == "manifest.json" and dir_fd is not None:
            observed_flags.append(flags)
        return original_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifact_module.os, "open", recording_open)
    load_encoded_artifact(summary.path)
    assert observed_flags
    assert all(flags & artifact_module.os.O_NONBLOCK for flags in observed_flags)

    monkeypatch.setattr(artifact_module, "_MAX_MANIFEST_BYTES", 32)
    with pytest.raises(ContractError, match="oversized"):
        load_encoded_artifact(summary.path)


def test_materialization_validates_every_shard_before_allocating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = write_encoded_artifact(
        _collection(count=2),
        tmp_path / "encoded",
        role="query",
        normalized=True,
        shard_size=1,
    )
    manifest = _manifest(summary.path)
    manifest["shards"][1]["rows"] = 1_000_000
    manifest["item_count"] = 1_000_001
    _write_manifest(summary.path, manifest)

    def fail_allocation(**_values: object) -> torch.Tensor:
        raise AssertionError("materialization allocated before every shard was verified")

    monkeypatch.setattr(
        artifact_module,
        "_allocate_materialized_embeddings",
        fail_allocation,
    )
    with pytest.raises(ContractError, match="shape or row count mismatch"):
        load_encoded_artifact(summary.path)


def test_materialization_enforces_an_exact_positive_byte_budget(tmp_path: Path) -> None:
    summary = write_encoded_artifact(
        _collection(count=2),
        tmp_path / "encoded",
        role="query",
        normalized=True,
    )
    with pytest.raises(ContractError, match=r"requires 32 bytes.*max_materialized_bytes=31"):
        load_encoded_artifact(summary.path, max_materialized_bytes=31)
    assert len(load_encoded_artifact(summary.path, max_materialized_bytes=32).item_ids) == 2
    for invalid in (True, 0, 1.0):
        with pytest.raises(ContractError, match="positive integer"):
            load_encoded_artifact(
                summary.path,
                max_materialized_bytes=cast("int", invalid),
            )
