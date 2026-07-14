"""Local media confinement, integrity, and canonicalization tests."""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

import pytest

from mira.cli._media import (
    resolve_item_media,
    resolve_record_media,
    validate_item_media,
    validate_media_reference,
    validate_record_media,
)
from mira.core import filesystem
from mira.core.exceptions import DataValidationError
from mira.data.schema import (
    CandidateItem,
    EmbeddingRecord,
    MediaRef,
    MultimodalItem,
    RetrievalMetadata,
)

if TYPE_CHECKING:
    from pathlib import Path


def _checksum(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _item(item_id: str, uri: str, *, checksum: str | None = None) -> MultimodalItem:
    return MultimodalItem(
        item_id=item_id,
        messages=({"role": "user", "content": "inspect <image>"},),
        images=(MediaRef(uri=uri, checksum=checksum),),
    )


def test_media_resolution_hashes_and_canonicalizes_every_record_axis(tmp_path: Path) -> None:
    payload = b"image-bytes"
    media = tmp_path / "media.bin"
    media.write_bytes(payload)
    query = _item("query", "media.bin", checksum=_checksum(payload))
    positive = CandidateItem.model_validate(_item("positive", "media.bin").model_dump())
    negative = CandidateItem.model_validate(_item("negative", "media.bin").model_dump())
    record = EmbeddingRecord(
        sample_id="sample",
        task_id="image-retrieval",
        query=query,
        positives=(positive,),
        negatives=(negative,),
        metadata=RetrievalMetadata(
            modality_pair="image_to_image",
            dataset_name="dataset",
        ),
    )

    resolved = resolve_record_media(record, tmp_path)

    for item in (resolved.query, *resolved.positives, *resolved.negatives):
        assert item.images[0].uri == str(media.resolve())
        assert item.images[0].checksum == _checksum(payload)
    validate_record_media(record, tmp_path)
    validate_item_media(query, tmp_path)
    validate_media_reference(query.images[0], tmp_path)


@pytest.mark.parametrize(
    ("uri", "match"),
    [
        ("https://example.invalid/image.png", "remote and data"),
        ("data:image/png;base64,AAAA", "remote and data"),
        ("/mira-absolute.png", "absolute media paths"),
        ("missing.png", "missing"),
        ("../outside.png", "outside the dataset directory"),
    ],
)
def test_media_resolution_rejects_remote_absolute_missing_and_escape(
    tmp_path: Path,
    uri: str,
    match: str,
) -> None:
    if uri == "../outside.png":
        (tmp_path.parent / "outside.png").write_bytes(b"outside")
    with pytest.raises(DataValidationError, match=match):
        resolve_item_media(_item("item", uri), tmp_path)


def test_media_resolution_rejects_directory_symlink_escape_and_checksum_drift(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(DataValidationError, match="non-regular"):
        resolve_item_media(_item("directory", "directory"), tmp_path)

    outside = tmp_path.parent / "outside-target.bin"
    outside.write_bytes(b"outside")
    symlink = tmp_path / "link.bin"
    symlink.symlink_to(outside)
    with pytest.raises(DataValidationError, match="symlinked"):
        resolve_item_media(_item("symlink", "link.bin"), tmp_path)

    media = tmp_path / "media.bin"
    media.write_bytes(b"changed")
    with pytest.raises(DataValidationError, match="checksum mismatch"):
        resolve_item_media(_item("checksum", "media.bin", checksum=_checksum(b"old")), tmp_path)


def test_missing_dataset_root_is_reported_as_data_validation_error(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing-root"
    with pytest.raises(DataValidationError, match="missing"):
        resolve_item_media(_item("item", "image.png"), missing_root)


def test_media_descriptor_walk_rejects_a_final_component_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"accepted")
    outside = tmp_path.parent / "outside-race.bin"
    outside.write_bytes(b"must-not-be-read")
    original_open = os.open
    swapped = False

    def racing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "media.bin" and dir_fd is not None and not swapped:
            swapped = True
            media.unlink()
            media.symlink_to(outside)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(filesystem.os, "open", racing_open)
    with pytest.raises(DataValidationError, match="symlinked"):
        resolve_item_media(_item("item", "media.bin"), tmp_path)
    assert swapped


def test_media_descriptor_walk_rejects_symlinked_directory_components(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-media-directory"
    outside.mkdir(exist_ok=True)
    (outside / "image.bin").write_bytes(b"outside")
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(DataValidationError, match="symlinked"):
        resolve_item_media(_item("item", "linked/image.bin"), tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="host has no POSIX FIFO support")
def test_media_resolution_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "image.fifo"
    os.mkfifo(fifo)

    with pytest.raises(DataValidationError, match="non-regular"):
        resolve_item_media(_item("item", fifo.name), tmp_path)
