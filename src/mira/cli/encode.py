"""Stream validated multimodal items through Qwen3-VL-Embedding inference."""

from __future__ import annotations

import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Annotated, Literal, cast

import typer
from pydantic import ValidationError

from mira.cli._common import build_encoded_provenance, emit_json
from mira.cli._media import resolve_item_media
from mira.config import resolve_config
from mira.core.exceptions import ContractError
from mira.core.filesystem import open_bound_regular_file
from mira.core.serialization import strict_json_loads
from mira.data.schema import MultimodalItem
from mira.inference import EncodedArtifactWriter, InferenceBackendName, load_encoder

if TYPE_CHECKING:
    from collections.abc import Iterator

_INFERENCE_BACKEND = "qwen3-vl-embedding-transformers"


def encode_command(  # noqa: PLR0913 - explicit inference artifacts are the CLI contract
    checkpoint: Annotated[
        str,
        typer.Option(
            "--checkpoint",
            help="Audited model ID, native MIRA checkpoint, or local MIRA export.",
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            readable=True,
            help="Resolved experiment contract for model and preprocessing semantics.",
        ),
    ],
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            readable=True,
            help="UTF-8 MultimodalItem JSONL to validate and encode.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New mira.encoded.v2 artifact directory."),
    ],
    role: Annotated[
        Literal["query", "document"],
        typer.Option("--role", help="Retrieval-axis role stored in the artifact manifest."),
    ],
    backend: Annotated[
        str,
        typer.Option("--backend", help="Explicit inference backend identifier."),
    ] = _INFERENCE_BACKEND,
    device: Annotated[
        str,
        typer.Option("--device", help="auto, cpu, mps, cuda, or a valid CUDA index."),
    ] = "auto",
    dtype: Annotated[
        str,
        typer.Option("--dtype", help="auto, float32, float16, or bfloat16."),
    ] = "auto",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Items per model forward batch."),
    ] = 8,
    shard_size: Annotated[
        int,
        typer.Option("--shard-size", min=1, help="Maximum vectors per artifact shard."),
    ] = 4096,
    dimension: Annotated[
        int | None,
        typer.Option("--dimension", min=1, help="Optional trained MRL prefix dimension."),
    ] = None,
    expected_checkpoint_manifest_sha256: Annotated[
        str | None,
        typer.Option(
            "--expected-checkpoint-manifest-sha256",
            help="Trusted SHA-256 for a native checkpoint manifest.",
        ),
    ] = None,
    expected_export_manifest_checksum: Annotated[
        str | None,
        typer.Option(
            "--expected-export-manifest-checksum",
            help="Trusted checksum printed by mira export.",
        ),
    ] = None,
) -> None:
    """Encode MultimodalItem JSONL into a checksummed v2 artifact directory."""
    if backend != _INFERENCE_BACKEND:
        raise ContractError(f"unsupported inference backend {backend!r}")
    if dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ContractError(f"unsupported inference dtype {dtype!r}")
    resolved = resolve_config(config)
    embedding_dimension = dimension or resolved.config.model.hidden_size
    if embedding_dimension is None:
        raise ContractError("Qwen3-VL-Embedding encoding requires model.hidden_size")
    started = time.monotonic()
    with _verified_input_snapshot(input_path) as snapshot:
        item_count = _preflight_items(snapshot, input_path)
        encoder = load_encoder(
            checkpoint,
            config,
            backend=cast("InferenceBackendName", backend),
            device=device,
            dtype=cast("Literal['auto', 'float32', 'float16', 'bfloat16']", dtype),
            batch_size=batch_size,
            expected_checkpoint_manifest_sha256=expected_checkpoint_manifest_sha256,
            expected_export_manifest_checksum=expected_export_manifest_checksum,
        )
        runtime = encoder.runtime
        if runtime.backend != backend:
            raise ContractError("encoder runtime identity differs from the selected backend")
        provenance = build_encoded_provenance(
            checkpoint,
            resolved.config,
            config_checksum=resolved.checksum,
            embedding_dimension=embedding_dimension,
            template_checksum=runtime.template_fingerprint,
            schema="mira.encoded.v2",
            backend=backend,
            template_profile=runtime.template_profile,
        )
        writer = EncodedArtifactWriter(
            output,
            provenance=provenance,
            role=role,
            normalized=True,
            shard_size=shard_size,
        )
        try:
            for batch in _item_batches(
                _iter_items(snapshot, input_path),
                batch_size=batch_size,
            ):
                embeddings = (
                    encoder.encode_query(
                        batch,
                        batch_size=batch_size,
                        dimension=dimension,
                        normalize=True,
                    )
                    if role == "query"
                    else encoder.encode_document(
                        batch,
                        batch_size=batch_size,
                        dimension=dimension,
                        normalize=True,
                    )
                )
                if embeddings.shape[1] != embedding_dimension:
                    raise ContractError(  # noqa: TRY301 - writer rollback boundary
                        "Qwen3-VL-Embedding encoder returned an unexpected dimension: "
                        f"{embeddings.shape[1]} != {embedding_dimension}"
                    )
                writer.append(
                    tuple(item.item_id for item in batch),
                    embeddings,
                    tuple({"source": str(input_path)} for _ in batch),
                )
            summary = writer.close()
        except Exception:
            writer.abort()
            raise
    elapsed = time.monotonic() - started
    emit_json(
        {
            "backend": backend,
            "config_checksum": resolved.checksum,
            "device": str(runtime.device),
            "dimension": summary.embedding_dimension,
            "dtype": summary.dtype,
            "elapsed_seconds": elapsed,
            "items": item_count,
            "items_per_second": item_count / elapsed if elapsed > 0 else None,
            "manifest_sha256": summary.manifest_sha256,
            "output": str(summary.path),
            "role": role,
            "shards": summary.shards,
            "status": "complete",
        }
    )


@contextmanager
def _verified_input_snapshot(path: Path) -> Iterator[Path]:
    """Copy one descriptor-bound input file so both validation passes see one value."""
    try:
        source = open_bound_regular_file(path)
    except OSError as error:
        raise ContractError("encode input is missing, symlinked, or non-regular") from error
    with source, TemporaryDirectory(prefix="mira-encode-input-") as temporary:
        opened = os.fstat(source.fileno())
        snapshot = Path(temporary) / "input.jsonl"
        with snapshot.open("xb") as output_stream:
            shutil.copyfileobj(source, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
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
            raise ContractError("encode input changed while it was snapshotted")
        yield snapshot


def _preflight_items(snapshot: Path, display_path: Path) -> int:
    seen: set[str] = set()
    count = 0
    for item in _iter_items(snapshot, display_path):
        if item.item_id in seen:
            raise ContractError("encode input item_id values must be unique")
        seen.add(item.item_id)
        count += 1
    if count == 0:
        raise ContractError("encode input contains no items")
    return count


def _iter_items(snapshot: Path, display_path: Path) -> Iterator[MultimodalItem]:
    try:
        with snapshot.open("r", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ContractError(f"blank item JSONL line at {display_path}:{line_number}")
                try:
                    payload = strict_json_loads(
                        line,
                        context=f"item at {display_path}:{line_number}",
                    )
                    item = MultimodalItem.model_validate(payload)
                    yield resolve_item_media(item, display_path.parent)
                except (ValidationError, ValueError) as error:
                    raise ContractError(
                        f"invalid item at {display_path}:{line_number}: {error}"
                    ) from error
    except UnicodeError as error:
        raise ContractError(f"encode input is not valid UTF-8: {display_path}") from error


def _item_batches(
    items: Iterator[MultimodalItem],
    *,
    batch_size: int,
) -> Iterator[list[MultimodalItem]]:
    batch: list[MultimodalItem] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _read_items(path: Path) -> list[MultimodalItem]:
    """Compatibility helper returning the same strictly preflighted item values."""
    with _verified_input_snapshot(path) as snapshot:
        _preflight_items(snapshot, path)
        return list(_iter_items(snapshot, path))
