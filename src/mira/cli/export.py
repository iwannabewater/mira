"""Export model weights, processor assets, and inference semantics."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Annotated

import typer

from mira.cli._common import emit_json, file_sha256
from mira.cli._model import (
    is_mira_checkpoint,
    load_configured_qwen_assets,
    merge_configured_adapter,
)
from mira.config import resolve_config
from mira.core.exceptions import ContractError
from mira.core.filesystem import atomic_rename_directory_no_replace
from mira.core.manifests import ExportManifest
from mira.inference import write_inference_manifest
from mira.models.base import ExportConfig


def export_command(
    checkpoint: Annotated[str, typer.Option("--checkpoint")],
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    expected_checkpoint_manifest_sha256: Annotated[
        str | None,
        typer.Option("--expected-checkpoint-manifest-sha256"),
    ] = None,
) -> None:
    """Export reloadable Qwen assets with checksum-protected semantics."""
    resolved = resolve_config(config)
    checkpoint_path = Path(checkpoint).expanduser()
    if not is_mira_checkpoint(checkpoint_path):
        raise typer.BadParameter(
            "export requires a native MIRA checkpoint directory with manifest.json and state.pt",
            param_hint="--checkpoint",
        )
    destination = output_dir.resolve()
    if destination.exists():
        raise ContractError(f"export output directory already exists: {destination}")
    assets = load_configured_qwen_assets(
        checkpoint,
        resolved.config,
        config_checksum=resolved.checksum,
        expected_checkpoint_manifest_sha256=expected_checkpoint_manifest_sha256,
    )
    merged_adapter = merge_configured_adapter(assets.adapter, resolved.config)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        model_manifest = assets.adapter.export(
            temporary,
            ExportConfig(metadata={"config_checksum": resolved.checksum}),
        )
        _save_processor(assets.processor, temporary)
        artifact_checksums = {
            path.relative_to(temporary).as_posix(): file_sha256(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        processor_name = type(assets.processor).__qualname__
        tokenizer = getattr(assets.processor, "tokenizer", assets.processor)
        checkpoint_checksum = file_sha256(checkpoint_path)
        inference_manifest = ExportManifest(
            model_name_or_path=resolved.config.model.name_or_path,
            model_revision=resolved.config.model.revision,
            pooling=model_manifest.pooling,
            normalized=True,
            embedding_dimension=model_manifest.embedding_dimension,
            representation_type="global",
            visual_preprocessing={
                "attention": resolved.config.runtime.attention,
                "image_max_tokens": resolved.config.data.media.image_max_tokens,
            },
            template="qwen3_vl_emb",
            tokenizer=type(tokenizer).__qualname__,
            processor=processor_name,
            artifact_checksums=artifact_checksums,
            instruction=model_manifest.instruction,
            model_checksum=checkpoint_checksum,
            adapter_name_or_path=(
                f"merged-{resolved.config.model.tuning.method.value}" if merged_adapter else None
            ),
            adapter_checksum=checkpoint_checksum if merged_adapter else None,
            licenses=resolved.config.model.licenses,
        )
        temporary_manifest_path = temporary / "inference-manifest.json"
        write_inference_manifest(inference_manifest, temporary_manifest_path)
        try:
            atomic_rename_directory_no_replace(temporary, destination)
        except FileExistsError as error:
            raise ContractError(f"export output directory already exists: {destination}") from error
        except OSError as error:
            raise ContractError(
                f"cannot atomically publish export directory: {destination}"
            ) from error
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    manifest_path = destination / "inference-manifest.json"
    emit_json(
        {
            "config_checksum": resolved.checksum,
            "manifest": str(manifest_path),
            "manifest_checksum": inference_manifest.checksum,
            "output_dir": str(output_dir),
            "status": "complete",
        }
    )


def _save_processor(processor: object, destination: Path) -> None:
    save_processor = getattr(processor, "save_pretrained", None)
    if not callable(save_processor):
        raise ContractError("the loaded processor does not expose save_pretrained")
    save_processor(destination)
