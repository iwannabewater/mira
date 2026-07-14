"""Encode validated multimodal items through the Qwen/MS-SWIFT adapter."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Typer resolves command annotations at runtime
from typing import Annotated, Any

import torch
import typer
from pydantic import ValidationError

from mira.cli._common import build_encoded_provenance, emit_json, save_encoded_collection
from mira.cli._media import resolve_item_media
from mira.cli._model import load_configured_qwen_assets
from mira.config import resolve_config
from mira.core.contracts import RepresentationRequest
from mira.core.exceptions import ContractError
from mira.core.serialization import strict_json_loads
from mira.data.schema import MultimodalItem
from mira.integrations.ms_swift import SwiftTemplateSpec
from mira.integrations.ms_swift.template_bridge import (
    template_fingerprint,
    validate_qwen_message_roles,
)
from mira.mining import EncodedCollection


def encode_command(  # noqa: PLR0913 - explicit inference artifacts are the CLI contract
    checkpoint: Annotated[str, typer.Option("--checkpoint")],
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
    input_path: Annotated[Path, typer.Option("--input", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 8,
    dimension: Annotated[int | None, typer.Option("--dimension", min=1)] = None,
    expected_checkpoint_manifest_sha256: Annotated[
        str | None,
        typer.Option("--expected-checkpoint-manifest-sha256"),
    ] = None,
    expected_export_manifest_checksum: Annotated[
        str | None,
        typer.Option("--expected-export-manifest-checksum"),
    ] = None,
) -> None:
    """Encode MultimodalItem JSONL into a safe tensor bundle."""
    resolved = resolve_config(config)
    items = _read_items(input_path)
    assets = load_configured_qwen_assets(
        checkpoint,
        resolved.config,
        config_checksum=resolved.checksum,
        expected_checkpoint_manifest_sha256=expected_checkpoint_manifest_sha256,
        expected_export_manifest_checksum=expected_export_manifest_checksum,
    )
    batches: list[torch.Tensor] = []
    assets.adapter.eval()
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            encoded = [assets.template.encode_item(item) for item in batch]
            collated = assets.template.collate_items(encoded)
            inputs = _tensor_inputs(collated, _model_device(assets.adapter.model))
            output_value = assets.adapter.forward_representations(
                inputs,
                RepresentationRequest(dimension=dimension, normalize=True),
            )
            if output_value.global_embeddings is None:
                raise ContractError("Qwen adapter returned no global embeddings")
            batches.append(output_value.global_embeddings.detach().cpu())
    embeddings = torch.cat(batches)
    collection = EncodedCollection(
        tuple(item.item_id for item in items),
        embeddings,
        tuple({"source": str(input_path)} for _ in items),
        build_encoded_provenance(
            checkpoint,
            resolved.config,
            config_checksum=resolved.checksum,
            embedding_dimension=int(embeddings.shape[1]),
            template_checksum=template_fingerprint(SwiftTemplateSpec()),
        ),
    )
    save_encoded_collection(collection, output)
    emit_json(
        {
            "config_checksum": resolved.checksum,
            "dimension": int(collection.embeddings.shape[1]),
            "items": len(items),
            "output": str(output),
            "status": "complete",
        }
    )


def _read_items(path: Path) -> list[MultimodalItem]:
    items: list[MultimodalItem] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ContractError(f"blank item JSONL line at {path}:{line_number}")
        try:
            payload = strict_json_loads(line, context=f"item at {path}:{line_number}")
            item = MultimodalItem.model_validate(payload)
        except (ValidationError, ValueError) as error:
            raise ContractError(f"invalid item at {path}:{line_number}: {error}") from error
        if item.instruction not in {None, SwiftTemplateSpec().instruction}:
            raise ContractError(
                f"item {item.item_id!r} instruction differs from the configured template"
            )
        if item.audios:
            raise ContractError(
                f"item {item.item_id!r} contains audio, but Qwen3-VL-Embedding does not "
                "declare audio support"
            )
        validate_qwen_message_roles(item)
        items.append(resolve_item_media(item, path.parent))
    if not items:
        raise ContractError("encode input contains no items")
    if len({item.item_id for item in items}) != len(items):
        raise ContractError("encode input item_id values must be unique")
    return items


def _tensor_inputs(values: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    non_tensors = tuple(
        name for name, value in values.items() if not isinstance(value, torch.Tensor)
    )
    if non_tensors:
        raise ContractError(f"Swift collator returned unsupported non-tensor fields: {non_tensors}")
    return {name: value.to(device) for name, value in values.items()}


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ContractError("loaded model contains no parameters") from error
