"""Shared Qwen3-VL-Embedding loading for upstream and native MIRA checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from torch import nn

from mira.config import TuningMethod
from mira.core.exceptions import ConfigurationValidationError, ContractError
from mira.inference.sources import (
    validate_exported_model,
)
from mira.inference.sources import (
    verified_export_snapshot as _verified_export_snapshot,
)
from mira.integrations.ms_swift import (
    SwiftBackendAssets,
    SwiftLoadSpec,
    SwiftTemplateSpec,
    load_qwen3_vl_embedding,
)
from mira.integrations.ms_swift.peft_bridge import LoraSpec, apply_lora
from mira.training import CheckpointManager

if TYPE_CHECKING:
    from mira.config import MiraConfig
    from mira.models.adapters import Qwen3VLEmbeddingAdapter


def load_configured_qwen_assets(
    checkpoint: str,
    config: MiraConfig,
    *,
    config_checksum: str,
    expected_checkpoint_manifest_sha256: str | None = None,
    expected_export_manifest_checksum: str | None = None,
) -> SwiftBackendAssets:
    """Load a model reference or reconstruct an exact native MIRA checkpoint."""
    checkpoint_path = Path(checkpoint).expanduser()
    native_checkpoint = is_mira_checkpoint(checkpoint_path)
    native_manifest: dict[str, object] | None = None
    local_export = False
    if native_checkpoint:
        if expected_export_manifest_checksum is not None:
            raise ContractError(
                "expected_export_manifest_checksum is only valid for a local exported model"
            )
        native_manifest = CheckpointManager().verify(
            checkpoint_path,
            expected_manifest_sha256=expected_checkpoint_manifest_sha256,
        )
        _validate_native_checkpoint_provenance(
            native_manifest,
            config,
            config_checksum=config_checksum,
        )
    elif checkpoint_path.exists():
        if expected_checkpoint_manifest_sha256 is not None:
            raise ContractError(
                "expected_checkpoint_manifest_sha256 is only valid for a native MIRA checkpoint"
            )
        if expected_export_manifest_checksum is None:
            raise ContractError(
                "local exported models require --expected-export-manifest-checksum from the "
                "trusted export result"
            )
        local_export = True
    else:
        if (
            expected_checkpoint_manifest_sha256 is not None
            or expected_export_manifest_checksum is not None
        ):
            raise ContractError("manifest checksums cannot be supplied for a remote model source")
        if checkpoint != config.model.name_or_path:
            raise ContractError(
                "remote checkpoint identifier must equal config.model.name_or_path so the "
                "audited revision remains authoritative"
            )
    load_spec = SwiftLoadSpec(
        model_id=config.model.name_or_path,
        revision=config.model.revision,
        torch_dtype=_precision_name(config.runtime.precision),
        attention=config.runtime.attention,
        image_max_tokens=config.data.media.image_max_tokens,
        embedding_dimension=config.model.hidden_size,
    )
    if local_export:
        if expected_export_manifest_checksum is None:
            raise ContractError("local export is missing its required trusted manifest checksum")
        with _verified_export_snapshot(
            checkpoint_path,
            config,
            expected_manifest_checksum=expected_export_manifest_checksum,
        ) as model_source:
            assets = load_qwen3_vl_embedding(
                load_spec,
                SwiftTemplateSpec(),
                model_source=str(model_source),
            )
    else:
        assets = load_qwen3_vl_embedding(load_spec, SwiftTemplateSpec())
    if native_checkpoint:
        apply_configured_lora(assets.adapter, config)
        CheckpointManager().load(
            checkpoint_path,
            model=assets.adapter,
            mode="resume_model_only",
            expected_manifest_sha256=expected_checkpoint_manifest_sha256,
        )
    return assets


def apply_configured_lora(
    adapter: Qwen3VLEmbeddingAdapter,
    config: MiraConfig,
) -> None:
    """Apply the exact configured PEFT topology without hidden defaults."""
    tuning = config.model.tuning
    if tuning.method is TuningMethod.FULL:
        return
    if tuning.method is TuningMethod.QLORA:
        raise ConfigurationValidationError(
            "QLoRA requires an explicit quantized load contract that is not implemented"
        )
    if tuning.lora is None:
        raise ConfigurationValidationError(
            f"MIRA does not implement tuning method {tuning.method.value!r} without LoRA values"
        )
    adapter.model = apply_lora(
        adapter.model,
        LoraSpec(
            rank=tuning.lora.rank,
            alpha=tuning.lora.alpha,
            dropout=tuning.lora.dropout,
            target_modules=tuning.lora.target_modules,
            use_dora=tuning.method is TuningMethod.DORA,
            use_rslora=tuning.method is TuningMethod.RS_LORA,
        ),
    )


def merge_configured_adapter(
    adapter: Qwen3VLEmbeddingAdapter,
    config: MiraConfig,
) -> bool:
    """Merge a configured PEFT adapter into the base model for standalone export."""
    if config.model.tuning.method is TuningMethod.FULL:
        return False
    merge = getattr(adapter.model, "merge_and_unload", None)
    if not callable(merge):
        raise ContractError("the configured PEFT model cannot produce a standalone merged export")
    merged = merge(safe_merge=True)
    if not isinstance(merged, nn.Module):
        raise ContractError("PEFT merge_and_unload returned an invalid model")
    adapter.model = merged
    return True


def is_mira_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "manifest.json").is_file() and (path / "state.pt").is_file()


def _validate_exported_model(
    path: Path,
    config: MiraConfig,
    *,
    expected_manifest_checksum: str,
) -> None:
    """Compatibility wrapper around the inference-owned export trust boundary."""
    validate_exported_model(
        path,
        config,
        expected_manifest_checksum=expected_manifest_checksum,
    )


def _validate_native_checkpoint_provenance(
    manifest: dict[str, object],
    config: MiraConfig,
    *,
    config_checksum: str,
) -> None:
    provenance = manifest.get("manifests")
    if not isinstance(provenance, dict):
        raise ContractError("native checkpoint has no provenance mapping")
    expected = {
        "config_checksum": config_checksum,
        "model_name_or_path": config.model.name_or_path,
        "model_revision": config.model.revision,
    }
    observed = {name: provenance.get(name) for name in expected}
    if observed != expected:
        raise ContractError(
            "native checkpoint provenance differs from the supplied config: "
            f"observed={observed}, expected={expected}"
        )


def _precision_name(value: str) -> Literal["bfloat16", "float16", "float32"]:
    if value == "bf16":
        return "bfloat16"
    if value == "fp16":
        return "float16"
    return "float32"
