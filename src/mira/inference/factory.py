"""Public construction of strict embedding encoders."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from mira.config import ModelAdapterType, TuningMethod, resolve_config
from mira.core.exceptions import ConfigurationValidationError, ContractError
from mira.inference.encoder import Encoder
from mira.inference.sources import verified_export_snapshot
from mira.integrations.peft import LoraSpec, apply_lora
from mira.integrations.qwen3_vl_embedding import (
    Qwen3VLEmbeddingLoadSpec,
    load_qwen3_vl_embedding_transformers,
    require_compatible,
)
from mira.training import CheckpointManager

if TYPE_CHECKING:
    from mira.config import MiraConfig, ResolvedConfig
    from mira.integrations.qwen3_vl_embedding import Qwen3VLEmbeddingBackendAssets

InferenceBackendName: TypeAlias = Literal["qwen3-vl-embedding-transformers"]


def load_encoder(  # noqa: PLR0913 - construction options are explicit public semantics
    checkpoint: str,
    config: str | Path,
    *,
    backend: InferenceBackendName = "qwen3-vl-embedding-transformers",
    device: str = "auto",
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto",
    batch_size: int = 8,
    expected_checkpoint_manifest_sha256: str | None = None,
    expected_export_manifest_checksum: str | None = None,
) -> Encoder:
    """Load a remote, native-checkpoint, or exported Qwen3-VL-Embedding encoder.

    Remote models are restricted to immutable audited revisions. Native MIRA
    checkpoints require their exact configuration checksum and may reconstruct
    PEFT before loading model state. Local exports require the checksum emitted
    by ``mira export`` through a trusted channel.
    """
    if backend != "qwen3-vl-embedding-transformers":
        raise ConfigurationValidationError(f"unsupported inference backend {backend!r}")
    if type(checkpoint) is not str or not checkpoint or checkpoint != checkpoint.strip():
        raise ConfigurationValidationError("checkpoint must be a canonical string")
    if type(batch_size) is not int or batch_size <= 0:
        raise ConfigurationValidationError("batch_size must be a positive integer")
    resolved = resolve_config(config)
    _require_qwen3_vl_embedding_config(resolved.config)
    assets = _load_assets(
        checkpoint,
        resolved,
        device=device,
        dtype=dtype,
        expected_checkpoint_manifest_sha256=expected_checkpoint_manifest_sha256,
        expected_export_manifest_checksum=expected_export_manifest_checksum,
    )
    instruction = assets.preprocessor.spec.instruction
    return Encoder(
        assets.backend,
        query_instruction=instruction,
        document_instruction=instruction,
        default_batch_size=batch_size,
    )


def _load_assets(  # noqa: PLR0913 - mirrors the public trust boundary
    checkpoint: str,
    resolved: ResolvedConfig,
    *,
    device: str,
    dtype: Literal["auto", "float32", "float16", "bfloat16"],
    expected_checkpoint_manifest_sha256: str | None,
    expected_export_manifest_checksum: str | None,
) -> Qwen3VLEmbeddingBackendAssets:
    config = resolved.config
    path = Path(checkpoint).expanduser()
    native = _is_mira_checkpoint(path)
    local_export = path.exists() and not native
    if native:
        if expected_export_manifest_checksum is not None:
            raise ContractError(
                "expected_export_manifest_checksum is only valid for a local export"
            )
        manifest = CheckpointManager().verify(
            path,
            expected_manifest_sha256=expected_checkpoint_manifest_sha256,
        )
        _validate_native_checkpoint_provenance(
            manifest,
            config,
            config_checksum=resolved.checksum,
        )
        assets = load_qwen3_vl_embedding_transformers(
            _load_spec(config, device=device, dtype=dtype, template_profile="mira_v1")
        )
        _apply_configured_lora(assets, config)
        CheckpointManager().load(
            path,
            model=assets.adapter,
            mode="resume_model_only",
            expected_manifest_sha256=expected_checkpoint_manifest_sha256,
        )
        return assets
    if local_export:
        if expected_checkpoint_manifest_sha256 is not None:
            raise ContractError(
                "expected_checkpoint_manifest_sha256 is only valid for a native checkpoint"
            )
        if expected_export_manifest_checksum is None:
            raise ContractError(
                "local Qwen3-VL-Embedding exports require their trusted manifest checksum"
            )
        with verified_export_snapshot(
            path,
            config,
            expected_manifest_checksum=expected_export_manifest_checksum,
        ) as snapshot:
            return load_qwen3_vl_embedding_transformers(
                _load_spec(config, device=device, dtype=dtype, template_profile="mira_v1"),
                model_source=str(snapshot),
            )
    if (
        expected_checkpoint_manifest_sha256 is not None
        or expected_export_manifest_checksum is not None
    ):
        raise ContractError("manifest checksums cannot be supplied for a remote model")
    if checkpoint != config.model.name_or_path:
        raise ContractError(
            "remote Qwen3-VL-Embedding checkpoint must equal config.model.name_or_path"
        )
    return load_qwen3_vl_embedding_transformers(
        _load_spec(config, device=device, dtype=dtype, template_profile="official")
    )


def _load_spec(
    config: MiraConfig,
    *,
    device: str,
    dtype: Literal["auto", "float32", "float16", "bfloat16"],
    template_profile: Literal["official", "mira_v1"],
) -> Qwen3VLEmbeddingLoadSpec:
    hidden_size = config.model.hidden_size
    if hidden_size is None:
        raise ConfigurationValidationError(
            "Qwen3-VL-Embedding inference requires model.hidden_size"
        )
    return Qwen3VLEmbeddingLoadSpec(
        model_id=config.model.name_or_path,
        revision=config.model.revision,
        embedding_dimension=hidden_size,
        device=device,
        dtype=dtype,
        attention=config.runtime.attention,
        image_max_tokens=config.data.media.image_max_tokens,
        instruction="Represent the user's input.",
        template_profile=template_profile,
        max_length=32768 if template_profile == "mira_v1" else 8192,
    )


def _apply_configured_lora(
    assets: Qwen3VLEmbeddingBackendAssets,
    config: MiraConfig,
) -> None:
    tuning = config.model.tuning
    if tuning.method is TuningMethod.FULL:
        return
    if tuning.method is TuningMethod.QLORA:
        raise ConfigurationValidationError(
            "QLoRA inference requires a quantized Qwen3-VL-Embedding load contract"
        )
    if tuning.lora is None:
        raise ConfigurationValidationError(
            f"Qwen3-VL-Embedding tuning method {tuning.method.value!r} lacks LoRA values"
        )
    require_compatible(require_peft=True)
    assets.adapter.model = apply_lora(
        assets.adapter.model,
        LoraSpec(
            rank=tuning.lora.rank,
            alpha=tuning.lora.alpha,
            dropout=tuning.lora.dropout,
            target_modules=tuning.lora.target_modules,
            use_dora=tuning.method is TuningMethod.DORA,
            use_rslora=tuning.method is TuningMethod.RS_LORA,
        ),
    )


def _require_qwen3_vl_embedding_config(config: MiraConfig) -> None:
    if config.model.adapter is not ModelAdapterType.QWEN3_VL_EMBEDDING:
        raise ConfigurationValidationError(
            "load_encoder currently requires model.adapter=qwen3_vl_embedding"
        )


def _is_mira_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "manifest.json").is_file() and (path / "state.pt").is_file()


def _validate_native_checkpoint_provenance(
    manifest: dict[str, object],
    config: MiraConfig,
    *,
    config_checksum: str,
) -> None:
    provenance = manifest.get("manifests")
    if not isinstance(provenance, dict):
        raise ContractError("native Qwen3-VL-Embedding checkpoint has no provenance mapping")
    expected = {
        "config_checksum": config_checksum,
        "model_name_or_path": config.model.name_or_path,
        "model_revision": config.model.revision,
    }
    observed = {name: provenance.get(name) for name in expected}
    if observed != expected:
        raise ContractError(
            "native Qwen3-VL-Embedding checkpoint provenance differs from config: "
            f"observed={observed}, expected={expected}"
        )
