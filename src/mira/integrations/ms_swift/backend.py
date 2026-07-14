"""Lazy model loading at the audited MS-SWIFT boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from threading import Lock, RLock
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from mira.core.exceptions import ContractError
from mira.integrations.ms_swift.compatibility import require_compatible
from mira.integrations.ms_swift.peft_bridge import apply_lora
from mira.integrations.ms_swift.template_bridge import SwiftTemplateBridge
from mira.integrations.ms_swift.trainer_bridge import create_swift_trainer
from mira.models.adapters import Qwen3VLEmbeddingAdapter, QwenEmbeddingSemantics
from mira.training import CheckpointManager

_IMAGE_LIMIT_ENVIRONMENT_NAME = "IMAGE_MAX_TOKEN_NUM"
# The upstream loader reads this process-global setting during model construction.
# Reentrancy is required because an upstream callback may load another model on the
# same thread while unrelated threads must remain excluded until restoration.
_IMAGE_LIMIT_ENVIRONMENT_LOCK = RLock()

if TYPE_CHECKING:
    from mira.integrations.ms_swift.arguments import SwiftLoadSpec, SwiftTemplateSpec
    from mira.integrations.ms_swift.peft_bridge import LoraSpec
    from mira.integrations.ms_swift.trainer_bridge import SwiftTrainerSpec
    from mira.models.base import ExportConfig, ExportManifest
    from mira.training import (
        CheckpointLoadResult,
        CheckpointRestoreTargets,
        CheckpointSnapshot,
        ResumeMode,
    )


@dataclass(frozen=True, slots=True)
class SwiftBackendAssets:
    """Loaded backend objects and the MIRA adapter that owns their semantics."""

    adapter: Qwen3VLEmbeddingAdapter
    processor: Any
    template: SwiftTemplateBridge


class SwiftBackendAdapter:
    """Stateful facade over the audited, dependency-lazy MS-SWIFT helpers.

    One facade owns one model/processor/template bundle. Accessing any of the
    three assets loads that bundle at most once, and a failed load is never
    cached as a partial result. Checkpoint and export operations are bound to
    the cached MIRA adapter so a caller cannot silently cross model instances.
    """

    def __init__(
        self,
        load_spec: SwiftLoadSpec,
        template_spec: SwiftTemplateSpec,
        *,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        """Bind immutable construction specs without importing MS-SWIFT."""
        self.load_spec = load_spec
        self.template_spec = template_spec
        self._checkpoint_manager = (
            checkpoint_manager if checkpoint_manager is not None else CheckpointManager()
        )
        self._assets: SwiftBackendAssets | None = None
        self._assets_lock = Lock()
        self._lifecycle_lock = Lock()
        self._peft_applied = False
        self._trainer_built = False

    def build_model(self) -> Qwen3VLEmbeddingAdapter:
        """Return the single cached MIRA model adapter."""
        return self._load_assets().adapter

    def build_processor(self) -> Any:
        """Return the processor paired with the cached model."""
        return self._load_assets().processor

    def build_template(self) -> SwiftTemplateBridge:
        """Return the template paired with the cached model and processor."""
        return self._load_assets().template

    def build_trainer(self, spec: SwiftTrainerSpec) -> Any:
        """Build a Swift Trainer for this facade's exact underlying model."""
        adapter = self.build_model()
        with self._lifecycle_lock:
            if spec.model is not adapter.model:
                raise ContractError(
                    "Swift trainer spec must use the model owned by this backend facade"
                )
            trainer = create_swift_trainer(spec)
            self._trainer_built = True
        return trainer

    def build_peft_model(self, spec: LoraSpec) -> Qwen3VLEmbeddingAdapter:
        """Apply one explicit PEFT topology to the cached model adapter."""
        adapter = self.build_model()
        with self._lifecycle_lock:
            if self._peft_applied:
                raise ContractError("PEFT has already been applied to this backend facade")
            if self._trainer_built:
                raise ContractError("PEFT must be applied before building a Swift trainer")
            wrapped_model = apply_lora(adapter.model, spec)
            adapter.model = wrapped_model
            self._peft_applied = True
        return adapter

    def save_checkpoint(
        self,
        output_dir: Path,
        *,
        snapshot: CheckpointSnapshot,
    ) -> dict[str, Any]:
        """Save a checkpoint only for the model owned by this facade."""
        if snapshot.model is not self.build_model():
            raise ContractError(
                "checkpoint snapshot must use the model owned by this backend facade"
            )
        return self._checkpoint_manager.save(output_dir, snapshot=snapshot)

    def resume_checkpoint(
        self,
        checkpoint_dir: Path,
        *,
        mode: ResumeMode,
        targets: CheckpointRestoreTargets | None = None,
        map_location: str | torch.device = "cpu",
        expected_manifest_sha256: str | None = None,
    ) -> CheckpointLoadResult:
        """Verify and resume into the cached model with an explicit mode."""
        return self._checkpoint_manager.load(
            checkpoint_dir,
            model=self.build_model(),
            mode=mode,
            targets=targets,
            map_location=map_location,
            expected_manifest_sha256=expected_manifest_sha256,
        )

    def export_model(
        self,
        output_dir: Path,
        *,
        export_config: ExportConfig,
    ) -> ExportManifest:
        """Export the cached model through its MIRA inference contract."""
        return self.build_model().export(output_dir, export_config)

    def _load_assets(self) -> SwiftBackendAssets:
        assets = self._assets
        if assets is not None:
            return assets
        with self._assets_lock:
            assets = self._assets
            if assets is None:
                assets = load_qwen3_vl_embedding(self.load_spec, self.template_spec)
                self._assets = assets
        return assets


def load_qwen3_vl_embedding(
    load_spec: SwiftLoadSpec,
    template_spec: SwiftTemplateSpec,
    *,
    model_source: str | None = None,
) -> SwiftBackendAssets:
    """Load a Qwen embedding model with an explicit embedding task type."""
    source = (
        load_spec.model_id if model_source is None else _validated_local_model_source(model_source)
    )
    require_compatible()
    swift_model = import_module("swift.model")
    get_model_processor = getattr(swift_model, "get_model_processor", None)
    if not callable(get_model_processor):
        raise ContractError("MS-SWIFT does not expose swift.model.get_model_processor")
    with _IMAGE_LIMIT_ENVIRONMENT_LOCK:
        previous_image_tokens = os.environ.get(_IMAGE_LIMIT_ENVIRONMENT_NAME)
        try:
            if load_spec.image_max_tokens is not None:
                os.environ[_IMAGE_LIMIT_ENVIRONMENT_NAME] = str(load_spec.image_max_tokens)
            model, processor = get_model_processor(
                source,
                revision=load_spec.revision,
                model_type=load_spec.model_type,
                task_type=load_spec.task_type,
                use_hf=load_spec.use_hf,
                device_map=load_spec.device_map,
                torch_dtype=_torch_dtype(load_spec.torch_dtype),
                attn_impl=_attention_name(load_spec.attention),
                load_model=load_spec.load_model,
            )
        finally:
            if previous_image_tokens is None:
                os.environ.pop(_IMAGE_LIMIT_ENVIRONMENT_NAME, None)
            else:
                os.environ[_IMAGE_LIMIT_ENVIRONMENT_NAME] = previous_image_tokens
    if not load_spec.load_model:
        raise ContractError("Qwen adapter construction requires load_model=True")
    if not isinstance(model, nn.Module):
        raise ContractError("MS-SWIFT returned no torch model")
    if processor is None:
        raise ContractError("MS-SWIFT returned no processor")
    template = SwiftTemplateBridge(processor, template_spec)
    eos_token_id = _resolve_eos_token_id(processor)
    adapter = Qwen3VLEmbeddingAdapter(
        model,
        semantics=QwenEmbeddingSemantics(
            model_id=load_spec.model_id,
            eos_token_id=eos_token_id,
            template_fingerprint=template.fingerprint,
            instruction=template_spec.instruction,
            embedding_dimension=load_spec.embedding_dimension,
        ),
    )
    return SwiftBackendAssets(adapter=adapter, processor=processor, template=template)


def _validated_local_model_source(source: str) -> str:
    if type(source) is not str or not source or source != source.strip():
        raise ContractError("local model source must be a canonical absolute directory path")
    path = Path(source)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ContractError("local model source must be a non-symlink absolute directory")
    for artifact in path.rglob("*"):
        if artifact.is_symlink():
            raise ContractError("local model source must not contain symlinked artifacts")
        if artifact.is_file() and artifact.suffix.lower() in {
            ".bin",
            ".dll",
            ".dylib",
            ".pickle",
            ".pkl",
            ".pt",
            ".pth",
            ".py",
            ".so",
        }:
            raise ContractError(
                f"local model source contains an executable or unsafe artifact: {artifact.name}"
            )
    return str(path)


def _torch_dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ContractError(f"unsupported torch dtype: {name!r}") from error


def _resolve_eos_token_id(processor: Any) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ContractError("MS-SWIFT processor has no valid eos_token_id")
    return eos_token_id


def _attention_name(name: str) -> str | None:
    mapping = {
        "auto": None,
        "eager": "eager",
        "flash_attention_2": "flash_attn",
        "sdpa": "sdpa",
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ContractError(f"unsupported attention implementation: {name!r}") from error
