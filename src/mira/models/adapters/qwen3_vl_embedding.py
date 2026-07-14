"""Qwen3-VL-Embedding representation adapter.

The adapter preserves the official EOS pooling and L2-normalization behavior.
MS-SWIFT is responsible only for model and template construction; tensor
semantics remain owned by MIRA.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
from torch import Tensor, nn

from mira.core.capabilities import Capability, ModelCapabilities
from mira.core.contracts import (
    EmbeddingBatch,
    EmbeddingOutput,
    RepresentationRequest,
    RepresentationType,
)
from mira.core.exceptions import CapabilityError, ContractError, InvalidEmbeddingError
from mira.core.serialization import canonicalize_json_value
from mira.models.base import (
    AdapterMode,
    ExportConfig,
    ExportManifest,
    ParameterGroupSpec,
    TuningSpec,
)
from mira.models.pooling import last_token_pool, normalize_embeddings, truncate_and_normalize
from mira.models.qwen_audit import AUDITED_QWEN_TARGETS

if TYPE_CHECKING:
    from pathlib import Path

_TOKEN_MATRIX_RANK = 2
_HIDDEN_STATE_RANK = 3
_COMPONENT_MARKERS: dict[str, tuple[str, ...]] = {
    "vision": (".visual.", ".vision_model.", ".vision_tower."),
    "aligner": (".aligner.", ".merger.", ".multi_modal_projector.", ".projector."),
    "language_model": (".language_model.", ".model.layers.", ".embed_tokens.", ".lm_head."),
    "embedding_projection": (".embedding_projection.", ".embedding_head."),
    "identity_head": (".identity_head.",),
    "late_interaction_adapter": (".late_interaction_adapter.",),
    "generative_matching_head": (".generative_matching_head.",),
}


@dataclass(frozen=True, slots=True)
class QwenEmbeddingSemantics:
    """Immutable Qwen prompt, EOS, and embedding-dimension semantics."""

    model_id: str
    eos_token_id: int
    template_fingerprint: str
    instruction: str = "Represent the user's input."
    embedding_dimension: int | None = None


class Qwen3VLEmbeddingAdapter(nn.Module):
    """Adapt a loaded Qwen3-VL embedding model to MIRA contracts."""

    DEFAULT_INSTRUCTION: ClassVar[str] = "Represent the user's input."
    SUPPORTED_MODEL_DIMENSIONS: ClassVar[Mapping[str, int]] = MappingProxyType(
        {model_id: target.embedding_dimension for model_id, target in AUDITED_QWEN_TARGETS.items()}
    )

    def __init__(
        self,
        model: nn.Module,
        *,
        semantics: QwenEmbeddingSemantics,
    ) -> None:
        """Bind a model to immutable prompt and representation semantics."""
        super().__init__()
        if not semantics.model_id.strip():
            raise ContractError("Qwen adapter model_id must not be empty")
        if semantics.eos_token_id < 0:
            raise ContractError("Qwen adapter eos_token_id must be non-negative")
        if not semantics.template_fingerprint.strip():
            raise ContractError("Qwen adapter template fingerprint must not be empty")
        if not semantics.instruction.strip():
            raise ContractError("Qwen adapter instruction must not be empty")
        if not _is_valid_embedding_dimension(semantics.embedding_dimension):
            raise ContractError("Qwen adapter embedding_dimension must be a positive integer")
        known_dimension = self.SUPPORTED_MODEL_DIMENSIONS.get(semantics.model_id)
        resolved_dimension = (
            known_dimension
            if semantics.embedding_dimension is None
            else semantics.embedding_dimension
        )
        if resolved_dimension is None:
            raise ContractError(
                "embedding_dimension is required for an unrecognized Qwen model identifier"
            )
        if known_dimension is not None and resolved_dimension != known_dimension:
            raise ContractError(
                f"model {semantics.model_id!r} has official dimension {known_dimension}, "
                f"not {resolved_dimension}"
            )
        self.model = model
        self.model_id = semantics.model_id
        self.eos_token_id = semantics.eos_token_id
        self.template_fingerprint = semantics.template_fingerprint
        self.instruction = semantics.instruction
        self.embedding_dimension = resolved_dimension
        self._capabilities = ModelCapabilities(
            supports_text=True,
            supports_image=True,
            supports_video=True,
            supports_audio=False,
            supports_mixed_modal=True,
            supports_global_embedding=True,
            supports_token_embedding=False,
            supports_hidden_states=True,
            supports_mrl=True,
            supports_instruction=True,
            supports_multi_embedding_tokens=False,
            supports_native_normalization=True,
            supports_vit_lora=True,
            supports_aligner_lora=True,
            supports_llm_lora=True,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return the Qwen3-VL-Embedding capability declaration."""
        return self._capabilities

    def prepare_inputs(
        self,
        batch: EmbeddingBatch,
        mode: AdapterMode,
    ) -> Mapping[str, Any]:
        """Check prompt provenance and return backend-ready tensors."""
        if mode not in {"train", "eval", "inference"}:
            raise ContractError(f"unsupported adapter mode: {mode!r}")
        observed_fingerprint = batch.metadata.extras.get("template_fingerprint")
        if observed_fingerprint != self.template_fingerprint:
            raise ContractError(
                "batch template fingerprint does not match the adapter; "
                f"expected {self.template_fingerprint!r}, got {observed_fingerprint!r}"
            )
        instruction = batch.metadata.extras.get("instruction")
        if instruction != self.instruction:
            raise ContractError(
                "batch instruction does not match the adapter; "
                f"expected {self.instruction!r}, got {instruction!r}"
            )
        _validate_token_inputs(batch.model_inputs)
        return dict(batch.model_inputs)

    def forward_representations(
        self,
        inputs: Mapping[str, Any],
        request: RepresentationRequest,
    ) -> EmbeddingOutput:
        """Run Qwen and extract the official last-EOS representation."""
        if request.representation is not RepresentationType.GLOBAL:
            raise CapabilityError(
                "Qwen3-VL token and hybrid representations are not stable in MIRA 0.1"
            )
        self.capabilities.require(
            Capability.GLOBAL_EMBEDDING,
            context="Qwen representation request",
        )
        _validate_token_inputs(inputs)
        input_ids = cast("Tensor", inputs["input_ids"])
        attention_mask = cast("Tensor", inputs["attention_mask"])
        _validate_eos(input_ids, attention_mask, self.eos_token_id)

        model_inputs = {name: value for name, value in inputs.items() if name != "labels"}
        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = _extract_hidden_states(outputs)
        final_hidden = hidden_states[-1]
        embeddings = last_token_pool(final_hidden, attention_mask)
        if embeddings.shape[1] != self.embedding_dimension:
            raise InvalidEmbeddingError(
                f"Qwen output dimension {embeddings.shape[1]} does not match declared "
                f"dimension {self.embedding_dimension}"
            )
        if request.dimension is not None and request.dimension > self.embedding_dimension:
            raise InvalidEmbeddingError(
                f"requested embedding dimension {request.dimension} exceeds Qwen output "
                f"dimension {self.embedding_dimension}"
            )
        if request.dimension is not None:
            embeddings = (
                truncate_and_normalize(embeddings, request.dimension)
                if request.normalize
                else embeddings[:, : request.dimension]
            )
        elif request.normalize:
            embeddings = normalize_embeddings(embeddings)

        requested_layers: dict[int, Tensor] | None = None
        if request.hidden_layers:
            requested_layers = {}
            for layer in request.hidden_layers:
                try:
                    layer_hidden = hidden_states[layer]
                except IndexError as error:
                    raise InvalidEmbeddingError(
                        f"requested hidden layer {layer} is unavailable; "
                        f"model returned {len(hidden_states)} layers"
                    ) from error
                layer_embedding = last_token_pool(layer_hidden, attention_mask)
                if request.dimension is not None:
                    layer_embedding = layer_embedding[:, : request.dimension]
                if request.normalize:
                    layer_embedding = normalize_embeddings(layer_embedding)
                requested_layers[layer] = layer_embedding

        return EmbeddingOutput(
            global_embeddings=embeddings,
            layer_embeddings=requested_layers,
            normalized=request.normalize,
            auxiliary={
                "instruction": self.instruction,
                "model_id": self.model_id,
                "pooling": "last_eos",
                "template_fingerprint": self.template_fingerprint,
            },
        )

    def get_parameter_groups(self, tuning: TuningSpec) -> list[ParameterGroupSpec]:
        """Resolve explicit component prefixes into disjoint parameter groups."""
        named_parameters = tuple(self.model.named_parameters())
        claimed: set[int] = set()
        groups: list[ParameterGroupSpec] = []
        for component, learning_rate in tuning.component_learning_rates.items():
            selected = tuple(
                parameter
                for name, parameter in named_parameters
                if parameter.requires_grad and _belongs_to_component(name, component)
            )
            if not selected:
                raise ContractError(f"tuning component {component!r} matched no model parameters")
            identities = {id(parameter) for parameter in selected}
            overlap = identities.intersection(claimed)
            if overlap:
                raise ContractError(
                    f"tuning component {component!r} overlaps an earlier parameter group"
                )
            claimed.update(identities)
            for parameter in selected:
                parameter.requires_grad = True
            groups.append(
                ParameterGroupSpec(
                    name=component,
                    parameters=selected,
                    learning_rate=learning_rate,
                    weight_decay=tuning.weight_decay,
                )
            )
        for parameter in self.model.parameters():
            if id(parameter) not in claimed:
                parameter.requires_grad = False
        return groups

    def export(self, output_dir: Path, export_config: ExportConfig) -> ExportManifest:
        """Export model files plus the exact MIRA inference contract."""
        if export_config.merge_adapter:
            raise CapabilityError(
                "adapter merging must be performed explicitly by the PEFT bridge before export"
            )
        destination = output_dir.resolve()
        if destination.exists() and any(destination.iterdir()):
            raise ContractError(f"export destination is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        save_pretrained = getattr(self.model, "save_pretrained", None)
        if not callable(save_pretrained):
            raise CapabilityError("the loaded Qwen model does not expose save_pretrained")
        save_pretrained(destination, safe_serialization=export_config.safe_serialization)
        files = tuple(sorted(path.name for path in destination.iterdir() if path.is_file()))
        if not files:
            raise ContractError("Qwen save_pretrained produced no export files")
        manifest = ExportManifest(
            model_id=self.model_id,
            pooling="last_eos",
            instruction=self.instruction,
            normalization="l2_after_optional_mrl_truncation",
            embedding_dimension=self.embedding_dimension,
            template_fingerprint=self.template_fingerprint,
            files=files,
            metadata=dict(export_config.metadata),
        )
        (destination / "mira-export.json").write_text(
            json.dumps(
                {
                    "model_id": manifest.model_id,
                    "pooling": manifest.pooling,
                    "instruction": manifest.instruction,
                    "normalization": manifest.normalization,
                    "embedding_dimension": manifest.embedding_dimension,
                    "template_fingerprint": manifest.template_fingerprint,
                    "files": manifest.files,
                    "metadata": canonicalize_json_value(
                        manifest.metadata,
                        context="Qwen export manifest metadata",
                    ),
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest


def _validate_token_inputs(inputs: Mapping[str, Any]) -> None:
    for key in ("input_ids", "attention_mask"):
        if key not in inputs or not isinstance(inputs[key], Tensor):
            raise ContractError(f"Qwen inputs require tensor field {key!r}")
    input_ids = cast("Tensor", inputs["input_ids"])
    attention_mask = cast("Tensor", inputs["attention_mask"])
    if input_ids.ndim != _TOKEN_MATRIX_RANK or input_ids.dtype not in {
        torch.int32,
        torch.int64,
    }:
        raise ContractError("input_ids must be an integer [batch, sequence] tensor")
    if tuple(attention_mask.shape) != tuple(input_ids.shape):
        raise ContractError("attention_mask shape must match input_ids")
    if input_ids.device != attention_mask.device:
        raise ContractError("input_ids and attention_mask must use the same device")


def _belongs_to_component(parameter_name: str, component: str) -> bool:
    normalized = f".{parameter_name}."
    direct = parameter_name == component or parameter_name.startswith(f"{component}.")
    markers = _COMPONENT_MARKERS.get(component, (f".{component}.",))
    return direct or any(marker in normalized for marker in markers)


def _is_valid_embedding_dimension(value: object) -> bool:
    """Accept omission or a positive integer, keeping booleans distinct."""
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value > 0)


def _validate_eos(input_ids: Tensor, attention_mask: Tensor, eos_token_id: int) -> None:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    last_positions = torch.where(
        attention_mask.bool(), positions, torch.full_like(positions, -1)
    ).amax(dim=1)
    if bool((last_positions < 0).any().item()):
        raise ContractError("Qwen attention mask contains an empty input row")
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    observed = input_ids[rows, last_positions]
    invalid = observed != eos_token_id
    if bool(invalid.any().item()):
        invalid_rows = invalid.nonzero(as_tuple=False).flatten().tolist()
        raise ContractError(
            f"Qwen template must terminate every input with EOS token {eos_token_id}; "
            f"invalid rows: {invalid_rows}"
        )


def _extract_hidden_states(outputs: Any) -> tuple[Tensor, ...]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and isinstance(outputs, Mapping):
        hidden_states = outputs.get("hidden_states")
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        raise InvalidEmbeddingError("Qwen model did not return hidden states")
    if not all(
        isinstance(value, Tensor) and value.ndim == _HIDDEN_STATE_RANK for value in hidden_states
    ):
        raise InvalidEmbeddingError("Qwen hidden states must be rank-3 tensors")
    return tuple(hidden_states)
