"""Lifecycle and model-capability declarations used at validation boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mira.core.exceptions import CapabilityError

_MIN_MIXED_MODALITIES = 2


class Stability(StrEnum):
    """Public lifecycle state of a complete feature."""

    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    PLANNED = "planned"


class Capability(StrEnum):
    """Names of capabilities that configuration may require from a model."""

    TEXT = "supports_text"
    IMAGE = "supports_image"
    VIDEO = "supports_video"
    AUDIO = "supports_audio"
    MIXED_MODAL = "supports_mixed_modal"
    GLOBAL_EMBEDDING = "supports_global_embedding"
    TOKEN_EMBEDDING = "supports_token_embedding"  # noqa: S105 - capability name
    HIDDEN_STATES = "supports_hidden_states"
    MRL = "supports_mrl"
    INSTRUCTION = "supports_instruction"
    MULTI_EMBEDDING_TOKENS = "supports_multi_embedding_tokens"
    NATIVE_NORMALIZATION = "supports_native_normalization"
    VIT_LORA = "supports_vit_lora"
    ALIGNER_LORA = "supports_aligner_lora"
    LLM_LORA = "supports_llm_lora"


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Declare model behavior without inspecting a model name in core code.

    Configuration validators call :meth:`require` before model construction.
    Impossible declarations raise :class:`CapabilityError` immediately.
    """

    supports_text: bool = False
    supports_image: bool = False
    supports_video: bool = False
    supports_audio: bool = False
    supports_mixed_modal: bool = False
    supports_global_embedding: bool = False
    supports_token_embedding: bool = False
    supports_hidden_states: bool = False
    supports_mrl: bool = False
    supports_instruction: bool = False
    supports_multi_embedding_tokens: bool = False
    supports_native_normalization: bool = False
    supports_vit_lora: bool = False
    supports_aligner_lora: bool = False
    supports_llm_lora: bool = False

    def __post_init__(self) -> None:
        """Reject internally contradictory capability declarations."""
        modality_count = sum(
            (self.supports_text, self.supports_image, self.supports_video, self.supports_audio)
        )
        if self.supports_mixed_modal and modality_count < _MIN_MIXED_MODALITIES:
            raise CapabilityError(
                "supports_mixed_modal requires at least two supported input modalities"
            )
        if self.supports_mrl and not self.supports_global_embedding:
            raise CapabilityError("supports_mrl requires supports_global_embedding")
        if self.supports_multi_embedding_tokens and not self.supports_token_embedding:
            raise CapabilityError(
                "supports_multi_embedding_tokens requires supports_token_embedding"
            )
        if self.supports_native_normalization and not (
            self.supports_global_embedding or self.supports_token_embedding
        ):
            raise CapabilityError(
                "supports_native_normalization requires an embedding output capability"
            )

    def supports(self, capability: Capability) -> bool:
        """Return whether one named capability is declared."""
        return bool(getattr(self, capability.value))

    def require(self, *capabilities: Capability, context: str = "configuration") -> None:
        """Fail if any requested capability is unavailable.

        Args:
            *capabilities: Capabilities needed by the pending operation.
            context: Human-readable configuration or operation name.

        Raises:
            CapabilityError: Lists every unsupported capability in one message.
        """
        missing = tuple(
            capability.value for capability in capabilities if not self.supports(capability)
        )
        if missing:
            names = ", ".join(missing)
            raise CapabilityError(f"{context} requires unsupported model capabilities: {names}")
