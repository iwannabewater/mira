"""Validated loading and preprocessing values for Qwen3-VL-Embedding inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

from mira.core.exceptions import ConfigurationValidationError
from mira.models.qwen3_vl_embedding_audit import AUDITED_QWEN3_VL_EMBEDDING_TARGETS

Qwen3VLEmbeddingTemplateProfile: TypeAlias = Literal["official", "mira_v1"]
Qwen3VLEmbeddingDType: TypeAlias = Literal["auto", "float32", "float16", "bfloat16"]
Qwen3VLEmbeddingAttention: TypeAlias = Literal["auto", "eager", "sdpa", "flash_attention_2"]

_DTYPES = frozenset({"auto", "float32", "float16", "bfloat16"})
_ATTENTION = frozenset({"auto", "eager", "sdpa", "flash_attention_2"})
_TEMPLATE_PROFILES = frozenset({"official", "mira_v1"})
_MAX_FPS = 60.0


@dataclass(frozen=True, slots=True)
class Qwen3VLEmbeddingLoadSpec:
    """Complete, immutable construction contract for one embedding runtime."""

    model_id: str
    revision: str
    embedding_dimension: int
    device: str = "auto"
    dtype: Qwen3VLEmbeddingDType = "auto"
    attention: Qwen3VLEmbeddingAttention = "sdpa"
    image_max_tokens: int = 768
    max_length: int = 8192
    max_frames: int = 64
    fps: float = 1.0
    instruction: str = "Represent the user's input."
    template_profile: Qwen3VLEmbeddingTemplateProfile = "official"

    def __post_init__(self) -> None:
        """Reject mutable identities, ambiguous devices, and silent truncation values."""
        if (
            type(self.model_id) is not str
            or self.model_id not in AUDITED_QWEN3_VL_EMBEDDING_TARGETS
        ):
            raise ConfigurationValidationError(
                "Qwen3-VL-Embedding model_id must be one of the audited model IDs"
            )
        target = AUDITED_QWEN3_VL_EMBEDDING_TARGETS[self.model_id]
        if type(self.revision) is not str or self.revision != target.revision:
            raise ConfigurationValidationError(
                f"Qwen3-VL-Embedding {self.model_id!r} requires immutable revision "
                f"{target.revision}"
            )
        if (
            type(self.embedding_dimension) is not int
            or self.embedding_dimension != target.embedding_dimension
        ):
            raise ConfigurationValidationError(
                f"Qwen3-VL-Embedding {self.model_id!r} requires embedding_dimension "
                f"{target.embedding_dimension}"
            )
        if type(self.device) is not str or not self.device or self.device != self.device.strip():
            raise ConfigurationValidationError("Qwen3-VL-Embedding device must be canonical")
        _require_choice(self.dtype, _DTYPES, "unsupported Qwen3-VL-Embedding dtype")
        _require_choice(
            self.attention,
            _ATTENTION,
            "unsupported Qwen3-VL-Embedding attention implementation",
        )
        _require_choice(
            self.template_profile,
            _TEMPLATE_PROFILES,
            "unsupported Qwen3-VL-Embedding template profile",
        )
        for name, value in (
            ("image_max_tokens", self.image_max_tokens),
            ("max_length", self.max_length),
            ("max_frames", self.max_frames),
        ):
            if type(value) is not int or value <= 0:
                raise ConfigurationValidationError(
                    f"Qwen3-VL-Embedding {name} must be a positive integer"
                )
        if type(self.fps) is not float or not 0.0 < self.fps <= _MAX_FPS:
            raise ConfigurationValidationError("Qwen3-VL-Embedding fps must be a float in (0, 60]")
        if (
            type(self.instruction) is not str
            or not self.instruction
            or self.instruction != self.instruction.strip()
        ):
            raise ConfigurationValidationError("Qwen3-VL-Embedding instruction must be canonical")


def resolve_device(name: str) -> torch.device:
    """Resolve the documented auto policy or fail for an unavailable explicit device."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as error:
        raise ConfigurationValidationError(f"invalid Qwen3-VL-Embedding device {name!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigurationValidationError("requested CUDA device is unavailable")
    if (
        device.type == "cuda"
        and device.index is not None
        and (device.index < 0 or device.index >= torch.cuda.device_count())
    ):
        raise ConfigurationValidationError(
            f"requested CUDA device index {device.index} is unavailable"
        )
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ConfigurationValidationError("requested MPS device is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ConfigurationValidationError(
            f"unsupported Qwen3-VL-Embedding device type {device.type!r}"
        )
    if device.type in {"cpu", "mps"} and device.index is not None:
        raise ConfigurationValidationError(
            f"Qwen3-VL-Embedding device {device.type!r} must not include an index"
        )
    return device


def resolve_dtype(name: Qwen3VLEmbeddingDType, device: torch.device) -> torch.dtype:
    """Resolve the documented auto policy without retrying failed precision modes."""
    if name == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]
    if device.type == "mps" and dtype is torch.bfloat16:
        raise ConfigurationValidationError(
            "bfloat16 is not a supported Qwen3-VL-Embedding MPS precision"
        )
    return dtype


def _require_choice(value: object, choices: frozenset[str], message: str) -> None:
    if type(value) is not str or value not in choices:
        raise ConfigurationValidationError(message)
