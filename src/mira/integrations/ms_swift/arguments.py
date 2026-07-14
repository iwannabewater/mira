"""Typed arguments accepted by the fixed MS-SWIFT boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mira.core.exceptions import ConfigurationValidationError
from mira.models.qwen_audit import AUDITED_QWEN_TARGETS

_TORCH_DTYPES = frozenset({"auto", "float32", "float16", "bfloat16"})
_ATTENTION_IMPLEMENTATIONS = frozenset({"auto", "eager", "sdpa", "flash_attention_2"})


@dataclass(frozen=True, slots=True)
class SwiftLoadSpec:
    """Model-loading values forwarded to ``swift.model.get_model_processor``."""

    model_id: str
    revision: str | None = None
    model_type: str = "qwen3_vl_emb"
    task_type: Literal["embedding"] = "embedding"
    use_hf: bool = True
    device_map: str | None = None
    torch_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    attention: Literal["auto", "eager", "sdpa", "flash_attention_2"] = "auto"
    image_max_tokens: int | None = None
    embedding_dimension: int | None = None
    load_model: bool = True

    def __post_init__(self) -> None:
        """Validate identifiers before importing the optional backend."""
        if type(self.model_id) is not str or self.model_id not in AUDITED_QWEN_TARGETS:
            raise ConfigurationValidationError(
                "MS-SWIFT model_id must be one of the two audited Qwen model IDs"
            )
        target = AUDITED_QWEN_TARGETS[self.model_id]
        if type(self.revision) is not str or self.revision != target.revision:
            raise ConfigurationValidationError(
                f"MS-SWIFT model {self.model_id!r} requires immutable revision {target.revision}"
            )
        if (
            type(self.embedding_dimension) is not int
            or self.embedding_dimension != target.embedding_dimension
        ):
            raise ConfigurationValidationError(
                f"MS-SWIFT model {self.model_id!r} requires embedding_dimension "
                f"{target.embedding_dimension}"
            )
        if type(self.model_type) is not str or self.model_type != "qwen3_vl_emb":
            raise ConfigurationValidationError(
                "the stable MS-SWIFT bridge only accepts model_type='qwen3_vl_emb'"
            )
        _require_string_choice(
            self.task_type,
            frozenset({"embedding"}),
            "MS-SWIFT task_type must be 'embedding'",
        )
        if type(self.use_hf) is not bool or not self.use_hf:
            raise ConfigurationValidationError("audited Qwen targets require use_hf=True")
        if self.device_map is not None and (
            type(self.device_map) is not str
            or not self.device_map
            or self.device_map != self.device_map.strip()
        ):
            raise ConfigurationValidationError("device_map must be null or a canonical string")
        _require_string_choice(
            self.torch_dtype,
            _TORCH_DTYPES,
            "unsupported MS-SWIFT torch_dtype",
        )
        _require_string_choice(
            self.attention,
            _ATTENTION_IMPLEMENTATIONS,
            "unsupported MS-SWIFT attention implementation",
        )
        if self.image_max_tokens is not None and (
            type(self.image_max_tokens) is not int or self.image_max_tokens <= 0
        ):
            raise ConfigurationValidationError("image_max_tokens must be a positive integer")
        if type(self.load_model) is not bool or not self.load_model:
            raise ConfigurationValidationError("Qwen adapter construction requires load_model=True")


@dataclass(frozen=True, slots=True)
class SwiftTemplateSpec:
    """Prompt behavior that must match across training and inference."""

    instruction: str = "Represent the user's input."
    template_type: str = "qwen3_vl_emb"
    max_length: int = 32768
    padding_side: Literal["left", "right"] = "right"
    truncation_strategy: Literal["raise"] = "raise"

    def __post_init__(self) -> None:
        """Reject silent prompt truncation and unsupported templates."""
        if (
            type(self.instruction) is not str
            or not self.instruction
            or self.instruction != self.instruction.strip()
        ):
            raise ConfigurationValidationError("embedding instruction must be a canonical string")
        if type(self.template_type) is not str or self.template_type != "qwen3_vl_emb":
            raise ConfigurationValidationError(
                "the stable Qwen bridge requires template_type='qwen3_vl_emb'"
            )
        if type(self.max_length) is not int or self.max_length <= 0:
            raise ConfigurationValidationError("template max_length must be a positive integer")
        _require_string_choice(
            self.padding_side,
            frozenset({"left", "right"}),
            "template padding_side must be 'left' or 'right'",
        )
        _require_string_choice(
            self.truncation_strategy,
            frozenset({"raise"}),
            "template truncation_strategy must be 'raise'",
        )


def _require_string_choice(value: object, choices: frozenset[str], message: str) -> None:
    if type(value) is not str or value not in choices:
        raise ConfigurationValidationError(message)
