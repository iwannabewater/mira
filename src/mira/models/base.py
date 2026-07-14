"""Backend-neutral model-adapter protocol and export contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from mira.core.serialization import canonicalize_json_value

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from mira.core.capabilities import ModelCapabilities
    from mira.core.contracts import EmbeddingBatch, EmbeddingOutput, RepresentationRequest

AdapterMode = Literal["train", "eval", "inference"]


@dataclass(frozen=True, slots=True)
class ParameterGroupSpec:
    """A named optimizer parameter group with auditable component ownership."""

    name: str
    parameters: tuple[Tensor, ...]
    learning_rate: float
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        """Validate optimizer values and reject empty groups."""
        if not self.name.strip():
            raise ValueError("parameter group name must not be empty")
        if not self.parameters:
            raise ValueError(f"parameter group {self.name!r} must contain parameters")
        if not _is_finite_number(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("parameter group learning rate must be a finite positive number")
        if not _is_finite_number(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("parameter group weight decay must be a finite non-negative number")


@dataclass(frozen=True, slots=True)
class TuningSpec:
    """Explicit learning rates keyed by adapter-owned component names."""

    component_learning_rates: Mapping[str, float]
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        """Validate and detach optimizer settings before model mutation."""
        if not isinstance(self.component_learning_rates, Mapping):
            raise TypeError("component_learning_rates must be a mapping")
        normalized: dict[str, float] = {}
        for component, learning_rate in self.component_learning_rates.items():
            if not isinstance(component, str) or not component.strip():
                raise ValueError("tuning component names must be non-empty strings")
            if not _is_finite_number(learning_rate) or learning_rate <= 0:
                raise ValueError(
                    f"learning rate for component {component!r} must be a finite positive number"
                )
            normalized[component] = float(learning_rate)
        if not _is_finite_number(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("tuning weight decay must be a finite non-negative number")
        object.__setattr__(self, "component_learning_rates", MappingProxyType(normalized))
        object.__setattr__(self, "weight_decay", float(self.weight_decay))


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Portable export choices that affect artifact semantics."""

    safe_serialization: bool = True
    merge_adapter: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject ambiguous options and freeze canonical JSON metadata."""
        if type(self.safe_serialization) is not bool:
            raise ValueError("safe_serialization must be boolean")
        if type(self.merge_adapter) is not bool:
            raise ValueError("merge_adapter must be boolean")
        object.__setattr__(
            self,
            "metadata",
            _canonical_metadata(self.metadata, context="export metadata"),
        )


@dataclass(frozen=True, slots=True)
class ExportManifest:
    """Inference semantics required to reload an exported adapter."""

    model_id: str
    pooling: str
    instruction: str
    normalization: str
    embedding_dimension: int
    template_fingerprint: str
    files: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze all inference semantics before serialization."""
        required_strings = {
            "model_id": self.model_id,
            "pooling": self.pooling,
            "instruction": self.instruction,
            "normalization": self.normalization,
            "template_fingerprint": self.template_fingerprint,
        }
        invalid_strings = sorted(
            name
            for name, value in required_strings.items()
            if not isinstance(value, str) or not value.strip()
        )
        if invalid_strings:
            raise ValueError(
                "export manifest fields must be non-empty strings: " + ", ".join(invalid_strings)
            )
        if type(self.embedding_dimension) is not int or self.embedding_dimension <= 0:
            raise ValueError("export manifest embedding_dimension must be a positive integer")
        object.__setattr__(self, "files", _canonical_export_files(self.files))
        object.__setattr__(
            self,
            "metadata",
            _canonical_metadata(self.metadata, context="export manifest metadata"),
        )


class EmbeddingModelAdapter(Protocol):
    """Uniform interface implemented by MIRA model adapters."""

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return capability declarations used during config validation."""

    def prepare_inputs(
        self,
        batch: EmbeddingBatch,
        mode: AdapterMode,
    ) -> Mapping[str, Any]:
        """Validate and prepare one structured batch for the underlying model."""

    def forward_representations(
        self,
        inputs: Mapping[str, Any],
        request: RepresentationRequest,
    ) -> EmbeddingOutput:
        """Return representations whose semantics match ``request``."""

    def get_parameter_groups(self, tuning: TuningSpec) -> list[ParameterGroupSpec]:
        """Return disjoint, named trainable parameter groups."""

    def export(self, output_dir: Path, export_config: ExportConfig) -> ExportManifest:
        """Persist weights and all inference semantics."""


def _is_finite_number(value: object) -> bool:
    """Return whether a value is a finite built-in number but not a boolean."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _canonical_metadata(value: object, *, context: str) -> Mapping[str, Any]:
    """Return an immutable, detached canonical JSON object."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    try:
        canonical = canonicalize_json_value(value, context=context)
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    return cast("Mapping[str, Any]", _freeze_json_value(canonical))


def _freeze_json_value(value: Any) -> Any:
    """Recursively freeze a value already normalized to canonical JSON types."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _canonical_export_files(value: object) -> tuple[str, ...]:
    """Validate portable, unique relative artifact paths."""
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("export manifest files must be a non-empty sequence")
    files: list[str] = []
    for file_name in value:
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError("export manifest files must contain non-empty strings")
        path = PurePosixPath(file_name)
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or "\\" in file_name
            or path.as_posix() != file_name
        ):
            raise ValueError(f"export manifest files contain invalid path {file_name!r}")
        files.append(file_name)
    if len(set(files)) != len(files):
        raise ValueError("export manifest files must be unique")
    return tuple(files)
