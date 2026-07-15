"""Pydantic v2 schema for fully resolved MIRA configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal, Never, Self, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from mira.core.capabilities import Capability, ModelCapabilities
from mira.core.exceptions import CapabilityError
from mira.models.qwen3_vl_embedding_audit import AUDITED_QWEN3_VL_EMBEDDING_TARGETS

CONFIG_SCHEMA_VERSION = 1
_WEIGHT_TOLERANCE = 1e-6
_METRIC_PATTERN = re.compile(r"^(recall|precision|hitrate|mrr|ndcg)@[1-9][0-9]*$|^map$")


class _StrictModel(BaseModel):
    """Forbid unknown configuration keys at every nesting level."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_ambiguous_scalar_inputs(cls, value: Any) -> Any:
        """Reject ambiguous scalar types and semantic whitespace before parsing."""
        if not isinstance(value, Mapping):
            return value
        _reject_edge_whitespace(value)
        for field_name, field in cls.model_fields.items():
            if field_name not in value:
                continue
            field_value = value[field_name]
            if field_value is None:
                continue
            annotation = field.annotation
            if _annotation_contains(annotation, bool):
                if type(field_value) is not bool:
                    raise ValueError(f"{field_name} must be a boolean")
            elif isinstance(field_value, bool) and (
                _annotation_contains(annotation, int) or _annotation_contains(annotation, float)
            ):
                raise ValueError(f"{field_name} must be numeric, not boolean")
            elif isinstance(field_value, float) and _annotation_contains(annotation, int):
                raise ValueError(f"{field_name} must be an integer")
            elif (
                _annotation_contains(annotation, float)
                and not isinstance(field_value, (Mapping, list, tuple))
                and type(field_value) not in (int, float)
            ):
                raise ValueError(f"{field_name} must be an integer or float")
            enum_type = _string_enum_annotation(annotation)
            if enum_type is not None and not isinstance(field_value, (str, enum_type)):
                raise ValueError(f"{field_name} must be a string enum value")
        return value


class BackendType(StrEnum):
    """Stable training backends accepted by schema version 1."""

    MS_SWIFT = "ms_swift"


class ModelAdapterType(StrEnum):
    """Implemented model adapter identifiers accepted by this release."""

    QWEN3_VL_EMBEDDING = "qwen3_vl_embedding"


class RepresentationKind(StrEnum):
    """Configured representation family."""

    GLOBAL = "global"
    MULTI_VECTOR = "multi_vector"
    HYBRID = "hybrid"


class TuningMethod(StrEnum):
    """Implemented parameter-tuning strategies."""

    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"
    DORA = "dora"
    RS_LORA = "rs_lora"
    ADAPTER_ONLY = "adapter_only"
    PROJECTION_ONLY = "projection_only"


class SamplerType(StrEnum):
    """Implemented sampler contracts exposed by stable configuration."""

    NO_DUPLICATE = "no_duplicate"
    MODALITY_BALANCED_NO_DUPLICATE = "modality_balanced_no_duplicate"


class CrossDeviceGradient(StrEnum):
    """Gradient semantics for cross-device candidate gathering."""

    NONE = "none"
    DETACHED_REMOTE = "detached_remote"
    AUTOGRAD = "autograd"


class DistributedStrategy(StrEnum):
    """Configured training distribution strategies."""

    SINGLE = "single"
    DDP = "ddp"
    DEEPSPEED_ZERO2 = "deepspeed_zero2"
    DEEPSPEED_ZERO3 = "deepspeed_zero3"
    FSDP = "fsdp"


class ProjectConfig(_StrictModel):
    """Experiment identity and deterministic seed."""

    name: StrictStr = Field(min_length=1)
    seed: StrictInt = Field(ge=0, le=2**32 - 1)


class BackendConfig(_StrictModel):
    """Backend selection kept separate from model and objective semantics."""

    type: BackendType


class RepresentationConfig(_StrictModel):
    """Shape, pooling, and normalization semantics requested from the adapter."""

    type: RepresentationKind
    pooling: StrictStr = Field(min_length=1)
    normalize: StrictBool


class LoraConfig(_StrictModel):
    """LoRA hyperparameters validated before backend construction."""

    rank: StrictInt = Field(gt=0)
    alpha: StrictFloat = Field(gt=0, allow_inf_nan=False)
    dropout: StrictFloat = Field(ge=0, lt=1, allow_inf_nan=False)
    target_modules: StrictStr | tuple[StrictStr, ...]

    @field_validator("target_modules")
    @classmethod
    def _target_modules_are_nonempty(cls, value: str | tuple[str, ...]) -> str | tuple[str, ...]:
        if isinstance(value, str):
            if not value:
                raise ValueError("target_modules must not be empty")
        elif not value or any(not module for module in value):
            raise ValueError("target_modules must contain non-empty module names")
        return value


class ComponentTuningConfig(_StrictModel):
    """Trainability and optional component-specific learning rate."""

    trainable: StrictBool
    learning_rate: StrictFloat | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _learning_rate_matches_trainability(self) -> Self:
        if self.trainable and self.learning_rate is None:
            raise ValueError("a trainable component requires learning_rate")
        if not self.trainable and self.learning_rate is not None:
            raise ValueError("a frozen component cannot declare learning_rate")
        return self


class TuningComponentsConfig(_StrictModel):
    """Independently controlled model component groups."""

    vision: ComponentTuningConfig = Field(
        default_factory=lambda: ComponentTuningConfig(trainable=False)
    )
    aligner: ComponentTuningConfig = Field(
        default_factory=lambda: ComponentTuningConfig(trainable=False)
    )
    language_model: ComponentTuningConfig = Field(
        default_factory=lambda: ComponentTuningConfig(trainable=False)
    )
    embedding_projection: ComponentTuningConfig = Field(
        default_factory=lambda: ComponentTuningConfig(trainable=False)
    )
    identity_head: ComponentTuningConfig = Field(
        default_factory=lambda: ComponentTuningConfig(trainable=False)
    )
    late_interaction_adapter: ComponentTuningConfig = Field(
        default_factory=lambda: ComponentTuningConfig(trainable=False)
    )
    generative_matching_head: ComponentTuningConfig = Field(
        default_factory=lambda: ComponentTuningConfig(trainable=False)
    )


class TuningConfig(_StrictModel):
    """Parameter-efficient or full-tuning configuration."""

    method: TuningMethod
    lora: LoraConfig | None = None
    components: TuningComponentsConfig

    @model_validator(mode="after")
    def _validate_method_payload(self) -> Self:
        unavailable = {
            TuningMethod.QLORA,
            TuningMethod.ADAPTER_ONLY,
            TuningMethod.PROJECTION_ONLY,
        }
        if self.method in unavailable:
            raise ValueError(
                f"tuning method {self.method.value} is planned but not implemented in "
                "schema version 1"
            )
        lora_methods = {
            TuningMethod.LORA,
            TuningMethod.DORA,
            TuningMethod.RS_LORA,
        }
        if self.method in lora_methods and self.lora is None:
            raise ValueError(f"tuning method {self.method.value} requires a lora section")
        if self.method not in lora_methods and self.lora is not None:
            raise ValueError(f"tuning method {self.method.value} cannot include a lora section")
        if not any(
            component.trainable
            for component in (
                self.components.vision,
                self.components.aligner,
                self.components.language_model,
                self.components.embedding_projection,
                self.components.identity_head,
                self.components.late_interaction_adapter,
                self.components.generative_matching_head,
            )
        ):
            raise ValueError("tuning.components must make at least one component trainable")
        return self


class ModelConfig(_StrictModel):
    """Model adapter, representation, and tuning selection."""

    name_or_path: StrictStr = Field(min_length=1)
    revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    licenses: tuple[StrictStr, ...] = Field(min_length=1)
    adapter: ModelAdapterType
    hidden_size: StrictInt | None = Field(default=None, gt=0)
    representation: RepresentationConfig
    tuning: TuningConfig

    @field_validator("licenses")
    @classmethod
    def _licenses_are_unique_and_nonempty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(not license_name for license_name in value):
            raise ValueError("model licenses must be unique non-empty identifiers")
        return value

    @model_validator(mode="after")
    def _qwen_target_is_immutable_and_audited(self) -> Self:
        expected = AUDITED_QWEN3_VL_EMBEDDING_TARGETS.get(self.name_or_path)
        if self.adapter is ModelAdapterType.QWEN3_VL_EMBEDDING and expected is None:
            raise ValueError("qwen3_vl_embedding accepts only the two audited model IDs")
        if expected is not None:
            if (
                self.revision != expected.revision
                or self.hidden_size != expected.embedding_dimension
            ):
                raise ValueError(
                    f"model {self.name_or_path!r} requires revision {expected.revision} "
                    f"and hidden_size {expected.embedding_dimension}"
                )
            if self.licenses != expected.licenses:
                raise ValueError(
                    "audited Qwen3-VL-Embedding targets require licenses: "
                    f"{list(expected.licenses)}"
                )
        return self


class DatasetSourceConfig(_StrictModel):
    """One weighted training dataset and its retrieval task semantics."""

    name: StrictStr = Field(min_length=1)
    path: StrictStr = Field(min_length=1)
    task_id: StrictStr = Field(min_length=1)
    licenses: tuple[StrictStr, ...] = Field(min_length=1)
    weight: StrictFloat = Field(gt=0, le=1, allow_inf_nan=False)
    negative_semantics: StrictStr | None = None

    @field_validator("licenses")
    @classmethod
    def _licenses_are_explicit(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(not item for item in value):
            raise ValueError("dataset licenses must be unique non-empty identifiers")
        return value


class ValidationDataConfig(_StrictModel):
    """Validation dataset source."""

    path: StrictStr = Field(min_length=1)
    name: StrictStr | None = None
    task_id: StrictStr | None = None
    licenses: tuple[StrictStr, ...] = Field(min_length=1)

    @field_validator("licenses")
    @classmethod
    def _licenses_are_explicit(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(not item for item in value):
            raise ValueError("validation dataset licenses must be unique non-empty identifiers")
        return value


class SamplerConfig(_StrictModel):
    """Global-batch sampling and duplicate-exclusion rules."""

    type: SamplerType
    homogeneous_global_batch: StrictBool = True
    duplicate_keys: tuple[StrictStr, ...] = ()

    @field_validator("duplicate_keys")
    @classmethod
    def _duplicate_keys_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("sampler duplicate_keys must be unique")
        if any(not key for key in value):
            raise ValueError("sampler duplicate_keys must be non-empty strings")
        return value

    @model_validator(mode="after")
    def _no_duplicate_sampler_has_rules(self) -> Self:
        if self.type is SamplerType.MODALITY_BALANCED_NO_DUPLICATE and not self.duplicate_keys:
            raise ValueError("modality_balanced_no_duplicate requires duplicate_keys")
        return self


class MediaConfig(_StrictModel):
    """Startup validation and visual token budget."""

    image_max_tokens: StrictInt = Field(gt=0)
    validate_on_startup: StrictBool = True


class DataConfig(_StrictModel):
    """Training, validation, sampling, and media configuration."""

    train: tuple[DatasetSourceConfig, ...] = Field(min_length=1)
    validation: ValidationDataConfig
    sampler: SamplerConfig
    media: MediaConfig

    @model_validator(mode="after")
    def _validate_dataset_mixture(self) -> Self:
        names = tuple(dataset.name for dataset in self.train)
        if len(set(names)) != len(names):
            raise ValueError("data.train dataset names must be unique")
        weight_sum = math.fsum(dataset.weight for dataset in self.train)
        if not math.isclose(weight_sum, 1.0, rel_tol=0, abs_tol=_WEIGHT_TOLERANCE):
            raise ValueError(f"data.train weights must sum to 1.0; got {weight_sum:.12g}")
        return self


class TemperatureConfig(_StrictModel):
    """Fixed or learnable contrastive temperature and safe bounds."""

    type: Literal["fixed", "learnable"]
    value: StrictFloat = Field(gt=0, le=1, allow_inf_nan=False)
    minimum: StrictFloat | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    maximum: StrictFloat | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    learning_rate: StrictFloat | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_temperature_mode(self) -> Self:
        if self.type == "fixed" and (
            self.minimum is not None or self.maximum is not None or self.learning_rate is not None
        ):
            raise ValueError("fixed temperature cannot declare learnable bounds or learning_rate")
        if self.type == "learnable":
            if self.minimum is None or self.maximum is None or self.learning_rate is None:
                raise ValueError(
                    "learnable temperature requires minimum, maximum, and learning_rate"
                )
            if self.minimum >= self.maximum:
                raise ValueError("temperature minimum must be less than maximum")
            if not self.minimum <= self.value <= self.maximum:
                raise ValueError("temperature value must lie within learnable bounds")
        return self


class NegativesConfig(_StrictModel):
    """Explicit, in-batch, and cross-device negative semantics."""

    in_batch: StrictBool
    cross_device: StrictBool
    cross_device_gradient: CrossDeviceGradient
    explicit_hard_negatives: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _validate_cross_device_gradient(self) -> Self:
        if self.cross_device and self.cross_device_gradient is CrossDeviceGradient.NONE:
            raise ValueError("cross_device=true requires detached_remote or autograd gradient mode")
        if not self.cross_device and self.cross_device_gradient is not CrossDeviceGradient.NONE:
            raise ValueError("cross_device=false requires cross_device_gradient=none")
        return self


class FalseNegativeMaskConfig(_StrictModel):
    """False-negative exclusion rules applied before loss construction."""

    enabled: StrictBool
    metadata_rules: StrictBool = True
    similarity_margin: StrictFloat | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)

    @model_validator(mode="after")
    def _enabled_mask_has_a_rule(self) -> Self:
        if self.enabled and not self.metadata_rules and self.similarity_margin is None:
            raise ValueError("enabled false_negative_mask requires at least one rule")
        return self


class MultiPositiveInfoNCEConfig(_StrictModel):
    """Stable multi-positive InfoNCE objective configuration."""

    name: StrictStr = Field(min_length=1)
    type: Literal["multi_positive_infonce"]
    weight: StrictFloat = Field(gt=0, allow_inf_nan=False)
    temperature: TemperatureConfig
    negatives: NegativesConfig
    false_negative_mask: FalseNegativeMaskConfig


class MatryoshkaConfig(_StrictModel):
    """MRL dimensions and unnormalized aggregation weights."""

    name: StrictStr = Field(min_length=1)
    type: Literal["matryoshka"]
    weight: StrictFloat = Field(gt=0, allow_inf_nan=False)
    dimensions: dict[StrictInt, StrictFloat]

    @field_validator("dimensions", mode="before")
    @classmethod
    def _dimensions_are_ordered(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            # Pydantic v2 deliberately propagates TypeError out of validators.
            raise ValueError(  # noqa: TRY004
                "matryoshka dimensions must be a dimension-to-weight mapping"
            )
        normalized: dict[int, Any] = {}
        for key, weight in value.items():
            dimension = _canonical_dimension(key)
            if dimension in normalized:
                raise ValueError("matryoshka dimensions must be unique")
            if isinstance(weight, bool) or type(weight) not in (int, float):
                raise ValueError("matryoshka weights must be numeric, not booleans or strings")
            normalized[dimension] = weight
        keys = list(normalized)
        if keys != sorted(keys):
            raise ValueError("matryoshka dimensions must be unique and strictly increasing")
        return normalized

    @model_validator(mode="after")
    def _validate_dimensions_and_weights(self) -> Self:
        if not self.dimensions:
            raise ValueError("matryoshka dimensions must not be empty")
        if any(dimension <= 0 for dimension in self.dimensions):
            raise ValueError("matryoshka dimensions must be positive")
        weights = tuple(self.dimensions.values())
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("matryoshka weights must be finite and non-negative")
        if not any(weight > 0 for weight in weights):
            raise ValueError("matryoshka weights cannot all be zero")
        object.__setattr__(self, "dimensions", _FrozenDimensionWeights(self.dimensions))
        return self


ObjectiveConfig = Annotated[
    MultiPositiveInfoNCEConfig | MatryoshkaConfig,
    Field(discriminator="type"),
]


class DistributedConfig(_StrictModel):
    """Runtime distribution strategy."""

    strategy: DistributedStrategy


class OptimizerConfig(_StrictModel):
    """Stable optimizer configuration."""

    type: Literal["adamw"]
    weight_decay: StrictFloat = Field(ge=0, allow_inf_nan=False)


class SchedulerConfig(_StrictModel):
    """Stable learning-rate schedule configuration."""

    type: Literal["cosine"]
    warmup_ratio: StrictFloat = Field(ge=0, lt=1, allow_inf_nan=False)


class RuntimeConfig(_StrictModel):
    """Numerical, batching, distribution, and optimization settings."""

    precision: Literal["bf16", "fp16", "fp32"]
    attention: Literal["eager", "sdpa", "flash_attention_2"]
    gradient_checkpointing: StrictBool
    padding_free: StrictBool
    train_batch_size_per_device: StrictInt = Field(gt=0)
    gradient_accumulation_steps: StrictInt = Field(gt=0)
    distributed: DistributedConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    epochs: StrictFloat = Field(gt=0, allow_inf_nan=False)


class EvaluationConfig(_StrictModel):
    """Retrieval metrics, MRL dimensions, and reporting slices."""

    metrics: tuple[StrictStr, ...] = Field(min_length=1)
    dimensions: tuple[StrictInt, ...] = Field(min_length=1)
    slices: tuple[StrictStr, ...] = ()

    @field_validator("metrics")
    @classmethod
    def _metrics_are_known_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(metric != metric.lower() for metric in value):
            raise ValueError("evaluation metrics must use canonical lowercase names")
        if any(_METRIC_PATTERN.fullmatch(metric) is None for metric in value):
            raise ValueError("unsupported evaluation metric")
        if len(set(value)) != len(value):
            raise ValueError("evaluation metrics must be unique")
        return value

    @field_validator("dimensions", mode="before")
    @classmethod
    def _evaluation_dimensions_are_ordered(cls, value: Any) -> Any:
        if not isinstance(value, (tuple, list)) or any(type(item) is not int for item in value):
            raise ValueError("evaluation dimensions must contain integers")
        if any(dimension <= 0 for dimension in value):
            raise ValueError("evaluation dimensions must be positive")
        if tuple(sorted(set(value))) != tuple(value):
            raise ValueError("evaluation dimensions must be unique and strictly increasing")
        return value

    @field_validator("slices")
    @classmethod
    def _slices_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(not item for item in value):
            raise ValueError("evaluation slices must be unique non-empty names")
        return value


_ADAPTER_CAPABILITIES: dict[ModelAdapterType, ModelCapabilities] = {
    ModelAdapterType.QWEN3_VL_EMBEDDING: ModelCapabilities(
        supports_text=True,
        supports_image=True,
        supports_video=True,
        supports_mixed_modal=True,
        supports_global_embedding=True,
        supports_hidden_states=True,
        supports_mrl=True,
        supports_instruction=True,
        supports_native_normalization=True,
        supports_vit_lora=True,
        supports_aligner_lora=True,
        supports_llm_lora=True,
    )
}


class MiraConfig(_StrictModel):
    """Complete schema-version-1 MIRA experiment configuration."""

    schema_version: Literal[1]
    project: ProjectConfig
    backend: BackendConfig
    model: ModelConfig
    data: DataConfig
    objectives: tuple[ObjectiveConfig, ...] = Field(min_length=1)
    runtime: RuntimeConfig
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def _validate_cross_field_semantics(self) -> Self:
        names = tuple(objective.name for objective in self.objectives)
        if len(set(names)) != len(names):
            raise ValueError("objective names must be unique")

        capabilities = _ADAPTER_CAPABILITIES[self.model.adapter]
        self._validate_representation(capabilities)
        self._validate_tuning_capabilities(capabilities)

        retrieval = next(
            (
                objective
                for objective in self.objectives
                if isinstance(objective, MultiPositiveInfoNCEConfig)
            ),
            None,
        )
        mrl = next(
            (objective for objective in self.objectives if isinstance(objective, MatryoshkaConfig)),
            None,
        )
        if retrieval is None:
            raise ValueError("stable configuration requires a multi_positive_infonce objective")
        if sum(isinstance(item, MultiPositiveInfoNCEConfig) for item in self.objectives) > 1:
            raise ValueError("only one multi_positive_infonce objective may be configured")
        if sum(isinstance(item, MatryoshkaConfig) for item in self.objectives) > 1:
            raise ValueError("only one matryoshka objective may be configured")
        self._validate_negative_sampling(retrieval)
        self._validate_mrl(mrl, capabilities)
        return self

    def _validate_representation(self, capabilities: ModelCapabilities) -> None:
        representation = self.model.representation
        if self.model.adapter is ModelAdapterType.QWEN3_VL_EMBEDDING and (
            representation.type is not RepresentationKind.GLOBAL
            or representation.pooling != "native"
            or not representation.normalize
        ):
            raise ValueError(
                "qwen3_vl_embedding requires global representation, pooling=native, "
                "and normalize=true"
            )
        try:
            capabilities.require(Capability.GLOBAL_EMBEDDING, context="model.representation")
            if representation.pooling == "native":
                capabilities.require(Capability.NATIVE_NORMALIZATION, context="native pooling")
        except CapabilityError as error:
            raise ValueError(str(error)) from error

    def _validate_tuning_capabilities(self, capabilities: ModelCapabilities) -> None:
        tuning = self.model.tuning
        components = self.model.tuning.components
        if (
            tuning.method
            in {
                TuningMethod.LORA,
                TuningMethod.DORA,
                TuningMethod.RS_LORA,
            }
            and (components.vision.trainable or components.aligner.trainable)
            and tuning.lora is not None
            and tuning.lora.target_modules != "all-linear"
        ):
            raise ValueError(
                "trainable Qwen3-VL-Embedding vision or aligner components require "
                "model.tuning.lora.target_modules=all-linear"
            )
        planned = {
            "embedding_projection": components.embedding_projection,
            "identity_head": components.identity_head,
            "late_interaction_adapter": components.late_interaction_adapter,
            "generative_matching_head": components.generative_matching_head,
        }
        enabled_planned = tuple(name for name, value in planned.items() if value.trainable)
        if enabled_planned:
            raise ValueError(
                "Phase-1 Qwen3-VL-Embedding training does not implement planned components: "
                + ", ".join(enabled_planned)
            )
        required: list[Capability] = []
        if components.vision.trainable:
            required.append(Capability.VIT_LORA)
        if components.aligner.trainable:
            required.append(Capability.ALIGNER_LORA)
        if components.language_model.trainable:
            required.append(Capability.LLM_LORA)
        try:
            capabilities.require(*required, context="model.tuning.components")
        except CapabilityError as error:
            raise ValueError(str(error)) from error

    def _validate_negative_sampling(self, objective: MultiPositiveInfoNCEConfig) -> None:
        negatives = objective.negatives
        sampler = self.data.sampler
        if negatives.in_batch and not sampler.homogeneous_global_batch:
            raise ValueError(
                "in-batch negatives require data.sampler.homogeneous_global_batch=true"
            )
        if (
            negatives.cross_device
            and self.runtime.distributed.strategy is DistributedStrategy.SINGLE
        ):
            raise ValueError("cross-device negatives require a distributed runtime strategy")

    def _validate_mrl(
        self, objective: MatryoshkaConfig | None, capabilities: ModelCapabilities
    ) -> None:
        if objective is None:
            if len(self.evaluation.dimensions) != 1:
                raise ValueError("multiple evaluation dimensions require a matryoshka objective")
            return
        try:
            capabilities.require(Capability.MRL, context="matryoshka objective")
        except CapabilityError as error:
            raise ValueError(str(error)) from error
        dimensions = tuple(objective.dimensions)
        if self.model.hidden_size is not None and dimensions[-1] > self.model.hidden_size:
            raise ValueError(
                f"matryoshka dimension {dimensions[-1]} exceeds model.hidden_size "
                f"{self.model.hidden_size}"
            )
        unknown_dimensions = sorted(set(self.evaluation.dimensions) - set(dimensions))
        if unknown_dimensions:
            raise ValueError(
                "evaluation dimensions are not trained by matryoshka: "
                + ", ".join(str(value) for value in unknown_dimensions)
            )


def _annotation_contains(annotation: Any, target: type[Any]) -> bool:
    """Return whether a field annotation directly accepts one scalar type."""
    if annotation is target:
        return True
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        return any(type(value) is target for value in arguments)
    return any(_annotation_contains(argument, target) for argument in arguments)


def _string_enum_annotation(annotation: Any) -> type[StrEnum] | None:
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return annotation
    return next(
        (
            enum_type
            for argument in get_args(annotation)
            if (enum_type := _string_enum_annotation(argument)) is not None
        ),
        None,
    )


def _reject_edge_whitespace(value: Any, *, active: set[int] | None = None) -> None:
    if isinstance(value, str):
        if value != value.strip():
            raise ValueError("configuration strings must not have edge whitespace")
        return
    if not isinstance(value, (Mapping, list, tuple)):
        return
    ancestors = set() if active is None else active
    identity = id(value)
    if identity in ancestors:
        raise ValueError("configuration containers must not be cyclic")
    ancestors.add(identity)
    items = value.values() if isinstance(value, Mapping) else value
    for item in items:
        _reject_edge_whitespace(item, active=ancestors)
    ancestors.remove(identity)


def _canonical_dimension(value: Any) -> int:
    if isinstance(value, bool):
        # Pydantic v2 deliberately propagates TypeError out of validators.
        raise ValueError(  # noqa: TRY004
            "matryoshka dimension keys must be integers, not booleans"
        )
    if type(value) is int:
        return value
    if isinstance(value, str):
        try:
            dimension = int(value)
        except ValueError as error:
            raise ValueError("matryoshka dimension keys must be integers") from error
        if str(dimension) == value:
            return dimension
    raise ValueError("matryoshka dimension keys must be canonical integers")


class _FrozenDimensionWeights(dict[int, float]):
    """Serializable dictionary whose validated values cannot drift after checksumming."""

    def __init__(self, value: Mapping[int, float]) -> None:
        if getattr(self, "_sealed", False):
            raise TypeError("matryoshka dimensions are immutable")
        super().__init__(value)
        self._sealed = True

    def __setitem__(self, key: int, value: float) -> Never:
        raise TypeError("matryoshka dimensions are immutable")

    def __delitem__(self, key: int) -> Never:
        raise TypeError("matryoshka dimensions are immutable")

    def __ior__(self, value: Mapping[int, float]) -> Never:  # type: ignore[override, misc]
        raise TypeError("matryoshka dimensions are immutable")

    def clear(self) -> Never:
        raise TypeError("matryoshka dimensions are immutable")

    def pop(self, key: int, default: Any = None) -> Never:
        del key, default
        raise TypeError("matryoshka dimensions are immutable")

    def popitem(self) -> Never:
        raise TypeError("matryoshka dimensions are immutable")

    def setdefault(self, key: int, default: float | None = None) -> Never:
        del key, default
        raise TypeError("matryoshka dimensions are immutable")

    def update(self, *args: Any, **kwargs: float) -> Never:
        del args, kwargs
        raise TypeError("matryoshka dimensions are immutable")
