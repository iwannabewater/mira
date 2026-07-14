"""Versioned, lossless data contracts for multimodal retrieval records.

Every :class:`MultimodalItem` validates its own placeholder/media alignment.
Consequently the query, each positive, and each negative fail independently and
the resulting error identifies the offending item.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, NoReturn, Self, TypeAlias, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from mira.core.serialization import canonicalize_json_value
from mira.data.errors import PlaceholderAlignmentError, RecordValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION: Literal[1] = 1
_PLACEHOLDER_PATTERN = re.compile(r"<(image|video|audio)>")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _reject_boolean_number(value: Any) -> Any:
    if isinstance(value, bool):
        # Pydantic propagates TypeError from validators instead of wrapping it.
        raise ValueError("numeric values must not be boolean")  # noqa: TRY004
    return value


_NonBooleanFloat: TypeAlias = Annotated[float, BeforeValidator(_reject_boolean_number)]


class _ImmutableJsonObject(dict[str, Any]):
    """A JSON object that Pydantic can serialize but callers cannot mutate."""

    @staticmethod
    def _mutation_error() -> NoReturn:
        raise TypeError("validated opaque JSON is immutable")

    def __setitem__(self, key: str, value: Any) -> None:
        del key, value
        self._mutation_error()

    def __delitem__(self, key: str) -> None:
        del key
        self._mutation_error()

    def __ior__(self, value: Any) -> Self:  # type: ignore[override,misc]
        del value
        self._mutation_error()

    def clear(self) -> None:
        self._mutation_error()

    def pop(self, key: str, default: Any = None) -> Any:
        del key, default
        self._mutation_error()

    def popitem(self) -> tuple[str, Any]:
        self._mutation_error()

    def setdefault(self, key: str, default: Any = None) -> Any:
        del key, default
        self._mutation_error()

    def update(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._mutation_error()

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        del memo
        return self


def _freeze_canonical_json(value: Any) -> Any:
    """Recursively freeze a detached value produced by the shared canonicalizer."""
    if isinstance(value, dict):
        return _ImmutableJsonObject(
            {key: _freeze_canonical_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_canonical_json(item) for item in value)
    return value


def _freeze_json_value(value: Any, *, path: str) -> Any:
    """Validate, detach, and recursively freeze one opaque JSON value."""
    try:
        canonical = canonicalize_json_value(value, context=path)
    except (TypeError, ValueError) as error:
        # Pydantic propagates TypeError from validators instead of wrapping it.
        raise ValueError(str(error)) from error
    return _freeze_canonical_json(canonical)


def _freeze_json_object(value: Mapping[str, Any], *, path: str) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", _freeze_json_value(value, path=path))


def _canonical_semantic_string(value: str, *, field: str) -> str:
    """Reject ambiguous semantic values without rewriting their input bytes."""
    if not value or value.isspace():
        raise ValueError(f"{field} must not be blank")
    if value != value.strip():
        raise ValueError(f"{field} must not have leading or trailing whitespace")
    return value


class _FrozenModel(BaseModel):
    """Common strict configuration for immutable record values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MessagePart(_FrozenModel):
    """One structured message content part.

    Unknown part attributes are retained because processor-specific structured
    content is intentionally opaque to the data contract.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    type: str
    text: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _processor_extras_are_json(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                # Pydantic propagates TypeError from validators instead of wrapping it.
                raise ValueError("message part keys must be strings")  # noqa: TRY004
            normalized[key] = (
                item
                if key in {"type", "text"}
                else _freeze_json_value(item, path=f"message part extra {key!r}")
            )
        return normalized

    @model_validator(mode="after")
    def _freeze_processor_extra_container(self) -> Self:
        extras = self.__pydantic_extra__
        if extras is not None:
            object.__setattr__(self, "__pydantic_extra__", _ImmutableJsonObject(extras))
        return self

    @field_validator("type")
    @classmethod
    def _type_is_canonical(cls, value: str) -> str:
        return _canonical_semantic_string(value, field="message part type")


class Message(_FrozenModel):
    """A conversation message accepted by multimodal model templates."""

    role: str
    content: str | tuple[MessagePart, ...]

    @field_validator("role")
    @classmethod
    def _role_is_nonempty(cls, value: str) -> str:
        return _canonical_semantic_string(value, field="message role")

    @field_validator("content")
    @classmethod
    def _content_is_nonempty(
        cls, value: str | tuple[MessagePart, ...]
    ) -> str | tuple[MessagePart, ...]:
        if not value:
            raise ValueError("message content must not be empty")
        return value

    def text_fragments(self) -> tuple[str, ...]:
        """Return textual fragments in which explicit media tags can occur."""
        if isinstance(self.content, str):
            return (self.content,)
        return tuple(part.text for part in self.content if part.text is not None)


class MediaRef(_FrozenModel):
    """A stable reference to one image, video, or audio payload."""

    uri: str
    checksum: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_string_reference(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"uri": value}
        return value

    @field_validator("uri")
    @classmethod
    def _uri_is_nonempty(cls, value: str) -> str:
        return _canonical_semantic_string(value, field="media URI")

    @field_validator("checksum")
    @classmethod
    def _checksum_is_tagged_sha256(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("media checksum must use the form sha256:<64 lowercase hex digits>")
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _freeze_json_object(value, path="media metadata")


class ObjectAnnotations(_FrozenModel):
    """Processor-agnostic object annotations associated with an item."""

    annotations: tuple[Mapping[str, Any], ...] = ()

    @field_validator("annotations")
    @classmethod
    def _annotations_are_json(
        cls, value: tuple[Mapping[str, Any], ...]
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            _freeze_json_object(annotation, path=f"object annotation {index}")
            for index, annotation in enumerate(value)
        )


class MultimodalItem(_FrozenModel):
    """One independently validated query or candidate item."""

    _MEDIA_FIELDS: ClassVar[dict[str, str]] = {
        "image": "images",
        "video": "videos",
        "audio": "audios",
    }

    item_id: str
    messages: tuple[Message, ...]
    images: tuple[MediaRef, ...] = ()
    videos: tuple[MediaRef, ...] = ()
    audios: tuple[MediaRef, ...] = ()
    objects: ObjectAnnotations | None = None
    instruction: str | None = None

    @field_validator("item_id")
    @classmethod
    def _item_id_is_nonempty(cls, value: str) -> str:
        return _canonical_semantic_string(value, field="item_id")

    @field_validator("messages")
    @classmethod
    def _messages_are_nonempty(cls, value: tuple[Message, ...]) -> tuple[Message, ...]:
        if not value:
            raise ValueError("messages must contain at least one message")
        return value

    @model_validator(mode="after")
    def _validate_placeholder_alignment(self) -> Self:
        counts = dict.fromkeys(self._MEDIA_FIELDS, 0)
        for message in self.messages:
            for fragment in message.text_fragments():
                for match in _PLACEHOLDER_PATTERN.finditer(fragment):
                    counts[match.group(1)] += 1

        for modality, field_name in self._MEDIA_FIELDS.items():
            placeholder_count = counts[modality]
            media_count = len(getattr(self, field_name))
            if placeholder_count != media_count:
                placeholder = f"<{modality}>"
                raise PlaceholderAlignmentError(
                    f"item {self.item_id!r}: found {placeholder_count} {placeholder} "
                    f"placeholder(s), but {field_name} contains {media_count} reference(s)"
                )
        return self

    def placeholder_counts(self) -> Mapping[str, int]:
        """Return explicit placeholder counts after successful validation."""
        counts = dict.fromkeys(self._MEDIA_FIELDS, 0)
        for message in self.messages:
            for fragment in message.text_fragments():
                for match in _PLACEHOLDER_PATTERN.finditer(fragment):
                    counts[match.group(1)] += 1
        return counts


class CandidateItem(MultimodalItem):
    """A positive or negative retrieval candidate."""


class RetrievalMetadata(_FrozenModel):
    """Retrieval semantics aligned to the ordered candidate sequence.

    ``relevance_labels`` and ``teacher_scores``, when supplied, follow
    ``positives + negatives`` exactly. Group-id tuples may be empty when that
    metadata is unavailable; a non-empty tuple must align one-to-one.
    """

    query_group_id: str | None = None
    positive_group_ids: tuple[str, ...] = ()
    negative_group_ids: tuple[str, ...] = ()
    modality_pair: str
    relevance_labels: tuple[_NonBooleanFloat, ...] | None = None
    teacher_scores: tuple[_NonBooleanFloat, ...] | None = None
    dataset_name: str
    category_id: str | None = None
    brand_id: str | None = None
    spu_id: str | None = None
    sku_id: str | None = None
    canonical_product_id: str | None = None
    image_hash_group: str | None = None
    cluster_id: str | None = None
    sampling_weight: _NonBooleanFloat = Field(default=1.0, gt=0.0, allow_inf_nan=False)

    @field_validator("modality_pair", "dataset_name")
    @classmethod
    def _required_strings_are_nonempty(cls, value: str) -> str:
        return _canonical_semantic_string(value, field="retrieval metadata value")

    @field_validator(
        "query_group_id",
        "category_id",
        "brand_id",
        "spu_id",
        "sku_id",
        "canonical_product_id",
        "image_hash_group",
        "cluster_id",
    )
    @classmethod
    def _optional_identifiers_are_canonical(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _canonical_semantic_string(value, field="retrieval metadata identifier")

    @field_validator("positive_group_ids", "negative_group_ids")
    @classmethod
    def _group_identifiers_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for identifier in value:
            _canonical_semantic_string(identifier, field="retrieval group identifier")
        return value

    @field_validator("relevance_labels", "teacher_scores")
    @classmethod
    def _scores_are_finite(cls, value: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if value is not None and any(not _is_finite(number) for number in value):
            raise ValueError("candidate-aligned scores must all be finite")
        return value


class EmbeddingRecord(_FrozenModel):
    """A query and its explicitly ordered positive and negative candidates."""

    schema_version: Literal[1] = SCHEMA_VERSION
    sample_id: str
    task_id: str
    query: MultimodalItem
    positives: tuple[CandidateItem, ...]
    negatives: tuple[CandidateItem, ...] = ()
    metadata: RetrievalMetadata

    @field_validator("sample_id", "task_id")
    @classmethod
    def _ids_are_nonempty(cls, value: str) -> str:
        return _canonical_semantic_string(value, field="record identifier")

    @field_validator("positives")
    @classmethod
    def _positive_set_is_nonempty(
        cls, value: tuple[CandidateItem, ...]
    ) -> tuple[CandidateItem, ...]:
        if not value:
            raise ValueError("every embedding record requires at least one positive")
        return value

    @model_validator(mode="after")
    def _validate_alignment_and_identity(self) -> Self:
        candidates = self.positives + self.negatives
        item_ids = [candidate.item_id for candidate in candidates]
        if len(set(item_ids)) != len(item_ids):
            raise RecordValidationError(
                f"record {self.sample_id!r}: candidate item_id values must be unique"
            )
        if self.query.item_id in item_ids:
            raise RecordValidationError(
                f"record {self.sample_id!r}: query item_id also appears as a candidate"
            )

        self._check_optional_alignment(
            "positive_group_ids", self.metadata.positive_group_ids, len(self.positives)
        )
        self._check_optional_alignment(
            "negative_group_ids", self.metadata.negative_group_ids, len(self.negatives)
        )
        candidate_count = len(candidates)
        if self.metadata.relevance_labels is not None:
            self._check_exact_alignment(
                "relevance_labels", self.metadata.relevance_labels, candidate_count
            )
        if self.metadata.teacher_scores is not None:
            self._check_exact_alignment(
                "teacher_scores", self.metadata.teacher_scores, candidate_count
            )

        positive_groups = set(self.metadata.positive_group_ids)
        conflicting_groups = positive_groups.intersection(self.metadata.negative_group_ids)
        if conflicting_groups:
            conflicts = ", ".join(sorted(conflicting_groups))
            raise RecordValidationError(
                f"record {self.sample_id!r}: positive and negative group IDs conflict: {conflicts}"
            )
        return self

    def _check_optional_alignment(self, name: str, values: Sequence[object], expected: int) -> None:
        if values and len(values) != expected:
            raise RecordValidationError(
                f"record {self.sample_id!r}: {name} has {len(values)} entries; expected {expected}"
            )

    def _check_exact_alignment(self, name: str, values: Sequence[object], expected: int) -> None:
        if len(values) != expected:
            raise RecordValidationError(
                f"record {self.sample_id!r}: {name} has {len(values)} entries; "
                f"expected {expected} for positives + negatives"
            )


def _is_finite(value: float) -> bool:
    return math.isfinite(value)
