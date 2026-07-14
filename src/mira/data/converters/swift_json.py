"""Lossless conversion between MIRA records and MS-SWIFT embedding JSON.

MS-SWIFT's classic embedding shape represents one positive. Callers must
therefore choose either ``expand`` (one Swift sample per positive) or
``preserve`` (MIRA's explicitly marked nested extension for a custom collator).
No conversion path selects or drops the first positive implicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, cast

from pydantic import ValidationError

from mira.data.errors import SwiftConversionError
from mira.data.schema import (
    CandidateItem,
    EmbeddingRecord,
    MediaRef,
    Message,
    MultimodalItem,
    RetrievalMetadata,
)


class MultiPositiveStrategy(StrEnum):
    """How a multi-positive MIRA record is represented for Swift."""

    EXPAND = "expand"
    PRESERVE = "preserve"


class SwiftJsonConverter:
    """Stateless converter exposing both format directions."""

    @staticmethod
    def to_swift(
        record: EmbeddingRecord,
        *,
        multi_positive: MultiPositiveStrategy | str = MultiPositiveStrategy.EXPAND,
    ) -> tuple[dict[str, Any], ...]:
        """Convert a MIRA record with an explicit multi-positive strategy."""
        return mira_to_swift(record, multi_positive=multi_positive)

    @staticmethod
    def from_swift(sample: Mapping[str, Any], **kwargs: Any) -> EmbeddingRecord:
        """Convert a Swift record, requiring missing MIRA identity as keywords."""
        return swift_to_mira(sample, **kwargs)


def mira_to_swift(
    record: EmbeddingRecord,
    *,
    multi_positive: MultiPositiveStrategy | str = MultiPositiveStrategy.EXPAND,
) -> tuple[dict[str, Any], ...]:
    """Convert one MIRA record into one or more Swift-compatible samples.

    ``expand`` is compatible with a backend limited to one positive and returns
    exactly ``len(record.positives)`` samples. ``preserve`` returns one sample
    containing nested positive fields and requires a MIRA-aware collator.
    """
    try:
        strategy = MultiPositiveStrategy(multi_positive)
    except ValueError as error:
        allowed = ", ".join(strategy.value for strategy in MultiPositiveStrategy)
        raise SwiftConversionError(
            f"unknown multi-positive strategy {multi_positive!r}; expected one of {allowed}"
        ) from error

    if strategy is MultiPositiveStrategy.PRESERVE:
        return (_preserved_sample(record),)
    return tuple(
        _expanded_sample(record, positive_index) for positive_index in range(len(record.positives))
    )


def swift_to_mira(
    sample: Mapping[str, Any],
    *,
    sample_id: str | None = None,
    task_id: str | None = None,
    dataset_name: str | None = None,
    modality_pair: str | None = None,
) -> EmbeddingRecord:
    """Convert a classic or MIRA-preserved Swift embedding sample.

    Missing MIRA-only identity and dataset fields must be supplied through the
    keyword arguments. Candidate item IDs may be deterministically generated
    from ``sample_id`` because classic Swift samples do not carry them.
    """
    mira_payload = _mapping_or_empty(sample.get("mira"), "mira")
    resolved_sample_id = _required_identifier(
        sample_id
        or _optional_string(sample.get("sample_id"), "sample_id")
        or _optional_string(mira_payload.get("source_sample_id"), "mira.source_sample_id"),
        "sample_id",
    )
    resolved_task_id = _required_identifier(
        task_id
        or _optional_string(sample.get("task_id"), "task_id")
        or _optional_string(mira_payload.get("task_id"), "mira.task_id"),
        "task_id",
    )

    raw_metadata = _mapping_or_empty(mira_payload.get("metadata"), "mira.metadata")
    metadata_payload = dict(raw_metadata)
    metadata_payload.setdefault(
        "dataset_name",
        dataset_name or _optional_string(sample.get("dataset_name"), "dataset_name") or None,
    )
    metadata_payload.setdefault(
        "modality_pair",
        modality_pair or _optional_string(sample.get("modality_pair"), "modality_pair") or None,
    )
    if metadata_payload["dataset_name"] is None:
        raise SwiftConversionError(
            "classic Swift sample has no dataset_name; pass dataset_name explicitly"
        )
    if metadata_payload["modality_pair"] is None:
        raise SwiftConversionError(
            "classic Swift sample has no modality_pair; pass modality_pair explicitly"
        )

    query_item_id = _optional_string(mira_payload.get("query_item_id"), "mira.query_item_id")
    query = _make_item(
        item_type=MultimodalItem,
        item_id=query_item_id or f"{resolved_sample_id}:query",
        messages=sample.get("messages"),
        images=sample.get("images", ()),
        videos=sample.get("videos", ()),
        audios=sample.get("audios", ()),
        instruction=sample.get("instruction"),
        context="query",
    )

    preserved = bool(mira_payload.get("multi_positive_preserved", False))
    positive_message_groups = _message_groups(
        sample.get("positive_messages"),
        "positive_messages",
        require_nested=preserved,
    )
    if not positive_message_groups:
        raise SwiftConversionError("positive_messages must describe at least one positive")
    negative_message_groups = _message_groups(
        sample.get("negative_messages", ()), "negative_messages", require_nested=False
    )

    positive_ids = _candidate_ids(
        mira_payload.get("positive_item_ids"),
        len(positive_message_groups),
        f"{resolved_sample_id}:positive",
        "mira.positive_item_ids",
    )
    negative_ids = _candidate_ids(
        mira_payload.get("negative_item_ids"),
        len(negative_message_groups),
        f"{resolved_sample_id}:negative",
        "mira.negative_item_ids",
    )

    positives = _make_candidates(
        positive_message_groups,
        positive_ids,
        sample,
        prefix="positive",
        preserved=preserved,
    )
    negatives = _make_candidates(
        negative_message_groups,
        negative_ids,
        sample,
        prefix="negative",
        preserved=len(negative_message_groups) > 1,
    )

    try:
        metadata = RetrievalMetadata.model_validate(metadata_payload)
        return EmbeddingRecord(
            sample_id=resolved_sample_id,
            task_id=resolved_task_id,
            query=query,
            positives=positives,
            negatives=negatives,
            metadata=metadata,
        )
    except (ValidationError, ValueError) as error:
        raise SwiftConversionError(
            f"Swift sample {resolved_sample_id!r} violates MIRA semantics: {error}"
        ) from error


def _preserved_sample(record: EmbeddingRecord) -> dict[str, Any]:
    result = _query_fields(record)
    result.update(_candidate_fields(record.positives, "positive", nested=True))
    if record.negatives:
        result.update(_candidate_fields(record.negatives, "negative", nested=True))
    result["mira"] = _mira_extension(record, multi_positive_preserved=True)
    return result


def _expanded_sample(record: EmbeddingRecord, positive_index: int) -> dict[str, Any]:
    positive = record.positives[positive_index]
    result = _query_fields(record)
    result.update(_candidate_fields((positive,), "positive", nested=False))
    if record.negatives:
        result.update(
            _candidate_fields(record.negatives, "negative", nested=len(record.negatives) > 1)
        )
    result["sample_id"] = f"{record.sample_id}::positive-{positive_index}"
    result["mira"] = _mira_extension(
        record,
        multi_positive_preserved=False,
        positive_index=positive_index,
    )
    return result


def _query_fields(record: EmbeddingRecord) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "sample_id": record.sample_id,
        "task_id": record.task_id,
        "dataset_name": record.metadata.dataset_name,
        "modality_pair": record.metadata.modality_pair,
        "messages": _serialize_messages(record.query.messages),
    }
    _add_item_media(fields, record.query, prefix="")
    if record.query.instruction is not None:
        fields["instruction"] = record.query.instruction
    return fields


def _candidate_fields(
    candidates: Sequence[CandidateItem], prefix: str, *, nested: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    messages = [_serialize_messages(candidate.messages) for candidate in candidates]
    result[f"{prefix}_messages"] = messages if nested else messages[0]
    for modality in ("images", "videos", "audios"):
        groups = [
            [
                reference.uri
                for reference in cast("tuple[MediaRef, ...]", getattr(candidate, modality))
            ]
            for candidate in candidates
        ]
        if any(groups):
            result[f"{prefix}_{modality}"] = groups if nested else groups[0]
    instructions = [candidate.instruction for candidate in candidates]
    if any(instruction is not None for instruction in instructions):
        result[f"{prefix}_instructions"] = instructions if nested else instructions[0]
    return result


def _add_item_media(fields: dict[str, Any], item: MultimodalItem, *, prefix: str) -> None:
    for modality in ("images", "videos", "audios"):
        references = cast("tuple[MediaRef, ...]", getattr(item, modality))
        if references:
            fields[f"{prefix}{modality}"] = [reference.uri for reference in references]


def _mira_extension(
    record: EmbeddingRecord,
    *,
    multi_positive_preserved: bool,
    positive_index: int | None = None,
) -> dict[str, Any]:
    if positive_index is None:
        positives = record.positives
        metadata = record.metadata
    else:
        positives = (record.positives[positive_index],)
        metadata = _metadata_for_expanded_positive(record, positive_index)
    extension: dict[str, Any] = {
        "schema_version": 1,
        "source_sample_id": record.sample_id,
        "task_id": record.task_id,
        "query_item_id": record.query.item_id,
        "positive_item_ids": [candidate.item_id for candidate in positives],
        "negative_item_ids": [candidate.item_id for candidate in record.negatives],
        "multi_positive_preserved": multi_positive_preserved,
        "metadata": metadata.model_dump(mode="json", exclude_none=True),
    }
    if positive_index is not None:
        extension["positive_index"] = positive_index
        extension["positive_count"] = len(record.positives)
    return extension


def _metadata_for_expanded_positive(
    record: EmbeddingRecord, positive_index: int
) -> RetrievalMetadata:
    metadata = record.metadata
    positive_groups = (
        (metadata.positive_group_ids[positive_index],) if metadata.positive_group_ids else ()
    )
    relevance = None
    if metadata.relevance_labels is not None:
        negative_values = metadata.relevance_labels[len(record.positives) :]
        relevance = (metadata.relevance_labels[positive_index], *negative_values)
    teacher_scores = None
    if metadata.teacher_scores is not None:
        negative_scores = metadata.teacher_scores[len(record.positives) :]
        teacher_scores = (metadata.teacher_scores[positive_index], *negative_scores)
    return metadata.model_copy(
        update={
            "positive_group_ids": positive_groups,
            "relevance_labels": relevance,
            "teacher_scores": teacher_scores,
        }
    )


def _serialize_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [message.model_dump(mode="json", exclude_none=True) for message in messages]


def _message_groups(value: Any, name: str, *, require_nested: bool) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not _is_sequence(value):
        raise SwiftConversionError(f"{name} must be a sequence")
    values = tuple(value)
    if not values:
        return ()

    first = values[0]
    if isinstance(first, Mapping):
        if require_nested:
            raise SwiftConversionError(
                f"{name} is marked as multi-positive-preserved but is not nested"
            )
        if not all(isinstance(entry, Mapping) for entry in values):
            raise SwiftConversionError(f"{name} mixes messages and message groups")
        return (values,)
    if _is_sequence(first):
        groups: list[tuple[Any, ...]] = []
        for group_index, group in enumerate(values):
            if not _is_sequence(group) or not group:
                raise SwiftConversionError(
                    f"{name}[{group_index}] must be a non-empty message sequence"
                )
            if not all(isinstance(message, Mapping) for message in group):
                raise SwiftConversionError(f"{name}[{group_index}] contains a non-message value")
            groups.append(tuple(group))
        return tuple(groups)
    raise SwiftConversionError(f"{name} does not contain valid message objects")


def _make_candidates(
    message_groups: Sequence[Any],
    item_ids: Sequence[str],
    sample: Mapping[str, Any],
    *,
    prefix: str,
    preserved: bool,
) -> tuple[CandidateItem, ...]:
    count = len(message_groups)
    media = {
        modality: _media_groups(
            sample.get(f"{prefix}_{modality}", ()),
            f"{prefix}_{modality}",
            group_count=count,
            nested=preserved,
        )
        for modality in ("images", "videos", "audios")
    }
    instructions = _instruction_groups(
        sample.get(f"{prefix}_instructions"), count, preserved=preserved
    )
    candidates: list[CandidateItem] = []
    for index, messages in enumerate(message_groups):
        candidates.append(
            _make_item(
                item_type=CandidateItem,
                item_id=item_ids[index],
                messages=messages,
                images=media["images"][index],
                videos=media["videos"][index],
                audios=media["audios"][index],
                instruction=instructions[index],
                context=f"{prefix}[{index}]",
            )
        )
    return tuple(candidates)


def _make_item(  # noqa: PLR0913
    *,
    item_type: type[MultimodalItem | CandidateItem],
    item_id: str,
    messages: Any,
    images: Any,
    videos: Any,
    audios: Any,
    instruction: Any,
    context: str,
) -> Any:
    if not _is_sequence(messages):
        raise SwiftConversionError(f"{context}.messages must be a sequence")
    if instruction is not None and not isinstance(instruction, str):
        raise SwiftConversionError(f"{context}.instruction must be a string or null")
    try:
        return item_type.model_validate(
            {
                "item_id": item_id,
                "messages": messages,
                "images": images,
                "videos": videos,
                "audios": audios,
                "instruction": instruction,
            }
        )
    except (ValidationError, ValueError) as error:
        raise SwiftConversionError(f"invalid {context}: {error}") from error


def _media_groups(
    value: Any, name: str, *, group_count: int, nested: bool
) -> tuple[tuple[Any, ...], ...]:
    if group_count == 0:
        if value not in (None, (), []):
            raise SwiftConversionError(f"{name} exists without corresponding messages")
        return ()
    if value is None:
        return tuple(() for _ in range(group_count))
    if not _is_sequence(value):
        raise SwiftConversionError(f"{name} must be a sequence")
    values = tuple(value)
    if not values:
        return tuple(() for _ in range(group_count))

    is_nested = _is_sequence(values[0]) and not isinstance(values[0], (str, bytes))
    if nested or group_count > 1:
        if not is_nested:
            raise SwiftConversionError(
                f"{name} must contain one media list per candidate; flat media are ambiguous"
            )
        if len(values) != group_count:
            raise SwiftConversionError(
                f"{name} has {len(values)} candidate groups; expected {group_count}"
            )
        return tuple(tuple(group) for group in values)
    if is_nested:
        if len(values) != 1:
            raise SwiftConversionError(f"{name} has unexpected nested candidate groups")
        return (tuple(values[0]),)
    return (values,)


def _instruction_groups(value: Any, count: int, *, preserved: bool) -> tuple[str | None, ...]:
    if value is None:
        return tuple(None for _ in range(count))
    if preserved:
        if not _is_sequence(value) or len(value) != count:
            raise SwiftConversionError(
                "candidate instructions must align one-to-one with candidate messages"
            )
        result = tuple(value)
    else:
        result = (value,) if count == 1 else tuple(value)
    invalid_entry = any(entry is not None and not isinstance(entry, str) for entry in result)
    if len(result) != count or invalid_entry:
        raise SwiftConversionError(
            "candidate instructions must be strings/null aligned with candidates"
        )
    return cast("tuple[str | None, ...]", result)


def _candidate_ids(value: Any, count: int, prefix: str, field_name: str) -> tuple[str, ...]:
    if value is None:
        return tuple(f"{prefix}:{index}" for index in range(count))
    if not _is_sequence(value) or any(not isinstance(item_id, str) for item_id in value):
        raise SwiftConversionError(f"{field_name} must be a sequence of strings")
    ids = tuple(value)
    if len(ids) != count:
        raise SwiftConversionError(f"{field_name} has {len(ids)} entries; expected {count}")
    return cast("tuple[str, ...]", ids)


def _mapping_or_empty(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SwiftConversionError(f"{name} must be an object")
    return cast("Mapping[str, Any]", value)


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SwiftConversionError(f"{name} must be a non-empty string")
    return value


def _required_identifier(value: str | None, name: str) -> str:
    if value is None or not value:
        raise SwiftConversionError(f"classic Swift sample has no {name}; pass {name} explicitly")
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


# Explicit aliases make call sites read naturally in either conversion direction.
to_swift_json = mira_to_swift
from_swift_json = swift_to_mira
