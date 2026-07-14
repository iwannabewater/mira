"""Adversarial validation coverage for the Phase-1 data surface."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import torch
from pydantic import ValidationError

from mira.core import DataValidationError
from mira.data.converters import SwiftJsonConverter
from mira.data.converters.swift_json import mira_to_swift, swift_to_mira
from mira.data.errors import JsonlDataError, SamplerStateError, SwiftConversionError
from mira.data.fingerprint import canonical_record_payload, records_checksum
from mira.data.layout import GroupLayoutBuilder, build_group_layout
from mira.data.readers import MiraJsonlReader, read_jsonl, write_jsonl
from mira.data.records import EmbeddingRecord
from mira.data.samplers import NoDuplicateBatchSampler
from mira.data.schema import (
    CandidateItem,
    MediaRef,
    Message,
    MessagePart,
    MultimodalItem,
    ObjectAnnotations,
    RetrievalMetadata,
)

if TYPE_CHECKING:
    from pathlib import Path


def _item(item_id: str, text: str) -> MultimodalItem:
    return MultimodalItem(item_id=item_id, messages=(Message(role="user", content=text),))


def _candidate(item_id: str, text: str) -> CandidateItem:
    return CandidateItem.model_validate(_item(item_id, text).model_dump())


def _record(
    sample_id: str,
    *,
    positive_count: int = 1,
    canonical: str | None = None,
    positive_groups: tuple[str, ...] = (),
) -> EmbeddingRecord:
    return EmbeddingRecord(
        sample_id=sample_id,
        task_id="text_to_text",
        query=_item(f"{sample_id}-q", "query"),
        positives=tuple(
            _candidate(f"{sample_id}-p{index}", f"positive {index}")
            for index in range(positive_count)
        ),
        negatives=(_candidate(f"{sample_id}-n", "negative"),),
        metadata=RetrievalMetadata(
            positive_group_ids=positive_groups,
            negative_group_ids=(f"{sample_id}-negative",),
            modality_pair="text_to_text",
            dataset_name="unit",
            canonical_product_id=canonical,
        ),
    )


def test_structured_message_and_string_media_are_canonicalized() -> None:
    item = MultimodalItem.model_validate(
        {
            "item_id": "structured",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "inspect <image>", "x": 1}],
                }
            ],
            "images": ["image.jpg"],
        }
    )
    assert item.images == (MediaRef(uri="image.jpg"),)
    assert item.placeholder_counts() == {"image": 1, "video": 0, "audio": 0}
    assert item.messages[0].text_fragments() == ("inspect <image>",)


@pytest.mark.parametrize(
    "payload",
    [
        {"role": "", "content": "text"},
        {"role": "user", "content": []},
    ],
)
def test_message_rejects_invalid_content(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Message.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"item_id": "", "messages": [{"role": "user", "content": "x"}]},
        {"item_id": "x", "messages": []},
        {
            "item_id": "x",
            "messages": [{"role": "user", "content": "<video> <audio>"}],
        },
    ],
)
def test_item_rejects_empty_identity_messages_or_media_mismatch(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        MultimodalItem.model_validate(payload)


@pytest.mark.parametrize(
    "update",
    [
        {"modality_pair": ""},
        {"dataset_name": ""},
        {"sampling_weight": 0.0},
        {"relevance_labels": [float("nan")]},
        {"teacher_scores": [float("inf")]},
    ],
)
def test_metadata_rejects_invalid_required_and_numeric_values(
    update: dict[str, object],
) -> None:
    payload: dict[str, object] = {"modality_pair": "t", "dataset_name": "d"}
    payload.update(update)
    with pytest.raises(ValidationError):
        RetrievalMetadata.model_validate(payload)


@pytest.mark.parametrize(
    "update",
    [
        {"sampling_weight": True},
        {"relevance_labels": [True]},
        {"teacher_scores": [False]},
    ],
)
def test_metadata_rejects_boolean_numeric_values(update: dict[str, object]) -> None:
    payload: dict[str, object] = {"modality_pair": "t", "dataset_name": "d"}
    payload.update(update)
    with pytest.raises(ValidationError, match="boolean"):
        RetrievalMetadata.model_validate(payload)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: MediaRef(uri="image.jpg", metadata={"nested": [{"score": float("nan")}]}),
            "contains non-finite number",
        ),
        (
            lambda: ObjectAnnotations(annotations=({"nested": {"score": float("inf")}},)),
            "contains non-finite number",
        ),
        (
            lambda: MessagePart(type="processor", processor={"scores": [0.0, float("-inf")]}),
            "contains non-finite number",
        ),
        (
            lambda: MediaRef(uri="image.jpg", metadata={"nested": {1: "value"}}),
            "contains non-string object key",
        ),
        (
            lambda: ObjectAnnotations(annotations=({"payload": object()},)),
            "contains unsupported value",
        ),
        (
            lambda: MessagePart(type="processor", processor={"tokens": {1, 2}}),
            "contains unsupported value",
        ),
    ],
)
def test_opaque_surfaces_reject_non_json_values(factory: object, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        factory()  # type: ignore[operator]


def test_opaque_surfaces_copy_json_primitives_and_reject_cycles() -> None:
    media = MediaRef(
        uri="image.jpg",
        checksum=f"sha256:{'a' * 64}",
        metadata={"label": "shoe", "count": 2, "score": 0.5, "flags": [True, None]},
    )
    assert media.metadata == {
        "label": "shoe",
        "count": 2,
        "score": 0.5,
        "flags": (True, None),
    }

    cyclic_mapping: dict[str, object] = {}
    cyclic_mapping["self"] = cyclic_mapping
    with pytest.raises(ValidationError, match="contains a cycle"):
        MediaRef(uri="image.jpg", metadata=cyclic_mapping)

    cyclic_list: list[object] = []
    cyclic_list.append(cyclic_list)
    with pytest.raises(ValidationError, match="contains a cycle"):
        MediaRef(uri="image.jpg", metadata={"items": cyclic_list})


def test_opaque_json_is_detached_deeply_frozen_and_checksum_stable() -> None:
    media_source: dict[str, Any] = {"processor": {"shape": [224, 224]}}
    part_source: dict[str, Any] = {"tokens": [" alpha "]}
    annotation_source: dict[str, Any] = {"bbox": [1, 2, 3, 4]}
    base = _record("immutable")
    query = MultimodalItem(
        item_id="immutable-q",
        messages=(
            Message(
                role="user",
                content=(
                    MessagePart(
                        type="processor",
                        text="inspect <image>",
                        processor=part_source,
                    ),
                ),
            ),
        ),
        images=(MediaRef(uri="image.jpg", metadata=media_source),),
        objects=ObjectAnnotations(annotations=(annotation_source,)),
    )
    record = base.model_copy(update={"query": query})
    checksum = records_checksum((record,))

    media_source["processor"]["shape"].append(448)
    part_source["tokens"].append("external")
    annotation_source["bbox"].append(5)

    assert record.query.images[0].metadata["processor"]["shape"] == (224, 224)
    part = record.query.messages[0].content[0]
    assert part.model_extra == {"processor": {"tokens": (" alpha ",)}}
    assert record.query.objects is not None
    assert record.query.objects.annotations[0]["bbox"] == (1, 2, 3, 4)
    assert records_checksum((record,)) == checksum

    with pytest.raises(TypeError, match="immutable"):
        record.query.images[0].metadata["new"] = True  # type: ignore[index]
    with pytest.raises(AttributeError):
        record.query.images[0].metadata["processor"]["shape"].append(448)
    assert part.model_extra is not None
    with pytest.raises(TypeError, match="immutable"):
        part.model_extra["new"] = True
    with pytest.raises(TypeError, match="immutable"):
        record.query.objects.annotations[0]["new"] = True  # type: ignore[index]
    assert records_checksum((record,)) == checksum


def test_model_input_strings_preserve_exact_whitespace_through_serialization() -> None:
    message = Message(role="user", content="  exact content\n")
    part = MessagePart(
        type="text",
        text="\tstructured text  ",
        processor={"prompt": "  opaque prompt  "},
    )
    item = MultimodalItem(
        item_id="item",
        messages=(message, Message(role="assistant", content=(part,))),
        instruction="  exact instruction\n",
    )

    assert message.content == "  exact content\n"
    assert part.text == "\tstructured text  "
    assert part.model_extra == {"processor": {"prompt": "  opaque prompt  "}}
    assert item.instruction == "  exact instruction\n"
    dumped = item.model_dump(mode="json")
    assert dumped["messages"][0]["content"] == "  exact content\n"
    assert dumped["messages"][1]["content"][0]["text"] == "\tstructured text  "
    assert dumped["instruction"] == "  exact instruction\n"
    assert MultimodalItem.model_validate(dumped) == item


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Message(role=" user", content="text"),
        lambda: MessagePart(type="text ", text="text"),
        lambda: MediaRef(uri=" image.jpg"),
        lambda: MultimodalItem(item_id="item ", messages=(Message(role="user", content="text"),)),
        lambda: RetrievalMetadata(modality_pair=" text", dataset_name="unit"),
        lambda: RetrievalMetadata(modality_pair="text", dataset_name="unit "),
        lambda: RetrievalMetadata(
            modality_pair="text", dataset_name="unit", positive_group_ids=(" group",)
        ),
        lambda: RetrievalMetadata(modality_pair="text", dataset_name="unit", sku_id="sku "),
        lambda: EmbeddingRecord.model_validate(
            _record("sample").model_dump() | {"sample_id": " sample"}
        ),
    ],
)
def test_semantic_strings_reject_edge_whitespace(factory: object) -> None:
    with pytest.raises((ValidationError, ValueError), match="leading or trailing whitespace"):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Message(role="\t", content="text"),
        lambda: MessagePart(type=" ", text="text"),
        lambda: MediaRef(uri="\n"),
        lambda: RetrievalMetadata(modality_pair=" ", dataset_name="unit"),
        lambda: RetrievalMetadata(modality_pair="text", dataset_name="unit", cluster_id=" "),
    ],
)
def test_semantic_strings_reject_blank_values(factory: object) -> None:
    with pytest.raises((ValidationError, ValueError), match="must not be blank"):
        factory()  # type: ignore[operator]


def test_message_part_and_media_reference_reject_malformed_boundaries() -> None:
    with pytest.raises(ValidationError):
        MessagePart.model_validate("not-an-object")
    with pytest.raises(ValidationError, match="message part keys must be strings"):
        MessagePart.model_validate({"type": "processor", 1: "bad"})
    with pytest.raises(ValidationError, match="media URI must not be blank"):
        MediaRef(uri="")
    with pytest.raises(ValidationError, match="sha256"):
        MediaRef(uri="image.jpg", checksum="sha256:ABC")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(sample_id=""), "identifier"),
        (lambda payload: payload.update(positives=[]), "at least one positive"),
        (
            lambda payload: payload.update(
                negatives=(*payload["negatives"], payload["positives"][0])
            ),
            "candidate item_id values must be unique",
        ),
        (
            lambda payload: payload["positives"][0].update(item_id="s-q"),
            "query item_id also appears",
        ),
        (
            lambda payload: payload["metadata"].update(positive_group_ids=["a", "b"]),
            "positive_group_ids has 2 entries",
        ),
        (
            lambda payload: payload["metadata"].update(relevance_labels=[1.0]),
            "relevance_labels has 1 entries",
        ),
        (
            lambda payload: payload["metadata"].update(teacher_scores=[1.0]),
            "teacher_scores has 1 entries",
        ),
        (
            lambda payload: payload["metadata"].update(
                positive_group_ids=["same"], negative_group_ids=["same"]
            ),
            "group IDs conflict",
        ),
    ],
)
def test_record_rejects_identity_and_alignment_errors(mutation: object, match: str) -> None:
    payload = _record("s", positive_groups=("positive",)).model_dump()
    mutation(payload)  # type: ignore[operator]
    with pytest.raises(ValidationError, match=match):
        EmbeddingRecord.model_validate(payload)


def test_record_accepts_exact_candidate_aligned_scores() -> None:
    payload = _record("aligned").model_dump()
    payload["metadata"].update(
        relevance_labels=[1.0, 0.0],
        teacher_scores=[2.0, -1.0],
    )
    record = EmbeddingRecord.model_validate(payload)
    assert record.metadata.relevance_labels == (1.0, 0.0)
    assert record.metadata.teacher_scores == (2.0, -1.0)


def test_jsonl_reader_reports_all_source_failures(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(JsonlDataError, match="cannot open dataset"):
        read_jsonl(missing)

    path = tmp_path / "bad.jsonl"
    path.write_text("{\n", encoding="utf-8")
    with pytest.raises(JsonlDataError, match="invalid JSON"):
        read_jsonl(path)
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(JsonlDataError, match="must be a JSON object"):
        read_jsonl(path)
    path.write_text('{"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(JsonlDataError, match="validation errors"):
        read_jsonl(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_jsonl_reader_rejects_non_finite_numbers_with_line_context(
    tmp_path: Path, constant: str
) -> None:
    path = tmp_path / "non-finite.jsonl"
    path.write_text(f'{{"value":{constant}}}\n', encoding="utf-8")
    with pytest.raises(JsonlDataError, match=r"non-finite\.jsonl:1: invalid JSON:.*finite"):
        read_jsonl(path)


@pytest.mark.parametrize("duplicate_field", ["sample_id", "dataset_name"])
def test_jsonl_reader_rejects_duplicate_object_keys_at_any_depth(
    tmp_path: Path, duplicate_field: str
) -> None:
    record = _record("duplicate")
    payload = record.model_dump_json()
    if duplicate_field == "sample_id":
        payload = payload.replace(
            '"sample_id":"duplicate"',
            '"sample_id":"duplicate","sample_id":"shadow"',
            1,
        )
    else:
        payload = payload.replace(
            '"dataset_name":"unit"',
            '"dataset_name":"unit","dataset_name":"shadow"',
            1,
        )
    path = tmp_path / "duplicate-key.jsonl"
    path.write_text(f"{payload}\n", encoding="utf-8")
    with pytest.raises(
        JsonlDataError,
        match=(
            rf"duplicate-key\.jsonl:1: invalid JSON: "
            rf".*duplicate JSON object key '{duplicate_field}'"
        ),
    ):
        read_jsonl(path)


def test_fingerprint_and_writer_reject_bypassed_non_finite_model_state(tmp_path: Path) -> None:
    record = _record("unsafe")
    unsafe_objects = ObjectAnnotations.model_construct(
        annotations=({"processor": {"score": float("nan")}},)
    )
    unsafe_record = record.model_copy(
        update={"query": record.query.model_copy(update={"objects": unsafe_objects})}
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        records_checksum((unsafe_record,))
    with pytest.raises(ValueError, match="Out of range float values"):
        NoDuplicateBatchSampler((unsafe_record,), batch_size=1)

    output = tmp_path / "unsafe.jsonl"
    with pytest.raises(ValueError, match="Out of range float values"):
        write_jsonl((unsafe_record,), output)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".unsafe.jsonl.tmp-*"))


def test_record_fingerprints_are_deterministic_ordered_and_root_safe(tmp_path: Path) -> None:
    first = _record("first")
    second = _record("second")
    checksum = records_checksum((first, second))
    assert records_checksum((first, second)) == checksum
    assert records_checksum((second, first)) != checksum

    media_root = tmp_path / "media"
    media_root.mkdir()
    relative_query = MultimodalItem(
        item_id="relative-q",
        messages=(Message(role="user", content="inspect <image>"),),
        images=(MediaRef(uri="relative.jpg"),),
    )
    relative_record = first.model_copy(update={"query": relative_query})
    payload = canonical_record_payload(relative_record, media_root=media_root)
    assert payload["query"]["images"][0]["uri"] == "relative.jpg"

    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    outside_query = relative_query.model_copy(update={"images": (MediaRef(uri=str(outside)),)})
    outside_record = first.model_copy(update={"query": outside_query})
    with pytest.raises(DataValidationError, match="outside fingerprint root"):
        canonical_record_payload(outside_record, media_root=media_root)


def test_canonical_record_payload_resolves_symlinked_media_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    media_path = alias_root / "image.jpg"
    media_path.write_bytes(b"image")
    record = _record("media")
    query = MultimodalItem(
        item_id=record.query.item_id,
        messages=(Message(role="user", content="inspect <image>"),),
        images=(MediaRef(uri=str(media_path)),),
    )
    record = record.model_copy(update={"query": query})

    payload = canonical_record_payload(record, media_root=alias_root)

    assert payload["query"]["images"][0]["uri"] == "image.jpg"


def test_jsonl_reader_class_duplicate_and_writer_guards(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    record = _record("s")
    path.write_text(f"{record.model_dump_json()}\n{record.model_dump_json()}\n", encoding="utf-8")
    reader = MiraJsonlReader(path)
    with pytest.raises(JsonlDataError, match="duplicate sample_id"):
        tuple(reader)
    with pytest.raises(JsonlDataError, match="duplicate sample_id"):
        reader.read()

    output = tmp_path / "output.jsonl"
    output.write_text("original\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate sample_id"):
        write_jsonl((record, record), output, overwrite=True)
    assert output.read_text(encoding="utf-8") == "original\n"
    with pytest.raises(FileExistsError):
        write_jsonl((record,), output)
    write_jsonl((record,), output, overwrite=True)
    assert output.read_text(encoding="utf-8").startswith('{"schema_version":1')


def test_classic_swift_conversion_and_converter_wrapper() -> None:
    sample = {
        "sample_id": "classic",
        "task_id": "image",
        "dataset_name": "data",
        "modality_pair": "image_to_image",
        "messages": [{"role": "user", "content": "<image>"}],
        "images": ["q.jpg"],
        "instruction": "query instruction",
        "positive_messages": [{"role": "user", "content": "<image>"}],
        "positive_images": ["p.jpg"],
        "positive_instructions": "candidate instruction",
        "negative_messages": [{"role": "user", "content": "negative"}],
    }
    record = SwiftJsonConverter.from_swift(sample)
    assert record.query.instruction == "query instruction"
    assert record.positives[0].instruction == "candidate instruction"
    converted = SwiftJsonConverter.to_swift(record, multi_positive="expand")
    assert len(converted) == 1


@pytest.mark.parametrize(
    ("sample", "match"),
    [
        ({}, "no sample_id"),
        ({"sample_id": "s"}, "no task_id"),
        ({"sample_id": "s", "task_id": "t"}, "no dataset_name"),
        (
            {"sample_id": "s", "task_id": "t", "dataset_name": "d"},
            "no modality_pair",
        ),
        (
            {
                "sample_id": "s",
                "task_id": "t",
                "dataset_name": "d",
                "modality_pair": "m",
                "messages": [],
            },
            "invalid query",
        ),
    ],
)
def test_swift_missing_contract_fields_fail(sample: dict[str, object], match: str) -> None:
    with pytest.raises(SwiftConversionError, match=match):
        swift_to_mira(sample)


def test_swift_rejects_invalid_strategy_and_malformed_extensions() -> None:
    record = _record("s")
    with pytest.raises(SwiftConversionError, match="unknown multi-positive strategy"):
        mira_to_swift(record, multi_positive="first")
    sample = {
        "sample_id": "s",
        "task_id": "t",
        "dataset_name": "d",
        "modality_pair": "m",
        "messages": [{"role": "user", "content": "q"}],
        "positive_messages": [{"role": "user", "content": "p"}],
        "mira": [],
    }
    with pytest.raises(SwiftConversionError, match="mira must be an object"):
        swift_to_mira(sample)


def _classic_sample() -> dict[str, object]:
    return {
        "sample_id": "s",
        "task_id": "t",
        "dataset_name": "d",
        "modality_pair": "m",
        "messages": [{"role": "user", "content": "q"}],
        "positive_messages": [{"role": "user", "content": "p"}],
    }


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"positive_messages": None}, "at least one positive"),
        ({"positive_messages": "bad"}, "must be a sequence"),
        ({"positive_messages": [{"role": "user", "content": "p"}, []]}, "mixes"),
        ({"positive_messages": [[]]}, "non-empty message sequence"),
        ({"positive_messages": [[1]]}, "non-message"),
        ({"positive_messages": [1]}, "valid message objects"),
        ({"messages": "query"}, "query.messages must be a sequence"),
        ({"instruction": 3}, "query.instruction"),
        ({"negative_images": ["orphan.jpg"]}, "without corresponding messages"),
        ({"positive_images": 3}, "positive_images must be a sequence"),
        ({"positive_images": [["a"], ["b"]]}, "unexpected nested"),
        ({"positive_instructions": 3}, "strings/null"),
        ({"sample_id": 3}, "sample_id must be a non-empty string"),
    ],
)
def test_swift_rejects_ambiguous_or_malformed_nested_fields(
    update: dict[str, object], match: str
) -> None:
    sample = _classic_sample()
    sample.update(update)
    with pytest.raises(SwiftConversionError, match=match):
        swift_to_mira(sample)


def test_swift_preserved_marker_and_candidate_ids_validate_alignment() -> None:
    sample = _classic_sample()
    sample["mira"] = {"multi_positive_preserved": True}
    with pytest.raises(SwiftConversionError, match="not nested"):
        swift_to_mira(sample)

    sample = _classic_sample()
    sample["mira"] = {"positive_item_ids": [3]}
    with pytest.raises(SwiftConversionError, match="sequence of strings"):
        swift_to_mira(sample)
    sample["mira"] = {"positive_item_ids": ["a", "b"]}
    with pytest.raises(SwiftConversionError, match="expected 1"):
        swift_to_mira(sample)


def test_swift_preserved_media_and_instruction_group_counts_validate() -> None:
    sample = _classic_sample()
    sample.update(
        positive_messages=[
            [{"role": "user", "content": "<image>"}],
            [{"role": "user", "content": "<image>"}],
        ],
        positive_images=[["one.jpg"]],
    )
    with pytest.raises(SwiftConversionError, match="candidate groups; expected 2"):
        swift_to_mira(sample)
    sample["positive_images"] = [["one.jpg"], ["two.jpg"]]
    sample["mira"] = {"multi_positive_preserved": True}
    sample["positive_instructions"] = ["only one"]
    with pytest.raises(SwiftConversionError, match="align one-to-one"):
        swift_to_mira(sample)


def test_swift_wraps_metadata_and_record_validation_and_omits_empty_negatives() -> None:
    sample = _classic_sample()
    sample["mira"] = {"metadata": {"sampling_weight": 0}}
    with pytest.raises(SwiftConversionError, match="violates MIRA semantics"):
        swift_to_mira(sample)

    record = _record("no-negatives").model_copy(
        update={
            "negatives": (),
            "metadata": _record("no-negatives").metadata.model_copy(
                update={"negative_group_ids": ()}
            ),
        }
    )
    expanded = mira_to_swift(record)
    preserved = mira_to_swift(record, multi_positive="preserve")
    assert "negative_messages" not in expanded[0]
    assert "negative_messages" not in preserved[0]


def test_expansion_keeps_aligned_relevance_and_teacher_scores() -> None:
    record = _record("scores", positive_count=2, positive_groups=("g0", "g1"))
    record = record.model_copy(
        update={
            "metadata": record.metadata.model_copy(
                update={
                    "relevance_labels": (0.5, 1.0, 0.0),
                    "teacher_scores": (2.0, 3.0, -1.0),
                }
            )
        }
    )
    samples = mira_to_swift(record)
    assert samples[1]["mira"]["metadata"]["relevance_labels"] == [1.0, 0.0]
    assert samples[0]["mira"]["metadata"]["teacher_scores"] == [2.0, -1.0]


def test_layout_builder_weights_scores_and_without_in_batch_negatives() -> None:
    first = _record("a", positive_groups=("ga",))
    second = _record("b", positive_groups=("gb",))
    records = tuple(
        record.model_copy(
            update={
                "metadata": record.metadata.model_copy(
                    update={
                        "relevance_labels": (0.5, 0.0),
                        "teacher_scores": (1.0, -1.0),
                    }
                )
            }
        )
        for record in (first, second)
    )
    builder = GroupLayoutBuilder(
        include_in_batch_negatives=False, use_relevance_as_positive_weights=True
    )
    result = builder.build(records)
    assert not result.layout.in_batch_negative_mask.any()
    assert torch.equal(
        result.layout.positive_weights[result.layout.positive_mask], torch.tensor([0.5, 0.5])
    )
    assert result.layout.relevance_scores is not None
    assert result.layout.teacher_scores is not None


def test_layout_rejects_empty_duplicate_and_conflicting_batch_semantics() -> None:
    record = _record("duplicate")
    with pytest.raises(DataValidationError, match="empty record sequence"):
        build_group_layout(())
    with pytest.raises(DataValidationError, match="unique sample_id"):
        build_group_layout((record, record))
    with pytest.raises(DataValidationError, match="not both"):
        build_group_layout(
            (record,),
            positive_weights={record.sample_id: (1.0,)},
            use_relevance_as_positive_weights=True,
        )


def test_layout_requires_relevance_labels_when_used_as_positive_weights() -> None:
    with pytest.raises(DataValidationError, match="has no relevance_labels"):
        build_group_layout((_record("missing-labels"),), use_relevance_as_positive_weights=True)


def test_layout_masks_explicit_negatives_equivalent_to_the_query_group() -> None:
    record = _record("explicit")
    negative_group = record.metadata.negative_group_ids[0]
    metadata = record.metadata.model_copy(update={"query_group_id": negative_group})
    record = record.model_copy(update={"metadata": metadata})

    result = build_group_layout((record,))

    assert result.layout.false_negative_mask[0, 1]
    assert not result.layout.candidate_mask[0, 1]
    assert f"group:{negative_group}" in result.cross_device_metadata.query_positive_keys[0]


def test_layout_accepts_explicit_negatives_without_group_metadata() -> None:
    record = _record("no-negative-groups")
    metadata = record.metadata.model_copy(update={"negative_group_ids": ()})
    result = build_group_layout((record.model_copy(update={"metadata": metadata}),))
    assert not result.layout.false_negative_mask.any()
    assert result.layout.explicit_negative_mask[0, 1]


def test_layout_masks_cross_record_item_group_and_product_equivalence() -> None:
    shared_item_first = _record("item-a")
    shared_item_second = _record("item-b")
    shared_item_second = EmbeddingRecord(
        sample_id=shared_item_second.sample_id,
        task_id=shared_item_second.task_id,
        query=shared_item_second.query,
        positives=(shared_item_first.positives[0],),
        negatives=shared_item_second.negatives,
        metadata=shared_item_second.metadata,
    )
    item_result = build_group_layout((shared_item_first, shared_item_second))
    assert item_result.layout.false_negative_mask[:, [0, 2]].sum().item() == 2

    group_query = _record("group-a", positive_groups=("own-group",))
    group_query = group_query.model_copy(
        update={"metadata": group_query.metadata.model_copy(update={"query_group_id": "shared"})}
    )
    group_candidate = _record("group-b", positive_groups=("shared",))
    group_result = build_group_layout((group_query, group_candidate))
    assert group_result.layout.false_negative_mask[0, 2]

    product_first = _record("product-a", canonical="same", positive_groups=("group-a",))
    product_second = _record("product-b", canonical="same", positive_groups=("group-b",))
    product_result = build_group_layout((product_first, product_second))
    assert product_result.layout.false_negative_mask[0, 2]
    assert product_result.layout.false_negative_mask[1, 0]


def test_layout_rejects_missing_or_misaligned_optional_weights_and_scores() -> None:
    first = _record("a")
    second = _record("b")
    with pytest.raises(Exception, match="unknown sample IDs"):
        build_group_layout((first,), positive_weights={"missing": (1.0,)})
    with pytest.raises(Exception, match="positive weights"):
        build_group_layout((first,), positive_weights={"a": (1.0, 0.5)})
    with pytest.raises(Exception, match="for every record"):
        build_group_layout(
            (
                first.model_copy(
                    update={
                        "metadata": first.metadata.model_copy(update={"teacher_scores": (1.0, 0.0)})
                    }
                ),
                second,
            )
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": 2, "global_batch_size": 3},
        {"batch_size": 1, "world_size": 2},
        {"batch_size": 2, "world_size": 0},
        {"batch_size": 2, "rank": 2, "world_size": 2},
        {"batch_size": 2, "epoch": -1},
        {"batch_size": 2, "duplicate_keys": ("sample_id", "sample_id")},
        {"batch_size": 2, "duplicate_keys": ("unknown",)},
    ],
)
def test_sampler_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r".+"):
        NoDuplicateBatchSampler((_record("a"),), **kwargs)


def test_sampler_epoch_distributed_partition_and_state_validation() -> None:
    records = tuple(_record(str(index), canonical=str(index)) for index in range(6))
    rank0 = NoDuplicateBatchSampler(
        records, batch_size=4, rank=0, world_size=2, seed=2, drop_last=True
    )
    rank1 = NoDuplicateBatchSampler(
        records, batch_size=4, rank=1, world_size=2, seed=2, drop_last=True
    )
    batch0 = next(iter(rank0))
    batch1 = next(iter(rank1))
    assert set(batch0).isdisjoint(batch1)
    rank0.set_epoch(3)
    assert rank0.cursor == 0
    assert rank0.epoch == 3
    with pytest.raises(ValueError, match="non-negative"):
        rank0.set_epoch(-1)

    state = rank0.state_dict()
    state.pop("version")
    with pytest.raises(SamplerStateError, match="missing required key"):
        rank0.load_state_dict(state)
    state = rank0.state_dict()
    state["epoch"] = "bad"
    with pytest.raises(SamplerStateError, match="invalid sampler epoch"):
        rank0.load_state_dict(state)
    state = rank0.state_dict()
    state["cursor"] = -1
    with pytest.raises(SamplerStateError, match="invalid sampler cursor"):
        rank0.load_state_dict(state)
    state = rank0.state_dict()
    state["cursor"] = 999
    with pytest.raises(SamplerStateError, match="exceeds epoch batch count"):
        rank0.load_state_dict(state)


def test_sampler_fingerprint_detects_content_change_not_only_duplicate_fields() -> None:
    original = _record("same")
    changed = original.model_copy(update={"query": _item("same-q", "different query text")})
    state = NoDuplicateBatchSampler((original,), batch_size=1).state_dict()
    with pytest.raises(SamplerStateError, match="dataset_fingerprint mismatch"):
        NoDuplicateBatchSampler((changed,), batch_size=1).load_state_dict(
            json.loads(json.dumps(state))
        )


def test_sampler_all_duplicate_key_extractors_and_partial_rank_guard() -> None:
    image_query = MultimodalItem(
        item_id="image-q",
        messages=(Message(role="user", content="<image>"),),
        images=(MediaRef(uri="image.jpg"),),
    )
    first = _record("a").model_copy(update={"query": image_query})
    keys = (
        "task_id",
        "query_item_id",
        "positive_item_ids",
        "candidate_item_ids",
        "media_uris",
        "negative_group_ids",
        "category_id",
    )
    sampler = NoDuplicateBatchSampler((first,), batch_size=1, duplicate_keys=keys)
    assert len(sampler) == 1
    assert sampler.global_batches() is sampler.global_batches()

    duplicates = (_record("x", canonical="same"), _record("y", canonical="same"))
    distributed = NoDuplicateBatchSampler(
        duplicates, batch_size=2, rank=0, world_size=2, drop_last=False
    )
    with pytest.raises(Exception, match="leave at least one rank empty"):
        distributed.global_batches()


def test_sampler_empty_records_missing_batch_size_drop_last_and_state_mismatch() -> None:
    with pytest.raises(ValueError, match="records must not be empty"):
        NoDuplicateBatchSampler((), batch_size=1)
    with pytest.raises(ValueError, match="provide batch_size"):
        NoDuplicateBatchSampler((_record("a"),))
    records = (_record("a", canonical="same"), _record("b", canonical="same"))
    sampler = NoDuplicateBatchSampler(records, batch_size=2, drop_last=True)
    assert sampler.global_batches() == ()
    state = NoDuplicateBatchSampler(records, batch_size=2).state_dict()
    state["seed"] = 9
    with pytest.raises(SamplerStateError, match="seed mismatch"):
        NoDuplicateBatchSampler(records, batch_size=2).load_state_dict(state)


def test_sampler_copies_input_and_always_enforces_identity_keys() -> None:
    first = _record("a")
    second = _record("b").model_copy(update={"positives": first.positives})
    records = [first, second]
    sampler = NoDuplicateBatchSampler(records, batch_size=2, duplicate_keys=())
    records.clear()
    assert len(sampler.global_batches()) == 2
    assert "sample_id" in sampler.duplicate_keys
    assert "positive_item_ids" in sampler.duplicate_keys
