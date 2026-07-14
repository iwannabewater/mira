"""Phase-1 data contract, conversion, layout, and sampler evidence."""

from __future__ import annotations

import json

import pytest
import torch
from pydantic import ValidationError

from mira.data.converters.swift_json import mira_to_swift, swift_to_mira
from mira.data.errors import JsonlDataError, SamplerStateError, SwiftConversionError
from mira.data.layout import build_group_layout
from mira.data.readers import read_jsonl, write_jsonl
from mira.data.samplers import NoDuplicateBatchSampler
from mira.data.schema import (
    CandidateItem,
    EmbeddingRecord,
    MediaRef,
    Message,
    MessagePart,
    MultimodalItem,
    ObjectAnnotations,
    RetrievalMetadata,
)


def _item(item_id: str, text: str, images: tuple[str, ...] = ()) -> MultimodalItem:
    return MultimodalItem(
        item_id=item_id,
        messages=(Message(role="user", content=text),),
        images=tuple(MediaRef(uri=uri) for uri in images),
    )


def _candidate(item_id: str, text: str, images: tuple[str, ...] = ()) -> CandidateItem:
    return CandidateItem.model_validate(_item(item_id, text, images).model_dump())


def _record(
    sample_id: str,
    *,
    positive_count: int = 1,
    canonical: str | None = None,
    positive_groups: tuple[str, ...] = (),
    image_hash: str | None = None,
) -> EmbeddingRecord:
    positives = tuple(
        _candidate(f"{sample_id}-p{index}", f"positive {index}") for index in range(positive_count)
    )
    return EmbeddingRecord(
        sample_id=sample_id,
        task_id="text_to_text",
        query=_item(f"{sample_id}-q", "query"),
        positives=positives,
        negatives=(_candidate(f"{sample_id}-n", "negative"),),
        metadata=RetrievalMetadata(
            positive_group_ids=positive_groups,
            negative_group_ids=(f"{sample_id}-negative-group",),
            modality_pair="text_to_text",
            dataset_name="unit",
            canonical_product_id=canonical,
            image_hash_group=image_hash,
        ),
    )


@pytest.mark.parametrize(
    ("text", "images", "match"),
    [
        ("<image> one", (), "item 'anchor'"),
        ("plain", ("image.jpg",), "item 'anchor'"),
        ("<video>", (), "<video>"),
        ("<audio>", (), "<audio>"),
    ],
)
def test_placeholder_alignment_is_strict(text: str, images: tuple[str, ...], match: str) -> None:
    payload: dict[str, object] = {
        "item_id": "anchor",
        "messages": ({"role": "user", "content": text},),
        "images": images,
    }
    with pytest.raises(ValidationError, match=match):
        MultimodalItem.model_validate(payload)


def test_each_positive_is_validated_independently() -> None:
    record = _record("sample")
    payload = record.model_dump()
    payload["positives"][0]["messages"][0]["content"] = "broken <image>"
    with pytest.raises(ValidationError, match="sample-p0"):
        EmbeddingRecord.model_validate(payload)


def test_jsonl_round_trip_and_line_context(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "records.jsonl"  # type: ignore[operator]
    record = _record("sample")
    write_jsonl((record,), path)
    assert read_jsonl(path) == (record,)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(JsonlDataError, match=r":1: blank JSONL"):
        read_jsonl(path)


def test_jsonl_round_trip_preserves_canonical_processor_extras(
    tmp_path: pytest.TempPathFactory,
) -> None:
    path = tmp_path / "opaque.jsonl"  # type: ignore[operator]
    base = _record("opaque")
    query = MultimodalItem(
        item_id="opaque-q",
        messages=(
            Message(
                role="user",
                content=(
                    MessagePart(
                        type="processor",
                        text="inspect <image>",
                        processor={"enabled": True, "thresholds": (0.25, None)},
                    ),
                ),
            ),
        ),
        images=(
            MediaRef(
                uri="image.jpg",
                metadata={"processor": {"enabled": False, "shape": (224, 224)}},
            ),
        ),
        objects=ObjectAnnotations(
            annotations=({"bbox": (1, 2, 3, 4), "visible": True, "label": None},)
        ),
    )
    record = EmbeddingRecord(
        sample_id=base.sample_id,
        task_id=base.task_id,
        query=query,
        positives=base.positives,
        negatives=base.negatives,
        metadata=base.metadata,
    )

    assert query.images[0].metadata["processor"] == {
        "enabled": False,
        "shape": (224, 224),
    }
    assert query.messages[0].content[0].model_extra == {
        "processor": {"enabled": True, "thresholds": (0.25, None)}
    }
    write_jsonl((record,), path)
    assert read_jsonl(path) == (record,)
    assert json.loads(path.read_text(encoding="utf-8"))["query"]["objects"]["annotations"] == [
        {"bbox": [1, 2, 3, 4], "label": None, "visible": True}
    ]


def test_swift_preserve_round_trip_keeps_every_positive() -> None:
    record = _record("multi", positive_count=2, positive_groups=("g0", "g1"))
    (sample,) = mira_to_swift(record, multi_positive="preserve")
    restored = swift_to_mira(sample)
    assert [item.item_id for item in restored.positives] == ["multi-p0", "multi-p1"]
    assert restored.metadata.positive_group_ids == ("g0", "g1")


def test_swift_expand_emits_one_sample_per_positive_without_loss() -> None:
    record = _record("multi", positive_count=3, positive_groups=("g0", "g1", "g2"))
    samples = mira_to_swift(record, multi_positive="expand")
    assert len(samples) == 3
    assert [sample["mira"]["positive_index"] for sample in samples] == [0, 1, 2]
    assert {sample["mira"]["positive_item_ids"][0] for sample in samples} == {
        "multi-p0",
        "multi-p1",
        "multi-p2",
    }


def test_swift_rejects_ambiguous_flat_media_for_multiple_positives() -> None:
    sample = {
        "sample_id": "s",
        "task_id": "t",
        "dataset_name": "d",
        "modality_pair": "image_to_image",
        "messages": [{"role": "user", "content": "query"}],
        "positive_messages": [
            [{"role": "user", "content": "<image>"}],
            [{"role": "user", "content": "<image>"}],
        ],
        "positive_images": ["a.jpg", "b.jpg"],
    }
    with pytest.raises(SwiftConversionError, match="ambiguous"):
        swift_to_mira(sample)


def test_layout_sources_and_false_negative_mask_are_explicit() -> None:
    first = _record("a", canonical="same", positive_groups=("shared",))
    second = _record("b", canonical="same", positive_groups=("shared",))
    result = build_group_layout((first, second))
    layout = result.layout
    assert layout.positive_mask.sum().item() == 2
    assert layout.explicit_negative_mask.sum().item() == 2
    assert layout.in_batch_negative_mask.sum().item() == 2
    assert layout.false_negative_mask.sum().item() == 2
    assert not torch.any(layout.candidate_mask & layout.false_negative_mask)
    assert len(result.flattened_items) == 6


def test_no_duplicate_sampler_is_deterministic_and_resume_exact() -> None:
    records = tuple(_record(str(index), canonical=f"p{index}") for index in range(8))
    sampler = NoDuplicateBatchSampler(records, global_batch_size=3, seed=17)
    iterator = iter(sampler)
    first_batch = next(iterator)
    state = sampler.state_dict()
    remaining = list(iterator)

    resumed = NoDuplicateBatchSampler(records, global_batch_size=3, seed=17)
    resumed.load_state_dict(state)
    assert list(resumed) == remaining
    fresh = NoDuplicateBatchSampler(records, global_batch_size=3, seed=17)
    assert [first_batch, *remaining] == list(fresh)


def test_no_duplicate_sampler_never_co_batches_configured_equivalence() -> None:
    records = (
        _record("a", canonical="same"),
        _record("b", canonical="same"),
        _record("c", canonical="other"),
    )
    sampler = NoDuplicateBatchSampler(records, global_batch_size=3, seed=1)
    for batch in sampler.global_batches():
        values = [records[index].metadata.canonical_product_id for index in batch]
        assert len(values) == len(set(values))


def test_sampler_resume_rejects_dataset_drift() -> None:
    original = (_record("a", canonical="one"), _record("b", canonical="two"))
    state = NoDuplicateBatchSampler(original, batch_size=2).state_dict()
    changed = (_record("a", canonical="changed"), _record("b", canonical="two"))
    sampler = NoDuplicateBatchSampler(changed, batch_size=2)
    with pytest.raises(SamplerStateError, match="dataset_fingerprint mismatch"):
        sampler.load_state_dict(json.loads(json.dumps(state)))
