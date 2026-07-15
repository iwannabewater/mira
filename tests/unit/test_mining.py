"""Offline mining filters, provenance, and deterministic retrieval tests."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import pytest
import torch

from mira.core.exceptions import (
    ChecksumMismatchError,
    ConfigurationValidationError,
    ContractError,
    ManifestError,
)
from mira.core.manifests import ManifestEnvelope, ManifestKind
from mira.mining import (
    EncodedCollection,
    MetadataFalseNegativeFilter,
    MiningManifest,
    MiningPipeline,
    MiningRequest,
    PositiveAwareSimilarityFilter,
    TeacherScoreFilter,
)
from mira.mining.manifests import checksum_positive_mapping

if TYPE_CHECKING:
    from pathlib import Path


def _manifest_values() -> dict[str, object]:
    return {
        "run_id": "mine-001",
        "model_path": "checkpoint",
        "model_checksum": "a" * 64,
        "adapter_checksum": None,
        "instruction_template_checksum": "b" * 64,
        "normalization_rule": "l2",
        "random_seed": 17,
        "teacher_version": None,
    }


def _collections() -> tuple[EncodedCollection, EncodedCollection]:
    queries = EncodedCollection(
        ("q1",),
        torch.tensor([[1.0, 0.0]]),
        ({"canonical_product_id": "query-product"},),
    )
    corpus = EncodedCollection(
        ("p1", "duplicate", "hard", "easy"),
        torch.tensor(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.8, 0.2],
                [0.1, 0.9],
            ]
        ),
        (
            {"canonical_product_id": "positive"},
            {"canonical_product_id": "query-product"},
            {"canonical_product_id": "hard"},
            {"canonical_product_id": "easy"},
        ),
    )
    return queries, corpus


def test_mining_removes_positives_and_metadata_false_negatives() -> None:
    queries, corpus = _collections()
    pipeline = MiningPipeline(metadata_filter=MetadataFalseNegativeFilter())
    result = pipeline.run(
        MiningRequest(
            queries,
            corpus,
            {"q1": ("p1",)},
            top_k=2,
            retrieve_k=4,
            manifest_values=_manifest_values(),
        )
    )
    assert [row.candidate_id for row in result.negatives] == ["hard", "easy"]
    assert result.manifest.index_type == "torch_exact_inner_product"
    assert result.manifest.filters[0]["type"] == "metadata"


def test_tied_scores_preserve_corpus_order_for_rows_strata_and_checksum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries = EncodedCollection(("q1",), torch.tensor([[1.0, 0.0]]), ({},))
    corpus = EncodedCollection(
        ("positive", "first", "second", "third", "fourth", "fifth", "sixth"),
        torch.tensor([[1.0, 0.0]] * 7),
        ({}, {}, {}, {}, {}, {}, {}),
    )
    request = MiningRequest(
        queries,
        corpus,
        {"q1": ("positive",)},
        top_k=6,
        retrieve_k=7,
        manifest_values=_manifest_values(),
    )
    expected = [
        ("first", 1, "hard"),
        ("second", 2, "hard"),
        ("third", 3, "medium"),
        ("fourth", 4, "medium"),
        ("fifth", 5, "easy"),
        ("sixth", 6, "easy"),
    ]
    baseline = MiningPipeline().run(request)

    def reversed_tie_topk(
        input_tensor: torch.Tensor,
        k: int,
        dim: int,
        **options: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert dim == 1
        assert options == {"largest": True, "sorted": True}
        indices = torch.arange(
            input_tensor.shape[1] - 1,
            -1,
            -1,
            device=input_tensor.device,
        ).expand(input_tensor.shape[0], -1)[:, :k]
        return input_tensor.gather(dim, indices), indices

    monkeypatch.setattr(torch, "topk", reversed_tie_topk)
    with_unstable_topk = MiningPipeline().run(request)

    assert [(row.candidate_id, row.rank, row.difficulty) for row in baseline.negatives] == expected
    assert with_unstable_topk.negatives == baseline.negatives
    assert with_unstable_topk.manifest.mined_rows_checksum == baseline.manifest.mined_rows_checksum


def test_chunked_exact_mining_matches_dense_reference_across_block_boundaries() -> None:
    generator = torch.Generator().manual_seed(19)
    query_values = torch.randn(7, 8, generator=generator)
    corpus_values = torch.randn(31, 8, generator=generator)
    queries = EncodedCollection(
        tuple(f"q{index}" for index in range(7)),
        query_values,
        tuple({} for _ in range(7)),
    )
    corpus = EncodedCollection(
        tuple(f"c{index}" for index in range(31)),
        corpus_values,
        tuple({} for _ in range(31)),
    )
    positives = dict.fromkeys(queries.item_ids, ("c0",))
    request = MiningRequest(
        queries,
        corpus,
        positives,
        top_k=5,
        retrieve_k=12,
        manifest_values=_manifest_values(),
    )
    result = MiningPipeline(query_block_size=3, corpus_block_size=7).run(request)
    dense_scores = queries.embeddings.float() @ corpus.embeddings.float().T
    dense_indices = torch.argsort(
        dense_scores,
        dim=1,
        descending=True,
        stable=True,
    )[:, :12]
    expected: list[str] = []
    for row in dense_indices:
        retained = [
            corpus.item_ids[int(index)] for index in row if corpus.item_ids[int(index)] != "c0"
        ]
        expected.extend(retained[:5])
    observed = [row.candidate_id for row in result.negatives]
    assert observed == expected


@pytest.mark.parametrize("name", ["query_block_size", "corpus_block_size"])
def test_mining_rejects_invalid_block_sizes(name: str) -> None:
    with pytest.raises(ContractError, match=name):
        MiningPipeline(**{name: 0})


def test_positive_aware_filter_applies_ratio_and_requires_exact_coverage() -> None:
    queries, corpus = _collections()
    pipeline = MiningPipeline(similarity_filter=PositiveAwareSimilarityFilter(alpha=0.9))
    result = pipeline.run(
        MiningRequest(
            queries,
            corpus,
            {"q1": ("p1",)},
            top_k=2,
            retrieve_k=4,
            manifest_values=_manifest_values(),
        )
    )
    assert [row.candidate_id for row in result.negatives] == ["hard", "easy"]
    with pytest.raises(ContractError, match="exactly cover"):
        pipeline.run(
            MiningRequest(
                queries,
                corpus,
                {},
                top_k=1,
                retrieve_k=4,
                manifest_values=_manifest_values(),
            )
        )


def test_mining_fails_when_filters_leave_too_few_rows() -> None:
    queries, corpus = _collections()
    pipeline = MiningPipeline(metadata_filter=MetadataFalseNegativeFilter())
    with pytest.raises(ContractError, match="retained 2 negatives; required 3"):
        pipeline.run(
            MiningRequest(
                queries,
                corpus,
                {"q1": ("p1",)},
                top_k=3,
                retrieve_k=4,
                manifest_values=_manifest_values(),
            )
        )


def test_manifest_is_versioned_strict_new_only_and_checksum_verified(tmp_path: Path) -> None:
    queries, corpus = _collections()
    result = MiningPipeline().run(
        MiningRequest(
            queries,
            corpus,
            {"q1": ("p1",)},
            top_k=1,
            retrieve_k=4,
            manifest_values=_manifest_values(),
        )
    )
    path = tmp_path / "manifest.json"
    result.manifest.write(path)
    document = json.loads(path.read_text())
    assert document["schema_version"] == 1
    assert document["kind"] == "mining"
    assert document["payload"]["run_id"] == "mine-001"
    assert document["payload"]["positive_mapping_checksum"] == (
        result.manifest.positive_mapping_checksum
    )
    assert document["payload"]["mined_rows_checksum"] == result.manifest.mined_rows_checksum
    assert document["checksum"] == result.manifest.checksum()
    assert MiningManifest.read(path) == result.manifest
    assert len(result.manifest.checksum()) == 64
    with pytest.raises(FileExistsError):
        result.manifest.write(path)

    failing_path = tmp_path / "write-failure.json"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "mira.mining.manifests.os.link",
            lambda *_args: (_ for _ in ()).throw(OSError("simulated link failure")),
        )
        with pytest.raises(ManifestError, match="cannot write mining manifest"):
            result.manifest.write(failing_path)
    assert not failing_path.exists()
    assert not tuple(tmp_path.glob(".write-failure.json.tmp-*"))

    document["payload"]["top_k"] = 2
    path.write_text(json.dumps(document))
    with pytest.raises(ChecksumMismatchError, match="checksum mismatch"):
        MiningManifest.read(path)

    envelope = result.manifest.to_envelope()
    extra_payload = dict(envelope.payload)
    extra_payload["unexpected"] = True
    strict_path = tmp_path / "strict.json"
    ManifestEnvelope(
        kind=ManifestKind.MINING,
        payload=extra_payload,
        created_at=envelope.created_at,
    ).write(strict_path)
    with pytest.raises(ManifestError, match="payload fields differ"):
        MiningManifest.read(strict_path)

    missing_payload = dict(envelope.payload)
    del missing_payload["mined_rows_checksum"]
    missing_path = tmp_path / "missing.json"
    ManifestEnvelope(
        kind=ManifestKind.MINING,
        payload=missing_payload,
        created_at=envelope.created_at,
    ).write(missing_path)
    with pytest.raises(ManifestError, match=r"missing=.*mined_rows_checksum"):
        MiningManifest.read(missing_path)


def test_positive_mapping_checksum_is_canonical_and_unambiguous() -> None:
    first = checksum_positive_mapping(
        {
            "q2": ("positive-c",),
            "q1": ("positive-b", "positive-a"),
        }
    )
    reordered = checksum_positive_mapping(
        {
            "q1": ("positive-a", "positive-b"),
            "q2": ("positive-c",),
        }
    )
    assert first == reordered
    assert checksum_positive_mapping({"ab": ("c",)}) != checksum_positive_mapping({"a": ("bc",)})
    assert checksum_positive_mapping({"q": ("positive",)}) != checksum_positive_mapping(
        {"q": ("tampered",)}
    )


@pytest.mark.parametrize(
    ("positives", "match"),
    [
        ({1: ("positive",)}, "query IDs"),
        ({"q": "positive"}, "candidate ID sequences"),
        ({"q": ("",)}, "non-empty strings"),
        ({"q": ("positive", "positive")}, "must be unique"),
    ],
)
def test_positive_mapping_checksum_rejects_noncanonical_values(
    positives: dict[object, object],
    match: str,
) -> None:
    with pytest.raises(ManifestError, match=match):
        checksum_positive_mapping(positives)  # type: ignore[arg-type]


def test_manifest_verifies_positive_membership_and_ordered_mined_rows(tmp_path: Path) -> None:
    queries, corpus = _collections()
    result = MiningPipeline().run(
        MiningRequest(
            queries,
            corpus,
            {"q1": ("p1",)},
            top_k=2,
            retrieve_k=4,
            manifest_values=_manifest_values(),
        )
    )
    result.manifest.verify_positive_mapping({"q1": ("p1",)})
    result.manifest.verify_mined_rows(result.negatives)

    with pytest.raises(ChecksumMismatchError, match="positive mapping checksum mismatch"):
        result.manifest.verify_positive_mapping({"q1": ("duplicate",)})
    with pytest.raises(ChecksumMismatchError, match="rows checksum mismatch"):
        result.manifest.verify_mined_rows(reversed(result.negatives))
    with pytest.raises(ChecksumMismatchError, match="rows checksum mismatch"):
        result.manifest.verify_mined_rows(
            (replace(result.negatives[0], candidate_id="tampered"), *result.negatives[1:])
        )

    output = tmp_path / "mined.jsonl"
    output.write_text(
        "".join(f"{json.dumps(asdict(row), sort_keys=True)}\n" for row in result.negatives),
        encoding="utf-8",
    )
    result.manifest.verify_output_file(output)
    output.write_text(
        "".join(
            f"{json.dumps(asdict(row), sort_keys=True)}\n" for row in reversed(result.negatives)
        ),
        encoding="utf-8",
    )
    with pytest.raises(ChecksumMismatchError, match="rows checksum mismatch"):
        result.manifest.verify_output_file(output)


def test_mining_output_reader_rejects_ambiguous_or_extra_json_fields(tmp_path: Path) -> None:
    queries, corpus = _collections()
    result = MiningPipeline().run(
        MiningRequest(
            queries,
            corpus,
            {"q1": ("p1",)},
            top_k=1,
            retrieve_k=4,
            manifest_values=_manifest_values(),
        )
    )
    output = tmp_path / "mined.jsonl"
    output.write_text(
        '{"candidate_id":"duplicate","difficulty":"hard","query_id":"q1",'
        '"query_id":"shadow","rank":1,"score":0.95}\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="duplicate JSON object key 'query_id'"):
        result.manifest.verify_output_file(output)

    row = asdict(result.negatives[0])
    row["unexpected"] = True
    output.write_text(f"{json.dumps(row)}\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="output row fields differ"):
        result.manifest.verify_output_file(output)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"query_id": ""}, "query_id"),
        ({"candidate_id": ""}, "candidate_id"),
        ({"score": True}, "score must be a number"),
        ({"score": float("inf")}, "score must be finite"),
        ({"rank": 0}, "rank must be a positive integer"),
        ({"difficulty": "impossible"}, "unknown mined row difficulty"),
    ],
)
def test_mined_rows_checksum_rejects_invalid_row_values(
    change: dict[str, object],
    match: str,
) -> None:
    queries, corpus = _collections()
    result = MiningPipeline().run(
        MiningRequest(
            queries,
            corpus,
            {"q1": ("p1",)},
            top_k=1,
            retrieve_k=4,
            manifest_values=_manifest_values(),
        )
    )
    with pytest.raises(ManifestError, match=match):
        result.manifest.verify_mined_rows((replace(result.negatives[0], **change),))


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("\n", "contains blank line"),
        ("{\n", "cannot decode mining output"),
        ("[]\n", "must be an object"),
        (
            '{"candidate_id":"duplicate","difficulty":"hard","query_id":"q1",'
            '"rank":2,"score":0.95}\n',
            "rank exceeds top_k",
        ),
    ],
)
def test_mining_output_reader_rejects_malformed_rows(
    tmp_path: Path,
    content: str,
    match: str,
) -> None:
    queries, corpus = _collections()
    result = MiningPipeline().run(
        MiningRequest(
            queries,
            corpus,
            {"q1": ("p1",)},
            top_k=1,
            retrieve_k=4,
            manifest_values=_manifest_values(),
        )
    )
    output = tmp_path / "mined.jsonl"
    output.write_text(content, encoding="utf-8")
    with pytest.raises(ManifestError, match=match):
        result.manifest.verify_output_file(output)

    with pytest.raises(ManifestError, match="cannot read mining output"):
        result.manifest.verify_output_file(tmp_path / "missing.jsonl")


@pytest.mark.parametrize(
    ("positives", "match"),
    [
        ({}, "exactly cover"),
        ({"q1": ()}, "at least one candidate"),
        ({"q1": ("p1", "p1")}, "unique candidate IDs"),
        ({"q1": ("",)}, "non-empty string candidate IDs"),
        ({"q1": ("p1",), "unknown": ("p1",)}, "unknown=.*unknown"),
    ],
)
def test_positive_mapping_requires_exact_nonempty_unique_coverage(
    positives: dict[str, tuple[str, ...]],
    match: str,
) -> None:
    queries, corpus = _collections()
    with pytest.raises(ContractError, match=match):
        MiningPipeline().run(
            MiningRequest(
                queries,
                corpus,
                positives,
                top_k=1,
                retrieve_k=4,
                manifest_values=_manifest_values(),
            )
        )


def test_manifest_values_cannot_override_mining_derived_bindings() -> None:
    queries, corpus = _collections()
    values = _manifest_values()
    values["positive_mapping_checksum"] = "0" * 64
    with pytest.raises(ContractError, match="cannot override mining-derived fields"):
        MiningPipeline().run(
            MiningRequest(
                queries,
                corpus,
                {"q1": ("p1",)},
                top_k=1,
                retrieve_k=4,
                manifest_values=values,
            )
        )


def test_encoded_collection_rejects_misaligned_metadata() -> None:
    with pytest.raises(ContractError, match="metadata"):
        EncodedCollection(("one",), torch.ones(1, 2), ())


def test_encoded_collection_rejects_invalid_ids_shapes_values_and_provenance() -> None:
    with pytest.raises(ContractError, match="IDs"):
        EncodedCollection((), torch.empty((0, 2)), ())
    with pytest.raises(ContractError, match="non-empty strings"):
        EncodedCollection(("",), torch.ones((1, 2)), ({},))
    with pytest.raises(ContractError, match="non-empty strings"):
        EncodedCollection((1,), torch.ones((1, 2)), ({},))  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="IDs"):
        EncodedCollection(("same", "same"), torch.ones((2, 2)), ({}, {}))
    with pytest.raises(ContractError, match="shape"):
        EncodedCollection(("one",), torch.ones(2), ({},))
    with pytest.raises(ContractError, match="finite floating-point"):
        EncodedCollection(("one",), torch.ones((1, 2), dtype=torch.int64), ({},))
    with pytest.raises(ContractError, match="finite floating-point"):
        EncodedCollection(("one",), torch.tensor([[float("nan"), 0.0]]), ({},))
    with pytest.raises(ContractError, match="provenance"):
        EncodedCollection(("one",), torch.ones((1, 2)), ({},), {"": "value"})


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (({"score": float("nan")},), "non-finite"),
        (({1: "coerced"},), "non-string object key"),
        (({"value": object()},), "unsupported value"),
        (("not-an-object",), "JSON objects"),
    ],
)
def test_encoded_collection_rejects_noncanonical_metadata(
    metadata: tuple[object, ...], message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        EncodedCollection(("one",), torch.ones((1, 2)), metadata)  # type: ignore[arg-type]


def test_encoded_collection_copies_and_canonicalizes_metadata() -> None:
    source = {"nested": {"values": (1, 2)}}
    collection = EncodedCollection(("one",), torch.ones((1, 2)), (source,))
    checksum = collection.checksum()
    source["nested"] = {"values": (3,)}

    assert collection.metadata == ({"nested": {"values": (1, 2)}},)
    assert collection.checksum() == checksum
    with pytest.raises(TypeError):
        collection.metadata[0]["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        collection.metadata[0]["nested"]["values"] += (3,)  # type: ignore[index,operator]


def test_encoded_collection_rejects_cyclic_metadata() -> None:
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    with pytest.raises(ContractError, match="cycle"):
        EncodedCollection(("one",), torch.ones((1, 2)), (cycle,))


def test_all_mining_filters_validate_configuration_and_nonfinite_scores() -> None:
    metadata = MetadataFalseNegativeFilter(("sku",))
    assert not metadata.keep({"sku": "same"}, {"sku": "same"})
    assert metadata.keep({"sku": "one"}, {"sku": "two"})
    with pytest.raises(ConfigurationValidationError, match="non-empty and unique"):
        MetadataFalseNegativeFilter(())
    with pytest.raises(ConfigurationValidationError, match="non-empty and unique"):
        MetadataFalseNegativeFilter(("sku", "sku"))
    with pytest.raises(ConfigurationValidationError, match="blank"):
        MetadataFalseNegativeFilter((" ",))

    for value in (0.0, -1.0, float("nan")):
        with pytest.raises(ConfigurationValidationError, match="finite and positive"):
            PositiveAwareSimilarityFilter(value)
    similarity = PositiveAwareSimilarityFilter(0.9)
    with pytest.raises(ValueError, match="finite"):
        similarity.keep(float("inf"), 1.0)
    with pytest.raises(ValueError, match="finite"):
        similarity.keep(1.0, float("nan"))

    with pytest.raises(ConfigurationValidationError, match="teacher threshold"):
        TeacherScoreFilter(float("inf"))
    teacher = TeacherScoreFilter(0.5)
    assert teacher.keep(0.5)
    assert not teacher.keep(0.6)
    with pytest.raises(ValueError, match="teacher score"):
        teacher.keep(float("nan"))


def test_mining_request_rejects_dimension_bounds_and_missing_corpus_positive() -> None:
    queries, corpus = _collections()
    wrong_dimension = EncodedCollection(
        corpus.item_ids,
        torch.ones((len(corpus.item_ids), 3)),
        corpus.metadata,
    )
    with pytest.raises(ContractError, match="dimensions"):
        MiningPipeline().run(
            MiningRequest(
                queries,
                wrong_dimension,
                {"q1": ("p1",)},
                top_k=1,
                retrieve_k=4,
                manifest_values=_manifest_values(),
            )
        )
    for top_k, retrieve_k in ((0, 1), (2, 1), (1, 5)):
        with pytest.raises(ContractError, match="top_k"):
            MiningPipeline().run(
                MiningRequest(
                    queries,
                    corpus,
                    {"q1": ("p1",)},
                    top_k=top_k,
                    retrieve_k=retrieve_k,
                    manifest_values=_manifest_values(),
                )
            )
    for top_k, retrieve_k in ((True, 1), (1, False), (1.0, 2)):
        with pytest.raises(ContractError, match="must be integers"):
            MiningPipeline().run(
                MiningRequest(
                    queries,
                    corpus,
                    {"q1": ("p1",)},
                    top_k=top_k,  # type: ignore[arg-type]
                    retrieve_k=retrieve_k,  # type: ignore[arg-type]
                    manifest_values=_manifest_values(),
                )
            )
    with pytest.raises(ContractError, match="absent from corpus"):
        MiningPipeline().run(
            MiningRequest(
                queries,
                corpus,
                {"q1": ("missing",)},
                top_k=1,
                retrieve_k=4,
                manifest_values=_manifest_values(),
            )
        )


def test_mining_request_rejects_noncanonical_mapping_inputs() -> None:
    queries, corpus = _collections()
    with pytest.raises(ContractError, match="positive mapping must be a mapping"):
        MiningPipeline().run(
            MiningRequest(
                queries,
                corpus,
                [("q1", ("p1",))],  # type: ignore[arg-type]
                top_k=1,
                retrieve_k=4,
                manifest_values=_manifest_values(),
            )
        )
    with pytest.raises(ContractError, match="tuple of candidate IDs"):
        MiningPipeline().run(
            MiningRequest(
                queries,
                corpus,
                {"q1": ["p1"]},  # type: ignore[dict-item]
                top_k=1,
                retrieve_k=4,
                manifest_values=_manifest_values(),
            )
        )
    with pytest.raises(ContractError, match="manifest_values must be a mapping"):
        MiningPipeline().run(
            MiningRequest(
                queries,
                corpus,
                {"q1": ("p1",)},
                top_k=1,
                retrieve_k=4,
                manifest_values=[],  # type: ignore[arg-type]
            )
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"schema_version": True}, "schema_version"),
        ({"run_id": ""}, "empty required"),
        ({"run_id": 1}, "must be strings"),
        ({"model_checksum": "bad"}, "SHA-256"),
        ({"adapter_checksum": "A" * 64}, "SHA-256"),
        ({"adapter_checksum": 7}, "SHA-256"),
        ({"positive_mapping_checksum": "bad"}, "SHA-256"),
        ({"positive_mapping_checksum": 7}, "SHA-256"),
        ({"mined_rows_checksum": "A" * 64}, "SHA-256"),
        ({"embedding_dimension": True}, "embedding_dimension must be an integer"),
        ({"top_k": True}, "top_k must be an integer"),
        ({"random_seed": True}, "random_seed must be an integer"),
        ({"embedding_dimension": 0}, "must be positive"),
        ({"index_parameters": []}, "index_parameters must be a mapping"),
        ({"filters": ({"ok": True}, "bad")}, "filters must be a tuple"),
        ({"teacher_version": ""}, "teacher_version"),
    ],
)
def test_mining_manifest_rejects_invalid_types_and_values(
    changes: dict[str, object],
    match: str,
) -> None:
    queries, corpus = _collections()
    manifest = (
        MiningPipeline()
        .run(
            MiningRequest(
                queries,
                corpus,
                {"q1": ("p1",)},
                top_k=1,
                retrieve_k=4,
                manifest_values=_manifest_values(),
            )
        )
        .manifest
    )
    with pytest.raises(ManifestError, match=match):
        replace(manifest, **changes)


def test_mining_manifest_rejects_wrong_kind_and_invalid_nested_payloads() -> None:
    queries, corpus = _collections()
    manifest = (
        MiningPipeline()
        .run(
            MiningRequest(
                queries,
                corpus,
                {"q1": ("p1",)},
                top_k=1,
                retrieve_k=4,
                manifest_values=_manifest_values(),
            )
        )
        .manifest
    )
    envelope = manifest.to_envelope()
    with pytest.raises(ManifestError, match="expected a mining manifest"):
        MiningManifest.from_envelope(
            ManifestEnvelope(
                kind=ManifestKind.EXPORT,
                payload=envelope.payload,
                created_at=envelope.created_at,
            )
        )

    for name, invalid, match in (
        ("index_parameters", [], "index_parameters"),
        ("filters", {}, "filters"),
    ):
        payload = dict(envelope.payload)
        payload[name] = invalid
        invalid_envelope = ManifestEnvelope(
            kind=ManifestKind.MINING,
            payload=payload,
            created_at=envelope.created_at,
        )
        with pytest.raises(ManifestError, match=match):
            MiningManifest.from_envelope(invalid_envelope)
