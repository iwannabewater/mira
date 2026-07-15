"""Artifact identity, provenance, and atomic CLI writer regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mira.cli import _common
from mira.cli._common import (
    build_encoded_provenance,
    emit_json,
    file_sha256,
    load_encoded_collection,
    require_matching_encoded_provenance,
    save_encoded_collection,
    validate_encoded_provenance,
    write_json_file,
    write_new_artifacts,
)
from mira.cli.evaluate import _evaluation_slices, _load_relevance, _metric_cutoffs
from mira.config import resolve_config
from mira.core.exceptions import ContractError
from mira.integrations.ms_swift import SwiftTemplateSpec
from mira.integrations.ms_swift.template_bridge import template_fingerprint
from mira.mining import EncodedCollection

_ROOT = Path(__file__).parents[2]
_CONFIG = _ROOT / "configs/experiments/qwen3_vl_embedding_2b_single_gpu_lora.yaml"


def _provenance(checkpoint: Path, dimension: int = 2) -> dict[str, str]:
    resolved = resolve_config(_CONFIG)
    return build_encoded_provenance(
        str(checkpoint),
        resolved.config,
        config_checksum=resolved.checksum,
        embedding_dimension=dimension,
        template_checksum=template_fingerprint(SwiftTemplateSpec()),
    )


def _collection(checkpoint: Path, *, suffix: str = "") -> EncodedCollection:
    return EncodedCollection(
        (f"item{suffix}",),
        torch.tensor([[1.0, 0.0]]),
        ({"slice": "text"},),
        _provenance(checkpoint),
    )


def test_encoded_collection_round_trip_binds_provenance_and_bfloat16(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"model")
    collection = _collection(checkpoint)
    path = tmp_path / "encoded.pt"

    save_encoded_collection(collection, path)
    loaded = load_encoded_collection(path)

    assert loaded.item_ids == collection.item_ids
    assert dict(loaded.provenance) == dict(collection.provenance)
    assert loaded.checksum() == collection.checksum()
    bf16 = EncodedCollection(
        ("bf16",),
        torch.ones((1, 2), dtype=torch.bfloat16),
        ({},),
        _provenance(checkpoint),
    )
    assert len(bf16.checksum()) == 64
    with pytest.raises(TypeError):
        collection.provenance["new"] = "value"  # type: ignore[index]


def test_encoded_provenance_rejects_drift_and_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"model")
    collection = _collection(checkpoint)
    resolved = resolve_config(_CONFIG)
    template_checksum = template_fingerprint(SwiftTemplateSpec())

    validate_encoded_provenance(
        collection,
        str(checkpoint),
        resolved.config,
        config_checksum=resolved.checksum,
        template_checksum=template_checksum,
    )
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ContractError, match="provenance differs"):
        validate_encoded_provenance(
            collection,
            str(checkpoint),
            resolved.config,
            config_checksum=resolved.checksum,
            template_checksum=template_checksum,
        )
    other = EncodedCollection(
        ("other",),
        torch.ones((1, 2)),
        ({},),
        {**dict(collection.provenance), "config_checksum": "f" * 64},
    )
    with pytest.raises(ContractError, match="different provenance"):
        require_matching_encoded_provenance(collection, other)


def test_l2_provenance_rejects_vectors_that_reverse_cosine_ranking(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"model")
    resolved = resolve_config(_CONFIG)
    template_checksum = template_fingerprint(SwiftTemplateSpec())
    query = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor([[0.8, 0.6], [10.0, 10.0]])
    collection = EncodedCollection(
        ("angular-best", "magnitude-best"),
        candidates,
        ({}, {}),
        _provenance(checkpoint),
    )

    raw_ranking = torch.argsort(query @ candidates.transpose(0, 1), descending=True)
    normalized_candidates = candidates / torch.linalg.vector_norm(candidates, dim=1, keepdim=True)
    cosine_ranking = torch.argsort(query @ normalized_candidates.transpose(0, 1), descending=True)

    assert raw_ranking.tolist() == [[1, 0]]
    assert cosine_ranking.tolist() == [[0, 1]]
    with pytest.raises(ContractError, match="not L2-normalized"):
        validate_encoded_provenance(
            collection,
            str(checkpoint),
            resolved.config,
            config_checksum=resolved.checksum,
            template_checksum=template_checksum,
        )

    overflow = EncodedCollection(
        ("overflow",),
        torch.tensor([[1e38, 1e38]], dtype=torch.float32),
        ({},),
        _provenance(checkpoint),
    )
    with pytest.raises(ContractError, match="non-finite embedding L2 norm"):
        validate_encoded_provenance(
            overflow,
            str(checkpoint),
            resolved.config,
            config_checksum=resolved.checksum,
            template_checksum=template_checksum,
        )


def test_invalid_or_missing_encoded_provenance_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="provenance"):
        EncodedCollection(("item",), torch.ones((1, 2)), ({},), {"empty": ""})

    path = tmp_path / "malformed.pt"
    torch.save(
        {
            "item_ids": ["item"],
            "embeddings": torch.ones((1, 2)),
            "metadata": [{}],
            "provenance": {"key": 1},
        },
        path,
    )
    with pytest.raises(ContractError, match="string mapping"):
        load_encoded_collection(path)

    empty_id_path = tmp_path / "empty-id.pt"
    torch.save(
        {
            "item_ids": [""],
            "embeddings": torch.ones((1, 2)),
            "metadata": [{}],
            "provenance": {},
        },
        empty_id_path,
    )
    with pytest.raises(ContractError, match="non-empty string sequence"):
        load_encoded_collection(empty_id_path)


def test_multi_artifact_writer_rolls_back_prepare_and_commit_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    def write_ok(path: Path) -> None:
        path.write_text("ok", encoding="utf-8")

    def write_failure(path: Path) -> None:
        path.write_text("partial", encoding="utf-8")
        raise OSError("writer failed")

    with pytest.raises(OSError, match="writer failed"):
        write_new_artifacts(((first, write_ok), (second, write_failure)))
    assert not first.exists()
    assert not second.exists()
    assert not tuple(tmp_path.glob(".*.tmp-*"))

    real_link = _common.os.link
    calls = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("commit failed")
        real_link(source, destination)

    monkeypatch.setattr(_common.os, "link", fail_second_link)
    with pytest.raises(OSError, match="commit failed"):
        write_new_artifacts(((first, write_ok), (second, write_ok)))
    assert not first.exists()
    assert not second.exists()


@pytest.mark.parametrize("value", [object(), float("nan"), float("inf")])
def test_cli_json_writers_reject_non_json_or_nonfinite_values(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        emit_json({"value": value})

    output = tmp_path / "report.json"
    with pytest.raises((TypeError, ValueError)):
        write_json_file({"value": value}, output)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".report.json.tmp-*"))


def test_directory_hash_is_unambiguous_and_file_hash_tracks_content(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a").write_bytes(b"bc")
    (second / "ab").write_bytes(b"c")

    assert file_sha256(first) != file_sha256(second)
    file_path = tmp_path / "file.bin"
    file_path.write_bytes(b"one")
    before = file_sha256(file_path)
    file_path.write_bytes(b"two")
    assert file_sha256(file_path) != before


def test_relevance_artifact_requires_exact_ids_and_collection_checksums(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"model")
    queries = _collection(checkpoint)
    candidates = EncodedCollection(
        ("first", "second"),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ({}, {}),
        _provenance(checkpoint),
    )
    artifact = tmp_path / "relevance.pt"
    payload = {
        "schema_version": 1,
        "query_ids": list(queries.item_ids),
        "candidate_ids": list(candidates.item_ids),
        "query_collection_checksum": queries.checksum(),
        "candidate_collection_checksum": candidates.checksum(),
        "relevance": torch.tensor([[1.0, 0.0]]),
    }
    torch.save(payload, artifact)
    assert torch.equal(_load_relevance(artifact, queries, candidates), payload["relevance"])

    payload["candidate_ids"] = ["second", "first"]
    torch.save(payload, artifact)
    with pytest.raises(ContractError, match="candidate_ids"):
        _load_relevance(artifact, queries, candidates)

    payload["candidate_ids"] = list(candidates.item_ids)
    payload["candidate_collection_checksum"] = "0" * 64
    torch.save(payload, artifact)
    with pytest.raises(ContractError, match="candidate collection checksum"):
        _load_relevance(artifact, queries, candidates)

    payload["candidate_collection_checksum"] = candidates.checksum()
    payload["query_ids"] = ["different"]
    torch.save(payload, artifact)
    with pytest.raises(ContractError, match="query_ids"):
        _load_relevance(artifact, queries, candidates)

    payload["query_ids"] = list(queries.item_ids)
    payload["query_collection_checksum"] = "0" * 64
    torch.save(payload, artifact)
    with pytest.raises(ContractError, match="query collection checksum"):
        _load_relevance(artifact, queries, candidates)

    payload["query_collection_checksum"] = queries.checksum()
    payload["schema_version"] = 2
    torch.save(payload, artifact)
    with pytest.raises(ContractError, match="schema_version"):
        _load_relevance(artifact, queries, candidates)

    payload["schema_version"] = True
    torch.save(payload, artifact)
    with pytest.raises(ContractError, match="schema_version"):
        _load_relevance(artifact, queries, candidates)

    payload["schema_version"] = 1
    payload["relevance"] = "not-a-tensor"
    torch.save(payload, artifact)
    with pytest.raises(ContractError, match="must be a tensor"):
        _load_relevance(artifact, queries, candidates)

    payload["relevance"] = torch.ones((1, 2))
    payload["unexpected"] = True
    torch.save(payload, artifact)
    with pytest.raises(ContractError, match="strict versioned mapping"):
        _load_relevance(artifact, queries, candidates)


def test_evaluation_cutoff_and_slice_contracts() -> None:
    assert _metric_cutoffs(("map",), 4) == (4,)
    assert _metric_cutoffs(("recall@1", "mrr@3"), 4) == (1, 3)
    with pytest.raises(ContractError, match="exceed candidate count"):
        _metric_cutoffs(("recall@5",), 4)

    per_query = {"recall@1": torch.tensor([1.0, 0.0])}
    metadata = ({"modality": "text"}, {"modality": "image"})
    slices = _evaluation_slices(per_query, metadata, ("modality",))
    assert slices == {
        "modality=image": {"recall@1": 0.0},
        "modality=text": {"recall@1": 1.0},
    }
    with pytest.raises(ContractError, match="missing evaluation slices"):
        _evaluation_slices(per_query, metadata, ("language",))
