"""Installed CLI workflows that do not require external model weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from typer.testing import CliRunner

from mira.cli._common import build_encoded_provenance, save_encoded_collection
from mira.cli.app import app
from mira.cli.mine import _load_positive_mapping
from mira.config import resolve_config
from mira.core.exceptions import ChecksumMismatchError, ContractError
from mira.inference import load_encoded_artifact, write_encoded_artifact
from mira.integrations.ms_swift import SwiftTemplateSpec
from mira.integrations.ms_swift.template_bridge import template_fingerprint
from mira.mining import EncodedCollection, MiningManifest

if TYPE_CHECKING:
    from typing import Literal

_ROOT = Path(__file__).parents[2]
_CONFIG = _ROOT / "configs/experiments/qwen3_vl_embedding_2b_single_gpu_lora.yaml"


def _save_collection(
    path: Path,
    item_ids: list[str],
    embeddings: torch.Tensor,
    checkpoint: Path,
    *,
    role: Literal["query", "document"],
) -> EncodedCollection:
    resolved = resolve_config(_CONFIG)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    collection = EncodedCollection(
        tuple(item_ids),
        embeddings,
        tuple({"modality_pair": "text_to_text"} for _ in item_ids),
        build_encoded_provenance(
            str(checkpoint),
            resolved.config,
            config_checksum=resolved.checksum,
            embedding_dimension=int(embeddings.shape[1]),
            template_checksum=template_fingerprint(SwiftTemplateSpec()),
        ),
    )
    write_encoded_artifact(
        collection,
        path,
        role=role,
        normalized=True,
    )
    return collection


def _assert_evaluation_failure(
    runner: CliRunner,
    arguments: list[str],
    *,
    output: Path,
    dimension: int,
    match: str,
) -> None:
    result = runner.invoke(
        app,
        [
            *arguments,
            "--output",
            str(output),
            "--dimension",
            str(dimension),
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ContractError)
    assert match in str(result.exception)
    assert not output.exists()


def test_cli_lists_every_required_command_and_inspects_config() -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    for name in (
        "inspect",
        "init-config",
        "validate-data",
        "train",
        "mine",
        "migrate-encoded",
        "evaluate",
        "encode",
        "export",
        "benchmark",
        "doctor",
    ):
        assert name in help_result.stdout

    result = runner.invoke(app, ["inspect", "--config", str(_CONFIG)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "valid"
    assert payload["model"] == "Qwen/Qwen3-VL-Embedding-2B"


@pytest.mark.parametrize(
    ("preset", "source_name", "model_id"),
    [
        (
            "qwen3-vl-embedding-2b-single-gpu-lora",
            "qwen3_vl_embedding_2b_single_gpu_lora.yaml",
            "Qwen/Qwen3-VL-Embedding-2B",
        ),
        (
            "qwen3-vl-embedding-8b-ecommerce",
            "qwen3_vl_embedding_ecommerce.yaml",
            "Qwen/Qwen3-VL-Embedding-8B",
        ),
    ],
)
def test_cli_init_config_publishes_exact_reference_without_replacement(
    tmp_path: Path,
    preset: str,
    source_name: str,
    model_id: str,
) -> None:
    runner = CliRunner()
    output = tmp_path / "nested" / "mira.yaml"
    result = runner.invoke(
        app,
        ["init-config", "--preset", preset, "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "created"
    assert payload["preset"] == preset
    assert output.read_bytes() == (_ROOT / "configs" / "experiments" / source_name).read_bytes()
    assert resolve_config(output).config.model.name_or_path == model_id

    repeated = runner.invoke(
        app,
        ["init-config", "--preset", preset, "--output", str(output)],
    )
    assert repeated.exit_code != 0
    assert isinstance(repeated.exception, ContractError)
    assert "already exists" in str(repeated.exception)


def test_cli_migrates_legacy_encoded_bundle_with_explicit_semantics(tmp_path: Path) -> None:
    runner = CliRunner()
    legacy = tmp_path / "legacy.pt"
    collection = EncodedCollection(
        ("one", "two"),
        torch.nn.functional.normalize(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1),
        ({"source": "legacy"}, {"source": "legacy"}),
        {"normalization_rule": "l2_normalized_inner_product", "schema": "mira.encoded.v1"},
    )
    save_encoded_collection(collection, legacy)
    output = tmp_path / "migrated"
    result = runner.invoke(
        app,
        [
            "migrate-encoded",
            "--input",
            str(legacy),
            "--output",
            str(output),
            "--role",
            "query",
            "--normalization",
            "normalized",
            "--shard-size",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["role"] == "query"
    assert payload["shards"] == 2
    migrated = load_encoded_artifact(
        output,
        expected_manifest_sha256=payload["manifest_sha256"],
        expected_role="query",
    )
    assert migrated.checksum() == collection.checksum()

    incompatible = runner.invoke(
        app,
        [
            "migrate-encoded",
            "--input",
            str(legacy),
            "--output",
            str(tmp_path / "invalid"),
            "--role",
            "query",
            "--normalization",
            "unnormalized",
        ],
    )
    assert incompatible.exit_code != 0
    assert isinstance(incompatible.exception, ContractError)
    assert "claims L2-normalized" in str(incompatible.exception)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ('{"q":["first"],"q":["second"]}', "duplicate JSON object key 'q'"),
        ('{"q":[NaN]}', "nonstandard JSON constant 'NaN'"),
        ('{"q":[Infinity]}', "nonstandard JSON constant 'Infinity'"),
        ('{"q":[-Infinity]}', "nonstandard JSON constant '-Infinity'"),
        ("{", "cannot read positive mapping"),
        ("[]", "root must be an object"),
        ('{"q":"positive"}', "values must be candidate ID lists"),
        ('{"q":[1]}', "candidate IDs must be strings"),
    ],
)
def test_positive_mapping_loader_rejects_ambiguous_json(
    tmp_path: Path,
    content: str,
    match: str,
) -> None:
    positives = tmp_path / "positives.json"
    positives.write_text(content, encoding="utf-8")
    with pytest.raises(ContractError, match=match):
        _load_positive_mapping(positives)


def test_cli_mine_and_evaluate_encoded_bundles(tmp_path: Path) -> None:
    runner = CliRunner()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"reference-checkpoint")
    queries = tmp_path / "queries.pt"
    candidates = tmp_path / "candidates.pt"
    query_embeddings = torch.zeros((1, 512))
    query_embeddings[0, 0] = 1.0
    candidate_embeddings = torch.zeros((12, 512))
    candidate_embeddings[0, 0] = 1.0
    candidate_embeddings[1, :2] = torch.tensor([0.8, 0.6])
    candidate_embeddings[2:, 1] = 1.0
    query_collection = _save_collection(
        queries,
        ["q"],
        query_embeddings,
        checkpoint,
        role="query",
    )
    candidate_collection = _save_collection(
        candidates,
        ["positive", "hard", "easy", *(f"filler-{index}" for index in range(9))],
        candidate_embeddings,
        checkpoint,
        role="document",
    )
    positives = tmp_path / "positives.json"
    positives.write_text(json.dumps({"q": ["positive"]}))
    mined = tmp_path / "mined.jsonl"
    mining_manifest = tmp_path / "mining-manifest.json"
    mine_arguments = [
        "mine",
        "--config",
        str(_CONFIG),
        "--queries",
        str(queries),
        "--corpus",
        str(candidates),
        "--positives",
        str(positives),
        "--output",
        str(mined),
        "--manifest",
        str(mining_manifest),
        "--model-checkpoint",
        str(checkpoint),
        "--top-k",
        "1",
        "--retrieve-k",
        "3",
        "--run-id",
        "test-mine",
    ]
    mine_result = runner.invoke(app, mine_arguments)
    assert mine_result.exit_code == 0, mine_result.output
    assert json.loads(mined.read_text())["candidate_id"] == "hard"
    manifest_document = json.loads(mining_manifest.read_text())
    assert manifest_document["payload"]["run_id"] == "test-mine"
    assert len(manifest_document["payload"]["positive_mapping_checksum"]) == 64
    assert len(manifest_document["payload"]["mined_rows_checksum"]) == 64
    loaded_manifest = MiningManifest.read(mining_manifest)
    loaded_manifest.verify_positive_mapping({"q": ("positive",)})
    loaded_manifest.verify_output_file(mined)

    repeated_mine = runner.invoke(app, mine_arguments)
    assert repeated_mine.exit_code != 0
    assert isinstance(repeated_mine.exception, ContractError)
    assert "must not already exist" in str(repeated_mine.exception)

    relevance = tmp_path / "relevance.pt"
    torch.save(
        {
            "schema_version": 1,
            "query_ids": list(query_collection.item_ids),
            "candidate_ids": list(candidate_collection.item_ids),
            "query_collection_checksum": query_collection.checksum(),
            "candidate_collection_checksum": candidate_collection.checksum(),
            "relevance": torch.tensor([[3.0, *([0.0] * 11)]]),
        },
        relevance,
    )
    report = tmp_path / "evaluation.json"
    evaluate_arguments = [
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(_CONFIG),
        "--queries",
        str(queries),
        "--candidates",
        str(candidates),
        "--relevance",
        str(relevance),
    ]
    _assert_evaluation_failure(
        runner,
        evaluate_arguments,
        output=tmp_path / "invalid-evaluation.json",
        dimension=2,
        match="not one of the configured trained evaluation dimensions",
    )
    _assert_evaluation_failure(
        runner,
        evaluate_arguments,
        output=tmp_path / "oversized-evaluation.json",
        dimension=1024,
        match="exceed encoded width 512",
    )

    evaluate_result = runner.invoke(
        app,
        [
            *evaluate_arguments,
            "--output",
            str(report),
            "--dimension",
            "512",
        ],
    )
    assert evaluate_result.exit_code == 0, evaluate_result.output
    metrics = json.loads(report.read_text())["metrics"]
    assert metrics["recall@1"] == 1.0
    assert metrics["mrr@10"] == 1.0

    mined.write_text(
        mined.read_text(encoding="utf-8").replace('"candidate_id":"hard"', '"candidate_id":"easy"'),
        encoding="utf-8",
    )
    with pytest.raises(ChecksumMismatchError, match="rows checksum mismatch"):
        loaded_manifest.verify_output_file(mined)


def test_cli_mine_rolls_back_both_artifacts_when_prepared_artifact_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"reference-checkpoint")
    queries = tmp_path / "queries.pt"
    candidates = tmp_path / "candidates.pt"
    _save_collection(
        queries,
        ["q"],
        torch.tensor([[1.0, 0.0]]),
        checkpoint,
        role="query",
    )
    _save_collection(
        candidates,
        ["positive", "hard", "easy"],
        torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]),
        checkpoint,
        role="document",
    )
    positives = tmp_path / "positives.json"
    positives.write_text(json.dumps({"q": ["positive"]}))
    mined = tmp_path / "mined.jsonl"
    mining_manifest = tmp_path / "mining-manifest.json"

    original_manifest_write = MiningManifest.write

    def fail_manifest_write(_manifest: MiningManifest, path: Path) -> None:
        path.write_text("partial", encoding="utf-8")
        raise OSError("simulated manifest failure")

    monkeypatch.setattr(MiningManifest, "write", fail_manifest_write)
    result = runner.invoke(
        app,
        [
            "mine",
            "--config",
            str(_CONFIG),
            "--queries",
            str(queries),
            "--corpus",
            str(candidates),
            "--positives",
            str(positives),
            "--output",
            str(mined),
            "--manifest",
            str(mining_manifest),
            "--model-checkpoint",
            str(checkpoint),
            "--top-k",
            "1",
            "--retrieve-k",
            "3",
        ],
    )

    assert result.exit_code != 0
    assert not mined.exists()
    assert not mining_manifest.exists()
    assert not tuple(tmp_path.glob(".*.tmp-*"))

    monkeypatch.setattr(MiningManifest, "write", original_manifest_write)

    def fail_output_verification(_manifest: MiningManifest, _path: Path) -> None:
        raise ChecksumMismatchError("simulated output checksum mismatch")

    monkeypatch.setattr(MiningManifest, "verify_output_file", fail_output_verification)
    result = runner.invoke(
        app,
        [
            "mine",
            "--config",
            str(_CONFIG),
            "--queries",
            str(queries),
            "--corpus",
            str(candidates),
            "--positives",
            str(positives),
            "--output",
            str(mined),
            "--manifest",
            str(mining_manifest),
            "--model-checkpoint",
            str(checkpoint),
            "--top-k",
            "1",
            "--retrieve-k",
            "3",
        ],
    )
    assert result.exit_code != 0
    assert not mined.exists()
    assert not mining_manifest.exists()
    assert not tuple(tmp_path.glob(".*.tmp-*"))


def test_cli_benchmark_writes_measured_report(tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "benchmark.json"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--config",
            str(_CONFIG),
            "--queries",
            "4",
            "--candidates",
            "8",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["status"] == "measured"
    assert payload["query_count"] == 4
