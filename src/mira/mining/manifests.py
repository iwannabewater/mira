"""Immutable, versioned provenance for an offline mining run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from mira.core.exceptions import ChecksumMismatchError, ManifestError
from mira.core.manifests import (
    MANIFEST_SCHEMA_VERSION,
    JSONValue,
    ManifestEnvelope,
    ManifestKind,
    canonical_json_bytes,
    sha256_digest,
)
from mira.core.serialization import strict_json_loads

if TYPE_CHECKING:
    from pathlib import Path

_SHA256_LENGTH = 64
_POSITIVE_MAPPING_DOMAIN = "mira.mining.positive-mapping.v1"
_MINED_ROWS_DOMAIN = b"mira.mining.mined-rows.v1\x00"
_MINED_ROW_FIELDS = {"candidate_id", "difficulty", "query_id", "rank", "score"}
_DIFFICULTIES = frozenset({"easy", "hard", "medium"})


class _MinedRowLike(Protocol):
    @property
    def query_id(self) -> str: ...

    @property
    def candidate_id(self) -> str: ...

    @property
    def score(self) -> float: ...

    @property
    def rank(self) -> int: ...

    @property
    def difficulty(self) -> str: ...


@dataclass(frozen=True, slots=True)
class _DecodedMinedRow:
    query_id: str
    candidate_id: str
    score: float
    rank: int
    difficulty: str


_PAYLOAD_FIELDS = {
    "adapter_checksum",
    "corpus_checksum",
    "embedding_dimension",
    "filters",
    "index_parameters",
    "index_type",
    "instruction_template_checksum",
    "model_checksum",
    "model_path",
    "normalization_rule",
    "mined_rows_checksum",
    "positive_mapping_checksum",
    "query_checksum",
    "random_seed",
    "run_id",
    "teacher_version",
    "top_k",
}


@dataclass(frozen=True, slots=True)
class MiningManifest:
    """All values required to reproduce and audit mined negatives."""

    run_id: str
    model_path: str
    model_checksum: str
    adapter_checksum: str | None
    instruction_template_checksum: str
    corpus_checksum: str
    query_checksum: str
    positive_mapping_checksum: str
    mined_rows_checksum: str
    embedding_dimension: int
    normalization_rule: str
    index_type: str
    index_parameters: dict[str, Any]
    top_k: int
    filters: tuple[dict[str, Any], ...]
    random_seed: int
    teacher_version: str | None
    generated_at: str
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate required provenance without guessing missing values."""
        if type(self.schema_version) is not int or self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestError(
                f"unsupported mining manifest schema_version {self.schema_version}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )
        required = {
            "run_id": self.run_id,
            "model_path": self.model_path,
            "model_checksum": self.model_checksum,
            "instruction_template_checksum": self.instruction_template_checksum,
            "corpus_checksum": self.corpus_checksum,
            "query_checksum": self.query_checksum,
            "normalization_rule": self.normalization_rule,
            "index_type": self.index_type,
            "generated_at": self.generated_at,
        }
        if any(not isinstance(value, str) for value in required.values()):
            raise ManifestError("mining manifest required string fields must be strings")
        missing = tuple(name for name, value in required.items() if not value.strip())
        if missing:
            raise ManifestError(f"mining manifest has empty required fields: {missing}")
        for name, value in (
            ("model_checksum", self.model_checksum),
            ("adapter_checksum", self.adapter_checksum),
            ("instruction_template_checksum", self.instruction_template_checksum),
            ("corpus_checksum", self.corpus_checksum),
            ("query_checksum", self.query_checksum),
            ("positive_mapping_checksum", self.positive_mapping_checksum),
            ("mined_rows_checksum", self.mined_rows_checksum),
        ):
            if value is not None:
                _validate_sha256(value, name)
        if isinstance(self.embedding_dimension, bool) or not isinstance(
            self.embedding_dimension, int
        ):
            raise ManifestError("mining embedding_dimension must be an integer")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ManifestError("mining top_k must be an integer")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ManifestError("mining random_seed must be an integer")
        if self.embedding_dimension <= 0 or self.top_k <= 0 or self.random_seed < 0:
            raise ManifestError("mining dimensions/top_k must be positive and seed non-negative")
        if not isinstance(self.index_parameters, dict):
            raise ManifestError("mining index_parameters must be a mapping")
        if not isinstance(self.filters, tuple) or any(
            not isinstance(value, dict) for value in self.filters
        ):
            raise ManifestError("mining filters must be a tuple of mappings")
        if self.teacher_version is not None and (
            not isinstance(self.teacher_version, str) or not self.teacher_version.strip()
        ):
            raise ManifestError("mining teacher_version must be null or a non-empty string")
        self.to_envelope()

    @classmethod
    def create(cls, **values: Any) -> MiningManifest:
        """Create a manifest with a timezone-aware generation timestamp."""
        values.setdefault("generated_at", datetime.now(UTC).isoformat())
        return cls(**values)

    def checksum(self) -> str:
        """Return the checksum persisted in the versioned envelope."""
        return self.to_envelope().checksum

    def to_envelope(self) -> ManifestEnvelope:
        """Return the strict, checksum-protected generic manifest envelope."""
        return ManifestEnvelope(
            kind=ManifestKind.MINING,
            payload={
                "adapter_checksum": self.adapter_checksum,
                "corpus_checksum": self.corpus_checksum,
                "embedding_dimension": self.embedding_dimension,
                "filters": [dict(value) for value in self.filters],
                "index_parameters": self.index_parameters,
                "index_type": self.index_type,
                "instruction_template_checksum": self.instruction_template_checksum,
                "model_checksum": self.model_checksum,
                "model_path": self.model_path,
                "normalization_rule": self.normalization_rule,
                "mined_rows_checksum": self.mined_rows_checksum,
                "positive_mapping_checksum": self.positive_mapping_checksum,
                "query_checksum": self.query_checksum,
                "random_seed": self.random_seed,
                "run_id": self.run_id,
                "teacher_version": self.teacher_version,
                "top_k": self.top_k,
            },
            created_at=self.generated_at,
            schema_version=self.schema_version,
        )

    def write(self, path: Path) -> None:
        """Atomically create a manifest without overwriting prior evidence."""
        destination = path.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    self.to_envelope().to_dict(),
                    stream,
                    allow_nan=False,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        except FileExistsError:
            raise
        except OSError as error:
            raise ManifestError(f"cannot write mining manifest {destination}: {error}") from error
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def read(cls, path: Path) -> MiningManifest:
        """Strictly read and checksum-verify a persisted mining manifest."""
        return cls.from_envelope(ManifestEnvelope.read(path))

    @classmethod
    def from_envelope(cls, envelope: ManifestEnvelope) -> MiningManifest:
        """Parse a verified mining envelope without accepting unknown fields."""
        if envelope.kind is not ManifestKind.MINING:
            raise ManifestError(
                f"expected a mining manifest, got manifest kind {envelope.kind.value!r}"
            )
        payload = dict(envelope.payload)
        actual = set(payload)
        if actual != _PAYLOAD_FIELDS:
            missing = sorted(_PAYLOAD_FIELDS - actual)
            extra = sorted(actual - _PAYLOAD_FIELDS)
            raise ManifestError(
                f"mining manifest payload fields differ; missing={missing}, extra={extra}"
            )
        index_parameters = payload["index_parameters"]
        filters = payload["filters"]
        if not isinstance(index_parameters, Mapping):
            raise ManifestError("mining index_parameters must be a mapping")
        if (
            isinstance(filters, (str, bytes))
            or not isinstance(filters, Sequence)
            or any(not isinstance(value, Mapping) for value in filters)
        ):
            raise ManifestError("mining filters must be a list of mappings")
        return cls(
            run_id=cast("str", payload["run_id"]),
            model_path=cast("str", payload["model_path"]),
            model_checksum=cast("str", payload["model_checksum"]),
            adapter_checksum=cast("str | None", payload["adapter_checksum"]),
            instruction_template_checksum=cast("str", payload["instruction_template_checksum"]),
            corpus_checksum=cast("str", payload["corpus_checksum"]),
            query_checksum=cast("str", payload["query_checksum"]),
            positive_mapping_checksum=cast("str", payload["positive_mapping_checksum"]),
            mined_rows_checksum=cast("str", payload["mined_rows_checksum"]),
            embedding_dimension=cast("int", payload["embedding_dimension"]),
            normalization_rule=cast("str", payload["normalization_rule"]),
            index_type=cast("str", payload["index_type"]),
            index_parameters=dict(cast("Mapping[str, Any]", index_parameters)),
            top_k=cast("int", payload["top_k"]),
            filters=tuple(dict(value) for value in cast("Sequence[Mapping[str, Any]]", filters)),
            random_seed=cast("int", payload["random_seed"]),
            teacher_version=cast("str | None", payload["teacher_version"]),
            generated_at=envelope.created_at,
            schema_version=envelope.schema_version,
        )

    def verify_positive_mapping(self, positives: Mapping[str, Sequence[str]]) -> None:
        """Verify that a logical positive mapping matches this mining run."""
        actual = checksum_positive_mapping(positives)
        if actual != self.positive_mapping_checksum:
            raise ChecksumMismatchError(
                "mining positive mapping checksum mismatch: "
                f"expected {self.positive_mapping_checksum}, computed {actual}"
            )

    def verify_mined_rows(self, rows: Iterable[_MinedRowLike]) -> None:
        """Verify ordered mined rows against the manifest binding."""
        actual = checksum_mined_rows(rows)
        if actual != self.mined_rows_checksum:
            raise ChecksumMismatchError(
                "mining rows checksum mismatch: "
                f"expected {self.mined_rows_checksum}, computed {actual}"
            )

    def verify_output_file(self, path: Path) -> None:
        """Strictly decode and checksum-verify a mined JSONL output file."""
        self.verify_mined_rows(_read_mined_rows(path, top_k=self.top_k))


def checksum_positive_mapping(positives: Mapping[str, Sequence[str]]) -> str:
    """Hash positive membership canonically, independent of mapping/set order."""
    canonical_positives: dict[str, JSONValue] = {}
    for query_id, candidate_ids in positives.items():
        if not isinstance(query_id, str) or not query_id:
            raise ManifestError("positive mapping query IDs must be non-empty strings")
        if isinstance(candidate_ids, (str, bytes)) or not isinstance(candidate_ids, Sequence):
            raise ManifestError("positive mapping values must be candidate ID sequences")
        if any(
            not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids
        ):
            raise ManifestError("positive mapping candidate IDs must be non-empty strings")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ManifestError("positive mapping candidate IDs must be unique")
        canonical_positives[query_id] = cast("list[JSONValue]", sorted(candidate_ids))
    return sha256_digest(
        {
            "checksum_domain": _POSITIVE_MAPPING_DOMAIN,
            "positives": canonical_positives,
        }
    )


def checksum_mined_rows(rows: Iterable[_MinedRowLike]) -> str:
    """Hash ordered mined rows with domain separation and length framing."""
    digest = hashlib.sha256(_MINED_ROWS_DOMAIN)
    count = 0
    for row in rows:
        encoded = canonical_json_bytes(_mined_row_payload(row))
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
        count += 1
    digest.update(count.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def _mined_row_payload(row: _MinedRowLike) -> dict[str, JSONValue]:
    if not isinstance(row.query_id, str) or not row.query_id:
        raise ManifestError("mined row query_id must be a non-empty string")
    if not isinstance(row.candidate_id, str) or not row.candidate_id:
        raise ManifestError("mined row candidate_id must be a non-empty string")
    if isinstance(row.score, bool) or not isinstance(row.score, (int, float)):
        raise ManifestError("mined row score must be a number")
    score = float(row.score)
    if not math.isfinite(score):
        raise ManifestError("mined row score must be finite")
    if isinstance(row.rank, bool) or not isinstance(row.rank, int) or row.rank <= 0:
        raise ManifestError("mined row rank must be a positive integer")
    if not isinstance(row.difficulty, str) or row.difficulty not in _DIFFICULTIES:
        raise ManifestError(f"unknown mined row difficulty {row.difficulty!r}")
    return {
        "candidate_id": row.candidate_id,
        "difficulty": row.difficulty,
        "query_id": row.query_id,
        "rank": row.rank,
        "score": score,
    }


def _read_mined_rows(path: Path, *, top_k: int) -> Iterable[_DecodedMinedRow]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ManifestError(f"mining output {path} contains blank line {line_number}")
                try:
                    decoded = strict_json_loads(
                        line,
                        context=f"mining output {path} line {line_number}",
                    )
                except ValueError as error:
                    raise ManifestError(
                        f"cannot decode mining output {path} line {line_number}: {error}"
                    ) from error
                if not isinstance(decoded, dict):
                    raise ManifestError(
                        f"mining output {path} line {line_number} must be an object"
                    )
                actual_fields = set(decoded)
                if actual_fields != _MINED_ROW_FIELDS:
                    missing = sorted(_MINED_ROW_FIELDS - actual_fields)
                    extra = sorted(actual_fields - _MINED_ROW_FIELDS)
                    raise ManifestError(
                        f"mining output row fields differ at line {line_number}; "
                        f"missing={missing}, extra={extra}"
                    )
                row = _DecodedMinedRow(
                    query_id=decoded["query_id"],
                    candidate_id=decoded["candidate_id"],
                    score=decoded["score"],
                    rank=decoded["rank"],
                    difficulty=decoded["difficulty"],
                )
                _mined_row_payload(row)
                if row.rank > top_k:
                    raise ManifestError(f"mining output rank exceeds top_k at line {line_number}")
                yield row
    except (OSError, UnicodeError) as error:
        raise ManifestError(f"cannot read mining output {path}: {error}") from error


def _validate_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ManifestError(f"{name} must be a lowercase 64-character SHA-256 digest")
