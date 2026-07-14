"""Deterministic global-batch construction with duplicate exclusion."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from torch.utils.data import Sampler

from mira.data.errors import RecordValidationError, SamplerStateError
from mira.data.fingerprint import canonical_record_payload

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from mira.data.schema import EmbeddingRecord

_STATE_VERSION: Final = 2
_DEFAULT_DUPLICATE_KEYS: Final = (
    "sample_id",
    "positive_item_ids",
    "canonical_product_id",
    "positive_group_ids",
    "image_hash_group",
)
_SCALAR_METADATA_KEYS: Final = frozenset(
    {
        "canonical_product_id",
        "spu_id",
        "sku_id",
        "image_hash_group",
        "cluster_id",
        "category_id",
        "brand_id",
        "dataset_name",
        "modality_pair",
    }
)
_SEQUENCE_METADATA_KEYS: Final = frozenset({"positive_group_ids", "negative_group_ids"})
_DIRECT_KEYS: Final = frozenset(
    {
        "sample_id",
        "task_id",
        "query_item_id",
        "positive_item_ids",
        "candidate_item_ids",
        "media_uris",
    }
)


class DeterministicNoDuplicateBatchSampler(Sampler[list[int]]):
    """Yield deterministic rank-local slices of duplicate-free global batches.

    The same seed, epoch, world size, global batch size, duplicate rules, and
    dataset content produce the same global batches. State restoration rejects
    any mismatch rather than resuming at an unknowable position.
    """

    def __init__(  # noqa: PLR0913
        self,
        records: Sequence[EmbeddingRecord],
        batch_size: int | None = None,
        *,
        global_batch_size: int | None = None,
        seed: int = 0,
        drop_last: bool = False,
        duplicate_keys: Sequence[str] = _DEFAULT_DUPLICATE_KEYS,
        rank: int = 0,
        world_size: int = 1,
        epoch: int = 0,
        homogeneous_keys: Sequence[str] = (),
        media_root: str | Path | None = None,
    ) -> None:
        """Configure deterministic global batching and rank slicing."""
        resolved_batch_size = _resolve_batch_size(batch_size, global_batch_size)
        if not records:
            raise ValueError("records must not be empty")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if resolved_batch_size < world_size:
            raise ValueError("global batch size must be at least world_size")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must satisfy 0 <= rank < {world_size}; got {rank}")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if len(set(duplicate_keys)) != len(duplicate_keys):
            raise ValueError("duplicate_keys must not contain repeated rules")
        if len(set(homogeneous_keys)) != len(homogeneous_keys):
            raise ValueError("homogeneous_keys must not contain repeated rules")
        supported_homogeneous = _SCALAR_METADATA_KEYS | {"task_id"}
        unsupported_homogeneous = set(homogeneous_keys).difference(supported_homogeneous)
        if unsupported_homogeneous:
            names = ", ".join(sorted(unsupported_homogeneous))
            raise ValueError(f"unsupported homogeneous key rule(s): {names}")
        effective_duplicate_keys = tuple(
            dict.fromkeys(("sample_id", "positive_item_ids", *duplicate_keys))
        )
        unsupported = set(effective_duplicate_keys).difference(
            _SCALAR_METADATA_KEYS | _SEQUENCE_METADATA_KEYS | _DIRECT_KEYS
        )
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"unsupported duplicate key rule(s): {names}")

        self._records = tuple(records)
        self.global_batch_size = resolved_batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.duplicate_keys = effective_duplicate_keys
        self.homogeneous_keys = tuple(homogeneous_keys)
        self._media_root = None if media_root is None else Path(media_root)
        self.rank = rank
        self.world_size = world_size
        self.epoch = epoch
        self._cursor = 0
        self._dataset_fingerprint = self._fingerprint_records()
        self._batches: tuple[tuple[int, ...], ...] | None = None

    @property
    def cursor(self) -> int:
        """Index of the next global batch to yield."""
        return self._cursor

    def __iter__(self) -> Iterator[list[int]]:
        """Yield remaining non-empty batches for this rank."""
        batches = self._global_batches()
        while self._cursor < len(batches):
            global_batch = batches[self._cursor]
            self._cursor += 1
            local_batch = list(global_batch[self.rank :: self.world_size])
            if local_batch:
                yield local_batch

    def __len__(self) -> int:
        """Return remaining non-empty rank-local batches."""
        return sum(
            bool(batch[self.rank :: self.world_size])
            for batch in self._global_batches()[self._cursor :]
        )

    def set_epoch(self, epoch: int) -> None:
        """Select a new deterministic epoch and rewind its cursor."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch
        self._cursor = 0
        self._batches = None

    def state_dict(self) -> dict[str, Any]:
        """Return all state required for an exact resume."""
        return {
            "version": _STATE_VERSION,
            "dataset_fingerprint": self._dataset_fingerprint,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self._cursor,
            "global_batch_size": self.global_batch_size,
            "drop_last": self.drop_last,
            "duplicate_keys": list(self.duplicate_keys),
            "homogeneous_keys": list(self.homogeneous_keys),
            "world_size": self.world_size,
            "rank": self.rank,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore an exact position, rejecting configuration drift."""
        expected = self.state_dict()
        immutable_keys = (
            "version",
            "dataset_fingerprint",
            "seed",
            "global_batch_size",
            "drop_last",
            "duplicate_keys",
            "homogeneous_keys",
            "world_size",
            "rank",
        )
        for key in immutable_keys:
            if key not in state:
                raise SamplerStateError(f"sampler state is missing required key {key!r}")
            if state[key] != expected[key]:
                raise SamplerStateError(
                    f"sampler state {key} mismatch: checkpoint={state[key]!r}, "
                    f"current={expected[key]!r}"
                )
        epoch = state.get("epoch")
        cursor = state.get("cursor")
        if not isinstance(epoch, int) or epoch < 0:
            raise SamplerStateError(f"invalid sampler epoch {epoch!r}")
        if not isinstance(cursor, int) or cursor < 0:
            raise SamplerStateError(f"invalid sampler cursor {cursor!r}")
        self.epoch = epoch
        self._batches = None
        batch_count = len(self._global_batches())
        if cursor > batch_count:
            raise SamplerStateError(
                f"sampler cursor {cursor} exceeds epoch batch count {batch_count}"
            )
        self._cursor = cursor

    def global_batches(self) -> tuple[tuple[int, ...], ...]:
        """Expose immutable global batches for validation and diagnostics."""
        return self._global_batches()

    def _global_batches(self) -> tuple[tuple[int, ...], ...]:
        if self._batches is not None:
            return self._batches
        indices = list(range(len(self._records)))
        random.Random(self.seed + self.epoch).shuffle(indices)  # noqa: S311
        pending = indices
        batches: list[tuple[int, ...]] = []
        while pending:
            selected: list[int] = []
            remaining: list[int] = []
            occupied: dict[str, set[str]] = {key: set() for key in self.duplicate_keys}
            homogeneous_signature: tuple[str, ...] | None = None
            for index in pending:
                values = self._duplicate_values(self._records[index])
                conflict = any(occupied[key].intersection(values[key]) for key in occupied)
                signature = self._homogeneous_signature(self._records[index])
                incompatible = (
                    homogeneous_signature is not None and signature != homogeneous_signature
                )
                if len(selected) >= self.global_batch_size or conflict or incompatible:
                    remaining.append(index)
                    continue
                selected.append(index)
                homogeneous_signature = signature
                for key, occupied_values in occupied.items():
                    occupied_values.update(values[key])
            if not selected:
                raise RuntimeError("no-duplicate batching made no progress")
            if len(selected) == self.global_batch_size or not self.drop_last:
                if self.world_size > 1 and len(selected) < self.world_size:
                    raise RecordValidationError(
                        "a distributed partial global batch would leave at least one rank "
                        "empty; enable drop_last or adjust duplicate rules"
                    )
                batch = tuple(selected)
                self._validate_global_batch(batch)
                batches.append(batch)
            pending = remaining
        self._batches = tuple(batches)
        return self._batches

    def _validate_global_batch(self, batch: Sequence[int]) -> None:
        occupied: dict[str, set[str]] = {key: set() for key in self.duplicate_keys}
        signatures = {self._homogeneous_signature(self._records[index]) for index in batch}
        if len(signatures) > 1:
            raise RecordValidationError("constructed batch violates homogeneous key rules")
        for index in batch:
            values = self._duplicate_values(self._records[index])
            for key, occupied_values in occupied.items():
                conflicts = occupied_values.intersection(values[key])
                if conflicts:
                    duplicate = min(conflicts)
                    raise RecordValidationError(
                        f"constructed batch contains duplicate {key}={duplicate!r}"
                    )
                occupied_values.update(values[key])

    def _duplicate_values(self, record: EmbeddingRecord) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for key in self.duplicate_keys:
            if key == "sample_id":
                result[key] = {record.sample_id}
            elif key == "task_id":
                result[key] = {record.task_id}
            elif key == "query_item_id":
                result[key] = {record.query.item_id}
            elif key == "positive_item_ids":
                result[key] = {candidate.item_id for candidate in record.positives}
            elif key == "candidate_item_ids":
                result[key] = {
                    candidate.item_id for candidate in record.positives + record.negatives
                }
            elif key == "media_uris":
                items = (record.query, *record.positives, *record.negatives)
                result[key] = {
                    reference.uri
                    for item in items
                    for modality in (item.images, item.videos, item.audios)
                    for reference in modality
                }
            elif key in _SEQUENCE_METADATA_KEYS:
                result[key] = set(getattr(record.metadata, key))
            else:
                value = getattr(record.metadata, key)
                result[key] = set() if value is None else {str(value)}
        return result

    def _homogeneous_signature(self, record: EmbeddingRecord) -> tuple[str, ...]:
        values: list[str] = []
        for key in self.homogeneous_keys:
            if key == "task_id":
                values.append(record.task_id)
            else:
                value = getattr(record.metadata, key)
                values.append("<missing>" if value is None else str(value))
        return tuple(values)

    def _fingerprint_records(self) -> str:
        payload = [
            {
                "sample_id": record.sample_id,
                "record": canonical_record_payload(record, media_root=self._media_root),
                "homogeneous": self._homogeneous_signature(record),
            }
            for record in self._records
        ]
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


def _resolve_batch_size(batch_size: int | None, global_batch_size: int | None) -> int:
    if batch_size is None and global_batch_size is None:
        raise ValueError("provide batch_size or global_batch_size")
    if batch_size is not None and global_batch_size is not None and batch_size != global_batch_size:
        raise ValueError("batch_size and global_batch_size disagree")
    value = global_batch_size if global_batch_size is not None else batch_size
    if value is None or value <= 0:
        raise ValueError("global batch size must be positive")
    return value


NoDuplicateBatchSampler = DeterministicNoDuplicateBatchSampler
NoDuplicateSampler = DeterministicNoDuplicateBatchSampler
