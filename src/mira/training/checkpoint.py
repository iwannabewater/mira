"""Atomic, checksummed training checkpoints with explicit resume semantics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import sys
import uuid
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import torch
from torch import Tensor, nn

from mira.core.exceptions import CheckpointError
from mira.core.filesystem import atomic_rename_directory_no_replace, open_bound_regular_file
from mira.core.serialization import strict_json_loads

if TYPE_CHECKING:
    from typing import BinaryIO

    from torch.optim import Optimizer

ResumeMode: TypeAlias = Literal[
    "resume_full",
    "resume_model_only",
    "load_adapter_for_new_stage",
]

_CHECKPOINT_VERSION = 1
_SHA256_LENGTH = 64
_STATE_FILE = "state.pt"
_MANIFEST_FILE = "manifest.json"
_CONFIG_FILE = "resolved-config.json"
_CHECKPOINT_FILES = frozenset({_STATE_FILE, _CONFIG_FILE})
_CHECKPOINT_ENTRIES = _CHECKPOINT_FILES | {_MANIFEST_FILE}
_MANIFEST_FIELDS = frozenset(
    {
        "checkpoint_version",
        "created_at",
        "files",
        "components",
        "dependency_versions",
        "source_revisions",
        "manifests",
        "training_state",
    }
)
_COMPONENT_FIELDS = frozenset(
    {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "sampler",
        "streaming_dataset",
        "gradient_cache",
        "objective",
        "curriculum",
        "adapter",
    }
)
_STATE_FIELDS = frozenset(
    {
        "checkpoint_version",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "training_state",
        "rng",
        "sampler_state",
        "dataset_state",
        "gradient_cache_state",
        "objective_state",
        "curriculum_state",
        "adapter_state",
    }
)
_RNG_FIELDS = frozenset({"python", "numpy", "torch_cpu", "torch_cuda"})
_NUMPY_RNG_FIELDS = frozenset({"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"})
_TRAINING_STATE_FIELDS = frozenset({"global_step", "epoch", "micro_step", "best_metric"})


@dataclass(frozen=True, slots=True)
class TrainingState:
    """Serializable progress counters required for an exact continuation."""

    global_step: int
    epoch: int = 0
    micro_step: int = 0
    best_metric: float | None = None

    def __post_init__(self) -> None:
        """Reject counters that cannot describe a valid training position."""
        if any(type(value) is not int for value in (self.global_step, self.epoch, self.micro_step)):
            raise CheckpointError("training progress counters must be integers")
        if self.global_step < 0 or self.epoch < 0 or self.micro_step < 0:
            raise CheckpointError("training progress counters must be non-negative")
        if self.best_metric is not None:
            if isinstance(self.best_metric, bool) or not isinstance(self.best_metric, (int, float)):
                raise CheckpointError("best_metric must be a number or null")
            if not math.isfinite(float(self.best_metric)):
                raise CheckpointError("best_metric must be finite")


@dataclass(frozen=True, slots=True)
class CheckpointLoadResult:
    """Evidence returned after a checkpoint has been verified and loaded."""

    mode: ResumeMode
    training_state: TrainingState | None
    restored_components: tuple[str, ...]
    manifest: dict[str, Any]


@dataclass(slots=True)
class CheckpointComponents:
    """Optional stateful components included in a full checkpoint."""

    optimizer: Optimizer | None = None
    scheduler: Any | None = None
    scaler: Any | None = None
    sampler_state: dict[str, Any] | None = None
    dataset_state: dict[str, Any] | None = None
    gradient_cache_state: dict[str, Any] | None = None
    objective_state: dict[str, Any] | None = None
    curriculum_state: dict[str, Any] | None = None
    adapter_state: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Provenance stored alongside serialized training state."""

    manifests: dict[str, Any] | None = None
    source_revisions: dict[str, str] | None = None


@dataclass(slots=True)
class CheckpointRestoreTargets:
    """Runtime objects that may receive restored state."""

    optimizer: Optimizer | None = None
    scheduler: Any | None = None
    scaler: Any | None = None
    adapter: nn.Module | None = None
    sampler: Any | None = None
    dataset: Any | None = None
    gradient_cache: Any | None = None
    objective: Any | None = None
    curriculum: Any | None = None


@dataclass(slots=True)
class CheckpointSnapshot:
    """All inputs needed to persist one training checkpoint."""

    model: nn.Module
    training_state: TrainingState
    resolved_config: dict[str, Any]
    components: CheckpointComponents | None = None
    metadata: CheckpointMetadata | None = None


@dataclass(frozen=True, slots=True)
class _RestoreOperation:
    name: str
    target: Any
    state: dict[str, Any]
    strict: bool = False


class CheckpointManager:
    """Save and restore complete training state without silent degradation.

    A checkpoint is first written into a sibling temporary directory. Checksums
    are computed after files are closed, then the directory is renamed into its
    final location. Existing destinations are never overwritten.
    """

    def save(
        self,
        output_dir: Path,
        *,
        snapshot: CheckpointSnapshot,
    ) -> dict[str, Any]:
        """Write an atomic checkpoint and return its verified manifest.

        Raises:
            CheckpointError: If the destination exists or serialization fails.
        """
        components = snapshot.components or CheckpointComponents()
        metadata = snapshot.metadata or CheckpointMetadata()
        destination = output_dir.resolve()
        if destination.exists():
            raise CheckpointError(f"checkpoint destination already exists: {destination}")
        temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
        temporary_created = False
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.mkdir()
            temporary_created = True
            config_path = temporary / _CONFIG_FILE
            config_path.write_text(
                json.dumps(
                    snapshot.resolved_config,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _sync_file(config_path)
            payload = {
                "checkpoint_version": _CHECKPOINT_VERSION,
                "model": snapshot.model.state_dict(),
                "optimizer": (
                    components.optimizer.state_dict() if components.optimizer is not None else None
                ),
                "scheduler": _state_dict_or_none(components.scheduler),
                "scaler": _state_dict_or_none(components.scaler),
                "training_state": asdict(snapshot.training_state),
                "rng": _capture_rng_state(),
                "sampler_state": components.sampler_state,
                "dataset_state": components.dataset_state,
                "gradient_cache_state": components.gradient_cache_state,
                "objective_state": components.objective_state,
                "curriculum_state": components.curriculum_state,
                "adapter_state": components.adapter_state,
            }
            state_path = temporary / _STATE_FILE
            torch.save(payload, state_path)
            _sync_file(state_path)

            manifest = {
                "checkpoint_version": _CHECKPOINT_VERSION,
                "created_at": datetime.now(UTC).isoformat(),
                "files": {
                    _STATE_FILE: _sha256_file(state_path),
                    _CONFIG_FILE: _sha256_file(config_path),
                },
                "components": {
                    "model": True,
                    "optimizer": components.optimizer is not None,
                    "scheduler": components.scheduler is not None,
                    "scaler": components.scaler is not None,
                    "sampler": components.sampler_state is not None,
                    "streaming_dataset": components.dataset_state is not None,
                    "gradient_cache": components.gradient_cache_state is not None,
                    "objective": components.objective_state is not None,
                    "curriculum": components.curriculum_state is not None,
                    "adapter": components.adapter_state is not None,
                },
                "dependency_versions": _dependency_versions(),
                "source_revisions": metadata.source_revisions or {},
                "manifests": metadata.manifests or {},
                "training_state": asdict(snapshot.training_state),
            }
            manifest_path = temporary / _MANIFEST_FILE
            manifest_path.write_text(
                json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _sync_file(manifest_path)
            _sync_directory(temporary)
            self.verify(temporary)
            try:
                atomic_rename_directory_no_replace(temporary, destination)
            except FileExistsError as error:
                raise CheckpointError(
                    f"checkpoint destination already exists: {destination}"
                ) from error
            _sync_directory(destination.parent)
        except Exception as error:
            if temporary_created and temporary.exists():
                shutil.rmtree(temporary)
            if isinstance(error, CheckpointError):
                raise
            raise CheckpointError(f"failed to save checkpoint at {destination}") from error

        return manifest

    def verify(
        self,
        checkpoint_dir: Path,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Validate structure and declared checksums, plus optional trusted provenance.

        The manifest is stored beside the files it protects, so its internal
        checksums detect accidental corruption, not a malicious replacement of
        both state and manifest. Supplying a digest obtained through a trusted
        channel authenticates the manifest bytes before they are parsed.
        """
        return _verify_checkpoint(
            checkpoint_dir.resolve(),
            expected_manifest_sha256=expected_manifest_sha256,
        )

    def load(  # noqa: PLR0913 - resume modes require explicit targets and evidence
        self,
        checkpoint_dir: Path,
        *,
        model: nn.Module,
        mode: ResumeMode,
        targets: CheckpointRestoreTargets | None = None,
        map_location: str | torch.device = "cpu",
        expected_manifest_sha256: str | None = None,
    ) -> CheckpointLoadResult:
        """Verify and restore a checkpoint according to an explicit mode."""
        targets = targets or CheckpointRestoreTargets()
        if mode not in {
            "resume_full",
            "resume_model_only",
            "load_adapter_for_new_stage",
        }:
            raise CheckpointError(f"unknown checkpoint load mode: {mode!r}")
        root = checkpoint_dir.resolve()
        state_path = root / _STATE_FILE
        state_stream = _open_checkpoint_file(state_path)
        with state_stream, TemporaryFile(mode="w+b") as verified_state_stream:
            manifest = _verify_checkpoint(
                root,
                expected_manifest_sha256=expected_manifest_sha256,
                state_stream=state_stream,
                state_snapshot=verified_state_stream,
            )
            try:
                verified_state_stream.seek(0)
                payload = torch.load(
                    verified_state_stream,
                    map_location=map_location,
                    weights_only=True,
                )
            except Exception as error:
                raise CheckpointError(
                    f"checkpoint state is unreadable: {checkpoint_dir}"
                ) from error
        if (
            not isinstance(payload, dict)
            or type(payload.get("checkpoint_version")) is not int
            or payload.get("checkpoint_version") != _CHECKPOINT_VERSION
        ):
            raise CheckpointError("checkpoint state has an invalid version or root type")
        _validate_checkpoint_payload(payload, manifest)

        if mode == "load_adapter_for_new_stage":
            if targets.adapter is None:
                raise CheckpointError("load_adapter_for_new_stage requires an adapter module")
            adapter_state = _require_mapping(payload, "adapter_state")
            _execute_restore_transaction(
                (_RestoreOperation("adapter", targets.adapter, adapter_state, strict=True),)
            )
            return CheckpointLoadResult(mode, None, ("adapter",), manifest)

        model_operation = _RestoreOperation(
            "model",
            model,
            _require_mapping(payload, "model"),
            strict=True,
        )
        if mode == "resume_model_only":
            _execute_restore_transaction((model_operation,))
            return CheckpointLoadResult(mode, None, ("model",), manifest)

        training_state = _parse_training_state(
            payload.get("training_state"), context="checkpoint state"
        )
        operations = (model_operation, *_full_restore_operations(payload, targets))
        _execute_restore_transaction(
            operations,
            rng_state=_require_mapping(payload, "rng"),
        )
        restored = (*tuple(operation.name for operation in operations), "training_state", "rng")
        return CheckpointLoadResult(mode, training_state, restored, manifest)


def _validate_checkpoint_payload(payload: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Validate the full v1 state schema before mutating any runtime object."""
    _require_exact_keys(payload, _STATE_FIELDS, "checkpoint state")
    _require_mapping(payload, "model")
    for field in (
        "optimizer",
        "scheduler",
        "scaler",
        "sampler_state",
        "dataset_state",
        "gradient_cache_state",
        "objective_state",
        "curriculum_state",
        "adapter_state",
    ):
        value = payload.get(field)
        if value is not None and not isinstance(value, dict):
            raise CheckpointError(f"checkpoint field {field!r} must be a mapping or null")
    _require_mapping(payload, "training_state")
    rng = _require_mapping(payload, "rng")

    component_fields = {
        "model": True,
        "optimizer": payload["optimizer"] is not None,
        "scheduler": payload["scheduler"] is not None,
        "scaler": payload["scaler"] is not None,
        "sampler": payload["sampler_state"] is not None,
        "streaming_dataset": payload["dataset_state"] is not None,
        "gradient_cache": payload["gradient_cache_state"] is not None,
        "objective": payload["objective_state"] is not None,
        "curriculum": payload["curriculum_state"] is not None,
        "adapter": payload["adapter_state"] is not None,
    }
    if manifest["components"] != component_fields:
        raise CheckpointError("checkpoint state and manifest component declarations do not match")
    if manifest["training_state"] != payload["training_state"]:
        raise CheckpointError(
            "checkpoint state and manifest training state declaration do not match"
        )
    _parse_training_state(payload["training_state"], context="checkpoint state")
    _validate_rng_payload(rng)


def _full_restore_operations(
    payload: dict[str, Any], targets: CheckpointRestoreTargets
) -> tuple[_RestoreOperation, ...]:
    operations = [_required_restore_operation(payload, "optimizer", "optimizer", targets.optimizer)]
    for payload_name, component_name, target in (
        ("scheduler", "scheduler", targets.scheduler),
        ("scaler", "scaler", targets.scaler),
        ("sampler_state", "sampler", targets.sampler),
        ("dataset_state", "streaming_dataset", targets.dataset),
        ("gradient_cache_state", "gradient_cache", targets.gradient_cache),
        ("objective_state", "objective", targets.objective),
        ("curriculum_state", "curriculum", targets.curriculum),
    ):
        operation = _optional_restore_operation(
            payload,
            payload_name,
            component_name,
            target,
        )
        if operation is not None:
            operations.append(operation)
    return tuple(operations)


def _required_restore_operation(
    payload: dict[str, Any], payload_name: str, component_name: str, target: Any | None
) -> _RestoreOperation:
    if target is None:
        raise CheckpointError(f"resume_full requires a {component_name} instance")
    state = payload.get(payload_name)
    if not isinstance(state, dict):
        raise CheckpointError(f"checkpoint does not contain required {component_name} state")
    return _RestoreOperation(component_name, target, state)


def _optional_restore_operation(
    payload: dict[str, Any], payload_name: str, component_name: str, target: Any | None
) -> _RestoreOperation | None:
    state = payload.get(payload_name)
    if state is None and target is None:
        return None
    if state is None:
        raise CheckpointError(f"checkpoint does not contain requested {component_name} state")
    if target is None:
        raise CheckpointError(
            f"checkpoint contains {component_name} state but no restore target was supplied"
        )
    if not isinstance(state, dict):
        raise CheckpointError(f"checkpoint field {payload_name!r} must be a mapping")
    return _RestoreOperation(component_name, target, state)


def _execute_restore_transaction(
    operations: tuple[_RestoreOperation, ...],
    *,
    rng_state: dict[str, Any] | None = None,
) -> None:
    rng_snapshot = _capture_rng_state()
    try:
        for operation in operations:
            _preflight_restore_operation(operation)
        snapshots = tuple(
            (operation, _snapshot_target_state(operation)) for operation in operations
        )
    except Exception as error:
        try:
            _restore_rng_state(rng_snapshot)
        except Exception:  # noqa: BLE001 - rollback failure must not hide the preflight error
            raise CheckpointError(
                "checkpoint restore preflight failed and RNG rollback was incomplete"
            ) from error
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError("checkpoint restore preflight failed") from error

    try:
        for operation in operations:
            _load_restore_operation(operation)
        if rng_state is not None:
            _restore_rng_state(rng_state)
        else:
            _restore_rng_state(rng_snapshot)
    except Exception as error:
        rollback_errors = _rollback_restore_transaction(snapshots, rng_snapshot)
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise CheckpointError(
                f"checkpoint restore failed and rollback was incomplete: {details}"
            ) from error
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError("checkpoint restore failed") from error


def _preflight_restore_operation(operation: _RestoreOperation) -> None:
    if isinstance(operation.target, nn.Module):
        _validate_module_state_compatible(operation)
        return
    try:
        probe = deepcopy(operation.target)
        probe_state = deepcopy(operation.state)
    except Exception as error:
        raise CheckpointError(
            f"checkpoint {operation.name} target cannot be transactionally preflighted"
        ) from error
    _load_restore_operation(operation, target=probe, state=probe_state)


def _validate_module_state_compatible(operation: _RestoreOperation) -> None:
    try:
        target_state = operation.target.state_dict()
    except Exception as error:
        raise CheckpointError(
            f"checkpoint {operation.name} target state could not be inspected"
        ) from error
    source_keys = set(operation.state)
    target_keys = set(target_state)
    if source_keys != target_keys:
        raise CheckpointError(
            f"checkpoint {operation.name} state is incompatible: state keys do not match"
        )
    for name, source_value in operation.state.items():
        target_value = target_state[name]
        if not isinstance(source_value, Tensor) or not isinstance(target_value, Tensor):
            raise CheckpointError(
                f"checkpoint {operation.name} state is incompatible: field {name!r} is not a tensor"
            )
        try:
            compatible = (
                source_value.shape == target_value.shape
                and source_value.layout == target_value.layout
            )
        except RuntimeError as error:
            raise CheckpointError(
                f"checkpoint {operation.name} state is incompatible: field {name!r} "
                "could not be inspected"
            ) from error
        if not compatible:
            raise CheckpointError(
                f"checkpoint {operation.name} state is incompatible: field {name!r} "
                "has different tensor semantics"
            )


def _snapshot_target_state(operation: _RestoreOperation) -> dict[str, Any]:
    state_dict = getattr(operation.target, "state_dict", None)
    if not callable(state_dict):
        raise CheckpointError(
            f"restore target {type(operation.target).__name__} has no state_dict method "
            "required for transactional restore"
        )
    try:
        state = state_dict()
    except Exception as error:
        raise CheckpointError(
            f"checkpoint {operation.name} target state could not be snapshotted"
        ) from error
    if not isinstance(state, dict):
        raise CheckpointError(
            f"checkpoint {operation.name} target state_dict did not return a mapping"
        )
    try:
        return deepcopy(state)
    except Exception as error:
        raise CheckpointError(
            f"checkpoint {operation.name} target state could not be snapshotted"
        ) from error


def _load_restore_operation(
    operation: _RestoreOperation,
    *,
    target: Any | None = None,
    state: dict[str, Any] | None = None,
) -> None:
    restore_target = operation.target if target is None else target
    restore_state = operation.state if state is None else state
    load_state_dict = getattr(restore_target, "load_state_dict", None)
    if not callable(load_state_dict):
        raise CheckpointError(
            f"restore target {type(restore_target).__name__} has no load_state_dict method"
        )
    try:
        if operation.strict:
            load_state_dict(restore_state, strict=True)
        else:
            load_state_dict(restore_state)
    except Exception as error:
        raise CheckpointError(f"checkpoint {operation.name} state is incompatible") from error


def _rollback_restore_transaction(
    snapshots: tuple[tuple[_RestoreOperation, dict[str, Any]], ...],
    rng_snapshot: dict[str, Any],
) -> list[str]:
    errors = []
    for operation, state in reversed(snapshots):
        try:
            _load_restore_operation(operation, state=state)
        except Exception:  # noqa: BLE001 - continue rolling back the remaining targets
            errors.append(operation.name)
    try:
        _restore_rng_state(rng_snapshot)
    except Exception:  # noqa: BLE001 - report incomplete RNG rollback explicitly
        errors.append("rng")
    return errors


def _capture_rng_state() -> dict[str, Any]:
    numpy_state: Any = np.random.get_state(legacy=True)
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    python_state, numpy_state, torch_cpu, torch_cuda = _validate_rng_payload(state)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_cpu.cpu())
    if torch_cuda:
        torch.cuda.set_rng_state_all([item.cpu() for item in torch_cuda])


def _validate_rng_payload(
    state: dict[str, Any],
) -> tuple[Any, tuple[Any, ...], Tensor, list[Tensor]]:
    _require_exact_keys(state, _RNG_FIELDS, "checkpoint RNG state")
    python_state = state["python"]
    numpy_state = state["numpy"]
    torch_cpu = state["torch_cpu"]
    torch_cuda = state["torch_cuda"]
    if not isinstance(numpy_state, dict) or not isinstance(torch_cpu, Tensor):
        raise CheckpointError("checkpoint RNG state contains invalid values")
    if not isinstance(torch_cuda, list) or not all(isinstance(item, Tensor) for item in torch_cuda):
        raise CheckpointError("checkpoint CUDA RNG state must be a list of tensors")
    if torch_cuda and not torch.cuda.is_available():
        raise CheckpointError("checkpoint contains CUDA RNG state but CUDA is unavailable")
    if torch_cuda and len(torch_cuda) != torch.cuda.device_count():
        raise CheckpointError("checkpoint CUDA RNG state count does not match visible devices")

    _require_exact_keys(numpy_state, _NUMPY_RNG_FIELDS, "checkpoint NumPy RNG state")
    bit_generator = numpy_state["bit_generator"]
    keys = numpy_state["keys"]
    position = numpy_state["position"]
    has_gauss = numpy_state["has_gauss"]
    cached_gaussian = numpy_state["cached_gaussian"]
    if (
        not isinstance(bit_generator, str)
        or not isinstance(keys, list)
        or not keys
        or any(type(item) is not int for item in keys)
        or type(position) is not int
        or type(has_gauss) is not int
        or has_gauss not in {0, 1}
        or isinstance(cached_gaussian, bool)
        or not isinstance(cached_gaussian, (int, float))
        or not math.isfinite(float(cached_gaussian))
    ):
        raise CheckpointError("checkpoint NumPy RNG state contains invalid values")
    normalized_numpy_state = (
        bit_generator,
        np.asarray(keys, dtype=np.uint32),
        position,
        has_gauss,
        float(cached_gaussian),
    )

    if not isinstance(python_state, tuple):
        raise CheckpointError("checkpoint Python RNG state contains invalid values")
    try:
        random.Random().setstate(python_state)  # noqa: S311 - validates reproducible RNG state
        np.random.RandomState().set_state(normalized_numpy_state)
        torch.Generator(device="cpu").set_state(torch_cpu.cpu())
    except (IndexError, RuntimeError, TypeError, ValueError) as error:
        raise CheckpointError("checkpoint RNG state is invalid") from error
    return python_state, normalized_numpy_state, torch_cpu, torch_cuda


def _verify_checkpoint(
    root: Path,
    *,
    expected_manifest_sha256: str | None,
    state_stream: BinaryIO | None = None,
    state_snapshot: BinaryIO | None = None,
) -> dict[str, Any]:
    manifest_path = root / _MANIFEST_FILE
    if not manifest_path.is_file():
        raise CheckpointError(f"checkpoint manifest is missing: {manifest_path}")
    if manifest_path.is_symlink():
        raise CheckpointError(f"checkpoint entry must be a regular file: {manifest_path}")
    manifest_stream = _open_checkpoint_file(manifest_path, manifest=True)
    with manifest_stream:
        if expected_manifest_sha256 is not None:
            try:
                actual_manifest_sha256 = _sha256_stream(manifest_stream)
            except OSError as error:
                raise CheckpointError(
                    f"checkpoint manifest is unreadable: {manifest_path}"
                ) from error
            if actual_manifest_sha256 != expected_manifest_sha256:
                raise CheckpointError(
                    "checkpoint manifest digest does not match the trusted digest: "
                    f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
                )
        raw = _read_checkpoint_manifest(manifest_stream, manifest_path)

    files = _validate_manifest_fields(raw)
    _verify_checkpoint_entries(root)

    config_path = root / _CONFIG_FILE
    config_stream = _open_checkpoint_file(config_path)
    with config_stream:
        _verify_stream_checksum(config_stream, config_path, files[_CONFIG_FILE])
        _validate_resolved_config(config_stream, config_path)

    if state_stream is None:
        if state_snapshot is not None:
            raise CheckpointError("checkpoint state snapshot requires an open state stream")
        state_path = root / _STATE_FILE
        opened_state = _open_checkpoint_file(state_path)
        with opened_state:
            _verify_stream_checksum(opened_state, state_path, files[_STATE_FILE])
    elif state_snapshot is not None:
        _snapshot_verified_state(
            state_stream,
            state_snapshot,
            root / _STATE_FILE,
            files[_STATE_FILE],
        )
    else:
        _verify_stream_checksum(state_stream, root / _STATE_FILE, files[_STATE_FILE])
    return raw


def _read_checkpoint_manifest(stream: BinaryIO, path: Path) -> dict[str, Any]:
    try:
        stream.seek(0)
        raw = _loads_strict_json(stream.read().decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise CheckpointError(f"checkpoint manifest is unreadable: {path}") from error
    if not isinstance(raw, dict):
        raise CheckpointError("checkpoint manifest root must be an object")
    _require_exact_keys(raw, _MANIFEST_FIELDS, "checkpoint manifest")
    version = raw.get("checkpoint_version")
    if type(version) is not int or version != _CHECKPOINT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint version {version!r}; expected {_CHECKPOINT_VERSION}"
        )
    return raw


def _validate_manifest_fields(raw: dict[str, Any]) -> dict[str, str]:
    files = raw.get("files")
    if not isinstance(files, dict) or not files:
        raise CheckpointError("checkpoint manifest must declare checksummed files")
    for name, expected in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise CheckpointError(f"invalid checkpoint file name in manifest: {name!r}")
        if not isinstance(expected, str) or not _is_sha256(expected):
            raise CheckpointError(f"invalid checksum for checkpoint file {name!r}")
    _require_exact_keys(files, _CHECKPOINT_FILES, "checkpoint manifest files")

    components = raw.get("components")
    if not isinstance(components, dict):
        raise CheckpointError("checkpoint component flags must be an object")
    _require_exact_keys(components, _COMPONENT_FIELDS, "checkpoint component flags")
    if not all(type(value) is bool for value in components.values()):
        raise CheckpointError("checkpoint component flags must be booleans")
    if components["model"] is not True:
        raise CheckpointError("checkpoint component flag for model must be true")

    _validate_created_at(raw.get("created_at"))
    _validate_string_mapping(raw.get("dependency_versions"), "dependency_versions")
    _validate_string_mapping(raw.get("source_revisions"), "source_revisions")
    if not isinstance(raw.get("manifests"), dict):
        raise CheckpointError("checkpoint manifests provenance must be an object")
    _parse_training_state(raw.get("training_state"), context="checkpoint manifest")
    return files


def _verify_checkpoint_entries(root: Path) -> None:
    try:
        entries = {entry.name for entry in root.iterdir()}
    except OSError as error:
        raise CheckpointError(f"checkpoint directory is unreadable: {root}") from error
    if entries == _CHECKPOINT_ENTRIES:
        return
    missing = _CHECKPOINT_ENTRIES - entries
    unexpected = entries - _CHECKPOINT_ENTRIES
    details = []
    if missing:
        details.append(f"missing entries: {_format_names(missing)}")
    if unexpected:
        details.append(f"unexpected entries: {_format_names(unexpected)}")
    raise CheckpointError(f"checkpoint directory has {'; '.join(details)}")


def _verify_stream_checksum(stream: BinaryIO, path: Path, expected: str) -> None:
    try:
        actual = _sha256_stream(stream)
    except OSError as error:
        raise CheckpointError(f"checkpoint file is unreadable: {path}") from error
    if actual != expected:
        raise CheckpointError(
            f"checkpoint checksum mismatch for {path.name}: expected {expected}, got {actual}"
        )


def _snapshot_verified_state(
    source: BinaryIO,
    snapshot: BinaryIO,
    path: Path,
    expected: str,
) -> None:
    digest = hashlib.sha256()
    try:
        source.seek(0)
        snapshot.seek(0)
        snapshot.truncate(0)
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            snapshot.write(chunk)
        snapshot.flush()
        snapshot.seek(0)
    except OSError as error:
        raise CheckpointError(f"checkpoint file is unreadable: {path}") from error
    actual = digest.hexdigest()
    if actual != expected:
        raise CheckpointError(
            f"checkpoint checksum mismatch for {path.name}: expected {expected}, got {actual}"
        )


def _validate_resolved_config(stream: BinaryIO, path: Path) -> None:
    try:
        stream.seek(0)
        resolved_config = _loads_strict_json(stream.read().decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise CheckpointError(f"checkpoint resolved config is unreadable: {path}") from error
    if not isinstance(resolved_config, dict):
        raise CheckpointError("checkpoint resolved config root must be an object")


def _require_exact_keys(value: dict[Any, Any], expected: frozenset[str], context: str) -> None:
    actual = set(value)
    missing = expected - actual
    if missing:
        raise CheckpointError(f"{context} is missing fields: {_format_names(missing)}")
    unexpected = actual - expected
    if unexpected:
        raise CheckpointError(f"{context} has unexpected fields: {_format_names(unexpected)}")


def _format_names(names: set[Any] | frozenset[Any]) -> str:
    return ", ".join(sorted(repr(name) for name in names))


def _parse_training_state(value: Any, *, context: str) -> TrainingState:
    if not isinstance(value, dict):
        raise CheckpointError(f"{context} training_state must be a mapping")
    _require_exact_keys(value, _TRAINING_STATE_FIELDS, f"{context} training_state")
    try:
        return TrainingState(**value)
    except (CheckpointError, TypeError, ValueError) as error:
        raise CheckpointError(f"{context} training state is invalid") from error


def _validate_created_at(value: Any) -> None:
    if not isinstance(value, str):
        raise CheckpointError("checkpoint created_at must be an ISO-8601 UTC timestamp")
    try:
        created_at = datetime.fromisoformat(value)
    except ValueError as error:
        raise CheckpointError("checkpoint created_at must be an ISO-8601 UTC timestamp") from error
    if created_at.tzinfo is None or created_at.utcoffset() != timedelta(0):
        raise CheckpointError("checkpoint created_at must be an ISO-8601 UTC timestamp")


def _validate_string_mapping(value: Any, field: str) -> None:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not key or not isinstance(item, str) or not item
        for key, item in value.items()
    ):
        raise CheckpointError(f"checkpoint {field} must map non-empty strings to strings")


def _loads_strict_json(content: str) -> Any:
    return strict_json_loads(content, context="checkpoint JSON")


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _require_mapping(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise CheckpointError(f"checkpoint field {name!r} must be a mapping")
    return value


def _state_dict_or_none(component: Any | None) -> dict[str, Any] | None:
    if component is None:
        return None
    state_dict = getattr(component, "state_dict", None)
    if not callable(state_dict):
        raise CheckpointError(f"component {type(component).__name__} has no state_dict method")
    state = state_dict()
    if not isinstance(state, dict):
        raise CheckpointError(
            f"component {type(component).__name__} returned an invalid state_dict"
        )
    return state


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _open_checkpoint_file(path: Path, *, manifest: bool = False) -> BinaryIO:
    if path.is_symlink():
        if not path.is_file():
            if manifest:
                raise CheckpointError(f"checkpoint manifest is missing: {path}")
            raise CheckpointError(f"checkpoint file is missing: {path}")
        raise CheckpointError(f"checkpoint entry must be a regular file: {path}")
    try:
        return _open_bound_regular_file(path)
    except FileNotFoundError as error:
        if manifest:
            raise CheckpointError(f"checkpoint manifest is missing: {path}") from error
        raise CheckpointError(f"checkpoint file is missing: {path}") from error
    except OSError as error:
        if path.is_symlink():
            raise CheckpointError(f"checkpoint entry must be a regular file: {path}") from error
        if manifest:
            raise CheckpointError(f"checkpoint manifest is unreadable: {path}") from error
        raise CheckpointError(f"checkpoint file is unreadable: {path}") from error


def _open_bound_regular_file(path: Path) -> BinaryIO:
    return open_bound_regular_file(path)


def _dependency_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "torch": torch.__version__,
    }


def _sync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    """Best-effort fsync on platforms that support directory descriptors."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        with suppress(OSError):
            os.close(descriptor)
