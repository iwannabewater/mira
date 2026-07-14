"""Strict lifecycle evidence for versioned, atomic training checkpoints."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import uuid
from copy import deepcopy
from dataclasses import dataclass
from threading import Event, Thread
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import torch
from torch import nn

import mira.training.checkpoint as checkpoint_module
from mira.core import filesystem
from mira.core.exceptions import CheckpointError
from mira.training import (
    CheckpointComponents,
    CheckpointManager,
    CheckpointMetadata,
    CheckpointRestoreTargets,
    CheckpointSnapshot,
    TrainingState,
)

_ORIGINAL_SNAPSHOT_VERIFIED_STATE = checkpoint_module._snapshot_verified_state

if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO


class _Stateful:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = deepcopy(state)
        self.loaded: dict[str, Any] | None = None

    def state_dict(self) -> dict[str, Any]:
        return deepcopy(self.state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loaded = deepcopy(state)


class _IncompatibleStateful(_Stateful):
    def load_state_dict(self, state: dict[str, Any]) -> None:
        del state
        raise ValueError("incompatible state")


class _TransactionalStateful:
    def __init__(self, state: dict[str, Any], *, probe: bool = False) -> None:
        self.state = deepcopy(state)
        self.probe = probe

    def __deepcopy__(self, memo: dict[int, Any]) -> _TransactionalStateful:
        return type(self)(deepcopy(self.state, memo), probe=True)

    def state_dict(self) -> dict[str, Any]:
        return deepcopy(self.state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.state = deepcopy(state)
        if state.get("fail") is True and not self.probe:
            raise ValueError("failure after mutation")


@dataclass(frozen=True)
class _MutationEvidence:
    attacker_state: bytes
    same_open_inode: list[bool]


class _CoordinatedSource:
    def __init__(self, source: Any, requested: Event, finished: Event) -> None:
        self._source = source
        self._requested = requested
        self._finished = finished
        self._first_read = True

    def seek(self, offset: int, whence: int = 0) -> int:
        return int(self._source.seek(offset, whence))

    def read(self, size: int = -1) -> bytes:
        chunk = bytes(self._source.read(size))
        if self._first_read:
            self._first_read = False
            self._requested.set()
            if not self._finished.wait(timeout=5):
                raise TimeoutError("in-place mutation did not finish")
        return chunk


def _snapshot_during_concurrent_mutation(
    source: Any,
    snapshot: Any,
    path: Path,
    expected: str,
    evidence: _MutationEvidence,
) -> None:
    mutation_requested = Event()
    mutation_finished = Event()
    writer_errors: list[OSError] = []
    writer_timed_out: list[bool] = []

    def overwrite_open_inode() -> None:
        if not mutation_requested.wait(timeout=5):
            writer_timed_out.append(True)
            mutation_finished.set()
            return
        try:
            evidence.same_open_inode.append(
                os.path.samestat(os.fstat(source.fileno()), path.stat())
            )
            path.write_bytes(evidence.attacker_state)
        except OSError as error:
            writer_errors.append(error)
        finally:
            mutation_finished.set()

    writer = Thread(target=overwrite_open_inode, daemon=True)
    writer.start()
    try:
        _ORIGINAL_SNAPSHOT_VERIFIED_STATE(
            cast("BinaryIO", _CoordinatedSource(source, mutation_requested, mutation_finished)),
            snapshot,
            path,
            expected,
        )
    finally:
        mutation_requested.set()
        writer.join(timeout=5)
    assert not writer.is_alive()
    assert not writer_timed_out
    if writer_errors:
        raise writer_errors[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(
    model: nn.Module,
    *,
    components: CheckpointComponents | None = None,
    state: TrainingState | None = None,
) -> CheckpointSnapshot:
    return CheckpointSnapshot(
        model=model,
        training_state=state or TrainingState(global_step=3, epoch=1, micro_step=2),
        resolved_config={"schema_version": 1, "seed": 17},
        components=components,
        metadata=CheckpointMetadata(
            manifests={"config_checksum": "a" * 64},
            source_revisions={"model": "b" * 40},
        ),
    )


def _save(
    tmp_path: Path,
    *,
    components: CheckpointComponents | None = None,
    state: TrainingState | None = None,
) -> tuple[CheckpointManager, Path, nn.Linear]:
    model = nn.Linear(2, 1)
    checkpoint = tmp_path / "checkpoint"
    manager = CheckpointManager()
    manager.save(checkpoint, snapshot=_snapshot(model, components=components, state=state))
    return manager, checkpoint, model


def _read_manifest(checkpoint: Path) -> dict[str, Any]:
    value = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_manifest(checkpoint: Path, manifest: dict[str, Any]) -> None:
    (checkpoint / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_payload(checkpoint: Path) -> dict[str, Any]:
    value = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
    assert isinstance(value, dict)
    return value


def _write_payload(checkpoint: Path, payload: object) -> None:
    torch.save(payload, checkpoint / "state.pt")
    manifest = _read_manifest(checkpoint)
    manifest["files"]["state.pt"] = _sha256(checkpoint / "state.pt")
    _write_manifest(checkpoint, manifest)


def _optimizer(model: nn.Module) -> torch.optim.Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    optimizer.zero_grad(set_to_none=True)
    model(torch.tensor([[1.0, 2.0]])).sum().backward()
    optimizer.step()
    return optimizer


def _assert_optimizer_state_equal(
    actual: torch.optim.Optimizer, expected: torch.optim.Optimizer
) -> None:
    actual_state = actual.state_dict()
    expected_state = expected.state_dict()
    assert actual_state["param_groups"] == expected_state["param_groups"]
    assert actual_state["state"].keys() == expected_state["state"].keys()
    for parameter_id, expected_values in expected_state["state"].items():
        actual_values = actual_state["state"][parameter_id]
        assert actual_values.keys() == expected_values.keys()
        for name, expected_value in expected_values.items():
            actual_value = actual_values[name]
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_nested_equal(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"global_step": -1}, "non-negative"),
        ({"global_step": True}, "integers"),
        ({"global_step": 1, "epoch": 1.5}, "integers"),
        ({"global_step": 1, "micro_step": "2"}, "integers"),
        ({"global_step": 1, "best_metric": float("nan")}, "finite"),
        ({"global_step": 1, "best_metric": "0.5"}, "number"),
    ],
)
def test_training_state_rejects_invalid_types_and_values(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(CheckpointError, match=message):
        TrainingState(**kwargs)


def test_full_resume_restores_every_declared_component_and_metadata(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = _optimizer(model)
    scheduler = _Stateful({"step": 4})
    scaler = _Stateful({"scale": 128.0})
    state_values: dict[str, dict[str, Any]] = {
        "sampler_state": {"epoch": 1, "cursor": 7},
        "dataset_state": {"shard": 2},
        "gradient_cache_state": {"chunk": 3},
        "objective_state": {"temperature": 0.07},
        "curriculum_state": {"stage": "hard"},
    }
    components = CheckpointComponents(
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        **state_values,
    )
    checkpoint = tmp_path / "all-components"
    manager = CheckpointManager()
    expected_state = TrainingState(global_step=9, epoch=2, micro_step=5, best_metric=0.75)
    manifest = manager.save(
        checkpoint, snapshot=_snapshot(model, components=components, state=expected_state)
    )

    resumed_model = nn.Linear(2, 1)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=1.0, momentum=0.9)
    targets = {name.removesuffix("_state"): _Stateful({}) for name in state_values}
    resumed_scheduler = _Stateful({})
    resumed_scaler = _Stateful({})
    result = manager.load(
        checkpoint,
        model=resumed_model,
        mode="resume_full",
        targets=CheckpointRestoreTargets(
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            scaler=resumed_scaler,
            sampler=targets["sampler"],
            dataset=targets["dataset"],
            gradient_cache=targets["gradient_cache"],
            objective=targets["objective"],
            curriculum=targets["curriculum"],
        ),
    )

    assert result.training_state == expected_state
    assert result.restored_components == (
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "sampler",
        "streaming_dataset",
        "gradient_cache",
        "objective",
        "curriculum",
        "training_state",
        "rng",
    )
    assert torch.equal(resumed_model.weight, model.weight)
    _assert_optimizer_state_equal(resumed_optimizer, optimizer)
    assert resumed_scheduler.loaded == {"step": 4}
    assert resumed_scaler.loaded == {"scale": 128.0}
    assert targets["sampler"].loaded == state_values["sampler_state"]
    assert targets["dataset"].loaded == state_values["dataset_state"]
    assert targets["gradient_cache"].loaded == state_values["gradient_cache_state"]
    assert targets["objective"].loaded == state_values["objective_state"]
    assert targets["curriculum"].loaded == state_values["curriculum_state"]
    assert manifest["source_revisions"] == {"model": "b" * 40}
    assert manifest["manifests"] == {"config_checksum": "a" * 64}


def test_adapter_stage_loads_explicit_adapter_state_and_not_full_model(tmp_path: Path) -> None:
    full_model = nn.Linear(2, 1)
    explicit_adapter = nn.Linear(2, 1)
    with torch.no_grad():
        full_model.weight.fill_(2.0)
        explicit_adapter.weight.fill_(7.0)
        explicit_adapter.bias.fill_(8.0)
    manager = CheckpointManager()
    checkpoint = tmp_path / "adapter"
    manager.save(
        checkpoint,
        snapshot=_snapshot(
            full_model,
            components=CheckpointComponents(adapter_state=explicit_adapter.state_dict()),
        ),
    )
    ignored_model = nn.Linear(2, 1)
    target_adapter = nn.Linear(2, 1)
    ignored_before = deepcopy(ignored_model.state_dict())

    result = manager.load(
        checkpoint,
        model=ignored_model,
        mode="load_adapter_for_new_stage",
        targets=CheckpointRestoreTargets(adapter=target_adapter),
    )

    assert result.training_state is None
    assert result.restored_components == ("adapter",)
    assert torch.equal(target_adapter.weight, explicit_adapter.weight)
    assert torch.equal(target_adapter.bias, explicit_adapter.bias)
    assert all(
        torch.equal(ignored_model.state_dict()[key], value) for key, value in ignored_before.items()
    )


def test_failed_model_only_restore_is_transactional_across_all_parameters(
    tmp_path: Path,
) -> None:
    source = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 3))
    target = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 4))
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(7.0)
        for parameter in target.parameters():
            parameter.zero_()
    manager = CheckpointManager()
    checkpoint = tmp_path / "partial-model"
    manager.save(checkpoint, snapshot=_snapshot(source))
    before = deepcopy(target.state_dict())

    with pytest.raises(CheckpointError, match="model state is incompatible"):
        manager.load(checkpoint, model=target, mode="resume_model_only")

    _assert_nested_equal(target.state_dict(), before)


def test_failed_adapter_restore_is_transactional_across_all_parameters(
    tmp_path: Path,
) -> None:
    source = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 3))
    target = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 4))
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(7.0)
        for parameter in target.parameters():
            parameter.zero_()
    manager = CheckpointManager()
    checkpoint = tmp_path / "partial-adapter"
    manager.save(
        checkpoint,
        snapshot=_snapshot(
            nn.Linear(1, 1),
            components=CheckpointComponents(adapter_state=source.state_dict()),
        ),
    )
    before = deepcopy(target.state_dict())

    with pytest.raises(CheckpointError, match="adapter state is incompatible"):
        manager.load(
            checkpoint,
            model=nn.Linear(1, 1),
            mode="load_adapter_for_new_stage",
            targets=CheckpointRestoreTargets(adapter=target),
        )

    _assert_nested_equal(target.state_dict(), before)


def test_failed_full_restore_rolls_back_model_optimizer_and_components(
    tmp_path: Path,
) -> None:
    source = nn.Linear(2, 1)
    source_optimizer = _optimizer(source)
    with torch.no_grad():
        source.weight.fill_(7.0)
        source.bias.fill_(7.0)
    manager = CheckpointManager()
    checkpoint = tmp_path / "full-rollback"
    manager.save(
        checkpoint,
        snapshot=_snapshot(
            source,
            components=CheckpointComponents(
                optimizer=source_optimizer,
                scheduler=_TransactionalStateful({"source": 1}),
                scaler=_TransactionalStateful({"fail": True}),
            ),
        ),
    )

    target = nn.Linear(2, 1)
    target_optimizer = _optimizer(target)
    with torch.no_grad():
        target.weight.zero_()
        target.bias.zero_()
    target_scheduler = _TransactionalStateful({"target": 2})
    target_scaler = _TransactionalStateful({"target": 3})
    model_before = deepcopy(target.state_dict())
    optimizer_before = deepcopy(target_optimizer.state_dict())
    scheduler_before = target_scheduler.state_dict()
    scaler_before = target_scaler.state_dict()
    python_rng_before = random.getstate()
    numpy_rng_before: Any = np.random.get_state()
    torch_rng_before = torch.get_rng_state().clone()

    with pytest.raises(CheckpointError, match="scaler state is incompatible"):
        manager.load(
            checkpoint,
            model=target,
            mode="resume_full",
            targets=CheckpointRestoreTargets(
                optimizer=target_optimizer,
                scheduler=target_scheduler,
                scaler=target_scaler,
            ),
        )

    _assert_nested_equal(target.state_dict(), model_before)
    _assert_nested_equal(target_optimizer.state_dict(), optimizer_before)
    assert target_scheduler.state_dict() == scheduler_before
    assert target_scaler.state_dict() == scaler_before
    assert random.getstate() == python_rng_before
    numpy_rng_after: Any = np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    assert np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
    assert numpy_rng_after[2:] == numpy_rng_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng_before)


def test_unexpected_restore_failure_rolls_back_targets_and_reports_incomplete_rng_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    target = nn.Linear(2, 1)
    before = deepcopy(target.state_dict())

    def fail_rng_restore(state: dict[str, Any]) -> None:
        del state
        raise RuntimeError("simulated runtime RNG failure")

    monkeypatch.setattr(checkpoint_module, "_restore_rng_state", fail_rng_restore)

    with pytest.raises(CheckpointError, match="rollback was incomplete: rng"):
        manager.load(checkpoint, model=target, mode="resume_model_only")

    _assert_nested_equal(target.state_dict(), before)


def test_module_key_mismatch_is_rejected_before_any_parameter_copy(tmp_path: Path) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    target = nn.Linear(2, 1, bias=False)
    before = deepcopy(target.state_dict())

    with pytest.raises(CheckpointError, match="state keys do not match"):
        manager.load(checkpoint, model=target, mode="resume_model_only")

    _assert_nested_equal(target.state_dict(), before)


def test_model_only_resume_ignores_training_components_and_progress(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    manager, checkpoint, _ = _save(
        tmp_path,
        components=CheckpointComponents(optimizer=_optimizer(model), scheduler=_Stateful({"x": 1})),
    )
    target = nn.Linear(2, 1)

    result = manager.load(checkpoint, model=target, mode="resume_model_only")

    assert result.training_state is None
    assert result.restored_components == ("model",)


def test_rng_streams_resume_at_the_exact_saved_position(tmp_path: Path) -> None:
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    model = nn.Linear(2, 1)
    optimizer = _optimizer(model)
    manager, checkpoint, _ = _save(
        tmp_path,
        components=CheckpointComponents(optimizer=optimizer),
    )
    expected = (random.random(), float(np.random.random()), torch.rand(4))  # noqa: S311
    for _ in range(10):
        random.random()  # noqa: S311
        np.random.random()
        torch.rand(4)

    target = nn.Linear(2, 1)
    manager.load(
        checkpoint,
        model=target,
        mode="resume_full",
        targets=CheckpointRestoreTargets(
            optimizer=torch.optim.SGD(target.parameters(), lr=0.05, momentum=0.9)
        ),
    )

    assert random.random() == expected[0]  # noqa: S311
    assert float(np.random.random()) == expected[1]
    assert torch.equal(torch.rand(4), expected[2])


def test_trusted_manifest_digest_authenticates_before_parsing(tmp_path: Path) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    digest = _sha256(checkpoint / "manifest.json")
    assert manager.verify(checkpoint, expected_manifest_sha256=digest)["checkpoint_version"] == 1

    manifest = _read_manifest(checkpoint)
    manifest["created_at"] = "2026-01-01T00:00:00+00:00"
    _write_manifest(checkpoint, manifest)
    with pytest.raises(CheckpointError, match="trusted digest"):
        manager.verify(checkpoint, expected_manifest_sha256=digest)


def test_trusted_manifest_hash_and_parse_use_the_same_open_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    manifest_path = checkpoint / "manifest.json"
    trusted_digest = _sha256(manifest_path)
    alternate = tmp_path / "alternate-manifest.json"
    alternate_document = _read_manifest(checkpoint)
    alternate_document["manifests"]["attacker"] = "same-path replacement"
    alternate.write_text(json.dumps(alternate_document, sort_keys=True) + "\n", encoding="utf-8")
    original_sha256_stream = checkpoint_module._sha256_stream
    call_count = 0

    def replace_after_hash(stream: Any) -> str:
        nonlocal call_count
        digest = original_sha256_stream(stream)
        call_count += 1
        if call_count == 1:
            alternate.replace(manifest_path)
        return digest

    monkeypatch.setattr(checkpoint_module, "_sha256_stream", replace_after_hash)

    verified = manager.verify(checkpoint, expected_manifest_sha256=trusted_digest)

    assert "attacker" not in verified["manifests"]


def test_config_hash_and_parse_use_the_same_open_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    config_path = checkpoint / "resolved-config.json"
    alternate = tmp_path / "alternate-config.json"
    alternate.write_text("[]\n", encoding="utf-8")
    original_sha256_stream = checkpoint_module._sha256_stream
    call_count = 0

    def replace_after_hash(stream: Any) -> str:
        nonlocal call_count
        digest = original_sha256_stream(stream)
        call_count += 1
        if call_count == 1:
            alternate.replace(config_path)
        return digest

    monkeypatch.setattr(checkpoint_module, "_sha256_stream", replace_after_hash)

    assert manager.verify(checkpoint)["checkpoint_version"] == 1


def test_state_hash_and_load_use_the_same_open_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = CheckpointManager()
    signed_model = nn.Linear(2, 1)
    alternate_model = nn.Linear(2, 1)
    with torch.no_grad():
        signed_model.weight.fill_(1.0)
        signed_model.bias.fill_(1.0)
        alternate_model.weight.fill_(9.0)
        alternate_model.bias.fill_(9.0)
    checkpoint = tmp_path / "signed"
    alternate_checkpoint = tmp_path / "alternate"
    manager.save(checkpoint, snapshot=_snapshot(signed_model))
    manager.save(alternate_checkpoint, snapshot=_snapshot(alternate_model))
    manifest_digest = _sha256(checkpoint / "manifest.json")
    alternate_state = tmp_path / "alternate-state.pt"
    alternate_state.write_bytes((alternate_checkpoint / "state.pt").read_bytes())
    state_path = checkpoint / "state.pt"
    original_snapshot = checkpoint_module._snapshot_verified_state

    def replace_after_snapshot(
        source: Any,
        snapshot: Any,
        path: Path,
        expected: str,
    ) -> None:
        original_snapshot(source, snapshot, path, expected)
        alternate_state.replace(state_path)

    monkeypatch.setattr(checkpoint_module, "_snapshot_verified_state", replace_after_snapshot)
    target = nn.Linear(2, 1)

    manager.load(
        checkpoint,
        model=target,
        mode="resume_model_only",
        expected_manifest_sha256=manifest_digest,
    )

    assert torch.equal(target.weight, signed_model.weight)
    assert torch.equal(target.bias, signed_model.bias)


@pytest.mark.parametrize("mutation_timing", ["during_snapshot", "after_snapshot"])
def test_concurrent_in_place_state_mutation_never_loads_attacker_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_timing: str,
) -> None:
    manager = CheckpointManager()
    signed_model = nn.Linear(400_000, 1)
    attacker_model = nn.Linear(400_000, 1)
    with torch.no_grad():
        signed_model.weight.fill_(1.0)
        signed_model.bias.fill_(1.0)
        attacker_model.weight.fill_(9.0)
        attacker_model.bias.fill_(9.0)
    checkpoint = tmp_path / "signed"
    attacker_checkpoint = tmp_path / "attacker"
    manager.save(checkpoint, snapshot=_snapshot(signed_model))
    manager.save(attacker_checkpoint, snapshot=_snapshot(attacker_model))
    manifest_digest = _sha256(checkpoint / "manifest.json")
    attacker_state = (attacker_checkpoint / "state.pt").read_bytes()
    assert len(attacker_state) > 1024 * 1024
    state_path = checkpoint / "state.pt"
    same_open_inode: list[bool] = []
    evidence = _MutationEvidence(attacker_state, same_open_inode)

    def mutate_open_inode(
        source: Any,
        snapshot: Any,
        path: Path,
        expected: str,
    ) -> None:
        if mutation_timing == "during_snapshot":
            _snapshot_during_concurrent_mutation(source, snapshot, path, expected, evidence)
            return

        _ORIGINAL_SNAPSHOT_VERIFIED_STATE(source, snapshot, path, expected)
        if mutation_timing == "after_snapshot":
            same_open_inode.append(os.path.samestat(os.fstat(source.fileno()), state_path.stat()))
            state_path.write_bytes(attacker_state)

    monkeypatch.setattr(checkpoint_module, "_snapshot_verified_state", mutate_open_inode)
    target = nn.Linear(400_000, 1)
    before = deepcopy(target.state_dict())

    if mutation_timing == "during_snapshot":
        with pytest.raises(CheckpointError, match="checksum mismatch"):
            manager.load(
                checkpoint,
                model=target,
                mode="resume_model_only",
                expected_manifest_sha256=manifest_digest,
            )
        _assert_nested_equal(target.state_dict(), before)
    else:
        manager.load(
            checkpoint,
            model=target,
            mode="resume_model_only",
            expected_manifest_sha256=manifest_digest,
        )
        assert torch.equal(target.weight, signed_model.weight)
        assert torch.equal(target.bias, signed_model.bias)

    assert same_open_inode == [True]
    assert not torch.equal(target.weight, attacker_model.weight)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"unexpected": True}), "unexpected fields"),
        (lambda value: value.pop("components"), "missing fields"),
        (lambda value: value.update({"checkpoint_version": 2}), "unsupported checkpoint version"),
        (lambda value: value.update({"files": []}), "checksummed files"),
        (lambda value: value["files"].update({"../state.pt": "0" * 64}), "file name"),
        (lambda value: value["files"].update({"state.pt": 7}), "checksum"),
        (lambda value: value["files"].update({"state.pt": "A" * 64}), "checksum"),
        (lambda value: value.update({"components": []}), "component flags"),
        (lambda value: value["components"].update({"model": "yes"}), "component flags"),
        (lambda value: value["components"].update({"model": False}), "model must be true"),
        (lambda value: value.update({"created_at": "not-a-time"}), "created_at"),
        (lambda value: value.update({"created_at": 7}), "created_at"),
        (lambda value: value.update({"created_at": "2026-01-01T00:00:00"}), "created_at"),
        (lambda value: value.update({"dependency_versions": []}), "dependency_versions"),
        (lambda value: value.update({"source_revisions": {"model": 4}}), "source_revisions"),
        (lambda value: value.update({"manifests": []}), "manifests provenance"),
        (lambda value: value.update({"training_state": []}), "training_state.*mapping"),
    ],
)
def test_manifest_rejects_extra_missing_type_and_version_errors(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    manifest = _read_manifest(checkpoint)
    mutation(manifest)
    _write_manifest(checkpoint, manifest)

    with pytest.raises(CheckpointError, match=message):
        manager.verify(checkpoint)


@pytest.mark.parametrize("document", [[], "manifest", None])
def test_manifest_root_must_be_an_object(tmp_path: Path, document: object) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    (checkpoint / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CheckpointError, match="root must be an object"):
        manager.verify(checkpoint)


def test_verify_rejects_duplicate_checkpoint_json_keys(tmp_path: Path) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    manifest_path = checkpoint / "manifest.json"
    content = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        content.replace(
            '"checkpoint_version": 1',
            '"checkpoint_version": 1,\n  "checkpoint_version": 1',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointError, match="manifest is unreadable"):
        manager.verify(checkpoint)

    manager, checkpoint, _ = _save(tmp_path / "config")
    config_path = checkpoint / "resolved-config.json"
    config_path.write_text('{"seed":17,"seed":18}\n', encoding="utf-8")
    manifest = _read_manifest(checkpoint)
    manifest["files"]["resolved-config.json"] = _sha256(config_path)
    _write_manifest(checkpoint, manifest)
    with pytest.raises(CheckpointError, match="resolved config is unreadable"):
        manager.verify(checkpoint)


def test_verify_rejects_unreadable_missing_untracked_and_symlink_files(tmp_path: Path) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    (checkpoint / "manifest.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(CheckpointError, match="manifest is unreadable"):
        manager.verify(checkpoint)

    checkpoint = tmp_path / "missing-manifest"
    checkpoint.mkdir()
    with pytest.raises(CheckpointError, match="manifest is missing"):
        manager.verify(checkpoint)

    manager, checkpoint, _ = _save(tmp_path / "extra")
    (checkpoint / "untracked.bin").write_bytes(b"not checksummed")
    with pytest.raises(CheckpointError, match="unexpected entries"):
        manager.verify(checkpoint)

    manager, checkpoint, _ = _save(tmp_path / "missing")
    (checkpoint / "state.pt").unlink()
    with pytest.raises(CheckpointError, match="missing entries"):
        manager.verify(checkpoint)

    manager, checkpoint, _ = _save(tmp_path / "symlink")
    target = tmp_path / "external-state.pt"
    (checkpoint / "state.pt").replace(target)
    (checkpoint / "state.pt").symlink_to(target)
    with pytest.raises(CheckpointError, match="regular file"):
        manager.verify(checkpoint)

    manager, checkpoint, _ = _save(tmp_path / "dangling")
    (checkpoint / "state.pt").unlink()
    (checkpoint / "state.pt").symlink_to(tmp_path / "does-not-exist.pt")
    with pytest.raises(CheckpointError, match="file is missing"):
        manager.verify(checkpoint)

    manager, checkpoint, _ = _save(tmp_path / "manifest-link")
    external_manifest = tmp_path / "external-manifest.json"
    (checkpoint / "manifest.json").replace(external_manifest)
    (checkpoint / "manifest.json").symlink_to(external_manifest)
    with pytest.raises(CheckpointError, match="regular file"):
        manager.verify(checkpoint)


def test_verify_rejects_checksum_mismatch_and_non_object_config(tmp_path: Path) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    with (checkpoint / "state.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(CheckpointError, match="checksum mismatch"):
        manager.verify(checkpoint)

    manager, checkpoint, _ = _save(tmp_path / "config")
    (checkpoint / "resolved-config.json").write_text("[]\n", encoding="utf-8")
    manifest = _read_manifest(checkpoint)
    manifest["files"]["resolved-config.json"] = _sha256(checkpoint / "resolved-config.json")
    _write_manifest(checkpoint, manifest)
    with pytest.raises(CheckpointError, match="resolved config root"):
        manager.verify(checkpoint)

    manager, checkpoint, _ = _save(tmp_path / "nonfinite-config")
    (checkpoint / "resolved-config.json").write_text('{"loss": NaN}\n', encoding="utf-8")
    manifest = _read_manifest(checkpoint)
    manifest["files"]["resolved-config.json"] = _sha256(checkpoint / "resolved-config.json")
    _write_manifest(checkpoint, manifest)
    with pytest.raises(CheckpointError, match="resolved config is unreadable"):
        manager.verify(checkpoint)


def test_verify_wraps_directory_and_file_read_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    path_type = type(checkpoint)
    original_iterdir = path_type.iterdir

    def fail_iterdir(path: Path) -> Any:
        del path
        raise OSError("directory unavailable")

    monkeypatch.setattr(path_type, "iterdir", fail_iterdir)
    with pytest.raises(CheckpointError, match="directory is unreadable"):
        manager.verify(checkpoint)
    monkeypatch.setattr(path_type, "iterdir", original_iterdir)

    def fail_hash(stream: Any) -> str:
        del stream
        raise OSError("file unavailable")

    monkeypatch.setattr(checkpoint_module, "_sha256_stream", fail_hash)
    with pytest.raises(CheckpointError, match="file is unreadable"):
        manager.verify(checkpoint)


def test_load_wraps_unreadable_state_after_checksum_verification(tmp_path: Path) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    (checkpoint / "state.pt").write_bytes(b"not a torch checkpoint")
    manifest = _read_manifest(checkpoint)
    manifest["files"]["state.pt"] = _sha256(checkpoint / "state.pt")
    _write_manifest(checkpoint, manifest)

    with pytest.raises(CheckpointError, match="state is unreadable"):
        manager.load(checkpoint, model=nn.Linear(2, 1), mode="resume_model_only")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"extra": {}}), "unexpected fields"),
        (lambda value: value.pop("adapter_state"), "missing fields"),
        (lambda value: value.update({"checkpoint_version": 99}), "invalid version"),
        (lambda value: value.update({"scheduler": []}), "scheduler.*mapping"),
        (lambda value: value.update({"training_state": []}), "training_state.*mapping"),
        (lambda value: value.update({"rng": []}), "rng.*mapping"),
    ],
)
def test_state_rejects_extra_missing_type_and_version_errors(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    payload = _read_payload(checkpoint)
    mutate(payload)
    _write_payload(checkpoint, payload)

    with pytest.raises(CheckpointError, match=message):
        manager.load(checkpoint, model=nn.Linear(2, 1), mode="resume_model_only")


@pytest.mark.parametrize("payload", [[], "state", None])
def test_state_root_must_be_a_versioned_object(tmp_path: Path, payload: object) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    _write_payload(checkpoint, payload)
    with pytest.raises(CheckpointError, match="invalid version or root type"):
        manager.load(checkpoint, model=nn.Linear(2, 1), mode="resume_model_only")


def test_state_manifest_component_and_progress_declarations_must_match(tmp_path: Path) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    manifest = _read_manifest(checkpoint)
    manifest["components"]["scheduler"] = True
    _write_manifest(checkpoint, manifest)
    with pytest.raises(CheckpointError, match="component declarations"):
        manager.load(checkpoint, model=nn.Linear(2, 1), mode="resume_model_only")

    manager, checkpoint, _ = _save(tmp_path / "progress")
    manifest = _read_manifest(checkpoint)
    manifest["training_state"]["global_step"] = 99
    _write_manifest(checkpoint, manifest)
    with pytest.raises(CheckpointError, match="training state declaration"):
        manager.load(checkpoint, model=nn.Linear(2, 1), mode="resume_model_only")


def test_resume_modes_require_their_explicit_targets_and_states(tmp_path: Path) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    with pytest.raises(CheckpointError, match="requires a optimizer"):
        manager.load(checkpoint, model=nn.Linear(2, 1), mode="resume_full")
    target = nn.Linear(2, 1)
    with pytest.raises(CheckpointError, match="does not contain required optimizer"):
        manager.load(
            checkpoint,
            model=target,
            mode="resume_full",
            targets=CheckpointRestoreTargets(
                optimizer=torch.optim.SGD(target.parameters(), lr=0.1)
            ),
        )
    with pytest.raises(CheckpointError, match="requires an adapter module"):
        manager.load(checkpoint, model=nn.Linear(2, 1), mode="load_adapter_for_new_stage")
    with pytest.raises(CheckpointError, match=r"adapter_state.*mapping"):
        manager.load(
            checkpoint,
            model=nn.Linear(2, 1),
            mode="load_adapter_for_new_stage",
            targets=CheckpointRestoreTargets(adapter=nn.Linear(2, 1)),
        )
    with pytest.raises(CheckpointError, match="unknown checkpoint load mode"):
        manager.load(checkpoint, model=nn.Linear(2, 1), mode="unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize("component", ["scheduler", "scaler"])
def test_resume_full_rejects_declared_component_without_target(
    tmp_path: Path, component: str
) -> None:
    model = nn.Linear(2, 1)
    optimizer = _optimizer(model)
    components = CheckpointComponents(optimizer=optimizer)
    setattr(components, component, _Stateful({"value": 1}))
    manager, checkpoint, _ = _save(tmp_path, components=components)
    target = nn.Linear(2, 1)
    with pytest.raises(CheckpointError, match=f"contains {component} state"):
        manager.load(
            checkpoint,
            model=target,
            mode="resume_full",
            targets=CheckpointRestoreTargets(
                optimizer=torch.optim.SGD(target.parameters(), lr=0.1)
            ),
        )


@pytest.mark.parametrize(
    ("payload_name", "target_name", "component_name"),
    [
        ("sampler_state", "sampler", "sampler"),
        ("dataset_state", "dataset", "streaming_dataset"),
        ("gradient_cache_state", "gradient_cache", "gradient_cache"),
        ("objective_state", "objective", "objective"),
        ("curriculum_state", "curriculum", "curriculum"),
    ],
)
def test_extended_resume_requires_state_and_target_in_both_directions(
    tmp_path: Path,
    payload_name: str,
    target_name: str,
    component_name: str,
) -> None:
    model = nn.Linear(2, 1)
    optimizer = _optimizer(model)
    components = CheckpointComponents(optimizer=optimizer)
    setattr(components, payload_name, {"position": 4})
    manager, checkpoint, _ = _save(tmp_path, components=components)
    target_model = nn.Linear(2, 1)
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.05, momentum=0.9)
    with pytest.raises(CheckpointError, match=f"contains {component_name} state"):
        manager.load(
            checkpoint,
            model=target_model,
            mode="resume_full",
            targets=CheckpointRestoreTargets(optimizer=target_optimizer),
        )

    manager, checkpoint, _ = _save(
        tmp_path / "missing", components=CheckpointComponents(optimizer=optimizer)
    )
    target_model = nn.Linear(2, 1)
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.05, momentum=0.9)
    targets = CheckpointRestoreTargets(optimizer=target_optimizer)
    setattr(targets, target_name, _Stateful({}))
    with pytest.raises(CheckpointError, match=f"does not contain requested {component_name} state"):
        manager.load(checkpoint, model=target_model, mode="resume_full", targets=targets)


def test_incompatible_model_adapter_and_stateful_targets_are_wrapped(tmp_path: Path) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    with pytest.raises(CheckpointError, match="model state is incompatible"):
        manager.load(checkpoint, model=nn.Linear(3, 1), mode="resume_model_only")

    adapter = nn.Linear(2, 1)
    manager, checkpoint, _ = _save(
        tmp_path / "adapter",
        components=CheckpointComponents(adapter_state=adapter.state_dict()),
    )
    with pytest.raises(CheckpointError, match="adapter state is incompatible"):
        manager.load(
            checkpoint,
            model=nn.Linear(2, 1),
            mode="load_adapter_for_new_stage",
            targets=CheckpointRestoreTargets(adapter=nn.Linear(3, 1)),
        )

    model = nn.Linear(2, 1)
    manager, checkpoint, _ = _save(
        tmp_path / "stateful",
        components=CheckpointComponents(
            optimizer=_optimizer(model),
            sampler_state={"cursor": 1},
        ),
    )
    target = nn.Linear(2, 1)
    with pytest.raises(CheckpointError, match="has no load_state_dict"):
        manager.load(
            checkpoint,
            model=target,
            mode="resume_full",
            targets=CheckpointRestoreTargets(
                optimizer=torch.optim.SGD(target.parameters(), lr=0.05, momentum=0.9),
                sampler=object(),
            ),
        )
    with pytest.raises(CheckpointError, match="sampler state is incompatible"):
        manager.load(
            checkpoint,
            model=target,
            mode="resume_full",
            targets=CheckpointRestoreTargets(
                optimizer=torch.optim.SGD(target.parameters(), lr=0.05, momentum=0.9),
                sampler=_IncompatibleStateful({}),
            ),
        )

    target = nn.Linear(2, 1)
    with pytest.raises(CheckpointError, match="optimizer state is incompatible"):
        manager.load(
            checkpoint,
            model=target,
            mode="resume_full",
            targets=CheckpointRestoreTargets(
                optimizer=torch.optim.SGD([target.weight], lr=0.05, momentum=0.9),
                sampler=_Stateful({}),
            ),
        )


def test_invalid_rng_payload_is_rejected_without_mutating_any_rng(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    manager, checkpoint, _ = _save(
        tmp_path,
        components=CheckpointComponents(optimizer=_optimizer(model)),
    )
    payload = _read_payload(checkpoint)
    payload["rng"]["numpy"].pop("position")
    _write_payload(checkpoint, payload)
    target = nn.Linear(2, 1)
    target_before = deepcopy(target.state_dict())
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    python_before = random.getstate()
    numpy_before: Any = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    with pytest.raises(CheckpointError, match=r"RNG state.*missing fields"):
        manager.load(
            checkpoint,
            model=target,
            mode="resume_full",
            targets=CheckpointRestoreTargets(
                optimizer=torch.optim.SGD(target.parameters(), lr=0.05, momentum=0.9)
            ),
        )

    assert random.getstate() == python_before
    numpy_after: Any = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert all(torch.equal(target.state_dict()[key], value) for key, value in target_before.items())


@pytest.mark.parametrize(
    ("rng_value", "message"),
    [
        ({"python": (), "numpy": {}, "torch_cpu": [], "torch_cuda": []}, "invalid values"),
        (
            {"python": (), "numpy": {}, "torch_cpu": torch.get_rng_state(), "torch_cuda": {}},
            "CUDA RNG state",
        ),
        (
            {
                "python": (),
                "numpy": {},
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": [torch.get_rng_state()],
            },
            "CUDA is unavailable",
        ),
    ],
)
def test_invalid_rng_value_types_are_wrapped(
    tmp_path: Path, rng_value: dict[str, Any], message: str
) -> None:
    if "unavailable" in message and torch.cuda.is_available():
        pytest.skip("CPU-only validation case")
    model = nn.Linear(2, 1)
    manager, checkpoint, _ = _save(
        tmp_path,
        components=CheckpointComponents(optimizer=_optimizer(model)),
    )
    payload = _read_payload(checkpoint)
    payload["rng"] = rng_value
    _write_payload(checkpoint, payload)
    target = nn.Linear(2, 1)
    with pytest.raises(CheckpointError, match=message):
        manager.load(
            checkpoint,
            model=target,
            mode="resume_full",
            targets=CheckpointRestoreTargets(
                optimizer=torch.optim.SGD(target.parameters(), lr=0.05, momentum=0.9)
            ),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("numpy.keys", [], "NumPy RNG state contains invalid values"),
        ("python", (), "RNG state is invalid"),
    ],
)
def test_malformed_rng_domain_states_are_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    model = nn.Linear(2, 1)
    manager, checkpoint, _ = _save(
        tmp_path,
        components=CheckpointComponents(optimizer=_optimizer(model)),
    )
    payload = _read_payload(checkpoint)
    if field == "python":
        payload["rng"]["python"] = value
    else:
        payload["rng"]["numpy"]["keys"] = value
    _write_payload(checkpoint, payload)
    target = nn.Linear(2, 1)

    with pytest.raises(CheckpointError, match=message):
        manager.load(
            checkpoint,
            model=target,
            mode="resume_full",
            targets=CheckpointRestoreTargets(
                optimizer=torch.optim.SGD(target.parameters(), lr=0.05, momentum=0.9)
            ),
        )


def test_cuda_rng_count_and_restore_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = nn.Linear(2, 1)
    manager, checkpoint, _ = _save(
        tmp_path,
        components=CheckpointComponents(optimizer=_optimizer(model)),
    )
    payload = _read_payload(checkpoint)
    payload["rng"]["torch_cuda"] = [torch.get_rng_state()]
    _write_payload(checkpoint, payload)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    target = nn.Linear(2, 1)
    with pytest.raises(CheckpointError, match="count does not match"):
        manager.load(
            checkpoint,
            model=target,
            mode="resume_full",
            targets=CheckpointRestoreTargets(
                optimizer=torch.optim.SGD(target.parameters(), lr=0.05, momentum=0.9)
            ),
        )

    restored: list[list[torch.Tensor]] = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def record_cuda_states(states: list[torch.Tensor]) -> None:
        restored.append(states)

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", record_cuda_states)
    target = nn.Linear(2, 1)
    manager.load(
        checkpoint,
        model=target,
        mode="resume_full",
        targets=CheckpointRestoreTargets(
            optimizer=torch.optim.SGD(target.parameters(), lr=0.05, momentum=0.9)
        ),
    )
    assert len(restored) == 1
    assert len(restored[0]) == 1
    assert torch.equal(restored[0][0], payload["rng"]["torch_cuda"][0])


def test_save_is_new_only_and_rolls_back_precommit_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, checkpoint, model = _save(tmp_path)
    with pytest.raises(CheckpointError, match="already exists"):
        manager.save(checkpoint, snapshot=_snapshot(model))

    invalid = tmp_path / "invalid-config"
    snapshot = _snapshot(model)
    snapshot.resolved_config["invalid"] = object()
    with pytest.raises(CheckpointError, match="failed to save"):
        manager.save(invalid, snapshot=snapshot)
    assert not invalid.exists()
    assert not list(tmp_path.glob(".invalid-config.tmp-*"))

    destination = tmp_path / "verification-failure"

    def fail_verify(*args: object, **kwargs: object) -> dict[str, Any]:
        del args, kwargs
        raise CheckpointError("simulated precommit verification failure")

    monkeypatch.setattr(manager, "verify", fail_verify)
    with pytest.raises(CheckpointError, match="precommit verification failure"):
        manager.save(destination, snapshot=_snapshot(model))
    assert not destination.exists()
    assert not list(tmp_path.glob(".verification-failure.tmp-*"))


@pytest.mark.parametrize("failure_point", ["parent", "temporary"])
def test_save_normalizes_directory_creation_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    destination = tmp_path / "nested" / "checkpoint"
    path_type = type(destination)
    original_mkdir = path_type.mkdir

    def fail_selected_directory(path: Path, *args: Any, **kwargs: Any) -> None:
        if failure_point == "parent" and path == destination.parent:
            raise PermissionError("parent denied")
        if failure_point == "temporary" and path.name.startswith(".checkpoint.tmp-"):
            raise PermissionError("temporary denied")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "mkdir", fail_selected_directory)

    with pytest.raises(CheckpointError, match="failed to save checkpoint"):
        CheckpointManager().save(destination, snapshot=_snapshot(nn.Linear(2, 1)))

    assert not destination.exists()
    if destination.parent.exists():
        assert not list(destination.parent.glob(".checkpoint.tmp-*"))


def test_save_does_not_clean_up_a_temporary_directory_it_did_not_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FixedUuid:
        hex = "fixed"

    destination = tmp_path / "checkpoint"
    foreign_temporary = tmp_path / ".checkpoint.tmp-fixed"
    foreign_temporary.mkdir()
    sentinel = foreign_temporary / "sentinel"
    sentinel.write_text("foreign", encoding="utf-8")
    monkeypatch.setattr(uuid, "uuid4", _FixedUuid)

    with pytest.raises(CheckpointError, match="failed to save checkpoint"):
        CheckpointManager().save(destination, snapshot=_snapshot(nn.Linear(2, 1)))

    assert sentinel.read_text(encoding="utf-8") == "foreign"


def test_save_never_replaces_destination_created_during_serialization(tmp_path: Path) -> None:
    destination = tmp_path / "raced-destination"
    created_inode: list[int] = []
    model = nn.Linear(2, 1)

    def create_competing_destination(module: nn.Module, prefix: str, keep_vars: object) -> None:
        del module, prefix, keep_vars
        destination.mkdir()
        created_inode.append(destination.stat().st_ino)

    model.register_state_dict_pre_hook(  # type: ignore[no-untyped-call]
        create_competing_destination
    )

    with pytest.raises(CheckpointError, match="already exists"):
        CheckpointManager().save(destination, snapshot=_snapshot(model))

    assert destination.is_dir()
    assert destination.stat().st_ino == created_inode[0]
    assert not list(destination.iterdir())
    assert not list(tmp_path.glob(".raced-destination.tmp-*"))


def test_save_rejects_components_with_invalid_state_dict_contract(tmp_path: Path) -> None:
    class MissingStateDict:
        pass

    class InvalidStateDict:
        def state_dict(self) -> list[object]:
            return []

    model = nn.Linear(2, 1)
    for component in (MissingStateDict(), InvalidStateDict()):
        with pytest.raises(CheckpointError, match="state_dict"):
            CheckpointManager().save(
                tmp_path / type(component).__name__,
                snapshot=_snapshot(model, components=CheckpointComponents(scheduler=component)),
            )
        assert not (tmp_path / type(component).__name__).exists()


def test_restore_preflight_normalizes_unexpected_and_rng_rollback_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = checkpoint_module._RestoreOperation("component", object(), {})

    def fail_preflight(value: object) -> None:
        del value
        raise RuntimeError("unexpected preflight failure")

    monkeypatch.setattr(checkpoint_module, "_preflight_restore_operation", fail_preflight)
    with pytest.raises(CheckpointError, match=r"restore preflight failed$"):
        checkpoint_module._execute_restore_transaction((operation,))

    def fail_rng_rollback(state: dict[str, Any]) -> None:
        del state
        raise RuntimeError("RNG rollback failure")

    monkeypatch.setattr(checkpoint_module, "_restore_rng_state", fail_rng_rollback)
    with pytest.raises(CheckpointError, match="RNG rollback was incomplete"):
        checkpoint_module._execute_restore_transaction((operation,))


def test_restore_preflight_and_snapshot_reject_untransactional_targets() -> None:
    class UncopyableTarget:
        def __deepcopy__(self, memo: dict[int, Any]) -> UncopyableTarget:
            del memo
            raise TypeError("cannot copy")

        def load_state_dict(self, state: dict[str, Any]) -> None:
            del state

    with pytest.raises(CheckpointError, match="cannot be transactionally preflighted"):
        checkpoint_module._preflight_restore_operation(
            checkpoint_module._RestoreOperation("uncopyable", UncopyableTarget(), {})
        )

    class MissingStateDict:
        pass

    class RaisingStateDict:
        def state_dict(self) -> dict[str, Any]:
            raise RuntimeError("cannot inspect")

    class InvalidStateDict:
        def state_dict(self) -> list[object]:
            return []

    class UncopyableValue:
        def __deepcopy__(self, memo: dict[int, Any]) -> UncopyableValue:
            del memo
            raise TypeError("cannot copy state")

    class UncopyableState:
        def state_dict(self) -> dict[str, Any]:
            return {"value": UncopyableValue()}

    cases = (
        (MissingStateDict(), "has no state_dict"),
        (RaisingStateDict(), "could not be snapshotted"),
        (InvalidStateDict(), "did not return a mapping"),
        (UncopyableState(), "could not be snapshotted"),
    )
    for target, message in cases:
        with pytest.raises(CheckpointError, match=message):
            checkpoint_module._snapshot_target_state(
                checkpoint_module._RestoreOperation("snapshot", target, {})
            )


def test_module_preflight_wraps_inspection_and_non_tensor_failures() -> None:
    class InspectionFailure(nn.Module):
        def state_dict(  # type: ignore[override]
            self, *args: Any, **kwargs: Any
        ) -> dict[str, Any]:
            del args, kwargs
            raise RuntimeError("inspection failed")

    with pytest.raises(CheckpointError, match="could not be inspected"):
        checkpoint_module._validate_module_state_compatible(
            checkpoint_module._RestoreOperation("model", InspectionFailure(), {})
        )

    target = nn.Linear(2, 1)
    state = dict(target.state_dict())
    state["weight"] = "not-a-tensor"
    with pytest.raises(CheckpointError, match="is not a tensor"):
        checkpoint_module._validate_module_state_compatible(
            checkpoint_module._RestoreOperation("model", target, state)
        )


def test_rollback_continues_after_a_target_refuses_its_snapshot() -> None:
    class RefusingTarget:
        def load_state_dict(self, state: dict[str, Any]) -> None:
            del state
            raise RuntimeError("rollback refused")

    operation = checkpoint_module._RestoreOperation("refusing", RefusingTarget(), {})
    errors = checkpoint_module._rollback_restore_transaction(
        ((operation, {}),),
        checkpoint_module._capture_rng_state(),
    )
    assert errors == ["refusing"]


def test_verify_open_state_stream_modes_and_manifest_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, checkpoint, _ = _save(tmp_path)
    with pytest.raises(CheckpointError, match="snapshot requires an open state stream"):
        checkpoint_module._verify_checkpoint(
            checkpoint,
            expected_manifest_sha256=None,
            state_snapshot=io.BytesIO(),
        )

    with (checkpoint / "state.pt").open("rb") as state_stream:
        manifest = checkpoint_module._verify_checkpoint(
            checkpoint,
            expected_manifest_sha256=None,
            state_stream=state_stream,
        )
    assert manifest["checkpoint_version"] == 1

    def fail_hash(stream: Any) -> str:
        del stream
        raise OSError("manifest read failed")

    monkeypatch.setattr(checkpoint_module, "_sha256_stream", fail_hash)
    with pytest.raises(CheckpointError, match="manifest is unreadable"):
        manager.verify(checkpoint, expected_manifest_sha256="0" * 64)


def test_state_snapshot_and_open_file_errors_are_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenReader(io.BytesIO):
        def seek(self, *args: Any, **kwargs: Any) -> int:
            del args, kwargs
            raise OSError("seek failed")

    with pytest.raises(CheckpointError, match="file is unreadable"):
        checkpoint_module._snapshot_verified_state(
            BrokenReader(b"state"),
            io.BytesIO(),
            tmp_path / "state.pt",
            "0" * 64,
        )

    dangling = tmp_path / "dangling-manifest.json"
    dangling.symlink_to(tmp_path / "missing.json")
    with pytest.raises(CheckpointError, match="manifest is missing"):
        checkpoint_module._open_checkpoint_file(dangling, manifest=True)

    regular = tmp_path / "regular.bin"
    regular.write_bytes(b"content")

    def missing_file(path: Path) -> Any:
        del path
        raise FileNotFoundError("gone")

    monkeypatch.setattr(checkpoint_module, "_open_bound_regular_file", missing_file)
    with pytest.raises(CheckpointError, match="checkpoint file is missing"):
        checkpoint_module._open_checkpoint_file(regular)
    with pytest.raises(CheckpointError, match="checkpoint manifest is missing"):
        checkpoint_module._open_checkpoint_file(regular, manifest=True)

    def unreadable_file(path: Path) -> Any:
        del path
        raise PermissionError("denied")

    monkeypatch.setattr(checkpoint_module, "_open_bound_regular_file", unreadable_file)
    with pytest.raises(CheckpointError, match="checkpoint file is unreadable"):
        checkpoint_module._open_checkpoint_file(regular)
    with pytest.raises(CheckpointError, match="checkpoint manifest is unreadable"):
        checkpoint_module._open_checkpoint_file(regular, manifest=True)


def test_bound_file_identity_and_directory_sync_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(OSError, match="not a regular file"):
        checkpoint_module._open_bound_regular_file(directory)

    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    with pytest.raises(OSError, match="changed while it was opened"):
        filesystem._require_same_regular_file(first.stat(), second.stat(), first)

    closed_descriptors: list[int] = []
    original_close = os.close

    def fail_identity(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError("identity failed")

    def record_close(descriptor: int) -> None:
        closed_descriptors.append(descriptor)
        original_close(descriptor)

    monkeypatch.setattr(filesystem, "_require_same_regular_file", fail_identity)
    monkeypatch.setattr(os, "close", record_close)
    with pytest.raises(OSError, match="identity failed"):
        checkpoint_module._open_bound_regular_file(first)
    assert closed_descriptors

    with monkeypatch.context() as context:
        context.setattr(os, "open", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
        checkpoint_module._sync_directory(tmp_path)

    with monkeypatch.context() as context:
        context.setattr(os, "fsync", lambda descriptor: (_ for _ in ()).throw(OSError()))
        checkpoint_module._sync_directory(tmp_path)
