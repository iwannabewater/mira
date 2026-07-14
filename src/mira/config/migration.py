"""Explicit, loss-aware configuration schema migrations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, Final

from mira.config.schema import CONFIG_SCHEMA_VERSION
from mira.core.exceptions import MigrationError

Migration = Callable[[dict[str, Any]], dict[str, Any]]
_OLDEST_SCHEMA_VERSION: Final = 0


def migrate_config(document: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate a configuration mapping to the current schema version.

    Version 0 is the only historical format. A missing ``schema_version`` is
    rejected because guessing a source schema could silently reinterpret fields.

    Raises:
        MigrationError: If the version is missing, future, unsupported, or a
            migration would overwrite both old and new spellings.
    """
    migrated = deepcopy(dict(document))
    version = migrated.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise MigrationError("configuration schema_version must be an explicit integer")
    if version > CONFIG_SCHEMA_VERSION:
        raise MigrationError(
            f"configuration schema_version {version} is newer than supported "
            f"version {CONFIG_SCHEMA_VERSION}"
        )
    if version < _OLDEST_SCHEMA_VERSION:
        raise MigrationError(f"unsupported configuration schema_version {version}")
    migrations: dict[int, Migration] = {0: _migrate_v0_to_v1}
    while version < CONFIG_SCHEMA_VERSION:
        migration = migrations.get(version)
        if migration is None:
            raise MigrationError(f"no migration path from schema_version {version}")
        migrated = migration(migrated)
        next_version = migrated.get("schema_version")
        if not isinstance(next_version, int) or next_version <= version:
            raise MigrationError(f"migration from version {version} did not advance the schema")
        version = next_version
    return migrated


def _migrate_v0_to_v1(document: dict[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(document)
    _rename_top_level(migrated, "loss", "objectives")

    project = _mapping_section(migrated, "project", create=True)
    _move_top_level(migrated, project, "project_name", "name")
    _move_top_level(migrated, project, "seed", "seed")

    _migrate_backend(migrated)
    _migrate_model(migrated)
    _migrate_objectives(migrated)
    _migrate_data(migrated)
    _migrate_runtime(migrated)

    migrated["schema_version"] = 1
    return migrated


def _migrate_backend(migrated: dict[str, Any]) -> None:
    backend = migrated.get("backend")
    if isinstance(backend, str):
        migrated["backend"] = {"type": backend}
    elif "backend" in migrated and not isinstance(backend, Mapping):
        raise MigrationError("v0 backend must be a string or mapping")


def _migrate_model(migrated: dict[str, Any]) -> None:
    model = migrated.get("model")
    if isinstance(model, str):
        adapter = migrated.pop("model_adapter", None)
        if not isinstance(adapter, str):
            raise MigrationError("v0 string model requires a model_adapter string")
        migrated["model"] = {"name_or_path": model, "adapter": adapter}
    elif isinstance(model, Mapping):
        model_section = dict(model)
        if "model_adapter" in migrated:
            if "adapter" in model_section:
                raise MigrationError(
                    "v0 configuration contains both model_adapter and model.adapter"
                )
            adapter = migrated.pop("model_adapter")
            if not isinstance(adapter, str):
                raise MigrationError("v0 model_adapter must be a string")
            model_section["adapter"] = adapter
        _rename_in_section(model_section, "path", "name_or_path", section="model")
        migrated["model"] = model_section
    elif "model" in migrated:
        raise MigrationError("v0 model must be a string or mapping")


def _migrate_objectives(migrated: dict[str, Any]) -> None:
    objectives = migrated.get("objectives")
    if isinstance(objectives, Mapping):
        migrated["objectives"] = [dict(objectives)]
    elif objectives is not None and not isinstance(objectives, list | tuple):
        raise MigrationError("v0 objectives must be a mapping or sequence")


def _migrate_data(migrated: dict[str, Any]) -> None:
    data = migrated.get("data")
    if isinstance(data, Mapping):
        data_section = dict(data)
        _rename_in_section(data_section, "val", "validation", section="data")
        migrated["data"] = data_section
    elif "data" in migrated:
        raise MigrationError("v0 data must be a mapping")


def _migrate_runtime(migrated: dict[str, Any]) -> None:
    runtime = migrated.get("runtime")
    if isinstance(runtime, Mapping):
        runtime_section = dict(runtime)
        _rename_in_section(
            runtime_section,
            "per_device_train_batch_size",
            "train_batch_size_per_device",
            section="runtime",
        )
        migrated["runtime"] = runtime_section
    elif "runtime" in migrated:
        raise MigrationError("v0 runtime must be a mapping")


def _rename_top_level(document: dict[str, Any], old: str, new: str) -> None:
    if old not in document:
        return
    if new in document:
        raise MigrationError(f"v0 configuration contains both {old!r} and {new!r}")
    document[new] = document.pop(old)


def _rename_in_section(section_value: dict[str, Any], old: str, new: str, *, section: str) -> None:
    if old not in section_value:
        return
    if new in section_value:
        raise MigrationError(f"v0 {section} contains both {old!r} and {new!r}")
    section_value[new] = section_value.pop(old)


def _mapping_section(document: dict[str, Any], name: str, *, create: bool) -> dict[str, Any]:
    if name not in document and create:
        section: dict[str, Any] = {}
        document[name] = section
        return section
    value = document.get(name)
    if not isinstance(value, Mapping):
        raise MigrationError(f"v0 {name} must be a mapping")
    section = dict(value)
    document[name] = section
    return section


def _move_top_level(document: dict[str, Any], target: dict[str, Any], old: str, new: str) -> None:
    if old not in document:
        return
    if new in target:
        raise MigrationError(f"v0 configuration contains both top-level {old!r} and project.{new}")
    target[new] = document.pop(old)
