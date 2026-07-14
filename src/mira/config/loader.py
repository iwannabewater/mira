"""Strict YAML composition, migration, override, and resolution pipeline."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Final, cast

import yaml
from pydantic import ValidationError

from mira.config.interpolation import interpolate_environment, interpolate_string
from mira.config.migration import migrate_config
from mira.config.schema import MiraConfig
from mira.core.exceptions import (
    ConfigurationFileError,
    ConfigurationValidationError,
)
from mira.core.filesystem import open_bound_directory

_OVERRIDE_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_MAX_CONFIG_SOURCE_BYTES: Final = 1 << 20
_MAX_CONTAINER_DEPTH: Final = 64
_MAX_COMPOSITION_DEPTH: Final = 32
_MAX_COMPOSED_FILE_LOADS: Final = 128


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects aliases and duplicate mapping keys."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        """Reject aliases before PyYAML can construct recursive object graphs."""
        if self.check_event(yaml.AliasEvent):  # type: ignore[no-untyped-call]
            event = self.peek_event()  # type: ignore[no-untyped-call]
            line = event.start_mark.line + 1
            raise ConfigurationFileError(f"YAML aliases are not supported (alias at line {line})")
        return super().compose_node(parent, index)


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, *, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ConfigurationFileError(
                f"unhashable YAML mapping key at line {key_node.start_mark.line + 1}"
            ) from error
        if duplicate:
            raise ConfigurationFileError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Validated configuration plus reproducible serialized evidence."""

    config: MiraConfig
    yaml: str
    checksum: str
    source_files: tuple[Path, ...]

    def write(self, path: str | Path) -> None:
        """Atomically persist the exact checksummed resolved YAML."""
        _atomic_write(Path(path), self.yaml)


@dataclass(slots=True)
class _CompositionBudget:
    """Per-resolution counter for selected-root and composed-file loads."""

    loaded_files: int = 0

    def claim_file_load(self) -> None:
        if self.loaded_files >= _MAX_COMPOSED_FILE_LOADS:
            raise ConfigurationFileError(
                f"configuration composition exceeds the {_MAX_COMPOSED_FILE_LOADS}-file limit"
            )
        self.loaded_files += 1


@dataclass(frozen=True, slots=True)
class _ConfigSource:
    """Immutable bytes and identity captured from one securely opened source."""

    path: Path
    relative_path: Path
    encoded: bytes
    identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _CompositionContext:
    """Root descriptor, environment, and budget shared by one resolution."""

    environment: Mapping[str, str]
    root_directory: Path
    root_directory_fd: int
    budget: _CompositionBudget


def load_config(
    path: str | Path,
    *,
    overrides: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
) -> MiraConfig:
    """Load and validate one composed configuration file.

    Args:
        path: Root YAML file. Relative ``extends`` and ``compose`` paths resolve
            from the file that declares them. Every composed file must remain under
            the selected root file's parent directory, and symbolic-link components
            are rejected during descriptor-relative traversal.
            Each YAML source is limited to 1 MiB and 64 container levels. One
            resolution permits at most 32 files in a composition chain and 128
            total file loads, including the selected root and repeated references.
        overrides: Dotted ``path=value`` CLI overrides parsed as YAML scalars.
        environ: Explicit environment mapping; defaults to ``os.environ``.

    Raises:
        ConfigurationFileError: For missing files, YAML errors, cycles, duplicate
            keys, malformed directives, or invalid overrides.
        InterpolationError: For missing or malformed ``${NAME}`` references.
        MigrationError: For unsupported or ambiguous schema migrations.
        ConfigurationValidationError: For Pydantic or cross-field violations.
    """
    return resolve_config(path, overrides=overrides, environ=environ).config


def resolve_config(
    path: str | Path,
    *,
    overrides: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
) -> ResolvedConfig:
    """Resolve a root-confined composition and return its reproducible evidence."""
    root_candidate = _resolve_root_path(path)
    environment = os.environ if environ is None else environ
    budget = _CompositionBudget()
    root_directory_fd, root_directory = _open_root_directory(root_candidate.parent)
    root = root_directory / root_candidate.name
    try:
        context = _CompositionContext(
            environment=environment,
            root_directory=root_directory,
            root_directory_fd=root_directory_fd,
            budget=budget,
        )
        budget.claim_file_load()
        root_source = _read_config_source(
            root.name,
            root_directory=root_directory,
            root_directory_fd=root_directory_fd,
            path_label="selected root configuration",
        )
        document, sources = _load_composed(
            root_source,
            context=context,
            stack=(),
        )
    finally:
        os.close(root_directory_fd)
    interpolated = interpolate_environment(document, environ=environment)
    migrated = migrate_config(interpolated)
    overridden = apply_overrides(migrated, overrides)
    config = validate_config_mapping(overridden, source=root)
    rendered = dump_resolved_yaml(config)
    checksum = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return ResolvedConfig(
        config=config,
        yaml=rendered,
        checksum=checksum,
        source_files=tuple(dict.fromkeys(sources)),
    )


def validate_config_mapping(
    document: Mapping[str, Any], *, source: str | Path = "<mapping>"
) -> MiraConfig:
    """Validate an already resolved version-1 mapping with contextual errors."""
    try:
        return MiraConfig.model_validate(document)
    except ValidationError as error:
        raise ConfigurationValidationError(
            f"invalid MIRA configuration from {source}:\n{error}"
        ) from error


def apply_overrides(document: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply unique dotted CLI overrides to a deep copy of ``document``.

    Override values use safe YAML scalar/list/mapping syntax. Unknown fields are
    retained here and rejected later by the Pydantic ``extra=forbid`` boundary.

    Raises:
        ConfigurationFileError: If syntax, paths, duplicate paths, or intermediate
            values are invalid.
    """
    _validate_container_structure(document)
    result = _deep_copy_mapping(document)
    seen: set[tuple[str, ...]] = set()
    for override in overrides:
        if not isinstance(override, str):
            raise ConfigurationFileError("CLI overrides must be strings")
        if "=" not in override:
            raise ConfigurationFileError(
                f"CLI override {override!r} must use dotted.path=value syntax"
            )
        dotted, raw_value = override.split("=", 1)
        parts = tuple(dotted.split("."))
        if not parts or any(_OVERRIDE_PATH.fullmatch(part) is None for part in parts):
            raise ConfigurationFileError(f"invalid CLI override path {dotted!r}")
        if parts in seen:
            raise ConfigurationFileError(f"duplicate CLI override path {dotted!r}")
        if any(
            parts[: len(previous)] == previous or previous[: len(parts)] == parts
            for previous in seen
        ):
            raise ConfigurationFileError(f"conflicting CLI override path {dotted!r}")
        seen.add(parts)
        _validate_source_size(raw_value, context="CLI override value")
        try:
            _validate_yaml_structure(raw_value, context="CLI override value")
            value = yaml.load(
                raw_value,
                Loader=_UniqueKeyLoader,  # noqa: S506 - subclass of yaml.SafeLoader
            )
        except (yaml.YAMLError, ConfigurationFileError) as error:
            raise ConfigurationFileError(
                f"invalid YAML value for CLI override {dotted!r}: {error}"
            ) from error
        _set_dotted(result, parts, value)
        _validate_container_structure(result)
    return result


def dump_resolved_yaml(config: MiraConfig) -> str:
    """Serialize a validated configuration deterministically as clean YAML."""
    data = _yaml_safe_value(config.model_dump(mode="python", by_alias=True, exclude_none=False))
    rendered = yaml.safe_dump(
        data,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    return rendered if rendered.endswith("\n") else f"{rendered}\n"


def config_checksum(config: MiraConfig) -> str:
    """Return SHA-256 of the exact deterministic resolved YAML serialization."""
    return hashlib.sha256(dump_resolved_yaml(config).encode("utf-8")).hexdigest()


def write_resolved_config(config: MiraConfig, path: str | Path) -> str:
    """Write ``resolved_config.yaml`` content and return its SHA-256 checksum."""
    rendered = dump_resolved_yaml(config)
    _atomic_write(Path(path), rendered)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _load_composed(
    source: _ConfigSource,
    *,
    context: _CompositionContext,
    stack: tuple[tuple[int, int], ...],
) -> tuple[dict[str, Any], list[Path]]:
    if source.identity in stack:
        raise ConfigurationFileError("configuration composition cycle detected")
    if len(stack) >= _MAX_COMPOSITION_DEPTH:
        raise ConfigurationFileError(
            f"configuration composition exceeds the {_MAX_COMPOSITION_DEPTH}-file depth limit"
        )
    path_label = "selected root configuration" if not stack else "composed configuration"
    document = _read_yaml_mapping(source.encoded, path_label=path_label)
    extends = document.pop("extends", ())
    compose = document.pop("compose", ())
    merged: dict[str, Any] = {}
    sources: list[Path] = []
    next_stack = (*stack, source.identity)

    for reference in _directive_references(extends, directive="extends"):
        resolved_source = _resolve_reference(
            source,
            reference,
            context=context,
        )
        base, base_sources = _load_composed(
            resolved_source,
            context=context,
            stack=next_stack,
        )
        merged = deep_merge(merged, base)
        sources.extend(base_sources)

    if isinstance(compose, Mapping):
        for section, reference_value in compose.items():
            if not isinstance(section, str) or not section:
                raise ConfigurationFileError("compose section name must be a non-empty string")
            references = _directive_references(reference_value, directive=f"compose.{section}")
            for reference in references:
                resolved_source = _resolve_reference(
                    source,
                    reference,
                    context=context,
                )
                fragment, fragment_sources = _load_composed(
                    resolved_source,
                    context=context,
                    stack=next_stack,
                )
                if section in fragment:
                    merged = deep_merge(merged, {section: fragment[section]})
                else:
                    merged = deep_merge(merged, {section: fragment})
                sources.extend(fragment_sources)
    else:
        for reference in _directive_references(compose, directive="compose"):
            resolved_source = _resolve_reference(
                source,
                reference,
                context=context,
            )
            fragment, fragment_sources = _load_composed(
                resolved_source,
                context=context,
                stack=next_stack,
            )
            merged = deep_merge(merged, fragment)
            sources.extend(fragment_sources)

    merged = deep_merge(merged, document)
    sources.append(source.path)
    return merged, sources


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; non-mapping values and sequences replace."""
    _validate_container_structure(base)
    _validate_container_structure(override)
    result = _deep_copy_mapping(base)
    for key, value in override.items():
        if not isinstance(key, str):
            raise ConfigurationFileError("configuration mapping keys must be strings")
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(
                cast("Mapping[str, Any]", existing), cast("Mapping[str, Any]", value)
            )
        else:
            result[key] = _deep_copy_value(value)
    _validate_container_structure(result)
    return result


def _read_yaml_mapping(encoded: bytes, *, path_label: str) -> dict[str, Any]:
    if len(encoded) > _MAX_CONFIG_SOURCE_BYTES:
        raise ConfigurationFileError(
            f"{path_label} exceeds the {_MAX_CONFIG_SOURCE_BYTES}-byte limit"
        )
    try:
        text = encoded.decode("utf-8")
    except UnicodeError as error:
        raise ConfigurationFileError(f"cannot read {path_label}: invalid UTF-8") from error
    _validate_yaml_structure(text, context=path_label)
    try:
        decoded = yaml.load(
            text,
            Loader=_UniqueKeyLoader,  # noqa: S506 - subclass of yaml.SafeLoader
        )
    except ConfigurationFileError:
        raise
    except RecursionError as error:
        raise ConfigurationFileError(
            f"{path_label} exceeds the {_MAX_CONTAINER_DEPTH}-level nesting limit"
        ) from error
    except yaml.YAMLError as error:
        raise ConfigurationFileError(f"invalid YAML in {path_label}: {error}") from error
    if not isinstance(decoded, Mapping):
        raise ConfigurationFileError(f"{path_label} must contain a YAML mapping")
    if any(not isinstance(key, str) for key in decoded):
        raise ConfigurationFileError(f"{path_label} root keys must be strings")
    _validate_container_structure(decoded)
    return _deep_copy_mapping(cast("Mapping[str, Any]", decoded))


def _validate_source_size(source: str, *, context: str) -> None:
    if len(source) > _MAX_CONFIG_SOURCE_BYTES:
        raise ConfigurationFileError(f"{context} exceeds the {_MAX_CONFIG_SOURCE_BYTES}-byte limit")
    try:
        encoded_size = len(source.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ConfigurationFileError(f"{context} must be valid UTF-8") from error
    if encoded_size > _MAX_CONFIG_SOURCE_BYTES:
        raise ConfigurationFileError(f"{context} exceeds the {_MAX_CONFIG_SOURCE_BYTES}-byte limit")


def _validate_yaml_structure(source: str, *, context: str) -> None:
    depth = 0
    try:
        events = yaml.parse(source, Loader=yaml.SafeLoader)
        for event in events:
            if isinstance(event, yaml.AliasEvent):
                line = event.start_mark.line + 1
                _raise_yaml_alias_error(line)
            if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
                depth += 1
                if depth > _MAX_CONTAINER_DEPTH:
                    _raise_yaml_depth_error(context)
            elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
                depth -= 1
    except ConfigurationFileError:
        raise
    except RecursionError as error:
        raise ConfigurationFileError(
            f"{context} exceeds the {_MAX_CONTAINER_DEPTH}-level nesting limit"
        ) from error
    except yaml.YAMLError as error:
        raise ConfigurationFileError(f"invalid YAML in {context}: {error}") from error


def _raise_yaml_alias_error(line: int) -> None:
    raise ConfigurationFileError(f"YAML aliases are not supported (alias at line {line})")


def _raise_yaml_depth_error(context: str) -> None:
    raise ConfigurationFileError(
        f"{context} exceeds the {_MAX_CONTAINER_DEPTH}-level nesting limit"
    )


def _validate_container_structure(value: Any) -> None:
    stack: list[tuple[Any, int, bool]] = [(value, 1, False)]
    ancestors: set[int] = set()
    while stack:
        current, depth, leaving = stack.pop()
        if not isinstance(current, (Mapping, list, tuple)):
            continue
        identity = id(current)
        if leaving:
            ancestors.remove(identity)
            continue
        if depth > _MAX_CONTAINER_DEPTH:
            raise ConfigurationFileError(
                f"configuration containers exceed the {_MAX_CONTAINER_DEPTH}-level depth limit"
            )
        if identity in ancestors:
            raise ConfigurationFileError("configuration containers must not be cyclic")
        ancestors.add(identity)
        stack.append((current, depth, True))
        children = current.values() if isinstance(current, Mapping) else current
        stack.extend((child, depth + 1, False) for child in reversed(tuple(children)))


def _directive_references(value: Any, *, directive: str) -> tuple[str, ...]:
    if value is None or value in ((), []):
        return ()
    if isinstance(value, str) and value:
        return (value,)
    if isinstance(value, list | tuple) and all(isinstance(item, str) and item for item in value):
        return tuple(cast("list[str] | tuple[str, ...]", value))
    raise ConfigurationFileError(f"{directive} must be a path string or sequence of path strings")


def _open_root_directory(root_directory: Path) -> tuple[int, Path]:
    """Open and identity-bind the directory that confines every config source."""
    if os.stat not in os.supports_dir_fd or os.stat not in os.supports_follow_symlinks:
        raise ConfigurationFileError(
            "secure descriptor-relative configuration traversal is unavailable"
        )
    try:
        return open_bound_directory(root_directory)
    except OSError as error:
        detail = error.strerror or "I/O error"
        raise ConfigurationFileError(
            f"cannot read selected root configuration: {detail}"
        ) from error


def _read_config_source(
    relative_path: str | Path,
    *,
    root_directory: Path,
    root_directory_fd: int,
    path_label: str,
) -> _ConfigSource:
    """Read one regular file through a symlink-free walk beneath the root fd."""
    relative = Path(relative_path)
    parts = relative.parts
    if relative.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ConfigurationFileError("invalid configuration reference path")
    try:
        parent_fd = os.dup(root_directory_fd)
    except OSError as error:
        detail = error.strerror or "I/O error"
        raise ConfigurationFileError(f"cannot read {path_label}: {detail}") from error
    try:
        for component in parts[:-1]:
            next_fd, _ = _open_bound_component(
                parent_fd,
                component,
                require_directory=True,
                path_label=path_label,
            )
            os.close(parent_fd)
            parent_fd = next_fd
        source_fd, opened = _open_bound_component(
            parent_fd,
            parts[-1],
            require_directory=False,
            path_label=path_label,
        )
    finally:
        os.close(parent_fd)
    try:
        encoded = _read_bounded_descriptor(source_fd, opened=opened, path_label=path_label)
    finally:
        os.close(source_fd)
    normalized = Path(*parts)
    return _ConfigSource(
        path=root_directory / normalized,
        relative_path=normalized,
        encoded=encoded,
        identity=(opened.st_dev, opened.st_ino),
    )


def _open_bound_component(
    parent_fd: int,
    component: str,
    *,
    require_directory: bool,
    path_label: str,
) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    if require_directory:
        flags |= os.O_DIRECTORY
    else:
        flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(component, flags, dir_fd=parent_fd)
    except OSError as error:
        if _is_symlink_component(parent_fd, component):
            raise ConfigurationFileError(
                "configuration reference traverses a symbolic link in the selected root "
                "configuration directory"
            ) from error
        detail = error.strerror or "I/O error"
        raise ConfigurationFileError(f"cannot read {path_label}: {detail}") from error
    try:
        opened = os.fstat(descriptor)
        named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        os.close(descriptor)
        detail = error.strerror or "I/O error"
        raise ConfigurationFileError(f"cannot read {path_label}: {detail}") from error
    expected_type = stat.S_ISDIR if require_directory else stat.S_ISREG
    if not expected_type(opened.st_mode) or not expected_type(named.st_mode):
        os.close(descriptor)
        kind = "directory" if require_directory else "regular file"
        raise ConfigurationFileError(f"cannot read {path_label}: source is not a {kind}")
    if not _same_file_identity(opened, named):
        os.close(descriptor)
        raise ConfigurationFileError(f"{path_label} changed while being opened")
    return descriptor, opened


def _is_symlink_component(parent_fd: int, component: str) -> bool:
    try:
        return stat.S_ISLNK(os.stat(component, dir_fd=parent_fd, follow_symlinks=False).st_mode)
    except OSError:
        return False


def _same_file_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _read_bounded_descriptor(descriptor: int, *, opened: os.stat_result, path_label: str) -> bytes:
    remaining = _MAX_CONFIG_SOURCE_BYTES + 1
    chunks: list[bytes] = []
    try:
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        final = os.fstat(descriptor)
    except OSError as error:
        detail = error.strerror or "I/O error"
        raise ConfigurationFileError(f"cannot read {path_label}: {detail}") from error
    encoded = b"".join(chunks)
    if len(encoded) > _MAX_CONFIG_SOURCE_BYTES:
        raise ConfigurationFileError(
            f"{path_label} exceeds the {_MAX_CONFIG_SOURCE_BYTES}-byte limit"
        )
    if (
        not _same_file_identity(opened, final)
        or opened.st_size != final.st_size
        or opened.st_mtime_ns != final.st_mtime_ns
        or opened.st_ctime_ns != final.st_ctime_ns
    ):
        raise ConfigurationFileError(f"{path_label} changed while being read")
    return encoded


def _resolve_reference(
    declaring_source: _ConfigSource,
    reference: str,
    *,
    context: _CompositionContext,
) -> _ConfigSource:
    resolved_reference = interpolate_string(
        reference, environ=context.environment, path="configuration directive"
    )
    if "\x00" in resolved_reference:
        raise ConfigurationFileError("invalid configuration reference path")
    try:
        candidate = Path(resolved_reference).expanduser()
        if not candidate.is_absolute():
            candidate = context.root_directory / declaring_source.relative_path.parent / candidate
        normalized = Path(os.path.normpath(candidate))
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigurationFileError("invalid configuration reference path") from error
    try:
        relative = normalized.relative_to(context.root_directory)
    except ValueError as error:
        raise ConfigurationFileError(
            "configuration reference escapes the selected root configuration directory"
        ) from error
    context.budget.claim_file_load()
    return _read_config_source(
        relative,
        root_directory=context.root_directory,
        root_directory_fd=context.root_directory_fd,
        path_label="composed configuration",
    )


def _set_dotted(document: dict[str, Any], parts: tuple[str, ...], value: Any) -> None:
    current = document
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            child: dict[str, Any] = {}
            current[part] = child
            current = child
        elif isinstance(existing, Mapping):
            child = _deep_copy_mapping(cast("Mapping[str, Any]", existing))
            current[part] = child
            current = child
        else:
            dotted = ".".join(parts)
            raise ConfigurationFileError(
                f"cannot apply override {dotted!r}: {part!r} is not a mapping"
            )
    current[parts[-1]] = _deep_copy_value(value)


def _deep_copy_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    original_keys: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str | int) or isinstance(key, bool):
            raise ConfigurationFileError(
                "configuration mapping keys must be strings or integer dimensions"
            )
        normalized = str(key)
        if normalized in result:
            previous = original_keys[normalized]
            raise ConfigurationFileError(
                f"configuration mapping keys {previous!r} and {key!r} normalize identically"
            )
        result[normalized] = _deep_copy_value(item)
        original_keys[normalized] = key
    return result


def _deep_copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _deep_copy_mapping(cast("Mapping[str, Any]", value))
    if isinstance(value, list):
        return [_deep_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_copy_value(item) for item in value)
    return value


def _yaml_safe_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _yaml_safe_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_yaml_safe_value(item) for item in value]
    return value


def _atomic_write(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except OSError as error:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
        raise ConfigurationFileError(
            f"cannot write resolved configuration {path}: {error}"
        ) from error


def _resolve_root_path(path: str | Path) -> Path:
    try:
        candidate = Path(path).expanduser()
        _reject_null_path(candidate)
        return candidate.absolute()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ConfigurationFileError("invalid root configuration path") from error


def _reject_null_path(path: Path) -> None:
    if "\x00" in os.fspath(path):
        raise ValueError("embedded null path component")
