"""Explicit, recursive ``${NAME}`` environment interpolation."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

from mira.core.exceptions import InterpolationError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def interpolate_environment(
    document: Mapping[str, Any], *, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Return a deep copy with explicit environment references resolved.

    Only ``${NAME}`` syntax is recognized. Bare ``$NAME`` strings remain literal,
    and a missing variable fails with a location-aware error without printing
    any environment values.

    Raises:
        InterpolationError: If a referenced variable is missing or syntax begins
            with ``${`` but is not a valid reference.
    """
    environment = os.environ if environ is None else environ
    result = _interpolate_value(dict(document), environment=environment, path="$")
    if not isinstance(result, dict):
        raise InterpolationError("configuration root must remain a mapping")
    return result


def interpolate_string(
    value: str, *, environ: Mapping[str, str] | None = None, path: str = "$"
) -> str:
    """Resolve explicit environment references inside one string."""
    environment = os.environ if environ is None else environ

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in environment:
            raise InterpolationError(f"missing environment variable {name!r} at {path}")
        return environment[name]

    resolved = _ENV_PATTERN.sub(replace, value)
    if "${" in resolved:
        raise InterpolationError(f"invalid environment interpolation syntax at {path}")
    return resolved


def _interpolate_value(value: Any, *, environment: Mapping[str, str], path: str) -> Any:
    if isinstance(value, str):
        return interpolate_string(value, environ=environment, path=path)
    if isinstance(value, list):
        return [
            _interpolate_value(item, environment=environment, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _interpolate_value(item, environment=environment, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise InterpolationError(f"configuration key at {path} must be a string")
            result[key] = _interpolate_value(item, environment=environment, path=f"{path}.{key}")
        return result
    return value
