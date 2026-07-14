"""Strict, standards-compliant serialization helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def canonicalize_json_value(value: Any, *, context: str = "JSON value") -> Any:
    """Return a detached JSON-compatible copy with finite numeric values.

    Tuples are normalized to arrays. Cycles, non-string object keys, non-finite
    floats, and Python-only objects are rejected instead of being coerced.
    """
    active: set[int] = set()

    def visit(item: Any, path: str) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{context} contains non-finite number at {path}")
            return item
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                raise ValueError(f"{context} contains a cycle at {path}")
            active.add(identity)
            try:
                result: dict[str, Any] = {}
                for key, nested in item.items():
                    if not isinstance(key, str):
                        raise TypeError(f"{context} contains non-string object key at {path}")
                    result[key] = visit(nested, f"{path}.{key}")
                return result
            finally:
                active.remove(identity)
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError(f"{context} contains a cycle at {path}")
            active.add(identity)
            try:
                return [visit(nested, f"{path}[{index}]") for index, nested in enumerate(item)]
            finally:
                active.remove(identity)
        raise TypeError(f"{context} contains unsupported value at {path}: {type(item).__name__}")

    return visit(value, "$")


def freeze_json_value(value: Any, *, context: str = "JSON value") -> Any:
    """Return a recursively immutable, detached canonical JSON value."""

    def freeze(item: Any) -> Any:
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(nested) for key, nested in item.items()})
        if isinstance(item, list):
            return tuple(freeze(nested) for nested in item)
        return item

    return freeze(canonicalize_json_value(value, context=context))


def strict_json_loads(content: str, *, context: str = "JSON document") -> Any:
    """Decode JSON while rejecting duplicate keys and nonstandard numbers.

    Args:
        content: Complete JSON document text.
        context: Human-readable artifact name included in ambiguity errors.

    Raises:
        ValueError: If an object repeats a key or uses ``NaN``/``Infinity``.
        json.JSONDecodeError: If the document is otherwise malformed.
    """

    def reject_constant(value: str) -> None:
        raise ValueError(f"{context} contains nonstandard JSON constant {value!r} (non-finite)")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{context} contains non-finite JSON number {value!r}")
        return parsed

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate JSON object key {key!r}")
            result[key] = value
        return result

    return json.loads(
        content,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )
