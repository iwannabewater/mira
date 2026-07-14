"""Typed, deterministic registry with explicit lifecycle semantics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, TypeVar

from mira.core.capabilities import Stability
from mira.core.exceptions import (
    DuplicateRegistrationError,
    InvalidRegistrationError,
    UnknownRegistrationError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

T = TypeVar("T")
_VALID_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class RegistryEntry(Generic[T]):
    """One immutable registry entry and its declared lifecycle state."""

    name: str
    value: T
    stability: Stability


class Registry(Generic[T]):
    """Map stable names to implemented objects without silent overwrites.

    Planned algorithms cannot be registered because a registry entry is a
    runtime availability claim. Duplicate and unknown lookups have distinct,
    actionable exception types.
    """

    def __init__(self, name: str) -> None:
        """Create an empty named registry.

        Raises:
            InvalidRegistrationError: If ``name`` is empty.
        """
        if not name.strip():
            raise InvalidRegistrationError("registry name must be non-empty")
        self._name = name.strip()
        self._entries: dict[str, RegistryEntry[T]] = {}

    @property
    def name(self) -> str:
        """Return the human-readable registry name used in errors."""
        return self._name

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic lexical order."""
        return tuple(sorted(self._entries))

    @property
    def entries(self) -> MappingProxyType[str, RegistryEntry[T]]:
        """Return an immutable snapshot view of registered entries."""
        return MappingProxyType(dict(self._entries))

    def register(self, name: str, value: T, *, stability: Stability = Stability.STABLE) -> T:
        """Register and return an implemented value.

        Raises:
            InvalidRegistrationError: If the name is malformed or state is planned.
            DuplicateRegistrationError: If ``name`` already exists.
        """
        self._validate_entry_name(name)
        if stability is Stability.PLANNED:
            raise InvalidRegistrationError(
                f"cannot register planned {self._name} {name!r}; planned work is documentation-only"
            )
        if name in self._entries:
            existing = self._entries[name]
            raise DuplicateRegistrationError(
                f"{self._name} {name!r} is already registered as {existing.stability.value}"
            )
        self._entries[name] = RegistryEntry(name=name, value=value, stability=stability)
        return value

    def get(self, name: str) -> T:
        """Resolve a registered value or raise an explicit unknown-name error."""
        try:
            return self._entries[name].value
        except KeyError as error:
            available = ", ".join(self.names) or "<none>"
            raise UnknownRegistrationError(
                f"unknown {self._name} {name!r}; available: {available}"
            ) from error

    def entry(self, name: str) -> RegistryEntry[T]:
        """Resolve the immutable entry, including its lifecycle state."""
        self.get(name)
        return self._entries[name]

    def __contains__(self, name: object) -> bool:
        """Return whether an exact name is registered."""
        return isinstance(name, str) and name in self._entries

    def __len__(self) -> int:
        """Return the number of registered entries."""
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        """Iterate registered names in deterministic lexical order."""
        return iter(self.names)

    @staticmethod
    def _validate_entry_name(name: str) -> None:
        if _VALID_NAME.fullmatch(name) is None:
            raise InvalidRegistrationError(
                "registry names must start with a lowercase letter and contain only "
                "lowercase letters, digits, '.', '_', or '-'"
            )
