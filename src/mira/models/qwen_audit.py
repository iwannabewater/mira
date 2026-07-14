"""Immutable identities for the Qwen models audited for MIRA 0.1."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True, slots=True)
class AuditedQwenTarget:
    """One exact model identity and its public representation contract."""

    revision: str
    embedding_dimension: int
    licenses: tuple[str, ...]


AUDITED_QWEN_TARGETS: Final = MappingProxyType(
    {
        "Qwen/Qwen3-VL-Embedding-2B": AuditedQwenTarget(
            revision="9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
            embedding_dimension=2048,
            licenses=("Apache-2.0",),
        ),
        "Qwen/Qwen3-VL-Embedding-8B": AuditedQwenTarget(
            revision="2c4565515e0f265c6511776e7193b22c0968ddc7",
            embedding_dimension=4096,
            licenses=("Apache-2.0",),
        ),
    }
)
