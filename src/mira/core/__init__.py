"""Dependency-light public contracts for MIRA."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from mira.core.capabilities import Capability, ModelCapabilities, Stability
from mira.core.contracts import (
    BatchMetadata,
    EmbeddingBatch,
    EmbeddingOutput,
    ObjectiveContext,
    ObjectiveResult,
    RepresentationRequest,
    RepresentationType,
)
from mira.core.exceptions import (
    CapabilityError,
    CheckpointError,
    ChecksumMismatchError,
    ConfigurationError,
    ConfigurationFileError,
    ConfigurationValidationError,
    ContractError,
    DataValidationError,
    DistributedError,
    DuplicateRegistrationError,
    ExportError,
    InterpolationError,
    InvalidEmbeddingError,
    InvalidRegistrationError,
    LayoutValidationError,
    ManifestError,
    MediaValidationError,
    MigrationError,
    MiraError,
    NumericalError,
    ObjectiveError,
    OptionalDependencyError,
    RegistryError,
    UnknownRegistrationError,
)
from mira.core.layout import CrossDeviceLayoutMetadata, GroupLayout
from mira.core.manifests import (
    MANIFEST_SCHEMA_VERSION,
    ExportManifest,
    ManifestEnvelope,
    ManifestKind,
    VerificationStatus,
    canonical_json_bytes,
    file_sha256,
    sha256_digest,
)
from mira.core.registry import Registry, RegistryEntry

_LAZY_CONTRACTS = frozenset(
    {
        "CandidateItem",
        "EmbeddingRecord",
        "MediaRef",
        "Message",
        "MultimodalItem",
        "ObjectAnnotations",
        "RetrievalMetadata",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve record-level compatibility aliases from ``mira.core.contracts``."""
    if name not in _LAZY_CONTRACTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module("mira.core.contracts"), name)


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "BatchMetadata",
    "CandidateItem",
    "Capability",
    "CapabilityError",
    "CheckpointError",
    "ChecksumMismatchError",
    "ConfigurationError",
    "ConfigurationFileError",
    "ConfigurationValidationError",
    "ContractError",
    "CrossDeviceLayoutMetadata",
    "DataValidationError",
    "DistributedError",
    "DuplicateRegistrationError",
    "EmbeddingBatch",
    "EmbeddingOutput",
    "EmbeddingRecord",
    "ExportError",
    "ExportManifest",
    "GroupLayout",
    "InterpolationError",
    "InvalidEmbeddingError",
    "InvalidRegistrationError",
    "LayoutValidationError",
    "ManifestEnvelope",
    "ManifestError",
    "ManifestKind",
    "MediaRef",
    "MediaValidationError",
    "Message",
    "MigrationError",
    "MiraError",
    "ModelCapabilities",
    "MultimodalItem",
    "NumericalError",
    "ObjectAnnotations",
    "ObjectiveContext",
    "ObjectiveError",
    "ObjectiveResult",
    "OptionalDependencyError",
    "Registry",
    "RegistryEntry",
    "RegistryError",
    "RepresentationRequest",
    "RepresentationType",
    "RetrievalMetadata",
    "Stability",
    "UnknownRegistrationError",
    "VerificationStatus",
    "canonical_json_bytes",
    "file_sha256",
    "sha256_digest",
]
