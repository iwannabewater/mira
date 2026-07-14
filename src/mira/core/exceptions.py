"""Explicit exception hierarchy for MIRA's public failure modes."""

from __future__ import annotations


class MiraError(Exception):
    """Base class for every error intentionally raised by MIRA."""


class ConfigurationError(MiraError, ValueError):
    """Base class for invalid or unreadable user configuration."""


class ConfigurationFileError(ConfigurationError):
    """Raised when a configuration source cannot be read or decoded."""


class InterpolationError(ConfigurationError):
    """Raised when explicit environment interpolation cannot be resolved."""


class MigrationError(ConfigurationError):
    """Raised when a versioned configuration cannot be migrated safely."""


class ConfigurationValidationError(ConfigurationError):
    """Raised when a resolved configuration violates schema or cross-field rules."""


class CapabilityError(ConfigurationValidationError):
    """Raised when configuration requests a capability the model does not expose."""


class ContractError(MiraError, ValueError):
    """Base class for an invalid core-domain value or tensor contract."""


class LayoutValidationError(ContractError):
    """Raised when a :class:`GroupLayout` violates shape or retrieval semantics."""


class DataValidationError(ContractError):
    """Raised when a retrieval record violates a data invariant."""


class MediaValidationError(DataValidationError):
    """Raised when media references and message placeholders do not align."""


class InvalidEmbeddingError(ContractError):
    """Raised when an embedding has an invalid shape, value, mask, or norm."""


class NumericalError(MiraError, ArithmeticError):
    """Raised when a finite, non-degenerate numerical result cannot be produced."""


class ObjectiveError(MiraError, ValueError):
    """Raised when an objective cannot produce a mathematically valid loss."""


class RegistryError(MiraError, LookupError):
    """Base class for named-registry failures."""


class DuplicateRegistrationError(RegistryError):
    """Raised when a registry name is registered more than once."""


class UnknownRegistrationError(RegistryError):
    """Raised when a registry name cannot be resolved."""


class InvalidRegistrationError(RegistryError, ValueError):
    """Raised when a name or lifecycle state is invalid for registration."""


class ManifestError(MiraError, ValueError):
    """Base class for malformed or unverifiable manifests."""


class ChecksumMismatchError(ManifestError):
    """Raised when persisted content does not match its recorded SHA-256 digest."""


class DistributedError(MiraError, RuntimeError):
    """Raised when distributed state or collective semantics are inconsistent."""


class CheckpointError(MiraError, RuntimeError):
    """Raised when a checkpoint cannot satisfy the requested resume contract."""


class ExportError(MiraError, RuntimeError):
    """Raised when model artifacts cannot be exported without losing semantics."""


class OptionalDependencyError(MiraError, ImportError):
    """Raised when an explicitly requested feature lacks an optional dependency."""
