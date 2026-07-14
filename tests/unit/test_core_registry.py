from __future__ import annotations

import pytest

from mira.core import (
    Capability,
    CapabilityError,
    DuplicateRegistrationError,
    InvalidRegistrationError,
    ModelCapabilities,
    Registry,
    Stability,
    UnknownRegistrationError,
)


def test_registry_is_typed_deterministic_and_explicit() -> None:
    registry = Registry[int]("objective")
    assert registry.register("zeta", 2, stability=Stability.EXPERIMENTAL) == 2
    assert registry.register("alpha", 1) == 1

    assert registry.names == ("alpha", "zeta")
    assert list(registry) == ["alpha", "zeta"]
    assert registry.get("alpha") == 1
    assert registry.entry("zeta").stability is Stability.EXPERIMENTAL
    assert "alpha" in registry

    with pytest.raises(DuplicateRegistrationError, match="already registered"):
        registry.register("alpha", 9)
    with pytest.raises(UnknownRegistrationError, match="available: alpha, zeta"):
        registry.get("missing")


@pytest.mark.parametrize("name", ["", "Upper", "white space", "_leading"])
def test_registry_rejects_malformed_names(name: str) -> None:
    registry = Registry[object]("adapter")
    with pytest.raises(InvalidRegistrationError):
        registry.register(name, object())


def test_registry_never_claims_planned_work_is_available() -> None:
    registry = Registry[object]("algorithm")
    with pytest.raises(InvalidRegistrationError, match="documentation-only"):
        registry.register("future", object(), stability=Stability.PLANNED)


def test_capabilities_fail_before_runtime_use() -> None:
    capabilities = ModelCapabilities(supports_text=True, supports_global_embedding=True)
    capabilities.require(Capability.TEXT, Capability.GLOBAL_EMBEDDING)

    with pytest.raises(CapabilityError, match="supports_video"):
        capabilities.require(Capability.VIDEO, context="video task")


def test_capability_declarations_cannot_contradict_themselves() -> None:
    with pytest.raises(CapabilityError, match="supports_mrl"):
        ModelCapabilities(supports_mrl=True)
    with pytest.raises(CapabilityError, match="at least two"):
        ModelCapabilities(supports_text=True, supports_mixed_modal=True)
    with pytest.raises(CapabilityError, match="multi_embedding_tokens"):
        ModelCapabilities(supports_multi_embedding_tokens=True)
    with pytest.raises(CapabilityError, match="native_normalization"):
        ModelCapabilities(supports_native_normalization=True)


def test_registry_metadata_is_read_only_and_registry_name_is_validated() -> None:
    with pytest.raises(InvalidRegistrationError, match="registry name"):
        Registry[int](" ")

    registry = Registry[int]("value")
    registry.register("one", 1)
    assert registry.name == "value"
    assert len(registry) == 1
    with pytest.raises(TypeError):
        registry.entries["two"] = registry.entry("one")  # type: ignore[index]
