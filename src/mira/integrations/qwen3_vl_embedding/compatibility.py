"""Optional-dependency audit boundary for Qwen3-VL-Embedding inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import metadata

from mira.core.exceptions import OptionalDependencyError

QWEN3_VL_EMBEDDING_EXTRA = "inference-qwen3-vl-embedding"

_REQUIREMENTS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "accelerate": ((1, 14, 0), (1, 15, 0)),
    "qwen-vl-utils": ((0, 0, 14), (0, 0, 15)),
    "safetensors": ((0, 8, 0), (0, 9, 0)),
    "torchvision": ((0, 28, 0), (0, 29, 0)),
    "transformers": ((5, 13, 1), (5, 14, 0)),
}
_PEFT_REQUIREMENT = ((0, 19, 1), (0, 20, 0))


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    """One optional dependency result with an exact supported range."""

    dependency: str
    installed_version: str | None
    compatible: bool
    requirement: str

    @property
    def reason(self) -> str:
        """Return a concise diagnostic compatible with CLI reporting."""
        if self.compatible:
            return "compatible"
        return f"installed version is {self.installed_version or 'missing'}"


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Complete dependency result for one requested loading path."""

    checks: tuple[DependencyCheck, ...]

    @property
    def compatible(self) -> bool:
        """Return true only when every dependency is in its audited range."""
        return all(check.compatible for check in self.checks)

    @property
    def failures(self) -> tuple[DependencyCheck, ...]:
        """Return every failed dependency check."""
        return tuple(check for check in self.checks if not check.compatible)


def check_compatibility(*, require_peft: bool = False) -> CompatibilityReport:
    """Inspect package metadata without importing model libraries."""
    requirements = dict(_REQUIREMENTS)
    if require_peft:
        requirements["peft"] = _PEFT_REQUIREMENT
    return CompatibilityReport(
        tuple(
            _check_range(distribution, minimum, maximum)
            for distribution, (minimum, maximum) in sorted(requirements.items())
        )
    )


def require_compatible(*, require_peft: bool = False) -> CompatibilityReport:
    """Fail with an actionable extra-install command when the tuple is incomplete."""
    report = check_compatibility(require_peft=require_peft)
    if report.compatible:
        return report
    details = "; ".join(
        f"{check.dependency}={check.installed_version or 'missing'} requires {check.requirement}"
        for check in report.failures
    )
    raise OptionalDependencyError(
        "Qwen3-VL-Embedding inference dependencies are incompatible: "
        f"{details}. Install mira-embed[{QWEN3_VL_EMBEDDING_EXTRA}]."
    )


def _check_range(
    distribution: str,
    minimum: tuple[int, int, int],
    maximum: tuple[int, int, int],
) -> DependencyCheck:
    try:
        installed = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        installed = None
    parsed = _release_tuple(installed) if installed is not None else None
    compatible = parsed is not None and minimum <= parsed < maximum
    requirement = f">={_format(minimum)},<{_format(maximum)}"
    return DependencyCheck(distribution, installed, compatible, requirement)


def _release_tuple(version: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def _format(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)
