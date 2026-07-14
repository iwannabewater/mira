"""Runtime compatibility checks for the audited MS-SWIFT tuple."""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from importlib import metadata

from mira.core.exceptions import ConfigurationValidationError

EXPECTED_MS_SWIFT_VERSION = "4.4.1"
MIN_TRANSFORMERS_VERSION = (4, 57, 0)
MAX_TRANSFORMERS_VERSION = (5, 13, 0)
MIN_QWEN_VL_UTILS_VERSION = (0, 0, 14)


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    """One optional dependency check with an actionable reason."""

    dependency: str
    installed_version: str | None
    compatible: bool
    requirement: str
    reason: str


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Complete result of validating the fixed upstream integration tuple."""

    checks: tuple[DependencyCheck, ...]

    @property
    def compatible(self) -> bool:
        """Return true only when every required check passed."""
        return all(check.compatible for check in self.checks)

    @property
    def failures(self) -> tuple[DependencyCheck, ...]:
        """Return failed checks without hiding optional-backend problems."""
        return tuple(check for check in self.checks if not check.compatible)


def check_compatibility() -> CompatibilityReport:
    """Inspect installed metadata without importing MS-SWIFT or Transformers."""
    checks = (
        _exact_version_check("ms-swift", EXPECTED_MS_SWIFT_VERSION),
        _range_version_check(
            "transformers",
            minimum=MIN_TRANSFORMERS_VERSION,
            maximum=MAX_TRANSFORMERS_VERSION,
        ),
        _minimum_version_check("qwen-vl-utils", MIN_QWEN_VL_UTILS_VERSION),
        _module_check("decord"),
        _module_check("swift"),
    )
    return CompatibilityReport(checks)


def require_compatible() -> CompatibilityReport:
    """Return a passing report or raise with every incompatibility listed."""
    report = check_compatibility()
    if report.compatible:
        return report
    details = "; ".join(
        f"{check.dependency}: {check.reason} (requires {check.requirement})"
        for check in report.failures
    )
    raise ConfigurationValidationError(f"MS-SWIFT compatibility check failed: {details}")


def _exact_version_check(distribution: str, expected: str) -> DependencyCheck:
    installed = _installed_version(distribution)
    compatible = installed == expected
    reason = "compatible" if compatible else f"installed version is {installed or 'missing'}"
    return DependencyCheck(distribution, installed, compatible, f"=={expected}", reason)


def _range_version_check(
    distribution: str,
    *,
    minimum: tuple[int, int, int],
    maximum: tuple[int, int, int],
) -> DependencyCheck:
    installed = _installed_version(distribution)
    parsed = _release_tuple(installed) if installed is not None else None
    compatible = parsed is not None and minimum <= parsed < maximum
    requirement = f">={_format_version(minimum)},<{_format_version(maximum)}"
    reason = "compatible" if compatible else f"installed version is {installed or 'missing'}"
    return DependencyCheck(distribution, installed, compatible, requirement, reason)


def _minimum_version_check(
    distribution: str,
    minimum: tuple[int, int, int],
) -> DependencyCheck:
    installed = _installed_version(distribution)
    parsed = _release_tuple(installed) if installed is not None else None
    compatible = parsed is not None and parsed >= minimum
    requirement = f">={_format_version(minimum)}"
    reason = "compatible" if compatible else f"installed version is {installed or 'missing'}"
    return DependencyCheck(distribution, installed, compatible, requirement, reason)


def _module_check(module: str) -> DependencyCheck:
    available = importlib.util.find_spec(module) is not None
    reason = "importable" if available else "module is not importable"
    return DependencyCheck(module, None, available, "importable module", reason)


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _release_tuple(version: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def _format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)
