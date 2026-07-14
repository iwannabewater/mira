"""Thin, version-pinned adapter for MS-SWIFT 4.4.1."""

from mira.integrations.ms_swift.arguments import SwiftLoadSpec, SwiftTemplateSpec
from mira.integrations.ms_swift.backend import (
    SwiftBackendAdapter,
    SwiftBackendAssets,
    load_qwen3_vl_embedding,
)
from mira.integrations.ms_swift.compatibility import (
    CompatibilityReport,
    check_compatibility,
    require_compatible,
)

__all__ = [
    "CompatibilityReport",
    "SwiftBackendAdapter",
    "SwiftBackendAssets",
    "SwiftLoadSpec",
    "SwiftTemplateSpec",
    "check_compatibility",
    "load_qwen3_vl_embedding",
    "require_compatible",
]
