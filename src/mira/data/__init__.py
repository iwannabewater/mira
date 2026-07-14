"""Canonical multimodal retrieval records and semantics-aware data utilities."""

from mira.data.converters import (
    MultiPositiveStrategy,
    SwiftJsonConverter,
    mira_to_swift,
    swift_to_mira,
)
from mira.data.layout import GroupLayoutBuilder, LayoutBuildResult, build_group_layout
from mira.data.readers import MiraJsonlReader, iter_jsonl, read_jsonl, write_jsonl
from mira.data.samplers import (
    DeterministicNoDuplicateBatchSampler,
    NoDuplicateBatchSampler,
    NoDuplicateSampler,
)
from mira.data.schema import (
    SCHEMA_VERSION,
    CandidateItem,
    EmbeddingRecord,
    MediaRef,
    Message,
    MessagePart,
    MultimodalItem,
    ObjectAnnotations,
    RetrievalMetadata,
)

__all__ = [
    "SCHEMA_VERSION",
    "CandidateItem",
    "DeterministicNoDuplicateBatchSampler",
    "EmbeddingRecord",
    "GroupLayoutBuilder",
    "LayoutBuildResult",
    "MediaRef",
    "Message",
    "MessagePart",
    "MiraJsonlReader",
    "MultiPositiveStrategy",
    "MultimodalItem",
    "NoDuplicateBatchSampler",
    "NoDuplicateSampler",
    "ObjectAnnotations",
    "RetrievalMetadata",
    "SwiftJsonConverter",
    "build_group_layout",
    "iter_jsonl",
    "mira_to_swift",
    "read_jsonl",
    "swift_to_mira",
    "write_jsonl",
]
