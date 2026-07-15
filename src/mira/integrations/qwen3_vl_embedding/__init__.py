"""Direct Transformers runtime for audited Qwen3-VL-Embedding models."""

from mira.integrations.qwen3_vl_embedding.arguments import (
    Qwen3VLEmbeddingLoadSpec,
    Qwen3VLEmbeddingTemplateProfile,
)
from mira.integrations.qwen3_vl_embedding.backend import (
    Qwen3VLEmbeddingBackendAssets,
    TransformersQwen3VLEmbeddingBackend,
    load_qwen3_vl_embedding_transformers,
)
from mira.integrations.qwen3_vl_embedding.compatibility import (
    QWEN3_VL_EMBEDDING_EXTRA,
    check_compatibility,
    require_compatible,
)
from mira.integrations.qwen3_vl_embedding.preprocessing import (
    preprocessing_fingerprint,
)

__all__ = [
    "QWEN3_VL_EMBEDDING_EXTRA",
    "Qwen3VLEmbeddingBackendAssets",
    "Qwen3VLEmbeddingLoadSpec",
    "Qwen3VLEmbeddingTemplateProfile",
    "TransformersQwen3VLEmbeddingBackend",
    "check_compatibility",
    "load_qwen3_vl_embedding_transformers",
    "preprocessing_fingerprint",
    "require_compatible",
]
