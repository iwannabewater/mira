"""Strict Transformers backend for audited Qwen3-VL-Embedding models."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from mira.core.contracts import EmbeddingOutput, RepresentationRequest, RepresentationType
from mira.core.exceptions import CapabilityError, ContractError
from mira.integrations.qwen3_vl_embedding.arguments import (
    Qwen3VLEmbeddingLoadSpec,
    resolve_device,
    resolve_dtype,
)
from mira.integrations.qwen3_vl_embedding.compatibility import require_compatible
from mira.integrations.qwen3_vl_embedding.preprocessing import (
    Qwen3VLEmbeddingPreprocessor,
)
from mira.models.adapters import Qwen3VLEmbeddingAdapter, Qwen3VLEmbeddingSemantics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mira.inference.encoder import InferenceRole, InferenceRuntimeDescriptor

_UNSAFE_MODEL_SUFFIXES = frozenset(
    {
        ".bin",
        ".dll",
        ".dylib",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".py",
        ".pyc",
        ".so",
    }
)


@dataclass(frozen=True, slots=True)
class Qwen3VLEmbeddingBackendAssets:
    """Loaded model, processor, preprocessor, and public inference backend."""

    adapter: Qwen3VLEmbeddingAdapter
    processor: Any
    preprocessor: Qwen3VLEmbeddingPreprocessor
    backend: TransformersQwen3VLEmbeddingBackend


class TransformersQwen3VLEmbeddingBackend:
    """Implement ``EncoderBackend`` with strict Qwen3-VL-Embedding semantics."""

    def __init__(
        self,
        adapter: Qwen3VLEmbeddingAdapter,
        preprocessor: Qwen3VLEmbeddingPreprocessor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Bind one loaded embedding model to one immutable preprocessing profile."""
        if not isinstance(adapter, Qwen3VLEmbeddingAdapter):
            raise ContractError("Qwen3-VL-Embedding backend requires Qwen3VLEmbeddingAdapter")
        self.adapter = adapter
        self.preprocessor = preprocessor
        self.device = device
        self.dtype = dtype

    @property
    def model_id(self) -> str:
        """Return the complete embedding-model identity."""
        return self.adapter.model_id

    @property
    def revision(self) -> str:
        """Return the immutable model revision."""
        return self.preprocessor.spec.revision

    @property
    def template_fingerprint(self) -> str:
        """Return the exact prompt and visual preprocessing fingerprint."""
        return self.preprocessor.fingerprint

    @property
    def runtime(self) -> InferenceRuntimeDescriptor:
        """Return backend-neutral provenance for CLI and artifact consumers."""
        from mira.inference.encoder import InferenceRuntimeDescriptor  # noqa: PLC0415

        return InferenceRuntimeDescriptor(
            backend="qwen3-vl-embedding-transformers",
            model_id=self.model_id,
            revision=self.revision,
            template_fingerprint=self.template_fingerprint,
            template_profile=self.preprocessor.spec.template_profile,
            device=str(self.device),
            dtype=str(self.dtype).removeprefix("torch."),
        )

    def encode_batch(
        self,
        items: Sequence[object],
        *,
        role: InferenceRole,
        instruction: str | None,
        request: RepresentationRequest,
    ) -> EmbeddingOutput:
        """Encode one ordered batch and return raw global embeddings."""
        if role not in {"query", "document"}:
            raise ContractError(f"unsupported Qwen3-VL-Embedding role {role!r}")
        if request.representation is not RepresentationType.GLOBAL:
            raise CapabilityError(
                "Qwen3-VL-Embedding Transformers inference currently supports only "
                "global embeddings"
            )
        selected_instruction = (
            self.preprocessor.spec.instruction if instruction is None else instruction
        )
        inputs = self.preprocessor.encode_batch(items, instruction=selected_instruction)
        model_inputs = {name: value.to(self.device) for name, value in inputs.items()}
        self.adapter.eval()
        with torch.inference_mode():
            output = self.adapter.forward_representations(
                model_inputs,
                RepresentationRequest(
                    representation=RepresentationType.GLOBAL,
                    dimension=None,
                    normalize=False,
                ),
            )
        return EmbeddingOutput(
            global_embeddings=output.global_embeddings,
            normalized=False,
            auxiliary={
                **dict(output.auxiliary),
                "backend": "qwen3-vl-embedding-transformers",
                "device": str(self.device),
                "dtype": str(self.dtype).removeprefix("torch."),
                "revision": self.revision,
                "role": role,
            },
        )


def load_qwen3_vl_embedding_transformers(
    spec: Qwen3VLEmbeddingLoadSpec,
    *,
    model_source: str | None = None,
) -> Qwen3VLEmbeddingBackendAssets:
    """Load one code-free Qwen3-VL-Embedding model through Transformers."""
    require_compatible()
    source, local = _resolve_model_source(spec.model_id, model_source)
    device = resolve_device(spec.device)
    dtype = resolve_dtype(spec.dtype, device)
    transformers = import_module("transformers")
    model_type = getattr(transformers, "Qwen3VLModel", None)
    processor_type = getattr(transformers, "AutoProcessor", None)
    if model_type is None or not callable(getattr(model_type, "from_pretrained", None)):
        raise ContractError(
            "Transformers does not expose the Qwen3VLModel backbone required by Qwen3-VL-Embedding"
        )
    if processor_type is None or not callable(getattr(processor_type, "from_pretrained", None)):
        raise ContractError("Transformers does not expose AutoProcessor for Qwen3-VL-Embedding")
    common: dict[str, Any] = {
        "local_files_only": local,
        "trust_remote_code": False,
    }
    if not local:
        common["revision"] = spec.revision
    model_options: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
        **common,
    }
    if spec.attention != "auto":
        model_options["attn_implementation"] = spec.attention
    model = model_type.from_pretrained(source, **model_options)
    processor = processor_type.from_pretrained(
        source,
        padding_side="right",
        **common,
    )
    if not isinstance(model, nn.Module):
        raise ContractError("Transformers returned no Qwen3-VL-Embedding torch model")
    if processor is None:
        raise ContractError("Transformers returned no Qwen3-VL-Embedding processor")
    model.eval()
    model.to(device)
    preprocessor = Qwen3VLEmbeddingPreprocessor(processor, spec)
    terminal_token_id = _resolve_terminal_token_id(processor)
    adapter = Qwen3VLEmbeddingAdapter(
        model,
        semantics=Qwen3VLEmbeddingSemantics(
            model_id=spec.model_id,
            terminal_token_id=terminal_token_id,
            template_fingerprint=preprocessor.fingerprint,
            instruction=spec.instruction,
            embedding_dimension=spec.embedding_dimension,
        ),
    )
    backend = TransformersQwen3VLEmbeddingBackend(
        adapter,
        preprocessor,
        device=device,
        dtype=dtype,
    )
    return Qwen3VLEmbeddingBackendAssets(
        adapter=adapter,
        processor=processor,
        preprocessor=preprocessor,
        backend=backend,
    )


def _resolve_model_source(model_id: str, model_source: str | None) -> tuple[str, bool]:
    if model_source is None:
        return model_id, False
    if type(model_source) is not str or not model_source or model_source != model_source.strip():
        raise ContractError("Qwen3-VL-Embedding model source must be canonical")
    path = Path(model_source)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ContractError(
            "local Qwen3-VL-Embedding source must be a non-symlink absolute directory"
        )
    for artifact in path.rglob("*"):
        if artifact.is_symlink():
            raise ContractError(
                "local Qwen3-VL-Embedding source must not contain symlinked artifacts"
            )
        if artifact.is_file() and artifact.suffix.lower() in _UNSAFE_MODEL_SUFFIXES:
            raise ContractError(
                f"local Qwen3-VL-Embedding source contains an unsafe artifact: {artifact.name}"
            )
    return str(path), True


def _resolve_terminal_token_id(processor: Any) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    value = convert("<|endoftext|>") if callable(convert) else None
    if type(value) is not int or value < 0:
        raise ContractError(
            "Qwen3-VL-Embedding tokenizer has no valid <|endoftext|> terminal token"
        )
    return value
