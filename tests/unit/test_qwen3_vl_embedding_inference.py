"""Production Qwen3-VL-Embedding inference boundary tests without model weights."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import pytest
import torch
from torch import nn

from mira.core import CapabilityError, ContractError, OptionalDependencyError
from mira.core.contracts import RepresentationRequest, RepresentationType
from mira.data import MediaRef, Message, MessagePart, MultimodalItem
from mira.inference import Encoder
from mira.integrations.qwen3_vl_embedding import (
    Qwen3VLEmbeddingLoadSpec,
    check_compatibility,
    compatibility,
    load_qwen3_vl_embedding_transformers,
    preprocessing,
    require_compatible,
)
from mira.integrations.qwen3_vl_embedding import backend as backend_module
from mira.integrations.qwen3_vl_embedding.arguments import resolve_device, resolve_dtype
from mira.integrations.qwen3_vl_embedding.preprocessing import (
    Qwen3VLEmbeddingPreprocessor,
    preprocessing_fingerprint,
)
from mira.models.adapters import Qwen3VLEmbeddingAdapter, Qwen3VLEmbeddingSemantics

_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
_DIMENSION = 2048


def _spec(**overrides: Any) -> Qwen3VLEmbeddingLoadSpec:
    values = {
        "model_id": _MODEL_ID,
        "revision": _REVISION,
        "embedding_dimension": _DIMENSION,
        "device": "cpu",
        "dtype": "float32",
    }
    values.update(overrides)
    return Qwen3VLEmbeddingLoadSpec(**values)


class _FakeProcessor:
    def __init__(self, *, sequence_length: int = 2) -> None:
        self.sequence_length = sequence_length
        self.tokenizer = SimpleNamespace(
            eos_token_id=2,
            convert_tokens_to_ids=lambda token: (
                2 if token == "<|endoftext|>" else -1  # noqa: S105 - terminal token
            ),
        )
        self.conversations: Any = None
        self.processor_kwargs: dict[str, Any] = {}

    def apply_chat_template(self, conversations: Any, **kwargs: Any) -> list[str]:
        self.conversations = conversations
        suffix = "generation" if kwargs["add_generation_prompt"] else "closed"
        return [f"row-{index}-{suffix}" for index in range(len(conversations))]

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.processor_kwargs = kwargs
        rows = len(kwargs["text"])
        return {
            "input_ids": torch.full((rows, self.sequence_length), 2, dtype=torch.int64),
            "attention_mask": torch.ones((rows, self.sequence_length), dtype=torch.int64),
        }


class _TinyQwen3VLEmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))

    def forward(self, input_ids: torch.Tensor, **_: Any) -> Any:
        rows, length = input_ids.shape
        hidden = torch.arange(
            rows * length * _DIMENSION,
            dtype=torch.float32,
            device=input_ids.device,
        ).reshape(rows, length, _DIMENSION)
        return SimpleNamespace(hidden_states=(hidden,))


def _install_fake_vision(monkeypatch: pytest.MonkeyPatch, *, failure: bool = False) -> None:
    def process(conversations: Any, **_: Any) -> tuple[Any, Any, dict[str, Any]]:
        if failure:
            raise RuntimeError("broken media")
        for conversation in conversations:
            for message in conversation:
                for part in message["content"]:
                    uri = part.get("image") or part.get("video")
                    if uri is not None:
                        assert Path(urlparse(uri).path).is_file()
        return None, None, {"do_sample_frames": False}

    monkeypatch.setattr(
        preprocessing,
        "import_module",
        lambda _name: SimpleNamespace(process_vision_info=process),
    )


def _item_with_image(path: Path) -> MultimodalItem:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return MultimodalItem(
        item_id="image-item",
        messages=(Message(role="user", content="before <image> after"),),
        images=(MediaRef(uri=str(path.resolve()), checksum=f"sha256:{digest}"),),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_id": "Qwen/Qwen3-VL"}, "audited model IDs"),
        ({"revision": "0" * 40}, "immutable revision"),
        ({"embedding_dimension": 1024}, "requires embedding_dimension"),
        ({"device": " cpu"}, "device must be canonical"),
        ({"dtype": "fp8"}, "unsupported.*dtype"),
        ({"attention": "unknown"}, "attention implementation"),
        ({"template_profile": "generic-qwen3-vl"}, "template profile"),
        ({"max_length": 0}, "max_length"),
        ({"fps": 0.0}, "fps"),
        ({"instruction": " "}, "instruction"),
    ],
)
def test_load_spec_rejects_generic_or_ambiguous_qwen_values(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _spec(**overrides)


def test_compatibility_reports_complete_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "accelerate": "1.14.0",
        "peft": "0.19.1",
        "qwen-vl-utils": "0.0.14",
        "safetensors": "0.8.0",
        "torchvision": "0.28.0",
        "transformers": "5.13.1",
    }
    monkeypatch.setattr(compatibility.metadata, "version", versions.__getitem__)

    report = check_compatibility(require_peft=True)

    assert report.compatible
    assert tuple(check.dependency for check in report.checks) == tuple(sorted(versions))


def test_compatibility_failure_names_install_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> str:
        raise compatibility.metadata.PackageNotFoundError

    monkeypatch.setattr(compatibility.metadata, "version", missing)

    with pytest.raises(
        OptionalDependencyError,
        match=r"mira-embed\[inference-qwen3-vl-embedding\]",
    ):
        require_compatible(require_peft=True)


def test_preprocessor_preserves_embedding_identity_and_media_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"not-decoded-by-fake")
    processor = _FakeProcessor()
    _install_fake_vision(monkeypatch)
    spec = _spec()

    output = Qwen3VLEmbeddingPreprocessor(processor, spec).encode_batch(
        ["plain text", _item_with_image(image)],
        instruction=spec.instruction,
    )

    assert output["input_ids"].shape == (2, 2)
    assert processor.conversations[0][0]["content"][0]["text"] == spec.instruction
    image_parts = processor.conversations[1][1]["content"]
    assert [part["type"] for part in image_parts] == ["text", "image", "text"]
    assert processor.processor_kwargs["truncation"] is False
    assert processor.processor_kwargs["do_resize"] is False


def test_preprocessor_never_substitutes_failed_vision_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"broken")
    _install_fake_vision(monkeypatch, failure=True)

    with pytest.raises(ContractError, match="vision preprocessing failed"):
        Qwen3VLEmbeddingPreprocessor(_FakeProcessor(), _spec()).encode_batch(
            [_item_with_image(image)],
            instruction="Represent the user's input.",
        )


def test_preprocessor_rejects_silent_truncation_and_ambiguous_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vision(monkeypatch)
    with pytest.raises(ContractError, match="does not silently truncate"):
        Qwen3VLEmbeddingPreprocessor(
            _FakeProcessor(sequence_length=3),
            _spec(max_length=2),
        ).encode_batch(["too long"], instruction="Represent the user's input.")

    item = MultimodalItem(
        item_id="structured",
        messages=(
            Message(
                role="user",
                content=(MessagePart(type="custom", text="opaque"),),
            ),
        ),
    )
    with pytest.raises(ContractError, match="structured message parts must be text"):
        Qwen3VLEmbeddingPreprocessor(_FakeProcessor(), _spec()).encode_batch(
            [item], instruction="Represent the user's input."
        )


def test_mira_v1_and_official_profiles_have_distinct_fingerprints_and_suffixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vision(monkeypatch)
    official_spec = _spec(template_profile="official")
    mira_spec = _spec(template_profile="mira_v1")
    official_processor = _FakeProcessor()
    mira_processor = _FakeProcessor()

    Qwen3VLEmbeddingPreprocessor(official_processor, official_spec).encode_batch(
        ["one"], instruction=official_spec.instruction
    )
    Qwen3VLEmbeddingPreprocessor(mira_processor, mira_spec).encode_batch(
        ["one"], instruction=mira_spec.instruction
    )

    assert preprocessing_fingerprint(official_spec) != preprocessing_fingerprint(mira_spec)
    assert official_processor.processor_kwargs["text"] == ["row-0-generation"]
    assert mira_processor.processor_kwargs["text"] == ["row-0-closed"]


def test_transformers_loader_uses_code_free_immutable_model_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict[str, Any]] = {}
    processor = _FakeProcessor()

    class FakeModelType:
        @staticmethod
        def from_pretrained(_source: str, **kwargs: Any) -> nn.Module:
            captured["model"] = kwargs
            return _TinyQwen3VLEmbeddingModel()

    class FakeProcessorType:
        @staticmethod
        def from_pretrained(_source: str, **kwargs: Any) -> Any:
            captured["processor"] = kwargs
            return processor

    monkeypatch.setattr(backend_module, "require_compatible", lambda: None)
    monkeypatch.setattr(
        backend_module,
        "import_module",
        lambda _name: SimpleNamespace(
            Qwen3VLModel=FakeModelType,
            AutoProcessor=FakeProcessorType,
        ),
    )

    assets = load_qwen3_vl_embedding_transformers(_spec())

    assert assets.backend.model_id == _MODEL_ID
    assert captured["model"]["revision"] == _REVISION
    assert captured["model"]["trust_remote_code"] is False
    assert captured["model"]["use_safetensors"] is True
    assert captured["model"]["low_cpu_mem_usage"] is True
    assert captured["processor"]["trust_remote_code"] is False


def test_transformers_backend_runs_through_unified_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _FakeProcessor()
    _install_fake_vision(monkeypatch)
    preprocessor = Qwen3VLEmbeddingPreprocessor(processor, _spec())
    adapter_model = _TinyQwen3VLEmbeddingModel()
    adapter = Qwen3VLEmbeddingAdapter(
        adapter_model,
        semantics=Qwen3VLEmbeddingSemantics(
            model_id=_MODEL_ID,
            terminal_token_id=None,
            template_fingerprint=preprocessor.fingerprint,
            embedding_dimension=_DIMENSION,
        ),
    )
    backend = backend_module.TransformersQwen3VLEmbeddingBackend(
        adapter,
        preprocessor,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    encoder = Encoder(
        backend,
        query_instruction="Represent the user's input.",
        default_batch_size=1,
    )

    embeddings = encoder.encode_query(["short", "longer"], dimension=512)

    assert embeddings.shape == (2, 512)
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(2))
    with pytest.raises(CapabilityError, match="global embeddings"):
        backend.encode_batch(
            ["unsupported"],
            role="document",
            instruction=None,
            request=RepresentationRequest(representation=RepresentationType.TOKEN),
        )


def test_explicit_unavailable_device_and_dtype_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA device is unavailable"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="bfloat16"):
        resolve_dtype("bfloat16", torch.device("mps"))
    with pytest.raises(ValueError, match="must not include an index"):
        resolve_device("cpu:1")


def test_preprocessor_rejects_empty_or_noncanonical_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vision(monkeypatch)
    preprocessor = Qwen3VLEmbeddingPreprocessor(_FakeProcessor(), _spec())
    for value in ("", " padded"):
        with pytest.raises(ContractError, match="text inputs must be canonical"):
            preprocessor.encode_batch([value], instruction="Represent the user's input.")


@pytest.mark.parametrize(
    "roles",
    [
        ("user", "user"),
        ("user", "assistant"),
        ("assistant",),
    ],
)
def test_preprocessor_rejects_ambiguous_conversation_roles(
    roles: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vision(monkeypatch)
    item = MultimodalItem(
        item_id="roles",
        messages=tuple(Message(role=role, content="text") for role in roles),
    )
    with pytest.raises(ContractError, match="alternate user/assistant"):
        Qwen3VLEmbeddingPreprocessor(_FakeProcessor(), _spec()).encode_batch(
            [item],
            instruction="Represent the user's input.",
        )
