"""Strict official and MIRA-v1 preprocessing for Qwen3-VL-Embedding."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

import torch

from mira.core.exceptions import ContractError
from mira.core.filesystem import open_bound_regular_file
from mira.data.schema import MultimodalItem

if TYPE_CHECKING:
    from mira.data.schema import MediaRef, Message
    from mira.integrations.qwen3_vl_embedding.arguments import Qwen3VLEmbeddingLoadSpec

_PLACEHOLDER_PATTERN = re.compile(r"<(image|video|audio)>")
_IMAGE_PATCH_SIZE = 16
_MERGED_PATCH_FACTOR = 32
_MIN_IMAGE_TOKENS = 4
_MAX_MEDIA_SUFFIX_LENGTH = 17
_TOKEN_MATRIX_RANK = 2


class Qwen3VLEmbeddingPreprocessor:
    """Convert validated MIRA inputs into one tensor-only embedding batch."""

    def __init__(self, processor: Any, spec: Qwen3VLEmbeddingLoadSpec) -> None:
        """Bind the exact processor and prompt-affecting load values."""
        if processor is None:
            raise ContractError("Qwen3-VL-Embedding processor must not be null")
        self.processor = processor
        self.spec = spec
        self.fingerprint = preprocessing_fingerprint(spec)

    def encode_batch(
        self,
        items: Sequence[object],
        *,
        instruction: str,
    ) -> dict[str, torch.Tensor]:
        """Snapshot media, format messages, and return a checked tensor mapping."""
        if not items:
            raise ContractError("cannot preprocess an empty Qwen3-VL-Embedding batch")
        if not instruction or instruction != instruction.strip():
            raise ContractError("Qwen3-VL-Embedding instruction must be canonical")
        with TemporaryDirectory(prefix="mira-qwen3-vl-embedding-media-") as temporary:
            root = Path(temporary)
            conversations = [
                self._conversation(item, instruction=instruction, root=root, row=row)
                for row, item in enumerate(items)
            ]
            texts = self._render_text(conversations)
            images, videos, video_metadata, video_kwargs = _process_vision(conversations)
            values = self.processor(
                text=texts,
                images=images,
                videos=videos,
                video_metadata=video_metadata,
                truncation=False,
                padding=True,
                do_resize=False,
                return_tensors="pt",
                **video_kwargs,
            )
        return _validate_processor_output(values, max_length=self.spec.max_length)

    def _conversation(
        self,
        item: object,
        *,
        instruction: str,
        root: Path,
        row: int,
    ) -> list[dict[str, Any]]:
        if isinstance(item, str):
            if not item or item != item.strip():
                raise ContractError("Qwen3-VL-Embedding text inputs must be canonical")
            messages = [{"role": "user", "content": [{"type": "text", "text": item}]}]
        elif isinstance(item, MultimodalItem):
            _validate_item(item, instruction)
            media = {
                "image": tuple(
                    _snapshot_verified_media(reference, root, row, "image", index)
                    for index, reference in enumerate(item.images)
                ),
                "video": tuple(
                    _snapshot_verified_media(reference, root, row, "video", index)
                    for index, reference in enumerate(item.videos)
                ),
            }
            positions = {"image": 0, "video": 0}
            messages = [
                {
                    "role": message.role,
                    "content": _expand_message(
                        message, media=media, positions=positions, spec=self.spec
                    ),
                }
                for message in item.messages
            ]
            if any(positions[name] != len(media[name]) for name in ("image", "video")):
                raise ContractError(
                    f"Qwen3-VL-Embedding item {item.item_id!r} did not consume all media"
                )
        else:
            raise ContractError(
                "Qwen3-VL-Embedding inputs must be strings or validated MultimodalItem values"
            )
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            *messages,
        ]

    def _render_text(self, conversations: list[list[dict[str, Any]]]) -> list[str]:
        apply_template = getattr(self.processor, "apply_chat_template", None)
        if not callable(apply_template):
            raise ContractError("Qwen3-VL-Embedding processor does not expose apply_chat_template")
        if self.spec.template_profile == "official":
            rendered = apply_template(
                conversations,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            prepared: list[list[dict[str, Any]]] = []
            for conversation in conversations:
                values = list(conversation)
                non_system_count = sum(message["role"] != "system" for message in values)
                if non_system_count % 2 == 1 and values[-1]["role"] != "assistant":
                    values.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})
                prepared.append(values)
            rendered = apply_template(
                prepared,
                add_generation_prompt=False,
                tokenize=False,
            )
        return _require_text_batch(rendered)


def preprocessing_fingerprint(spec: Qwen3VLEmbeddingLoadSpec) -> str:
    """Hash every value that can change Qwen3-VL-Embedding model inputs."""
    document = {
        "fps": spec.fps,
        "image_max_tokens": spec.image_max_tokens,
        "instruction": spec.instruction,
        "max_frames": spec.max_frames,
        "max_length": spec.max_length,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "profile": spec.template_profile,
        "schema": "mira.qwen3_vl_embedding.preprocessing.v1",
    }
    return hashlib.sha256(
        json.dumps(document, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _validate_item(item: MultimodalItem, instruction: str) -> None:
    roles = tuple(message.role for message in item.messages)
    expected = tuple("user" if index % 2 == 0 else "assistant" for index in range(len(roles)))
    if roles != expected or roles[-1] != "user":
        raise ContractError(
            "Qwen3-VL-Embedding items must start with user, alternate user/assistant roles, "
            "and end with user; "
            f"got {roles}"
        )
    if item.audios:
        raise ContractError(f"Qwen3-VL-Embedding item {item.item_id!r} contains unsupported audio")
    if item.instruction is not None and item.instruction != instruction:
        raise ContractError(
            f"Qwen3-VL-Embedding item {item.item_id!r} instruction differs from the encoder"
        )


def _expand_message(
    message: Message,
    *,
    media: dict[str, tuple[Path, ...]],
    positions: dict[str, int],
    spec: Qwen3VLEmbeddingLoadSpec,
) -> list[dict[str, Any]]:
    if isinstance(message.content, str):
        fragments: tuple[str, ...] = (message.content,)
    else:
        fragments_list: list[str] = []
        for part in message.content:
            if part.type != "text" or part.text is None:
                raise ContractError(
                    "Qwen3-VL-Embedding structured message parts must be text; "
                    "use explicit <image>/<video> placeholders and MediaRef values"
                )
            fragments_list.append(part.text)
        fragments = tuple(fragments_list)
    content: list[dict[str, Any]] = []
    for fragment in fragments:
        start = 0
        for match in _PLACEHOLDER_PATTERN.finditer(fragment):
            if match.start() > start:
                content.append({"type": "text", "text": fragment[start : match.start()]})
            modality = match.group(1)
            if modality == "audio":
                raise ContractError("Qwen3-VL-Embedding does not support audio placeholders")
            position = positions[modality]
            try:
                path = media[modality][position]
            except IndexError as error:
                raise ContractError(
                    f"Qwen3-VL-Embedding {modality} placeholder has no aligned MediaRef"
                ) from error
            positions[modality] += 1
            content.append(_media_part(modality, path, spec))
            start = match.end()
        if start < len(fragment):
            content.append({"type": "text", "text": fragment[start:]})
    if not content:
        raise ContractError("Qwen3-VL-Embedding message content became empty")
    return content


def _media_part(
    modality: str,
    path: Path,
    spec: Qwen3VLEmbeddingLoadSpec,
) -> dict[str, Any]:
    uri = path.as_uri()
    pixels_per_token = _MERGED_PATCH_FACTOR * _MERGED_PATCH_FACTOR
    if modality == "image":
        return {
            "type": "image",
            "image": uri,
            "min_pixels": _MIN_IMAGE_TOKENS * pixels_per_token,
            "max_pixels": spec.image_max_tokens * pixels_per_token,
        }
    return {
        "type": "video",
        "video": uri,
        "fps": spec.fps,
        "max_frames": spec.max_frames,
        "total_pixels": 10 * spec.image_max_tokens * pixels_per_token,
    }


def _snapshot_verified_media(
    reference: MediaRef,
    root: Path,
    row: int,
    modality: str,
    index: int,
) -> Path:
    source = Path(reference.uri)
    if not source.is_absolute() or reference.checksum is None:
        raise ContractError(
            "Qwen3-VL-Embedding media must use an absolute verified path and SHA-256"
        )
    suffix = source.suffix
    if suffix and (len(suffix) > _MAX_MEDIA_SUFFIX_LENGTH or not suffix[1:].isalnum()):
        suffix = ""
    destination = root / f"row-{row:06d}-{modality}-{index:06d}{suffix.lower()}"
    digest = hashlib.sha256()
    try:
        with open_bound_regular_file(source) as input_stream, destination.open("xb") as output:
            for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise ContractError(
            "verified Qwen3-VL-Embedding media is unavailable or changed"
        ) from error
    observed = f"sha256:{digest.hexdigest()}"
    if observed != reference.checksum:
        destination.unlink(missing_ok=True)
        raise ContractError(
            "Qwen3-VL-Embedding media checksum changed before preprocessing: "
            f"expected {reference.checksum}, got {observed}"
        )
    destination.chmod(0o400)
    return destination


def _process_vision(
    conversations: list[list[dict[str, Any]]],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    module = import_module("qwen_vl_utils.vision_process")
    process = getattr(module, "process_vision_info", None)
    if not callable(process):
        raise ContractError("qwen-vl-utils does not expose Qwen3-VL-Embedding vision preprocessing")
    try:
        images, video_inputs, video_kwargs = process(
            conversations,
            image_patch_size=_IMAGE_PATCH_SIZE,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
    except Exception as error:
        raise ContractError("Qwen3-VL-Embedding vision preprocessing failed") from error
    if video_inputs is None:
        video_values = None
        video_metadata_values = None
    else:
        try:
            videos, video_metadata = zip(*video_inputs, strict=True)
        except (TypeError, ValueError) as error:
            raise ContractError(
                "Qwen3-VL-Embedding video preprocessing returned malformed metadata"
            ) from error
        video_values = list(videos)
        video_metadata_values = list(video_metadata)
    if not isinstance(video_kwargs, Mapping):
        raise ContractError("Qwen3-VL-Embedding video preprocessing returned malformed options")
    return images, video_values, video_metadata_values, dict(video_kwargs)


def _require_text_batch(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence) or any(not isinstance(item, str) for item in value):
        raise ContractError("Qwen3-VL-Embedding chat template returned invalid text")
    return list(cast("Sequence[str]", value))


def _validate_processor_output(value: Any, *, max_length: int) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ContractError("Qwen3-VL-Embedding processor returned no tensor mapping")
    output: dict[str, torch.Tensor] = {}
    for name, item in value.items():
        if type(name) is not str or not isinstance(item, torch.Tensor):
            raise ContractError("Qwen3-VL-Embedding processor outputs must map names to tensors")
        output[name] = item
    input_ids = output.get("input_ids")
    attention_mask = output.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise ContractError("Qwen3-VL-Embedding processor must return input_ids and attention_mask")
    if input_ids.ndim != _TOKEN_MATRIX_RANK or tuple(attention_mask.shape) != tuple(
        input_ids.shape
    ):
        raise ContractError("Qwen3-VL-Embedding token and attention tensors must align as [N, L]")
    if input_ids.shape[1] > max_length:
        raise ContractError(
            f"Qwen3-VL-Embedding input length {input_ids.shape[1]} exceeds {max_length}; "
            "MIRA does not silently truncate inference inputs"
        )
    output.pop("labels", None)
    return output
