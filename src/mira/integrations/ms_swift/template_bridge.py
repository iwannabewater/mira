"""Exact Qwen3-VL-Embedding template construction through MS-SWIFT."""

from __future__ import annotations

import hashlib
import json
import os
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Protocol, cast

from mira.core.exceptions import ContractError
from mira.core.filesystem import open_bound_regular_file
from mira.integrations.ms_swift.compatibility import EXPECTED_MS_SWIFT_VERSION

_MAX_MEDIA_SUFFIX_LENGTH = 17

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mira.data.schema import MediaRef, MultimodalItem
    from mira.integrations.ms_swift.arguments import SwiftTemplateSpec


class _SwiftTemplate(Protocol):
    task_type: str

    def set_mode(self, mode: str) -> None:
        """Set the upstream encode mode."""

    def encode(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Encode one Swift-compatible item."""

    def data_collator(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad and collate individually encoded items."""


class SwiftTemplateBridge:
    """Own one immutable upstream template and its reproducible fingerprint."""

    def __init__(self, processor: Any, spec: SwiftTemplateSpec) -> None:
        """Construct the audited Qwen3-VL-Embedding template without fallback."""
        swift_template = import_module("swift.template")
        get_template = getattr(swift_template, "get_template", None)
        if not callable(get_template):
            raise ContractError("MS-SWIFT does not expose swift.template.get_template")
        template = get_template(
            processor,
            default_system=spec.instruction,
            max_length=spec.max_length,
            template_type=spec.template_type,
            truncation_strategy=spec.truncation_strategy,
            padding_side=spec.padding_side,
        )
        if not hasattr(template, "encode") or not hasattr(template, "set_mode"):
            raise ContractError("MS-SWIFT returned an incompatible template object")
        self._template = cast("_SwiftTemplate", template)
        # MIRA encodes each item independently, then owns query/candidate layout.
        # Using Swift's embedding branch here would select positive[0].
        self._template.task_type = "causal_lm"
        self._template.set_mode("transformers")
        self.spec = spec
        self.fingerprint = template_fingerprint(spec)

    def encode_item(self, item: MultimodalItem) -> dict[str, Any]:
        """Encode one independently validated MIRA item with exact media order."""
        validate_qwen_message_roles(item)
        with TemporaryDirectory(prefix="mira-media-") as temporary_directory:
            sample: dict[str, Any] = {
                "messages": [
                    message.model_dump(mode="json", exclude_none=True) for message in item.messages
                ],
            }
            snapshot_root = Path(temporary_directory)
            for field in ("images", "videos", "audios"):
                references = getattr(item, field)
                if references:
                    sample[field] = [
                        str(_snapshot_verified_media(reference, snapshot_root, field, index))
                        for index, reference in enumerate(references)
                    ]
            encoded = self._template.encode(sample)
        if not isinstance(encoded, dict) or "input_ids" not in encoded:
            raise ContractError("MS-SWIFT template returned no input_ids")
        encoded.pop("labels", None)
        return encoded

    def collate_items(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate independently encoded items without positional group inference."""
        if not items:
            raise ContractError("cannot collate an empty MIRA item batch")
        collated = self._template.data_collator(items)
        if not isinstance(collated, dict) or "input_ids" not in collated:
            raise ContractError("MS-SWIFT template collator returned no input_ids")
        collated.pop("labels", None)
        return collated


def template_fingerprint(spec: SwiftTemplateSpec) -> str:
    """Hash every prompt-affecting value plus the audited upstream version."""
    payload = {
        "instruction": spec.instruction,
        "max_length": spec.max_length,
        "ms_swift_version": EXPECTED_MS_SWIFT_VERSION,
        "padding_side": spec.padding_side,
        "suffix": "<|endoftext|>",
        "task_type": "mira_item_encoding_v1",
        "template_type": spec.template_type,
        "truncation_strategy": spec.truncation_strategy,
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def validate_qwen_message_roles(item: MultimodalItem) -> None:
    """Reject roles that can replace or bypass the fixed embedding instruction."""
    roles = tuple(message.role for message in item.messages)
    expected = tuple("user" if index % 2 == 0 else "assistant" for index in range(len(roles)))
    if roles != expected or roles[-1] != "user":
        raise ContractError(
            "Qwen3-VL-Embedding items must start with user, alternate user/assistant roles, "
            "and end with user; "
            f"got {roles}"
        )


def _snapshot_verified_media(
    reference: MediaRef,
    snapshot_root: Path,
    field: str,
    index: int,
) -> Path:
    source = Path(reference.uri)
    if not source.is_absolute() or reference.checksum is None:
        raise ContractError(
            "MS-SWIFT media must be resolved to an absolute path with a SHA-256 checksum"
        )
    suffix = source.suffix
    if suffix and (len(suffix) > _MAX_MEDIA_SUFFIX_LENGTH or not suffix[1:].isalnum()):
        suffix = ""
    destination = snapshot_root / f"{field}-{index:06d}{suffix.lower()}"
    digest = hashlib.sha256()
    try:
        with (
            open_bound_regular_file(source) as input_stream,
            destination.open("xb") as output_stream,
        ):
            for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                digest.update(chunk)
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except OSError as error:
        raise ContractError("verified media source is unavailable or changed") from error
    observed = f"sha256:{digest.hexdigest()}"
    if observed != reference.checksum:
        destination.unlink(missing_ok=True)
        raise ContractError(
            "verified media checksum changed before MS-SWIFT template encoding: "
            f"expected {reference.checksum}, got {observed}"
        )
    destination.chmod(0o400)
    return destination
