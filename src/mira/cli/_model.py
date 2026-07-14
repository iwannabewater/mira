"""Shared Qwen loading for upstream models and native MIRA checkpoints."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

from torch import nn

from mira.config import TuningMethod
from mira.core.exceptions import ConfigurationValidationError, ContractError
from mira.core.filesystem import open_bound_directory
from mira.core.serialization import strict_json_loads
from mira.inference import load_inference_manifest
from mira.integrations.ms_swift import (
    SwiftBackendAssets,
    SwiftLoadSpec,
    SwiftTemplateSpec,
    load_qwen3_vl_embedding,
)
from mira.integrations.ms_swift.peft_bridge import LoraSpec, apply_lora
from mira.training import CheckpointManager

if TYPE_CHECKING:
    from mira.config import MiraConfig
    from mira.core.manifests import ExportManifest
    from mira.models.adapters import Qwen3VLEmbeddingAdapter

_EXPORT_MANIFEST_NAME = "inference-manifest.json"
_MAX_EXPORT_FILES = 4096
_MAX_EXPORT_METADATA_BYTES = 8 * 1024 * 1024
_UNSAFE_EXPORT_SUFFIXES = frozenset(
    {
        ".bin",
        ".dll",
        ".dylib",
        ".gz",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".py",
        ".pyc",
        ".so",
        ".tar",
        ".whl",
        ".zip",
    }
)


def load_configured_qwen_assets(
    checkpoint: str,
    config: MiraConfig,
    *,
    config_checksum: str,
    expected_checkpoint_manifest_sha256: str | None = None,
    expected_export_manifest_checksum: str | None = None,
) -> SwiftBackendAssets:
    """Load a model reference or reconstruct an exact native MIRA checkpoint."""
    checkpoint_path = Path(checkpoint).expanduser()
    native_checkpoint = is_mira_checkpoint(checkpoint_path)
    native_manifest: dict[str, object] | None = None
    local_export = False
    if native_checkpoint:
        if expected_export_manifest_checksum is not None:
            raise ContractError(
                "expected_export_manifest_checksum is only valid for a local exported model"
            )
        native_manifest = CheckpointManager().verify(
            checkpoint_path,
            expected_manifest_sha256=expected_checkpoint_manifest_sha256,
        )
        _validate_native_checkpoint_provenance(
            native_manifest,
            config,
            config_checksum=config_checksum,
        )
    elif checkpoint_path.exists():
        if expected_checkpoint_manifest_sha256 is not None:
            raise ContractError(
                "expected_checkpoint_manifest_sha256 is only valid for a native MIRA checkpoint"
            )
        if expected_export_manifest_checksum is None:
            raise ContractError(
                "local exported models require --expected-export-manifest-checksum from the "
                "trusted export result"
            )
        local_export = True
    else:
        if (
            expected_checkpoint_manifest_sha256 is not None
            or expected_export_manifest_checksum is not None
        ):
            raise ContractError("manifest checksums cannot be supplied for a remote model source")
        if checkpoint != config.model.name_or_path:
            raise ContractError(
                "remote checkpoint identifier must equal config.model.name_or_path so the "
                "audited revision remains authoritative"
            )
    load_spec = SwiftLoadSpec(
        model_id=config.model.name_or_path,
        revision=config.model.revision,
        torch_dtype=_precision_name(config.runtime.precision),
        attention=config.runtime.attention,
        image_max_tokens=config.data.media.image_max_tokens,
        embedding_dimension=config.model.hidden_size,
    )
    if local_export:
        if expected_export_manifest_checksum is None:
            raise ContractError("local export is missing its required trusted manifest checksum")
        with _verified_export_snapshot(
            checkpoint_path,
            config,
            expected_manifest_checksum=expected_export_manifest_checksum,
        ) as model_source:
            assets = load_qwen3_vl_embedding(
                load_spec,
                SwiftTemplateSpec(),
                model_source=str(model_source),
            )
    else:
        assets = load_qwen3_vl_embedding(load_spec, SwiftTemplateSpec())
    if native_checkpoint:
        apply_configured_lora(assets.adapter, config)
        CheckpointManager().load(
            checkpoint_path,
            model=assets.adapter,
            mode="resume_model_only",
            expected_manifest_sha256=expected_checkpoint_manifest_sha256,
        )
    return assets


def apply_configured_lora(
    adapter: Qwen3VLEmbeddingAdapter,
    config: MiraConfig,
) -> None:
    """Apply the exact configured PEFT topology without hidden defaults."""
    tuning = config.model.tuning
    if tuning.method is TuningMethod.FULL:
        return
    if tuning.method is TuningMethod.QLORA:
        raise ConfigurationValidationError(
            "QLoRA requires an explicit quantized load contract that is not implemented"
        )
    if tuning.lora is None:
        raise ConfigurationValidationError(
            f"MIRA does not implement tuning method {tuning.method.value!r} without LoRA values"
        )
    adapter.model = apply_lora(
        adapter.model,
        LoraSpec(
            rank=tuning.lora.rank,
            alpha=tuning.lora.alpha,
            dropout=tuning.lora.dropout,
            target_modules=tuning.lora.target_modules,
            use_dora=tuning.method is TuningMethod.DORA,
            use_rslora=tuning.method is TuningMethod.RS_LORA,
        ),
    )


def merge_configured_adapter(
    adapter: Qwen3VLEmbeddingAdapter,
    config: MiraConfig,
) -> bool:
    """Merge a configured PEFT adapter into the base model for standalone export."""
    if config.model.tuning.method is TuningMethod.FULL:
        return False
    merge = getattr(adapter.model, "merge_and_unload", None)
    if not callable(merge):
        raise ContractError("the configured PEFT model cannot produce a standalone merged export")
    merged = merge(safe_merge=True)
    if not isinstance(merged, nn.Module):
        raise ContractError("PEFT merge_and_unload returned an invalid model")
    adapter.model = merged
    return True


def is_mira_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "manifest.json").is_file() and (path / "state.pt").is_file()


def _validate_native_checkpoint_provenance(
    manifest: dict[str, object],
    config: MiraConfig,
    *,
    config_checksum: str,
) -> None:
    provenance = manifest.get("manifests")
    if not isinstance(provenance, dict):
        raise ContractError("native checkpoint has no provenance mapping")
    expected = {
        "config_checksum": config_checksum,
        "model_name_or_path": config.model.name_or_path,
        "model_revision": config.model.revision,
    }
    observed = {name: provenance.get(name) for name in expected}
    if observed != expected:
        raise ContractError(
            "native checkpoint provenance differs from the supplied config: "
            f"observed={observed}, expected={expected}"
        )


def _validate_exported_model(
    path: Path,
    config: MiraConfig,
    *,
    expected_manifest_checksum: str,
) -> None:
    with _verified_export_snapshot(
        path,
        config,
        expected_manifest_checksum=expected_manifest_checksum,
    ):
        return


@contextmanager
def _verified_export_snapshot(
    path: Path,
    config: MiraConfig,
    *,
    expected_manifest_checksum: str,
) -> Iterator[Path]:
    """Yield a private copy of the exact externally trusted export artifact set."""
    with TemporaryDirectory(prefix="mira-export-") as temporary_directory:
        snapshot = Path(temporary_directory)
        root_fd = _open_export_directory(path)
        try:
            manifest_destination = snapshot / _EXPORT_MANIFEST_NAME
            _copy_bound_artifact(
                root_fd,
                Path(_EXPORT_MANIFEST_NAME),
                manifest_destination,
                expected_checksum=None,
            )
            manifest = load_inference_manifest(manifest_destination)
            if manifest.checksum != expected_manifest_checksum:
                raise ContractError(
                    "export inference manifest checksum does not match the trusted checksum: "
                    f"expected {expected_manifest_checksum}, got {manifest.checksum}"
                )
            _validate_export_semantics(manifest, config)
            artifact_names = set(manifest.artifact_checksums)
            if _EXPORT_MANIFEST_NAME in artifact_names:
                raise ContractError("export artifact checksums must not include their own manifest")
            actual_files = _list_bound_export_files(root_fd)
            expected_files = artifact_names | {_EXPORT_MANIFEST_NAME}
            if actual_files != expected_files:
                raise ContractError(
                    "local export file set differs from artifact_checksums: "
                    f"actual={sorted(actual_files)}, expected={sorted(expected_files)}"
                )
            for relative, expected_checksum in sorted(manifest.artifact_checksums.items()):
                _validate_export_artifact_name(relative)
                _copy_bound_artifact(
                    root_fd,
                    Path(relative),
                    snapshot / relative,
                    expected_checksum=expected_checksum,
                )
            _reject_remote_code_metadata(snapshot, artifact_names)
        finally:
            os.close(root_fd)
        yield snapshot


def _validate_export_semantics(manifest: ExportManifest, config: MiraConfig) -> None:
    expected = {
        "embedding_dimension": config.model.hidden_size,
        "instruction": SwiftTemplateSpec().instruction,
        "licenses": config.model.licenses,
        "model_name_or_path": config.model.name_or_path,
        "model_revision": config.model.revision,
        "normalized": config.model.representation.normalize,
        "pooling": "last_eos",
        "representation_type": "global",
        "template": "qwen3_vl_emb",
        "visual_preprocessing": {
            "attention": config.runtime.attention,
            "image_max_tokens": config.data.media.image_max_tokens,
        },
    }
    observed = {
        "embedding_dimension": manifest.embedding_dimension,
        "instruction": manifest.instruction,
        "licenses": manifest.licenses,
        "model_name_or_path": manifest.model_name_or_path,
        "model_revision": manifest.model_revision,
        "normalized": manifest.normalized,
        "pooling": manifest.pooling,
        "representation_type": manifest.representation_type,
        "template": manifest.template,
        "visual_preprocessing": dict(manifest.visual_preprocessing),
    }
    if observed != expected:
        raise ContractError(
            f"local export semantics differ from config: observed={observed}, expected={expected}"
        )


def _open_export_directory(path: Path) -> int:
    try:
        descriptor, _absolute = open_bound_directory(path)
    except OSError as error:
        raise ContractError("local export path is missing, symlinked, or unreadable") from error
    return descriptor


def _open_export_artifact(root_fd: int, relative: Path) -> tuple[int, os.stat_result]:
    parts = relative.parts
    if relative.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ContractError(f"invalid local export artifact path: {relative.as_posix()!r}")
    nofollow = os.O_NOFOLLOW
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.dup(root_fd)
    except OSError as error:
        raise ContractError("cannot bind the local export directory") from error
    try:
        for component in parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            try:
                _require_directory_descriptor(
                    child,
                    "local export artifact parent is not a directory",
                )
            except Exception:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        file_flags = (
            os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        artifact_fd = os.open(parts[-1], file_flags, dir_fd=descriptor)
        try:
            opened = os.fstat(artifact_fd)
            named = os.stat(parts[-1], dir_fd=descriptor, follow_symlinks=False)
            _require_bound_regular_artifact(opened, named)
        except Exception:
            os.close(artifact_fd)
            raise
    except OSError as error:
        raise ContractError(
            f"local export artifact is missing, symlinked, or unreadable: {relative.as_posix()}"
        ) from error
    finally:
        os.close(descriptor)
    return artifact_fd, opened


def _copy_bound_artifact(
    root_fd: int,
    relative: Path,
    destination: Path,
    *,
    expected_checksum: str | None,
) -> None:
    artifact_fd, opened = _open_export_artifact(root_fd, relative)
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with os.fdopen(artifact_fd, "rb") as source, destination.open("xb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
            final = os.fstat(source.fileno())
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise ContractError(
            f"cannot snapshot local export artifact: {relative.as_posix()}"
        ) from error
    if (
        not os.path.samestat(opened, final)
        or opened.st_size != final.st_size
        or opened.st_mtime_ns != final.st_mtime_ns
        or opened.st_ctime_ns != final.st_ctime_ns
    ):
        destination.unlink(missing_ok=True)
        raise ContractError(f"local export artifact changed while read: {relative.as_posix()}")
    observed = digest.hexdigest()
    if expected_checksum is not None and observed != expected_checksum:
        destination.unlink(missing_ok=True)
        raise ContractError(f"local export artifact checksum mismatch: {relative.as_posix()}")
    try:
        destination.chmod(0o400)
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise ContractError(f"cannot seal local export artifact: {relative.as_posix()}") from error


def _list_bound_export_files(root_fd: int) -> set[str]:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    pending: list[tuple[int, Path]] = [(os.dup(root_fd), Path())]
    files: set[str] = set()
    entries = 0
    try:
        while pending:
            descriptor, prefix = pending.pop()
            try:
                for name in os.listdir(descriptor):  # noqa: PTH208 - descriptor-bound walk
                    entries += 1
                    if entries > _MAX_EXPORT_FILES:
                        raise ContractError(
                            f"local export exceeds the {_MAX_EXPORT_FILES}-entry limit"
                        )
                    named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    relative = prefix / name
                    if stat.S_ISREG(named.st_mode):
                        files.add(relative.as_posix())
                    elif stat.S_ISDIR(named.st_mode):
                        child = os.open(name, directory_flags, dir_fd=descriptor)
                        if not stat.S_ISDIR(os.fstat(child).st_mode):
                            os.close(child)
                            raise ContractError(
                                "local export directory changed while opened: "
                                f"{relative.as_posix()}"
                            )
                        pending.append((child, relative))
                    else:
                        raise ContractError(
                            f"local export contains a non-regular artifact: {relative.as_posix()}"
                        )
            finally:
                os.close(descriptor)
    except Exception:
        for descriptor, _prefix in pending:
            os.close(descriptor)
        raise
    return files


def _require_directory_descriptor(descriptor: int, message: str) -> None:
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        raise ContractError(message)


def _require_bound_regular_artifact(
    opened: os.stat_result,
    named: os.stat_result,
) -> None:
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or not os.path.samestat(opened, named)
    ):
        raise ContractError("local export artifact is not a bound regular file")


def _validate_export_artifact_name(relative: str) -> None:
    if Path(relative).suffix.lower() in _UNSAFE_EXPORT_SUFFIXES:
        raise ContractError(f"local export contains an unsafe artifact: {relative}")


def _reject_remote_code_metadata(snapshot: Path, artifact_names: set[str]) -> None:
    for relative in artifact_names:
        path = snapshot / relative
        if not path.name.endswith("config.json"):
            continue
        if path.stat().st_size > _MAX_EXPORT_METADATA_BYTES:
            raise ContractError(f"local export metadata is oversized: {relative}")
        try:
            decoded = strict_json_loads(
                path.read_text(encoding="utf-8"),
                context=f"local export metadata {relative}",
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise ContractError(f"local export metadata is invalid: {relative}") from error
        if _contains_remote_code_metadata(decoded):
            raise ContractError(f"local export metadata requests remote code: {relative}")


def _contains_remote_code_metadata(value: object) -> bool:
    if isinstance(value, Mapping):
        if "auto_map" in value:
            return True
        return any(_contains_remote_code_metadata(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_remote_code_metadata(item) for item in value)
    return False


def _precision_name(value: str) -> Literal["bfloat16", "float16", "float32"]:
    if value == "bf16":
        return "bfloat16"
    if value == "fp16":
        return "float16"
    return "float32"
