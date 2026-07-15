"""Publish an installed reference experiment configuration."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Annotated

import typer

from mira.cli._common import emit_json, write_new_artifacts
from mira.core.exceptions import ContractError


class ReferenceConfigPreset(StrEnum):
    """Stable names for packaged Qwen3-VL-Embedding experiment presets."""

    QWEN3_VL_EMBEDDING_2B_SINGLE_GPU_LORA = "qwen3-vl-embedding-2b-single-gpu-lora"
    QWEN3_VL_EMBEDDING_8B_ECOMMERCE = "qwen3-vl-embedding-8b-ecommerce"


_PRESET_FILES = {
    ReferenceConfigPreset.QWEN3_VL_EMBEDDING_2B_SINGLE_GPU_LORA: (
        "qwen3_vl_embedding_2b_single_gpu_lora.yaml"
    ),
    ReferenceConfigPreset.QWEN3_VL_EMBEDDING_8B_ECOMMERCE: ("qwen3_vl_embedding_ecommerce.yaml"),
}


def init_config_command(
    output: Annotated[
        Path,
        typer.Option("--output", help="New YAML path; existing files are never replaced."),
    ] = Path("mira.yaml"),
    preset: Annotated[
        ReferenceConfigPreset,
        typer.Option("--preset", help="Packaged Qwen3-VL-Embedding reference experiment."),
    ] = ReferenceConfigPreset.QWEN3_VL_EMBEDDING_2B_SINGLE_GPU_LORA,
) -> None:
    """Create a validated reference config that works outside the source tree."""
    filename = _PRESET_FILES[preset]
    content = _read_reference_config(filename)

    def writer(temporary: Path) -> None:
        temporary.write_bytes(content)

    try:
        write_new_artifacts(((output, writer),))
    except FileExistsError as error:
        raise ContractError(f"reference config output already exists: {output}") from error
    emit_json(
        {
            "path": str(output.resolve()),
            "preset": preset.value,
            "sha256": hashlib.sha256(content).hexdigest(),
            "status": "created",
        }
    )


def _read_reference_config(filename: str) -> bytes:
    """Read wheel data, with a source-tree path for editable development installs."""
    packaged = resources.files("mira").joinpath("reference_configs", filename)
    try:
        return packaged.read_bytes()
    except FileNotFoundError:
        repository_copy = Path(__file__).parents[3] / "configs" / "experiments" / filename
        try:
            return repository_copy.read_bytes()
        except OSError as error:
            raise ContractError(
                f"installed MIRA package is missing reference config {filename!r}"
            ) from error
