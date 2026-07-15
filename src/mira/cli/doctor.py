"""Runtime and dependency diagnostics for a selected experiment."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Literal

import torch
import torch.distributed as dist
import typer

from mira.cli._common import emit_json
from mira.config import resolve_config
from mira.integrations.ms_swift import (
    check_compatibility as check_training_compatibility,
)
from mira.integrations.qwen3_vl_embedding import (
    check_compatibility as check_inference_compatibility,
)
from mira.integrations.qwen3_vl_embedding.arguments import resolve_device, resolve_dtype

check_compatibility = check_training_compatibility
"""Compatibility alias retained for callers of the legacy training workflow."""


def doctor_command(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, readable=True, help="Experiment config to check."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory tested for writable artifact output."),
    ] = Path("output"),
    workflow: Annotated[
        Literal["inference", "training"],
        typer.Option("--workflow", help="Dependency and hardware contract to verify."),
    ] = "training",
    device: Annotated[
        str,
        typer.Option("--device", help="Inference device policy; ignored for training checks."),
    ] = "auto",
    dtype: Annotated[
        Literal["auto", "float32", "float16", "bfloat16"],
        typer.Option("--dtype", help="Inference dtype policy; ignored for training checks."),
    ] = "auto",
) -> None:
    """Check inference or training dependencies, devices, and artifact access."""
    resolved = resolve_config(config)
    checks: list[dict[str, Any]] = []
    compatibility = (
        check_inference_compatibility() if workflow == "inference" else check_compatibility()
    )
    checks.extend(
        {
            "detail": check.reason,
            "name": f"dependency:{check.dependency}",
            "requirement": check.requirement,
            "status": "pass" if check.compatible else "fail",
            "version": check.installed_version,
        }
        for check in compatibility.checks
    )
    if workflow == "inference":
        checks.append(_inference_device_check(device, dtype))
    else:
        cuda = torch.cuda.is_available()
        checks.append(
            {
                "detail": f"device_count={torch.cuda.device_count()}",
                "name": "cuda",
                "status": "pass" if cuda else "fail",
            }
        )
    nccl_required = (
        workflow == "training" and resolved.config.runtime.distributed.strategy.value != "single"
    )
    nccl_available = dist.is_nccl_available()
    checks.append(
        {
            "detail": f"required={nccl_required}",
            "name": "nccl",
            "status": ("pass" if nccl_available else "fail" if nccl_required else "not_required"),
        }
    )
    flash_required = resolved.config.runtime.attention == "flash_attention_2"
    flash_available = importlib.util.find_spec("flash_attn") is not None
    checks.append(
        {
            "detail": f"required={flash_required}",
            "name": "flash_attention",
            "status": ("pass" if flash_available else "fail" if flash_required else "not_required"),
        }
    )
    checks.append(_distributed_environment_check())
    checks.append(_model_location_check(resolved.config.model.name_or_path))
    checks.append(_output_directory_check(output_dir))
    failures = [check for check in checks if check["status"] == "fail"]
    emit_json(
        {
            "checks": checks,
            "config_checksum": resolved.checksum,
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "status": "fail" if failures else "pass",
            "torch": torch.__version__,
            "workflow": workflow,
        }
    )
    if failures:
        raise typer.Exit(code=1)


def _inference_device_check(
    device: str,
    dtype: Literal["auto", "float32", "float16", "bfloat16"],
) -> dict[str, str]:
    try:
        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(dtype, resolved_device)
    except ValueError as error:
        return {"detail": str(error), "name": "inference_device", "status": "fail"}
    return {
        "detail": f"device={resolved_device}, dtype={str(resolved_dtype).removeprefix('torch.')}",
        "name": "inference_device",
        "status": "pass",
    }


def _distributed_environment_check() -> dict[str, str]:
    names = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    supplied = {name: os.environ.get(name) for name in names}
    present = [name for name, value in supplied.items() if value is not None]
    if not present:
        return {
            "detail": "single-process environment",
            "name": "distributed_environment",
            "status": "not_required",
        }
    missing = [name for name, value in supplied.items() if value is None]
    return {
        "detail": "complete" if not missing else f"missing={missing}",
        "name": "distributed_environment",
        "status": "pass" if not missing else "fail",
    }


def _model_location_check(model: str) -> dict[str, str]:
    path = Path(model).expanduser()
    local_like = path.is_absolute() or model.startswith((".", "~"))
    if local_like:
        status = "pass" if path.exists() else "fail"
        detail = str(path)
    else:
        status = "pass"
        detail = "remote model identifier; availability checked during explicit load"
    return {"detail": detail, "name": "model_location", "status": status}


def _output_directory_check(path: Path) -> dict[str, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path, prefix=".mira-doctor-", delete=True):
            pass
    except OSError as error:
        return {"detail": str(error), "name": "output_directory", "status": "fail"}
    return {"detail": str(path.resolve()), "name": "output_directory", "status": "pass"}
