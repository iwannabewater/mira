from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.distributed
def test_two_rank_gloo_collectives_and_global_mean() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--master-addr=127.0.0.1",
        f"--master-port={_unused_loopback_port()}",
        str(Path(__file__).with_name("_worker.py")),
    ]

    completed = subprocess.run(  # noqa: S603 - fixed interpreter and worker path
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, (
        f"torchrun failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
