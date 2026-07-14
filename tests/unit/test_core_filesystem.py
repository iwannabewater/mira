"""Cross-platform contract tests for fail-closed artifact publication."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from typing import TYPE_CHECKING

import pytest

from mira.core import filesystem

if TYPE_CHECKING:
    from pathlib import Path


class _FakeFunction:
    argtypes: list[object]
    restype: object

    def __init__(self, result: int, calls: list[tuple[object, ...]]) -> None:
        self.result = result
        self.calls = calls

    def __call__(self, *args: object) -> int:
        self.calls.append(args)
        return self.result


class _FakeLibc:
    def __init__(
        self,
        name: str,
        result: int,
        calls: list[tuple[object, ...]],
    ) -> None:
        setattr(self, name, _FakeFunction(result, calls))


def test_real_host_publish_is_atomic_and_never_replaces(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "complete").write_text("ready", encoding="utf-8")

    filesystem.atomic_rename_directory_no_replace(source, destination)

    assert not source.exists()
    assert (destination / "complete").read_text(encoding="utf-8") == "ready"

    competing = tmp_path / "competing"
    competing.mkdir()
    inode = destination.stat().st_ino
    with pytest.raises(FileExistsError):
        filesystem.atomic_rename_directory_no_replace(competing, destination)
    assert competing.is_dir()
    assert destination.stat().st_ino == inode


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="host has no POSIX FIFO support")
def test_confined_regular_open_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "source.fifo"
    os.mkfifo(fifo)

    with pytest.raises(OSError, match="regular file"):
        filesystem.open_confined_regular_file(tmp_path, fifo.relative_to(tmp_path))


def test_windows_uses_non_replacing_os_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    calls: list[tuple[object, object]] = []

    def refuse_replace(old: object, new: object) -> None:
        calls.append((old, new))
        raise FileExistsError(errno.EEXIST, "exists", new)

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os, "rename", refuse_replace)

    with pytest.raises(FileExistsError):
        filesystem.atomic_rename_directory_no_replace(source, destination)
    assert calls == [(source, destination)]


def test_darwin_uses_renamex_exclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    libc = _FakeLibc("renamex_np", 0, calls)
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: libc)

    filesystem.atomic_rename_directory_no_replace(tmp_path / "source", tmp_path / "destination")

    assert calls[-1][-1] == 0x00000004


def test_linux_uses_renameat2_no_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    libc = _FakeLibc("renameat2", 0, calls)
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: libc)

    filesystem.atomic_rename_directory_no_replace(tmp_path / "source", tmp_path / "destination")

    assert calls[-1][-1] == 1


def test_unsupported_platform_and_native_errors_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _UnsupportedLibc:
        pass

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(sys, "platform", "other")
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: _UnsupportedLibc())
    with pytest.raises(OSError, match="no-replace") as unsupported:
        filesystem.atomic_rename_directory_no_replace(source, destination)
    assert unsupported.value.errno == errno.ENOTSUP

    calls: list[tuple[object, ...]] = []
    libc = _FakeLibc("renameat2", -1, calls)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: libc)
    monkeypatch.setattr(ctypes, "get_errno", lambda: errno.EACCES)
    with pytest.raises(OSError, match="Permission denied") as denied:
        filesystem.atomic_rename_directory_no_replace(source, destination)
    assert denied.value.errno == errno.EACCES
