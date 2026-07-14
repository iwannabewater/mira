"""Fail-closed filesystem publication primitives."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import sys
from pathlib import Path
from typing import BinaryIO

_LINUX_AT_FDCWD = -100
_RENAME_EXCL = 0x00000004
_RENAME_NOREPLACE = 1
_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd


def open_bound_regular_file(path: Path) -> BinaryIO:
    """Open a regular file through a symlink-free absolute descriptor walk."""
    parent_fd, _ = open_bound_directory(path.parent)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        try:
            opened = os.fstat(descriptor)
            linked = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            _require_same_regular_file(opened, linked, path)
            return os.fdopen(descriptor, "rb")
        except Exception:
            os.close(descriptor)
            raise
    finally:
        os.close(parent_fd)


def open_bound_directory(path: Path) -> tuple[int, Path]:
    """Open an absolute directory path without following any component symlink."""
    absolute = _lexical_absolute_path(path)
    directory_flags, _nofollow = _secure_directory_flags(absolute)
    descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            try:
                _require_directory_descriptor(child, component)
            except Exception:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, absolute


def open_confined_regular_file(root: Path, relative: Path) -> tuple[BinaryIO, Path]:
    """Open a regular file through a no-symlink descriptor walk beneath ``root``.

    The returned display path is informational. The returned stream is the only
    authority for content: every directory component and the final file are
    opened relative to an already bound directory descriptor.
    """
    components = _confined_components(relative)
    directory_flags, nofollow = _secure_directory_flags(relative)
    descriptor, canonical_root = open_bound_directory(root)
    try:
        for component in components[:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            try:
                _require_directory_descriptor(child, component)
            except Exception:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | nofollow
        )
        file_descriptor = os.open(components[-1], file_flags, dir_fd=descriptor)
        try:
            _require_regular_descriptor(file_descriptor, components[-1])
            stream = os.fdopen(file_descriptor, "rb")
        except Exception:
            os.close(file_descriptor)
            raise
    finally:
        os.close(descriptor)
    return stream, canonical_root.joinpath(*components)


def _confined_components(relative: Path) -> tuple[str, ...]:
    if relative.is_absolute():
        raise OSError(errno.EPERM, "confined path must be relative", os.fspath(relative))
    components: list[str] = []
    for component in relative.parts:
        if component in {"", "."}:
            continue
        if component == "..":
            if not components:
                raise OSError(errno.EPERM, "path escapes confined root", os.fspath(relative))
            components.pop()
        else:
            components.append(component)
    if not components:
        raise OSError(errno.EINVAL, "confined path does not name a file", os.fspath(relative))
    return tuple(components)


def _secure_directory_flags(relative: Path) -> tuple[int, int]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not _OPEN_SUPPORTS_DIR_FD or not nofollow or not directory_flag:
        raise OSError(
            errno.ENOTSUP,
            "secure descriptor-relative file opening is unavailable",
            os.fspath(relative),
        )
    return (
        os.O_RDONLY | directory_flag | nofollow | getattr(os, "O_CLOEXEC", 0),
        nofollow,
    )


def _lexical_absolute_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else Path.cwd() / path
    if candidate.anchor != os.sep:
        raise OSError(errno.ENOTSUP, "secure absolute path traversal is unavailable", path)
    components: list[str] = []
    for component in candidate.parts[1:]:
        if component in {"", "."}:
            continue
        if component == "..":
            if not components:
                raise OSError(errno.EPERM, "absolute path escapes filesystem root", path)
            components.pop()
        else:
            components.append(component)
    return Path(candidate.anchor, *components)


def _require_same_regular_file(
    opened: os.stat_result,
    linked: os.stat_result,
    path: Path,
) -> None:
    if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(linked.st_mode):
        raise OSError(errno.EINVAL, "path is not a regular file", os.fspath(path))
    if not os.path.samestat(opened, linked):
        raise OSError(errno.ESTALE, "path changed while it was opened", os.fspath(path))


def _require_directory_descriptor(descriptor: int, component: str) -> None:
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        raise OSError(errno.ENOTDIR, "path component is not a directory", component)


def _require_regular_descriptor(descriptor: int, component: str) -> None:
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise OSError(errno.EINVAL, "confined path is not a regular file", component)


def atomic_rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` while refusing to replace ``destination``.

    The operation fails closed when the host has no atomic no-replace rename
    primitive. A check followed by ``os.replace`` is not equivalent because a
    competing process can create the destination between those operations.
    """
    if os.name == "nt":
        # Windows Path.rename uses MoveFileEx without MOVEFILE_REPLACE_EXISTING.
        source.rename(destination)
        return

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename_exclusive = libc.renamex_np
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(source_bytes, destination_bytes, _RENAME_EXCL)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename_exclusive = libc.renameat2
        rename_exclusive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(
            _LINUX_AT_FDCWD,
            source_bytes,
            _LINUX_AT_FDCWD,
            destination_bytes,
            _RENAME_NOREPLACE,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory rename is unavailable on this platform",
            os.fspath(destination),
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)
