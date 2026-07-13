# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Ported from ROCm/rocm-systems projects/hipfile/python/hipfile/file.py
# (commit cbbf349092, "[hipFile] Polish the python project prior to early
# release on PyPI", #5089) — original author Riley Dixon
# <riley.dixon@amd.com>. The port re-routes imports to consume hip-python's
# auto-generated rocm.bindings.hipfile and adapts the (retval, error) shape
# differences:
#
#   1. The auto-generated hipFileHandleRegister takes a hipFileDescr struct
#      object. The generated wrapper is fully settable: `.type` takes a
#      hipFileFileHandleType, `.handle.fd` writes the POSIX union member in
#      place, and `.fs_ops` (a pointer-to-record field) is zero-initialized
#      to NULL by the constructor — so no ctypes shim is needed.
#   2. hipFileRead / hipFileWrite return (retval, errno, hip_drv_err): the
#      robust hip-python bindings snapshot POSIX errno and
#      hipPeekAtLastError() inside the same with-nogil block as the C call
#      (see the overrides in generators_systems.generate_hipfile), matching
#      upstream's Cython glue. Negative retval is interpreted as upstream
#      (-1 → POSIX errno; <-1 → -hipFileOpError, with the HIP driver error in
#      hip_drv_err), and both side-channels are recoverable, so the raised
#      OSError / HipFileException carry the real error detail.

__author__ = (
    "Riley Dixon <riley.dixon@amd.com> (original); "
    "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com> (port)"
)

import os
import stat
from sys import stderr

from rocm.bindings.hipfile import (
    hipFileDescr,
    hipFileFileHandleType,
    hipFileHandleRegister as _handle_register,
    hipFileHandleDeregister as _handle_deregister,
    hipFileRead as _read,
    hipFileWrite as _write,
)

from .enums import FileHandleType, OpError
from .error import HipFileException


class FileHandle:
    """Lifecycle manager for a hipFile-registered open file.

    Wraps `~.hipFileHandleRegister` / `~.hipFileHandleDeregister` plus
    synchronous `~.hipFileRead` / `~.hipFileWrite`.

    Use as a context manager:

        with FileHandle(path, os.O_RDWR | os.O_DIRECT) as fh:
            fh.read(buf, size, file_offset, buffer_offset)
            fh.write(buf, size, file_offset, buffer_offset)
    """

    DEFAULT_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH

    def __init__(
        self,
        path,
        flags,
        mode=DEFAULT_MODE,
        handle_type=FileHandleType.OPAQUE_FD,
    ):
        self._fd = None
        self._flags = flags
        self._handle = None
        self._handle_type = None
        self._mode = mode
        self._path = path

        # Goes through the validating setter below.
        self.handle_type = handle_type

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 — suppress in destructor
            print(
                "Failed to deregister hipFile.FileHandle at destruction time.",
                file=stderr,
            )

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # --- Read-only properties ----------------------------------------------

    @property
    def flags(self):
        return self._flags

    @property
    def handle(self):
        return self._handle

    @property
    def mode(self):
        return self._mode

    @property
    def path(self):
        return self._path

    # --- handle_type with validation ---------------------------------------

    @property
    def handle_type(self):
        return self._handle_type

    @handle_type.setter
    def handle_type(self, _handle_type):
        if self._handle is not None:
            raise RuntimeError(
                "Cannot modify handle_type while FileHandle is open"
            )
        if _handle_type not in FileHandleType:
            raise ValueError(
                f"'{_handle_type}' is not a member of enum FileHandleType"
            )
        if _handle_type == FileHandleType.OPAQUE_WIN32:
            raise NotImplementedError(
                "FileHandle does not currently support Win32 Handles"
            )
        self._handle_type = _handle_type

    # --- Lifecycle ---------------------------------------------------------

    def open(self):
        if self._handle is not None:
            raise RuntimeError("The FileHandle is already open.")
        self._fd = os.open(self._path, self._flags, self._mode)

        # Build the descriptor with the generated wrapper: the constructor
        # zero-inits the buffer (so fs_ops stays NULL), `.type` takes a
        # hipFileFileHandleType, and `.handle.fd` writes the POSIX union
        # member in place.
        descr = hipFileDescr()
        descr.type = hipFileFileHandleType(int(self._handle_type))
        descr.handle.fd = self._fd

        err, fh = _handle_register(descr)
        if err.err != OpError.SUCCESS:
            os.close(self._fd)
            self._fd = None
            raise HipFileException(err.err, err.hip_drv_err)
        self._handle = fh

    def close(self):
        if self._handle is not None:
            _handle_deregister(self._handle)
            self._handle = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    # --- I/O ---------------------------------------------------------------

    def read(self, buffer, size, file_offset, buffer_offset):
        """Synchronous read into a registered `~.buffer.Buffer`.

        Returns the number of bytes read on success. Raises
        `~.error.HipFileException` (with the parsed
        `~.enums.OpError` and HIP driver error) on a hipFile-level
        error, or ``OSError`` (with the real ``errno``) on a
        POSIX-level error.
        """
        if self._handle is None:
            raise RuntimeError("The FileHandle is not open.")
        n, err, drv = _read(
            self._handle, buffer.ptr, size, file_offset, buffer_offset
        )
        return self._check_io_result(n, err, drv)

    def write(self, buffer, size, file_offset, buffer_offset):
        """Synchronous write from a registered `~.buffer.Buffer`.

        Returns the number of bytes written on success. Same error
        semantics as `~.FileHandle.read`.
        """
        if self._handle is None:
            raise RuntimeError("The FileHandle is not open.")
        n, err, drv = _write(
            self._handle, buffer.ptr, size, file_offset, buffer_offset
        )
        return self._check_io_result(n, err, drv)

    @staticmethod
    def _check_io_result(n, err, drv):
        # hipFileRead/hipFileWrite return (retval, errno, hip_drv_err); the
        # robust bindings snapshot errno / hipPeekAtLastError() in the same
        # with-nogil block as the C call, so both are recoverable here.
        if n == -1:
            # POSIX-level failure — surface the real errno.
            raise OSError(err, os.strerror(err))
        if n < -1:
            # `-n` is the hipFileOpError code; when it is hipFileHipDriverError
            # the HIP driver error is carried in `drv`.
            raise HipFileException(OpError(-n), drv)
        return n
