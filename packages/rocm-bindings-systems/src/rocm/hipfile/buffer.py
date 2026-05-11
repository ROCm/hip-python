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

# Ported from ROCm/rocm-systems projects/hipfile/python/hipfile/buffer.py
# (commit cbbf349092, "[hipFile] Polish the python project prior to early
# release on PyPI", #5089) — original author Riley Dixon
# <riley.dixon@amd.com>. The port re-routes imports to consume hip-python's
# auto-generated rocm.bindings.hipfile and adapts the hipFileError struct
# return shape (`.err` / `.hip_drv_err` attributes) the auto-generated
# wrappers use.

from __future__ import annotations

__author__ = (
    "Riley Dixon <riley.dixon@amd.com> (original); "
    "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com> (port)"
)

from sys import stderr
from typing import TYPE_CHECKING

from rocm.bindings.hipfile import (
    hipFileBufDeregister as _buf_deregister,
    hipFileBufRegister as _buf_register,
)

from .enums import OpError
from .error import HipFileException

if TYPE_CHECKING:
    from ctypes import c_void_p


class Buffer:
    """Lifecycle manager for a hipFile-registered GPU memory region.

    The caller pre-allocates the underlying device buffer (typically
    via :py:func:`rocm.bindings.hip.hipMalloc` or any equivalent
    GPU-allocator) and constructs a :py:class:`Buffer` to register it
    with the hipFile driver. Deregistration happens on context exit
    (or via explicit :py:meth:`deregister`).

    Use as a context manager:

        with Buffer(ptr, length, flags=0) as registered_buf:
            ...
    """

    @classmethod
    def from_ctypes_void_p(cls, ctypes_void_p: c_void_p, length, flags):
        """Construct a :py:class:`Buffer` from a :py:class:`ctypes.c_void_p`."""
        return cls(ctypes_void_p.value, length, flags)

    def __init__(self, buffer_ptr, length, flags) -> None:
        self._buffer_ptr = buffer_ptr
        self._flags = flags
        self._length = length
        self._registered = False

    def __del__(self):
        # We did not create the underlying buffer. Don't try to free it.
        try:
            self.deregister()
        except Exception:  # noqa: BLE001 — suppress in destructor
            print(
                "Failed to deregister hipFile.Buffer at destruction time.",
                file=stderr,
            )

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.deregister()

    @property
    def ptr(self):
        return self._buffer_ptr

    def deregister(self):
        if self._registered:
            err = _buf_deregister(self._buffer_ptr)
            if err.err != OpError.SUCCESS:
                raise HipFileException(err.err, err.hip_drv_err)
            self._registered = False

    def register(self):
        err = _buf_register(self._buffer_ptr, self._length, self._flags)
        if err.err != OpError.SUCCESS:
            raise HipFileException(err.err, err.hip_drv_err)
        self._registered = True
