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

# Ported from ROCm/rocm-systems projects/hipfile/python/hipfile/driver.py
# (commit cbbf349092, "[hipFile] Polish the python project prior to early
# release on PyPI", #5089) — original author Riley Dixon
# <riley.dixon@amd.com>. The port re-routes imports to consume hip-python's
# auto-generated rocm.bindings.hipfile and adapts the hipFileError struct
# return shape (`.err` / `.hip_drv_err` attributes) the auto-generated
# wrappers use.

__author__ = (
    "Riley Dixon <riley.dixon@amd.com> (original); "
    "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com> (port)"
)

from rocm.bindings.hipfile import (
    hipFileDriverOpen as _driver_open,
    hipFileDriverClose as _driver_close,
    hipFileUseCount as _use_count,
)

from .enums import OpError
from .error import HipFileException


class Driver:
    """Lifecycle manager for the hipFile driver.

    Each instance brackets one ``hipFileDriverOpen`` /
    ``hipFileDriverClose`` pair. The driver is reference-counted by the
    library; multiple ``Driver`` instances coexist safely.

    Use as a context manager:

        with Driver():
            ...

    or call :py:meth:`open` / :py:meth:`close` explicitly.
    """

    @staticmethod
    def use_count():
        """Return the current driver reference count."""
        # hipFileUseCount returns int64_t; the always-return-tuple wrappers
        # hand it back as a 1-tuple.
        (count,) = _use_count()
        return count

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        # hipFileDriverClose returns only a hipFileError; the
        # always-return-tuple wrappers wrap it in a 1-tuple.
        (err,) = _driver_close()
        if err.err != OpError.SUCCESS:
            raise HipFileException(err.err, err.hip_drv_err)

    def open(self):
        # hipFileDriverOpen returns only a hipFileError; the
        # always-return-tuple wrappers wrap it in a 1-tuple.
        (err,) = _driver_open()
        if err.err != OpError.SUCCESS:
            raise HipFileException(err.err, err.hip_drv_err)
