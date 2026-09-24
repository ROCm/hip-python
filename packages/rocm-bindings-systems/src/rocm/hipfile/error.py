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

# Ported from ROCm/rocm-systems projects/hipfile/python/hipfile/error.py
# (commit cbbf349092, "[hipFile] Polish the python project prior to early
# release on PyPI", #5089) — original author Riley Dixon
# <riley.dixon@amd.com>. The port re-routes imports to consume hip-python's
# auto-generated rocm.bindings.hipfile instead of the upstream's in-tree
# _hipfile.pyx Cython extension.

__author__ = (
    "Riley Dixon <riley.dixon@amd.com> (original); "
    "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com> (port)"
)

from rocm.bindings.hipfile import (
    hipFileGetOpErrorString as _get_op_error_string,
)

from .enums import OpError


class HipFileException(Exception):
    """Exception raised on a non-success hipFile error.

    Carries both the hipFile-level error code (``hipfile_err``) and the
    underlying HIP driver error (``hip_err``) when the former is
    ``OpError.HIP_DRIVER_ERROR``.
    """

    def __init__(self, hipfile_err, hip_err):
        self._hipfile_err = hipfile_err
        self._hip_err = hip_err

    @property
    def hipfile_err(self):
        return self._hipfile_err

    @property
    def hip_err(self):
        return self._hip_err

    def __str__(self):
        # rocm.bindings.hipfile.hipFileGetOpErrorString returns a
        # ``const char *`` as a CStr, wrapped in a 1-tuple by the
        # always-return-tuple wrappers. Decode for friendlier rendering.
        # The OpError enum coerces to int when fed to the C wrapper.
        (descr,) = _get_op_error_string(OpError(int(self._hipfile_err)))
        if isinstance(descr, (bytes, bytearray)):
            descr = descr.decode("utf-8", "replace")
        else:
            descr = str(descr)
        err_msg = f"{self._hipfile_err} - {descr}"
        if int(self._hipfile_err) == OpError.HIP_DRIVER_ERROR:
            err_msg += f" {self._hip_err}"
        return err_msg
