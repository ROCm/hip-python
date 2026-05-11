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

# Ported from ROCm/rocm-systems projects/hipfile/python/hipfile/properties.py
# (commit cbbf349092, "[hipFile] Polish the python project prior to early
# release on PyPI", #5089) — original author Riley Dixon
# <riley.dixon@amd.com>. The port re-routes imports to consume hip-python's
# auto-generated rocm.bindings.hipfile and adapts the (retval, hipFileError)
# return shape the auto-generated wrappers use.

__author__ = (
    "Riley Dixon <riley.dixon@amd.com> (original); "
    "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com> (port)"
)

from rocm.bindings.hipfile import (
    hipFileDriverGetProperties as _driver_get_properties,
    hipFileGetVersion as _get_version,
)

from .error import HipFileException
from .enums import OpError


def driver_get_properties():
    """Return the driver properties struct as set by `hipFileDriverGetProperties`."""
    err, props = _driver_get_properties()
    if err.err != OpError.SUCCESS:
        raise HipFileException(err.err, err.hip_drv_err)
    return props


def get_version():
    """Return ``(major, minor, patch)`` for the loaded libhipfile.so."""
    err, major, minor, patch = _get_version()
    if err.err != OpError.SUCCESS:
        raise HipFileException(err.err, err.hip_drv_err)
    return (major, minor, patch)
