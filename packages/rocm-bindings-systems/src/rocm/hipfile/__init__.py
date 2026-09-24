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

# Ported from ROCm/rocm-systems projects/hipfile/python/hipfile/__init__.py
# (commit cbbf349092, "[hipFile] Polish the python project prior to early
# release on PyPI", #5089) — original author Riley Dixon
# <riley.dixon@amd.com>. The port re-routes the version constants to
# hip-python's auto-generated rocm.bindings.hipfile and re-exports the
# same public-API surface upstream's package shipped with.

"""High-level Pythonic interface to the hipFile (Accelerated I/O Storage)
library.

This sub-package wraps the lower-level `~.rocm.bindings.hipfile`
auto-generated bindings with idiomatic Python classes:

* `~.Driver` — context manager for the hipFile driver lifecycle.
* `~.FileHandle` — context manager for an open + registered file.
* `~.Buffer` — context manager for a registered GPU memory region.

plus the mirrored enums (`~.enums.OpError`,
`~.enums.FileHandleType`), the `~.HipFileException` type, and
the standalone `~.properties.driver_get_properties` /
`~.properties.get_version` helpers.

The complete copy-via-GPU-memory example lives in
``examples/0_Basic_Usage/hipfile_copy.py``.
"""

__author__ = (
    "Riley Dixon <riley.dixon@amd.com> (original); "
    "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com> (port)"
)

from rocm.bindings.hipfile import (
    HIPFILE_VERSION_MAJOR as _VERSION_MAJOR,
    HIPFILE_VERSION_MINOR as _VERSION_MINOR,
    HIPFILE_VERSION_PATCH as _VERSION_PATCH,
)

from .buffer import Buffer
from .driver import Driver
from .enums import FileHandleType, OpError
from .error import HipFileException
from .file import FileHandle
from .properties import driver_get_properties, get_version

__all__ = [
    "__version__",
    "Driver",
    "FileHandle",
    "Buffer",
    "HipFileException",
    "FileHandleType",
    "OpError",
    "driver_get_properties",
    "get_version",
]
__version__ = (
    f"{int(_VERSION_MAJOR)}.{int(_VERSION_MINOR)}.{int(_VERSION_PATCH)}"
)
