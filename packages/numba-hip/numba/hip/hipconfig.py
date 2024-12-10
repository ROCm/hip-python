# MIT License
#
# Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

"""Configuration options for Numba HIP

Attributes (Controllable via Environment Variables ``NUMBA_HIP_<attribute>``):
    ENABLE_MIDEND_OPT (`bool`):
        Enable midend optimizations.
        Defaults to ``False``.
    OPT_LEVEL (`int`):
        Default level when doing optimizations.
        Defaults to ``3``.
    DEFAULT_ARCH_WITH_FEATURES (`bool`):
        Use a device's default arch with features, e.g. 'gfx90a:xnack-'
        instead of 'gfx90a'. Note that features
        enabled for Numba-generated IR must match those of all
        external dependencies that should be linked in.
        As external dependencies are usually compiled with
        no additional features, e.g. with a simple `--offload-arch=gfx90a`,
        this option defaults to ``False``.
        (Note that `hipcc` per default also specifies the first device's
        architecture without an appended feature set as offload arch.)
    USE_LINKER_CACHE (`bool`):
        Enable the caching of link-time dependencies that are files or file buffers in a dictionary.
        Defaults to ``True``.
    USE_DEVICE_LIB_FILE_CACHE (`bool`):
        Store architecture-dependent device library LLVM IR files in a filesystem cache, so that
        the next Numba program can reuse the files from this cache. This caching
        has a significant impact on programs with short runtime, e.g., on
        tests and mini benchmarks. Defaults to ``True``.
    CLEAR_DEVICE_LIB_CACHE (`bool`):
        Clear the filesystem cache used for storing architecture-dependent device library LLVM IR.
        Defaults to ``False``.
    MINIMIZE_IR (`bool`):
        Apply a couple of steps to minimize the produced LLVM IR.
        Warning enabling this feature can have significant impact on performance.
        Defaults to ``False``.

Note:
    We currently don't want to break out of subfolder
    ``numba/hip``with the changes that we apply to an
    Numba installation in order to make patching
    and upgrading as frictionless as possible.
"""

import os

import logging

_log = logging.getLogger(__name__)

ENABLE_MIDEND_OPT = bool(
    int(os.environ.get("NUMBA_HIP_MIDEND_OPT", False))
)  # enable midend optimizations
OPT_LEVEL = int(os.environ.get("NUMBA_HIP_OPT_LEVEL", 3))  # default optimization level
DEFAULT_ARCH_WITH_FEATURES = int(
    os.environ.get("NUMBA_HIP_DEFAULT_ARCH_WITH_FEATURES", 0)
)  # Use a device's default arch with features, e.g. 'gfx90a:xnack-'
# instead of 'gfx90a'. Note that features
# enabled for Numba-generated IR must match those of all
# external dependencies that should be linked in.
# As external dependencies are usually compiled with `gfx90a`.
# this option defaults to `False`.

USE_LINKER_CACHE = bool(
    int(os.environ.get("NUMBA_HIP_USE_LINKER_CACHE", True))
)  # Enable the caching of link-time dependencies that are files or file buffers in a dictionary. Defaults to ``True``.

USE_DEVICE_LIB_CACHE = bool(
    int(os.environ.get("NUMBA_HIP_USE_DEVICE_LIB_CACHE", True))
)  # Store architecture-dependent device library LLVM IR files in a filesystem cache, so that
# the next Numba program can reuse the files from this cache. Defaults to True.

CLEAR_DEVICE_LIB_CACHE = bool(
    int(os.environ.get("NUMBA_HIP_CLEAR_DEVICE_LIB_CACHE", False))
)  # Clear the filesystem cache used for storing architecture-dependent device library LLVM IR.
# Defaults to False.

MINIMIZE_IR = bool(
    int(os.environ.get("NUMBA_HIP_MINIMIZE_IR", False))
)  # Apply a couple of steps to minimize the produced LLVM IR.
# Warning enabling this feature can have significant impact on performance.
# Defaults to ``False``.


def get_rocm_path(*subdirs):
    """Get paths of ROCM_PATH.

    Args:
        subdirs (optional):
            Either:

            * Parts (`str`) of a subdirectory path:
              Example: 'get_rocm_path("llvm","lib")' gives '<ROCM_PATH>/llvm/lib'.
            * A list of `tuple` / `list` objects, which each describe parts of a subdirectory path.
              In this case, all of the subdirectories are checked and the first existing one is returned.

    """
    rocm_path = os.environ.get("ROCM_HOME", os.environ.get("ROCM_PATH"))
    if rocm_path is None:
        _log.info(
            "neither 'ROCM_PATH' nor 'ROCM_HOME' environment variable specified, trying default path '/opt/rocm'"
        )
        rocm_path = "/opt/rocm/"
    if not os.path.exists(rocm_path):
        msg = "no ROCm installation found, checked 'ROCM_PATH' and 'ROCM_HOME' and tried '/opt/rocm'"
        _log.error(msg)
        raise FileNotFoundError(msg)

    if all((isinstance(s, str) for s in subdirs)):
        subdirs = [subdirs]
    elif all((isinstance(s, (tuple, list)) for s in subdirs)):
        pass
    else:
        raise TypeError(
            "inputs to 'get_rocm_path' must be all of type `str` or all of type 'tuple' or 'list'"
        )

    rocm_subdirs = []
    for subdir in subdirs:
        rocm_subdir = os.path.join(rocm_path, *subdir)
        if os.path.exists(rocm_subdir):
            return rocm_subdir
        rocm_subdirs.append(rocm_subdir)

    rocm_subdirs = ", ".join(f"'{f}'" for f in rocm_subdirs)
    msg = f"found ROCm installation at '{rocm_path}' but none of the subdirectories: {rocm_subdirs}"
    _log.error(msg)
    raise FileNotFoundError(msg)


def get_rocm_inc_dir():
    """Path to ROCm include directory."""
    return get_rocm_path("include")
