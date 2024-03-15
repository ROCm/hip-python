# MIT License
#
# Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

__author__ = "Advanced Micro Devices, Inc."

"""Filesystem cache for architecture-specific temporary compilation results.
"""

import os
import tempfile
import shutil
import logging
from pathlib import Path

from numba.hip import hipconfig as _hipconfig

_log = logging.getLogger(__name__)


def get_cache_dir() -> str:
    """Returns the cache directory."""
    return os.path.join(tempfile.gettempdir(), "numba", "hip")


def get_cached_file_path(amdgpu_arch: str, prefix: str, ext: str) -> str:
    """Returns a (to be) cached file's name given an AMD GPU architecture."""
    amdgpu_arch = amdgpu_arch.replace(" ", "")
    return os.path.join(get_cache_dir(), f"{prefix}_{amdgpu_arch}.{ext}")


def read_cached_file(amdgpu_arch: str, prefix: str, ext: str):
    """Loads a cached file or throws FileNotFoundError if file doesn't exist.

    See:
        `_write_cached_file`.
    """
    with open(get_cached_file_path(amdgpu_arch, prefix, ext), "rb") as infile:
        content = infile.read()
    return content


def write_cached_file(content: str, amdgpu_arch: str, prefix: str, ext: str):
    """
    Loads a cached file or throws FileNotFoundError if file doesn't exist.

    Note:
        We apply a write-replace/rename strategy to ensure that
        different processes do not write into the same file at the same time.

        We use `os.replace(src, dest, ...)` as [it has the following properties](https://docs.python.org/3/library/os.html#os.replace):

        > If dst exists and is a file, it will be replaced silently if the user has permission.[...]
        > If successful, the renaming will be an atomic operation (this is a POSIX requirement).
    Note:
        Caller is reponsible for locking this operation with a threading lock if necessary.
    """
    dest = get_cached_file_path(amdgpu_arch, prefix, ext)
    tmp_dest = f"{dest}-{os.getpid()}"
    with open(tmp_dest, "wb") as outfile:
        outfile.write(content)
    os.replace(tmp_dest, dest)


def init_cache():
    cache_dir = get_cache_dir()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    _log.info(f"created/reuse Numba HIP cache directory '{cache_dir}'")


def clear_cache():
    cache_dir = get_cache_dir()
    _log.info(f"clear Numba HIP cache directory '{cache_dir}'")
    shutil.rmtree(cache_dir, ignore_errors=True)


# note: must come before init_cache()
if _hipconfig.CLEAR_DEVICE_LIB_CACHE:
    clear_cache()

if _hipconfig.USE_DEVICE_LIB_CACHE:
    init_cache()
