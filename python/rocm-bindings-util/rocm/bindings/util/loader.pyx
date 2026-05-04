# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

"""
Platform-agnostic dynamic library loader.

This module provides a unified interface for loading dynamic libraries across
platforms, using the appropriate platform-specific implementation:
- Windows: LoadLibrary/GetProcAddress/FreeLibrary
- Linux/POSIX: dlopen/dlsym/dlclose

Platform selection is done at compile time using Cython's IF UNAME_SYSNAME.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# Import platform-specific implementation
IF UNAME_SYSNAME == "Windows":
    from rocm.bindings.util.win32loader cimport (
        open_library as _open_library,
        close_library as _close_library,
        load_symbol as _load_symbol
    )
ELSE:
    from rocm.bindings.util.posixloader cimport (
        open_library as _open_library,
        close_library as _close_library,
        load_symbol as _load_symbol
    )

# Re-export with standard names (thin wrapper for proper PyInit_loader)
cdef int open_library(void** lib_handle, const char* path) except 1 nogil:
    """Opens a dynamic library and returns a handle for it via out parameter."""
    return _open_library(lib_handle, path)

cdef int close_library(void* lib_handle) except 1 nogil:
    """Closes the given dynamic library."""
    return _close_library(lib_handle)

cdef int load_symbol(void** handle, void* lib_handle, const char* name) except 1 nogil:
    """Returns a symbol handle from an opened dynamic library via out parameter."""
    return _load_symbol(handle, lib_handle, name)
