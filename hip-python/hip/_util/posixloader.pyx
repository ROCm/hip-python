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

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

cimport posix.dlfcn

cdef int open_library(void** lib_handle, const char* path) except 1 nogil:
    """Opens a shared object and returns a handle for it via out parameter.

    Args:
        lib_handle (void**, out):
            The library handle, the result.
            If an error has occured, the dereferenced value is NULL.
    Returns:
        Positive number if something has gone wrong, '0' otherwise.
    """
    lib_handle[0] = posix.dlfcn.dlopen(path, posix.dlfcn.RTLD_NOW)
    cdef char* reason = NULL
    if lib_handle[0] == NULL:
        reason = posix.dlfcn.dlerror()
        raise RuntimeError(f"failed to dlopen '{str(path)}': {str(reason)}")
    return 0

cdef int close_library(void* lib_handle) except 1 nogil:
    """Closes the given shared object.

    Args:
        lib_handle (void*, in):
            Handle to the library to close.
    Returns:
        Positive number if something has gone wrong, '0' otherwise.
    """
    if lib_handle == NULL:
        raise RuntimeError(f"handle is NULL")
    cdef int rtype = posix.dlfcn.dlclose(lib_handle)
    cdef char* reason = NULL
    if rtype != 0:
        reason = posix.dlfcn.dlerror()
        raise RuntimeError(f"failed to dclose given handle: {reason}")
    return 0

cdef int load_symbol(void** handle, void* lib_handle, const char* name) except 1 nogil:
    """Returns a symbol handle from an openend shared object via out parameter.

    Args:
        handle (void**, in):
            The symbol handle, the result.
            If an error has occured, the dereferenced value is NULL.
        lib_handle (void*, in):
            Shared object handle.
        name (char*, in):
            Name of the symbol.
    Returns:
        Positive number if something has gone wrong, '0' otherwise.
    """
    handle[0] = posix.dlfcn.dlsym(lib_handle, name)
    cdef char* reason = NULL
    if handle[0] == NULL:
        reason = posix.dlfcn.dlerror()
        raise RuntimeError(f"failed to dlsym '{name}': {reason}")
    return 0
