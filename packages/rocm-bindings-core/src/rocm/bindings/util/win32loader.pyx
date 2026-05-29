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

# Windows DLL loader using LoadLibrary/GetProcAddress/FreeLibrary APIs

cdef extern from "windows.h":
    ctypedef void* HMODULE
    ctypedef void* FARPROC
    ctypedef unsigned long DWORD

    HMODULE LoadLibraryA(const char* lpLibFileName) nogil
    FARPROC GetProcAddress(HMODULE hModule, const char* lpProcName) nogil
    int FreeLibrary(HMODULE hLibModule) nogil
    DWORD GetLastError() nogil


cdef int open_library(void** lib_handle, const char* path) except 1 nogil:
    """Opens a DLL and returns a handle for it via out parameter.

    Args:
        lib_handle (void**, out):
            The library handle, the result.
            If an error has occured, the dereferenced value is NULL.
    Returns:
        Positive number if something has gone wrong, '0' otherwise.
    """
    lib_handle[0] = <void*>LoadLibraryA(path)
    cdef DWORD error_code = 0
    if lib_handle[0] == NULL:
        error_code = GetLastError()
        with gil:
            raise RuntimeError(f"failed to LoadLibrary '{path.decode('utf-8')}': error code {error_code}")
    return 0

cdef int close_library(void* lib_handle) except 1 nogil:
    """Closes the given DLL.

    Args:
        lib_handle (void*, in):
            Handle to the library to close.
    Returns:
        Positive number if something has gone wrong, '0' otherwise.
    """
    if lib_handle == NULL:
        with gil:
            raise RuntimeError("handle is NULL")
    cdef int result = FreeLibrary(<HMODULE>lib_handle)
    cdef DWORD error_code = 0
    if result == 0:
        error_code = GetLastError()
        with gil:
            raise RuntimeError(f"failed to FreeLibrary: error code {error_code}")
    return 0

cdef int load_symbol(void** handle, void* lib_handle, const char* name) except 1 nogil:
    """Returns a symbol handle from an opened DLL via out parameter.

    Args:
        handle (void**, in):
            The symbol handle, the result.
            If an error has occured, the dereferenced value is NULL.
        lib_handle (void*, in):
            DLL handle.
        name (char*, in):
            Name of the symbol.
    Returns:
        Positive number if something has gone wrong, '0' otherwise.
    """
    handle[0] = <void*>GetProcAddress(<HMODULE>lib_handle, name)
    cdef DWORD error_code = 0
    if handle[0] == NULL:
        error_code = GetLastError()
        with gil:
            raise RuntimeError(f"failed to GetProcAddress '{name.decode('utf-8')}': error code {error_code}")
    return 0

cdef bint has_symbol(void* lib_handle, const char* name) noexcept nogil:
    """Probe whether a symbol is exported by an opened DLL.

    Non-raising counterpart to ``load_symbol`` — returns ``True`` if
    ``GetProcAddress`` resolves the symbol, ``False`` otherwise.
    Useful for feature detection against libraries that ship in two
    flavours.

    Args:
        lib_handle (void*, in):
            DLL handle (must be non-NULL — call ``open_library`` first).
        name (char*, in):
            Name of the symbol.
    Returns:
        True if the symbol resolves, False otherwise.
    """
    if lib_handle == NULL:
        return False
    cdef FARPROC sym = GetProcAddress(<HMODULE>lib_handle, name)
    return sym != NULL
