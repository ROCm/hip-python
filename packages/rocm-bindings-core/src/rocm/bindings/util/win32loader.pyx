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
    ctypedef void* HANDLE
    ctypedef void* FARPROC
    ctypedef unsigned long DWORD

    HMODULE LoadLibraryA(const char* lpLibFileName) nogil
    HMODULE LoadLibraryExA(const char* lpLibFileName, HANDLE hFile, DWORD dwFlags) nogil
    FARPROC GetProcAddress(HMODULE hModule, const char* lpProcName) nogil
    int FreeLibrary(HMODULE hLibModule) nogil
    DWORD GetLastError() nogil


cdef enum:
    # Spelled out rather than taken from windows.h, where they appear only for a
    # high enough _WIN32_WINNT.
    #
    # DLL_LOAD_DIR adds the directory of the DLL being opened, which is where a
    # ROCm library's siblings are. DEFAULT_DIRS adds the system directories and
    # any directory registered with AddDllDirectory, which is how a dependency in
    # a *different* tree is reached: the rocm_sdk wheels put the HIP runtime and
    # hiprtc in the core tree while rocFFT and friends live in the libraries tree,
    # and paths.py registers both. Note that DLL_LOAD_DIR is inert on its own --
    # these flags only take effect in combination.
    _LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
    _LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000

    # Character codes, spelled out because ord() needs the GIL.
    _CH_SLASH = 47      # /
    _CH_COLON = 58      # :
    _CH_BACKSLASH = 92  # \


cdef bint _is_absolute(const char* path) noexcept nogil:
    """True for a drive-qualified path (C:\\...) or a UNC path (\\\\host\\share)."""
    if path[0] == 0:
        return False
    # Reading path[1] is safe once path[0] is not the terminator, and path[2]
    # likewise once path[1] is not.
    if path[1] == _CH_COLON:
        return path[2] == _CH_BACKSLASH or path[2] == _CH_SLASH
    return path[0] == _CH_BACKSLASH and path[1] == _CH_BACKSLASH


cdef int open_library(void** lib_handle, const char* path) except 1 nogil:
    """Opens a DLL and returns a handle for it via out parameter.

    Args:
        lib_handle (void**, out):
            The library handle, the result.
            If an error has occured, the dereferenced value is NULL.
    Returns:
        Positive number if something has gone wrong, '0' otherwise.
    """
    # A ROCm DLL depends on other ROCm DLLs, and naming this one by absolute path
    # does not tell Windows where to look for those: the default search order
    # covers the *process* directory, the system directories and PATH, so the
    # dependencies come up missing (error 126) even though the file named here was
    # found. Passing the search-order flags instead makes a ROCm installation
    # self-sufficient, without requiring its bin directory on PATH.
    #
    # The flags are meaningful only for an absolute path, and get_library_path
    # does return a bare DLL name for a driver-installed ROCm that is expected to
    # be found in System32, so they are applied only where they apply.
    #
    # Unlike the flags, PATH is not consulted in that search order, so a load that
    # only ever worked because of PATH would start failing. The plain call is kept
    # as a fallback for exactly that case.
    cdef DWORD error_code = 0
    if _is_absolute(path):
        lib_handle[0] = <void*>LoadLibraryExA(
            path,
            NULL,
            _LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | _LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
        if lib_handle[0] == NULL:
            lib_handle[0] = <void*>LoadLibraryA(path)
    else:
        lib_handle[0] = <void*>LoadLibraryA(path)
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
