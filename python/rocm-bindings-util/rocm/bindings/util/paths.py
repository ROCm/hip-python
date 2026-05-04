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
ROCm library path resolution utilities.

This module provides cross-platform library path discovery for ROCm libraries,
supporting both traditional installations (/opt/rocm) and TheRock/rocm_sdk
package-based installations.

The get_library_path() function is designed for lazy evaluation - it should be
called when a library is first accessed, not at module import time, to preserve
the lazy loading behavior of the ROCm bindings.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

import sys
import os
from pathlib import Path
from typing import Optional


def get_library_path(shortname: str, bundled_location: Optional[Path] = None) -> bytes:
    """
    Get platform-appropriate library path for ROCm library (lazy evaluation).

    This function uses a fallback chain to locate ROCm libraries across different
    installation methods and platforms:

    1. Bundled location - Recursively search rocm package for library
    2. rocm_sdk.find_libraries() - TheRock Python package installation
    3. ROCM_PATH environment variable - Traditional Linux installation
    4. Basename fallback - Rely on system loader (PATH on Windows, LD_LIBRARY_PATH on Linux)

    This is designed to be called lazily when a library is first accessed,
    preserving the lazy loading behavior of the bindings.

    Args:
        shortname: Library short name without platform-specific prefix/suffix
                   Examples: 'amdhip64', 'hiprtc', 'hipblas', 'hipsolver', 'LLVM'
        bundled_location: Optional path to bundled library directory.
                         If None, recursively searches from rocm package root.

    Returns:
        Absolute path to library as bytes, or basename if not found.
        The returned bytes can be passed directly to open_library().

    Platform-specific behavior:
        Windows:
            - Traditional HIP SDK: DLLs are in C:\\Windows\\System32 (installed by GPU driver)
              We return basename and rely on Windows DLL search path
            - rocm_sdk (TheRock): DLLs are in site-packages\\_rocm_sdk_core\\bin
              The rocm_sdk.find_libraries() API returns the full path

        macOS (Darwin):
            - Traditional: Libraries in /opt/rocm/lib or ROCM_PATH/lib (.dylib extension)
            - rocm_sdk (TheRock): Libraries in site-packages/_rocm_sdk_core/lib
              The rocm_sdk.find_libraries() API returns the full path

        Linux:
            - Traditional: Libraries in /opt/rocm/lib or ROCM_PATH/lib (.so extension)
            - rocm_sdk (TheRock): Libraries in site-packages/_rocm_sdk_core/lib
              The rocm_sdk.find_libraries() API returns the full path

    Examples:
        >>> # Get path for HIP runtime
        >>> path = get_library_path('amdhip64')
        >>> # path might be:
        >>> # - b'/path/to/site-packages/_rocm_sdk_core/lib/libamdhip64.so' (rocm_sdk on Linux)
        >>> # - b'/path/to/site-packages/_rocm_sdk_core/lib/libamdhip64.dylib' (rocm_sdk on macOS)
        >>> # - b'C:\\\\Users\\\\...\\\\site-packages\\\\_rocm_sdk_core\\\\bin\\\\amdhip64.dll' (rocm_sdk on Windows)
        >>> # - b'/opt/rocm/lib/libamdhip64.so' (traditional Linux)
        >>> # - b'/opt/rocm/lib/libamdhip64.dylib' (traditional macOS)
        >>> # - b'amdhip64.dll' (traditional Windows, relies on System32)
    """
    # Determine library extension based on platform
    if sys.platform == 'win32':
        lib_ext = 'dll'
        lib_prefix = ''
    elif sys.platform == 'darwin':
        lib_ext = 'dylib'
        lib_prefix = 'lib'
    else:
        lib_ext = 'so'
        lib_prefix = 'lib'

    lib_name = f'{lib_prefix}{shortname}.{lib_ext}'

    # 1. Try bundled location first (for wheel-packaged libraries)
    if bundled_location is not None:
        bundled_path = bundled_location / lib_name
        if bundled_path.exists():
            return str(bundled_path).encode('utf-8')
    else:
        # Auto-detect: recursively search from rocm package root
        # paths.py is at rocm/bindings/util/paths.py
        # Package root is rocm/, so go up 3 levels: util -> bindings -> rocm
        package_root = Path(__file__).parent.parent.parent

        # Recursively search for library file in rocm package
        for lib_file in package_root.rglob(lib_name):
            if lib_file.is_file():
                return str(lib_file).encode('utf-8')

    # 2. Try rocm_sdk API (TheRock installation)
    #    This is the official recommended approach for ROCm 7.9+
    try:
        from rocm_sdk import find_libraries
        paths = find_libraries(shortname)
        if paths and len(paths) > 0:
            return paths[0].encode('utf-8')
    except (ImportError, Exception):
        # rocm_sdk not installed or find_libraries failed
        # Continue with fallback options
        pass

    # 3. Try ROCM_PATH environment variable (Unix traditional install)
    #    On Windows, traditional HIP SDK installs DLLs to System32, not a ROCm directory
    if sys.platform not in ('win32', 'cygwin'):
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        lib_name = f'{lib_prefix}{shortname}.{lib_ext}'

        # Special case for LLVM: check llvm/lib subdirectory
        if shortname == 'LLVM':
            lib_file = Path(rocm_path) / 'llvm' / 'lib' / lib_name
            if lib_file.exists():
                return str(lib_file).encode('utf-8')

        # Standard location: lib subdirectory
        lib_file = Path(rocm_path) / 'lib' / lib_name
        if lib_file.exists():
            return str(lib_file).encode('utf-8')

    # 4. Fall back to basename (rely on system loader)
    #    Windows: DLLs found via PATH (typically System32 for GPU drivers)
    #    macOS: Libraries found via DYLD_LIBRARY_PATH or standard paths
    #    Linux: Libraries found via LD_LIBRARY_PATH or standard paths
    return f'{lib_prefix}{shortname}.{lib_ext}'.encode('utf-8')
