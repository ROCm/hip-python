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
    2. rocm_sdk.find_libraries() - TheRock Python package installation.
       For the LLVM toolchain (clang/LLVM), which rocm_sdk does not register,
       anchor on 'amdhip64' and resolve the sibling <core>/lib/llvm/lib, then
       fall back to rocm_sdk._devel.get_devel_root() (the rocm-sdk-devel tree).
    3. ROCM_PATH / ROCM_HOME environment variable - Traditional Linux install
       (LLVM and clang resolve under <rocm>/llvm/lib, versioned soname allowed)
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

    # Every tier below resolves into this single result. The conversion to the
    # bytes contract required by the Cython consumer (posixloader.open_library,
    # which takes a const char*) happens exactly once, at the end, via
    # os.fsencode - which accepts both str and Path. Centralizing the encode
    # removes the class of bug where an individual tier forgets the str()/Path
    # bridge (e.g. calling .encode() directly on a PosixPath from rocm_sdk).
    resolved = None

    # 1. Try bundled location first (for wheel-packaged libraries)
    if bundled_location is not None:
        bundled_path = bundled_location / lib_name
        if bundled_path.exists():
            resolved = bundled_path
    else:
        # Auto-detect: recursively search from rocm package root
        # paths.py is at rocm/bindings/util/paths.py
        # Package root is rocm/, so go up 3 levels: util -> bindings -> rocm
        package_root = Path(__file__).parent.parent.parent

        # Recursively search for library file in rocm package
        for lib_file in package_root.rglob(lib_name):
            if lib_file.is_file():
                resolved = lib_file
                break

    # 2b. LLVM toolchain libs (clang, LLVM) are NOT registered in
    #     rocm_sdk.ALL_LIBRARIES, so find_libraries(shortname) cannot find them
    #     (see ROCM_SDK_PACKAGING_BUG_REPORT.md). For TheRock/rocm_sdk wheel
    #     installs, anchor on a library that IS registered and shipped by
    #     rocm-sdk-core, then walk to the sibling llvm/lib directory.
    if resolved is None and shortname in ('LLVM', 'clang'):
        try:
            from rocm_sdk import find_libraries
            # Forward-compat: prefer a direct hit if these ever get registered.
            try:
                direct = find_libraries(shortname)
                if direct:
                    resolved = direct[0]
            except Exception:
                pass
            # Cheap tier first: anchor on core (today's toolchain home). This
            # avoids forcing the devel-root tarball expansion below when core
            # still carries the toolchain.
            anchor = find_libraries('amdhip64')
            if anchor:
                llvm_lib = Path(anchor[0]).parent / 'llvm' / 'lib'
                matches = sorted(llvm_lib.glob(f'{lib_name}*'))
                if matches:
                    return str(matches[0]).encode('utf-8')
        except Exception:
            pass  # rocm_sdk not installed / not a wheel install; fall through

        # 2c. Canonical (post-fix) home: the rocm-sdk-devel tree. Triggers a
        #     lazy _devel.tar expansion on first use, so it runs only after the
        #     cheap core anchor above misses. numba-hip needs the toolchain
        #     anyway, so requiring rocm[devel] in that future is acceptable.
        if resolved is None:
            try:
                from rocm_sdk._devel import get_devel_root
                llvm_lib = Path(get_devel_root()) / 'lib' / 'llvm' / 'lib'
                matches = sorted(llvm_lib.glob(f'{lib_name}*'))
                if matches:
                    resolved = matches[0]
            except ImportError:
                pass  # devel not installed; fall through

    # 2. Try rocm_sdk API (TheRock installation)
    #    This is the official recommended approach for ROCm 7.9+
    if resolved is None:
        try:
            from rocm_sdk import find_libraries
            paths = find_libraries(shortname)
            if paths and len(paths) > 0:
                resolved = paths[0]
        except ImportError:
            # rocm_sdk not installed; continue with fallback options
            pass

    # 3. Try ROCM_PATH / ROCM_HOME environment variables (Unix traditional install)
    #    On Windows, traditional HIP SDK installs DLLs to System32, not a ROCm directory
    if resolved is None and sys.platform not in ('win32', 'cygwin'):
        rocm_path = (
            os.environ.get('ROCM_PATH')
            or os.environ.get('ROCM_HOME')
            or '/opt/rocm'
        )

        # LLVM toolchain libs (LLVM, clang) live under llvm/lib.
        if shortname in ('LLVM', 'clang'):
            llvm_lib = Path(rocm_path) / 'llvm' / 'lib'
            lib_file = llvm_lib / lib_name
            if lib_file.exists():
                resolved = lib_file
            else:
                # ROCm often ships only a versioned soname (e.g. libclang.so.23.0git).
                matches = sorted(llvm_lib.glob(f'{lib_name}*'))
                if matches:
                    resolved = matches[0]

        # Standard location: lib subdirectory
        if resolved is None:
            lib_file = Path(rocm_path) / 'lib' / lib_name
            if lib_file.exists():
                resolved = lib_file

    # 4. Fall back to basename (rely on system loader)
    #    Windows: DLLs found via PATH (typically System32 for GPU drivers)
    #    macOS: Libraries found via DYLD_LIBRARY_PATH or standard paths
    #    Linux: Libraries found via LD_LIBRARY_PATH or standard paths
    if resolved is None:
        resolved = lib_name

    return os.fsencode(resolved)
