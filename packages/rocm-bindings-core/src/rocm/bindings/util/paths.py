# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

import os
import sys
from pathlib import Path
from typing import Iterator, Optional, Sequence


def _windows_dll_candidates(shortname: str) -> Iterator[str]:
    """Yield the filenames ROCm may use for ``shortname`` on Windows.

    Windows ROCm does not follow the flat ``<shortname>.dll`` rule that the
    ``lib<shortname>.so`` convention maps onto elsewhere. Across the pip
    rocm-sdk tree, the TheRock tarball and the DLLs the GPU driver installs into
    System32, three shapes occur:

      * most libraries are plain -- ``hipblas.dll``, ``amd_comgr.dll``;
      * some keep the Unix ``lib`` prefix -- ``libhipblaslt.dll``, ``libclang.dll``;
      * the HIP runtime and hipRTC carry the ROCm version -- ``amdhip64_7.dll``,
        ``hiprtc0714.dll``.

    The version-bearing forms are derived from ``rocm.version`` rather than
    matched by wildcard, so they name the ROCm release these bindings were
    generated against. That matters in System32, where several majors coexist
    (``amdhip64_6.dll`` beside ``amdhip64_7.dll``) and loading the newest one
    would silently pair the bindings with the wrong runtime.

    Candidates are yielded most-likely first; the shapes that do not apply to a
    given library simply never exist on disk, so no per-library table is needed.
    """
    yield f"{shortname}.dll"
    yield f"lib{shortname}.dll"
    try:
        from rocm.version import ROCM_VERSION_TUPLE
    except ImportError:
        return
    major, minor = ROCM_VERSION_TUPLE[0], ROCM_VERSION_TUPLE[1]
    yield f"{shortname}_{major}.dll"
    yield f"{shortname}{major:02d}{minor:02d}.dll"


# Libraries rocm_sdk leaves out of ALL_LIBRARIES, each mapped to one it does
# register plus the path from that library's directory to where it sits. A tree
# install needs no entry here, because ROCM_PATH names the directory to scan;
# only a wheel install does, where find_libraries is the only thing that knows
# which tree a library landed in and it refuses every name outside its registry.
#
# hiprtc-builtins holds the header hipRTC prepends to every runtime-compiled
# translation unit (see rocm.comgr.hiprtc_header) and is installed beside
# hipRTC itself.
_ANCHORED_SHORTNAMES = {
    "hiprtc-builtins": ("hiprtc", ()),
}


# Handles returned by os.add_dll_directory. They have to be held: dropping one
# takes its directory back off the search path.
_dll_directory_handles = []
_dll_directories_registered = False


def _register_rocm_sdk_dll_directories() -> None:
    """Put every rocm_sdk tree's library directory on the DLL search path.

    The wheels split ROCm across trees -- the HIP runtime and hipRTC in
    rocm-sdk-core, rocFFT and the other maths libraries in rocm-sdk-libraries --
    and a library in one tree imports libraries from another. Windows satisfies
    those imports by searching directories, not by reusing the path its dependent
    was opened with, so the other trees have to be named ahead of the load or it
    fails with 'module not found' naming the library that *was* found. Only
    Windows needs this; the ELF loader has the sonames' directories from DT_RUNPATH.

    Registering directories does nothing on its own: it pairs with the
    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS that win32loader.open_library asks for.
    """
    global _dll_directories_registered
    if _dll_directories_registered:
        return
    _dll_directories_registered = True

    try:
        from rocm_sdk import find_libraries
        from rocm_sdk._dist_info import ALL_LIBRARIES
    except ImportError:
        return  # not a wheel install; nothing to register

    seen = set()
    for shortname in ALL_LIBRARIES:
        try:
            found = find_libraries(shortname)
        except Exception:
            continue  # the package providing it is not installed
        for path in found:
            directory = str(Path(path).parent)
            if directory in seen:
                continue
            seen.add(directory)
            try:
                _dll_directory_handles.append(os.add_dll_directory(directory))
            except OSError:
                pass  # vanished between find_libraries and here


def _first_existing(
    directories: Sequence[Path], filenames: Sequence[str]
) -> Optional[Path]:
    """First ``directory/filename`` that exists, filename-major.

    Ordered so that a better-matching name in a later directory beats a
    worse-matching one in an earlier directory.
    """
    for filename in filenames:
        for directory in directories:
            candidate = directory / filename
            if candidate.exists():
                return candidate
    return None


def _highest_version_dir(clang_root: Path) -> Optional[str]:
    """Highest-versioned subdirectory of a clang resource root, or None."""
    if clang_root.is_dir():
        versions = sorted(p for p in clang_root.iterdir() if p.is_dir())
        if versions:
            return str(versions[-1])  # e.g. '23' or '23.0.0'
    return None


def get_clang_resource_dir(
    libclang_file: Optional[str] = None,
) -> Optional[str]:
    """Directory holding clang's own headers, as '-resource-dir' expects it.

    Anything that parses HIP or C++ sources through libclang needs this
    directory, and where it sits depends on the installation, so the same
    layout knowledge get_library_path carries applies here.

    The resource directory belongs to a specific libclang and normally sits
    beside it, at <dir>/clang/<version>. That holds wherever shared libraries
    and their data share a directory, which is every platform but Windows:
    there the DLLs live in a bin directory while the resource directory stays
    in the sibling lib, so both are searched.

    Args:
        libclang_file: The libclang to resolve the directory for. Defaults to
                       the one get_library_path finds.

    Returns:
        The resource directory, or None if no installation could be found.
    """
    if libclang_file is None:
        libclang_file = os.fsdecode(get_library_path("clang"))

    roots = []
    libclang_path = Path(libclang_file)
    if libclang_path.is_absolute() and libclang_path.exists():
        libdir = libclang_path.resolve().parent
        roots.append(libdir / "clang")
        # Windows: <llvm>/bin/libclang.dll, resource dir in <llvm>/lib/clang.
        roots.append(libdir.parent / "lib" / "clang")

    try:
        from rocm_sdk._devel import get_devel_root

        roots.append(Path(get_devel_root()) / "lib" / "llvm" / "lib" / "clang")
    except ImportError:
        pass  # devel tree not installed

    rocm_path = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
    if rocm_path is None and sys.platform not in ("win32", "cygwin"):
        rocm_path = "/opt/rocm"
    if rocm_path:
        roots.append(Path(rocm_path) / "llvm" / "lib" / "clang")
        roots.append(Path(rocm_path) / "lib" / "llvm" / "lib" / "clang")

    for root in roots:
        resource_dir = _highest_version_dir(root)
        if resource_dir:
            return resource_dir
    return None


def get_library_path(
    shortname: str, bundled_location: Optional[Path] = None
) -> bytes:
    """
    Get platform-appropriate library path for ROCm library (lazy evaluation).

    This function uses a fallback chain to locate ROCm libraries across different
    installation methods and platforms:

    1. Bundled location - Recursively search rocm package for library
    2. rocm_sdk.find_libraries() - TheRock Python package installation.
       For the LLVM toolchain (clang/LLVM), which rocm_sdk does not register,
       anchor on 'amdhip64' and resolve the sibling <core>/lib/llvm/lib, then
       fall back to rocm_sdk._devel.get_devel_root() (the rocm-sdk-devel tree).
    3. ROCM_PATH / ROCM_HOME environment variable - Traditional install.
       On Unix, LLVM and clang resolve under <rocm>/llvm/lib and a versioned
       soname is allowed. On Windows the DLLs live in <rocm>/bin (LLVM in
       <rocm>/lib/llvm/bin) under names that are not simply <shortname>.dll,
       so the tree is scanned for the candidates _windows_dll_candidates lists.
    4. Basename fallback - Rely on system loader (LD_LIBRARY_PATH on Linux,
       DYLD_LIBRARY_PATH on macOS). On Windows PATH is walked here instead,
       because the bare name the loader would be given is often not the name
       on disk.

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
            - DLL names differ from the other platforms: the HIP runtime and
              hipRTC are version-suffixed (amdhip64_7.dll, hiprtc0714.dll) and a
              few libraries keep the Unix 'lib' prefix (libhipblaslt.dll). See
              _windows_dll_candidates.
            - Traditional HIP SDK: DLLs are in C:\\Windows\\System32 (installed
              by the GPU driver), found by walking PATH
            - Tarball / TheRock unpacked: DLLs are in <ROCM_PATH>\\bin
            - rocm_sdk (TheRock): DLLs are in site-packages\\_rocm_sdk_core\\bin
              The rocm_sdk.find_libraries() API returns the full path, already
              carrying the correct Windows name

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
    if sys.platform == "win32":
        lib_ext = "dll"
        lib_prefix = ""
    elif sys.platform == "darwin":
        lib_ext = "dylib"
        lib_prefix = "lib"
    else:
        lib_ext = "so"
        lib_prefix = "lib"

    lib_name = f"{lib_prefix}{shortname}.{lib_ext}"

    if sys.platform == "win32":
        _register_rocm_sdk_dll_directories()

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
    if resolved is None and shortname in ("LLVM", "clang"):
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
            anchor = find_libraries("amdhip64")
            if anchor:
                llvm_lib = Path(anchor[0]).parent / "llvm" / "lib"
                matches = sorted(llvm_lib.glob(f"{lib_name}*"))
                if matches:
                    return str(matches[0]).encode("utf-8")
        except Exception:
            pass  # rocm_sdk not installed / not a wheel install; fall through

        # 2c. Canonical (post-fix) home: the rocm-sdk-devel tree. Triggers a
        #     lazy _devel.tar expansion on first use, so it runs only after the
        #     cheap core anchor above misses. numba-hip needs the toolchain
        #     anyway, so requiring rocm[devel] in that future is acceptable.
        if resolved is None:
            try:
                from rocm_sdk._devel import get_devel_root

                llvm_lib = Path(get_devel_root()) / "lib" / "llvm" / "lib"
                matches = sorted(llvm_lib.glob(f"{lib_name}*"))
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

    # 2d. Libraries outside rocm_sdk's registry that are not part of the LLVM
    #     toolchain: resolve a library that IS registered and look next to it.
    #     Same packaging gap as LLVM/clang above, but these need no separate
    #     tree, so one anchor is enough.
    if resolved is None and shortname in _ANCHORED_SHORTNAMES:
        anchor_shortname, subdir = _ANCHORED_SHORTNAMES[shortname]
        anchor = Path(os.fsdecode(get_library_path(anchor_shortname)))
        if anchor.is_absolute() and anchor.exists():
            directory = anchor.parent.joinpath(*subdir)
            if sys.platform == "win32":
                names = list(_windows_dll_candidates(shortname))
            else:
                names = [lib_name]
            resolved = _first_existing([directory], names)
            if resolved is None:
                # Unix ships versioned sonames alongside the bare one.
                matches = sorted(directory.glob(f"{lib_name}*"))
                if matches:
                    resolved = matches[0]

    # 3w. Windows ROCm tree named by ROCM_PATH / ROCM_HOME. Its DLLs live in
    #     bin/ (with the LLVM toolchain in its own nested bin/) rather than
    #     lib/, and are not necessarily named <shortname>.dll -- see
    #     _windows_dll_candidates. Scanning the tree resolves the real filename
    #     without having to trust a naming rule, and one tree only ever holds a
    #     single ROCm release, so the match is unambiguous.
    if resolved is None and sys.platform in ("win32", "cygwin"):
        rocm_path = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
        if rocm_path:
            resolved = _first_existing(
                [
                    Path(rocm_path) / "bin",
                    Path(rocm_path) / "lib" / "llvm" / "bin",
                ],
                list(_windows_dll_candidates(shortname)),
            )

    # 3. Try ROCM_PATH / ROCM_HOME environment variables (Unix traditional install)
    if resolved is None and sys.platform not in ("win32", "cygwin"):
        rocm_path = (
            os.environ.get("ROCM_PATH")
            or os.environ.get("ROCM_HOME")
            or "/opt/rocm"
        )

        # LLVM toolchain libs (LLVM, clang) live under llvm/lib.
        if shortname in ("LLVM", "clang"):
            llvm_lib = Path(rocm_path) / "llvm" / "lib"
            lib_file = llvm_lib / lib_name
            if lib_file.exists():
                resolved = lib_file
            else:
                # ROCm often ships only a versioned soname (e.g. libclang.so.23.0git).
                matches = sorted(llvm_lib.glob(f"{lib_name}*"))
                if matches:
                    resolved = matches[0]

        # Standard location: lib subdirectory
        if resolved is None:
            lib_file = Path(rocm_path) / "lib" / lib_name
            if lib_file.exists():
                resolved = lib_file

    # 4w. Windows without a ROCm tree to inspect -- a driver-only install, whose
    #     DLLs land in System32. Handing the loader a bare name only works if we
    #     guess that name right, and for the HIP runtime we would not:
    #     System32 holds amdhip64_7.dll, never amdhip64.dll. So walk PATH (which
    #     includes System32) ourselves and take the first candidate that exists,
    #     which resolves the name and the directory in one pass.
    if resolved is None and sys.platform in ("win32", "cygwin"):
        search_dirs = [
            Path(entry)
            for entry in os.environ.get("PATH", "").split(os.pathsep)
            if entry
        ]
        resolved = _first_existing(
            search_dirs, list(_windows_dll_candidates(shortname))
        )

    # 4. Fall back to basename (rely on system loader)
    #    Windows: DLLs found via PATH (typically System32 for GPU drivers)
    #    macOS: Libraries found via DYLD_LIBRARY_PATH or standard paths
    #    Linux: Libraries found via LD_LIBRARY_PATH or standard paths
    if resolved is None:
        resolved = lib_name

    return os.fsencode(resolved)
