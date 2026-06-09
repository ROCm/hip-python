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

"""Provides access to types and functions in the HIP device library.

Attributes:
    thestubs:
        TODO document 'stubs'
    unsupported_stubs:
        TODO document 'unsupported_stubs'
"""

import threading
from pathlib import Path

import rocm.bindings.clang.cindex as ci

# isort: off
from rocm.bindings.llvm.config.llvm_config import (
    LLVM_VERSION_MAJOR as _LLVM_VERSION_MAJOR,
    LLVM_VERSION_MINOR as _LLVM_VERSION_MINOR,
    LLVM_VERSION_PATCH as _LLVM_VERSION_PATCH,
)

# isort: on

from numba.hip import hipconfig as _hipconfig
from numba.hip.typing_lowering.registries import (
    impl_registry,
    typing_registry,
)
from numba.hip.util import fscache as _fscache

from . import cparser as _cparser
from . import hipdevicelib as _hipdevicelib
from .hipdevicelib import DEVICE_FUN_PREFIX
from .hipdevicelib import HIPDeviceLib as _HIPDeviceLib

_lock = threading.Lock()

_LLVM_VERSION_STRING = (
    f"{_LLVM_VERSION_MAJOR}.{_LLVM_VERSION_MINOR}.{_LLVM_VERSION_PATCH}"
)


def _highest_version_dir(clang_root: Path):
    """Return the highest-versioned subdir of a clang resource root, or None."""
    if clang_root.is_dir():
        versions = sorted(p for p in clang_root.iterdir() if p.is_dir())
        if versions:
            return str(versions[-1])  # e.g. '23' or '23.0.0'
    return None


def _resolve_clang_res_dir(libclang_file):
    """Resolve the clang resource dir.

    The resource dir is always a sibling of libclang: <dir>/clang/<version>.
    Deriving it from the resolved libclang makes it track the install method
    (traditional /opt/rocm, TheRock/rocm_sdk wheel, or env override). Falls
    back to the rocm-sdk-devel tree and then the traditional ROCm layout.
    """
    # Primary: sibling of the resolved libclang (covers core-today and the
    # likely post-fix case where libclang and its resource dir move together).
    if libclang_file:
        clang_root = Path(libclang_file).resolve().parent / "clang"
        res = _highest_version_dir(clang_root)
        if res:
            return res
    # Fallback A: the rocm-sdk-devel tree (covers the "split" case where the
    # resource dir lives in devel even though libclang was found elsewhere).
    try:
        from rocm_sdk._devel import get_devel_root

        res = _highest_version_dir(
            Path(get_devel_root()) / "lib" / "llvm" / "lib" / "clang"
        )
        if res:
            return res
    except Exception:
        pass
    # Fallback B: traditional /opt/rocm layout.
    return _hipconfig.get_rocm_path(
        (  # variant 1
            "llvm",
            "lib",
            "clang",
            f"{_LLVM_VERSION_MAJOR}.{_LLVM_VERSION_MINOR}.{_LLVM_VERSION_PATCH}",
        ),
        (  # variant 2
            "llvm",
            "lib",
            "clang",
            f"{_LLVM_VERSION_MAJOR}",
        ),
    )


def _setup_libclang():
    """Initialize libclang."""
    # `ci.Config` is process-global; once libclang has been loaded its
    # `set_library_*` setters raise. Guard against a second import of this
    # package (e.g. a shadow copy on sys.path) re-running setup and crashing.
    libclang_file = None
    if not ci.Config.loaded:
        if _hipconfig.LIBCLANG_FILE:
            # Highest priority: explicit file override (NUMBA_HIP_LIBCLANG_FILE).
            libclang_file = _hipconfig.LIBCLANG_FILE
            ci.conf.set_library_file(libclang_file)
        elif _hipconfig.LIBCLANG_PATH:
            # Next: explicit directory override (NUMBA_HIP_LIBCLANG_PATH).
            matches = sorted(Path(_hipconfig.LIBCLANG_PATH).glob("libclang.so*"))
            if matches:
                libclang_file = str(matches[0])
                ci.conf.set_library_file(libclang_file)
            else:
                ci.conf.set_library_path(_hipconfig.LIBCLANG_PATH)
        else:
            # Otherwise reuse the shared rocm-bindings resolver (handles
            # ROCM_PATH/ROCM_HOME, the rocm_sdk wheel anchor, and the
            # versioned libclang soname).
            from rocm.bindings.util.paths import get_library_path

            libclang_file = get_library_path("clang").decode("utf-8")
            ci.conf.set_library_file(libclang_file)
        _ = ci.conf.get_cindex_library()  # validate the binding loads

    _cparser.CParser.set_clang_res_dir(_resolve_clang_res_dir(libclang_file))


_setup_libclang()


def _create_stubs():

    all_stubs = _HIPDeviceLib().create_stubs_decls_impls(
        typing_registry, impl_registry
    )

    unsupported_stubs = {}
    thestubs = {}
    for name, stub in all_stubs.items():
        if stub.is_supported():
            thestubs[name] = (
                stub  # allows to easily add them to numba.hip globals() in __init__.py
            )
        else:
            unsupported_stubs[name] = stub
    return thestubs, unsupported_stubs


thestubs, unsupported_stubs = _create_stubs()
globals().update(thestubs)


def reload():
    """Reload the HIP device library.

    Main purpose of this routine is to allow
    reloading stubs and refilling the typing and impl
    registries after a user has registered custom
    extensions with the HIP device library.

    Note:
        Clears all caches, i.e., the per-instance cache
        of the HIPDeviceLib instances per architecture
        as well as the filesystem cache.

    Note (Implementation details):
        The hip device lib calls impl_registry.lower as follows:

        ``impl_registry.lower(stub, *argtys)(impl)```

        This routine is implemented as shown below

        ```
        def lower(self, stub, *argtys):
            # [...]
            self.functions.append((impl, stub, argtys))
        ```
    """
    global thestubs
    global unsupported_stubs
    from numba.hip.typing_lowering.registries import (
        impl_registry,
        typing_registry,
    )

    del globals()["unsupported_stubs"]
    for k, stub in globals()["thestubs"]:
        typing_registry.functions.remove(stub._template_)
        impl_registry.functions.remove(
            next(tup for tup in impl_registry.functions if tup[1] == stub)
        )
        del globals()[k]  # finally remove the stub from the globals

    _thestubs, _unsupported_stubs = _create_stubs()
    thestubs.clear()
    unsupported_stubs.clear()
    thestubs.update(_thestubs)
    unsupported_stubs.update(_unsupported_stubs)
    globals().update(thestubs)
    # reload the HIPDeviceLib input source and
    _hipdevicelib.HIPDeviceLib.reload()
    # finally clean the filesystem cache
    _fscache.clear_cache()


_HIPDEVICELIB = "hipdevicelib"
_EXT = "bc"


def get_llvm_bc(amdgpu_arch: str):
    """Returns LLVM BC for the given AMD GPU architecture.

    Note:
        If `numba.hip.hipconfig.USE_DEVICE_LIB_CACHE` is ``True``,
        this routine first tries to lookup a cached file for the
        given AMD GPU architecture, which it expects to be stored at location
        `os.path.join(tempfile.gettempdir(), "numba","hip")`.
        If there is no such file, the LLVM BC library is compiled
        from the HIP C++ source of the HIP device library. Before returning
        the result, it is stored into the aforementioned directory
        so that the next lookup (by a different process) will find it.

        The cache key includes both the architecture and LLVM version
        to prevent loading incompatible bitcode when switching ROCm versions.

    Args:
        amdgpu_arch (`str`):
            An AMD GPU arch identifier such as `gfx90a` (MI200 series) or `gfx942` (MI300 series).
            Can also have target features appended that are separated via ":".
            These are stripped away where not needed.
    """
    found_cached_file = False
    instance = _HIPDeviceLib(amdgpu_arch)
    if _hipconfig.USE_DEVICE_LIB_CACHE:
        # file system caching
        if instance._bitcode is None:  # ! uses hidden attribute '_bitcode'
            try:
                with _lock:
                    instance._bitcode = _fscache.read_cached_file(
                        amdgpu_arch,
                        prefix=_HIPDEVICELIB,
                        ext=_EXT,
                        version=_LLVM_VERSION_STRING,
                    )  # ! uses hidden attribute '_bitcode'
                found_cached_file = True
            except FileNotFoundError:
                pass
    # instance internally caches the IR too
    bc = instance.bitcode
    if not found_cached_file and _hipconfig.USE_DEVICE_LIB_CACHE:
        with _lock:
            _fscache.write_cached_file(
                bc,
                amdgpu_arch,
                prefix=_HIPDEVICELIB,
                ext=_EXT,
                version=_LLVM_VERSION_STRING,
            )
    return bc


def get_llvm_module(amdgpu_arch: str):
    """Returns the ROCm LLVM module derived from the HIP device lib.

    Note:
        If `numba.hip.hipconfig.USE_DEVICE_LIB_CACHE` is ``True``,
        this routine first tries to lookup a cached file for the
        given AMD GPU architecture, which it expects to be stored at location
        `os.path.join(tempfile.gettempdir(), "numba","hip")`.
        If there is no such file, the LLVM BC library is compiled
        from the HIP C++ source of the HIP device library. Before returning
        the result, it is stored into the aforementioned directory
        so that the next lookup (by a different process) will find it.

     Args:
        amdgpu_arch (`str`):
            An AMD GPU arch identifier such as `gfx90a` (MI200 series) or `gfx942` (MI300 series).
            Can also have target features appended that are separated via ":".
            These are stripped away where not needed.
    """
    _ = get_llvm_bc(
        amdgpu_arch
    )  # initializes/loads cached bitcode if not already done
    return _HIPDeviceLib(amdgpu_arch).module


__all__ = [
    "thestubs",
    "unsupported_stubs",
    "reload",
    "get_llvm_bc",
    "get_llvm_module",
    "DEVICE_FUN_PREFIX",
]
