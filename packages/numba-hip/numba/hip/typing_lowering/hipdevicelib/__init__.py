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
import logging

_lock = threading.Lock()
_log = logging.getLogger(__name__)

import rocm.clang.cindex as ci
from rocm.llvm.config.llvm_config import (
    LLVM_VERSION_MAJOR as _LLVM_VERSION_MAJOR,
    LLVM_VERSION_MINOR as _LLVM_VERSION_MINOR,
    LLVM_VERSION_PATCH as _LLVM_VERSION_PATCH,
)

from numba.hip import hipconfig as _hipconfig
from numba.hip.util import fscache as _fscache

ci.Config.set_library_path(_hipconfig.get_rocm_path("llvm", "lib"))

from . import cparser as _cparser

_cparser.CParser.set_clang_res_dir(
    _hipconfig.get_rocm_path(
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
)

from .hipdevicelib import DEVICE_FUN_PREFIX
from .hipdevicelib import HIPDeviceLib as _HIPDeviceLib


def _create_stubs():

    from numba.hip.typing_lowering.registries import (
        typing_registry,
        impl_registry,
    )

    all_stubs = _HIPDeviceLib().create_stubs_decls_impls(typing_registry, impl_registry)

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
        typing_registry,
        impl_registry,
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
    hipdevicelib.HIPDeviceLib.reload()
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
        if instance._bitcode == None:  # ! uses hidden attribute '_bitcode'
            try:
                with _lock:
                    instance._bitcode = _fscache.read_cached_file(
                        amdgpu_arch,
                        prefix=_HIPDEVICELIB,
                        ext=_EXT,
                    )  # ! uses hidden attribute '_bitcode'
                found_cached_file = True
            except FileNotFoundError:
                pass
    # instance internally caches the IR too
    bc = instance.bitcode
    if not found_cached_file and _hipconfig.USE_DEVICE_LIB_CACHE:
        with _lock:
            _fscache.write_cached_file(bc, amdgpu_arch, prefix=_HIPDEVICELIB, ext=_EXT)
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
    _ = get_llvm_bc(amdgpu_arch)  # initializes/loads cached bitcode if not already done
    return _HIPDeviceLib(amdgpu_arch).module


__all__ = [
    "thestubs",
    "unsupported_stubs",
    "reload",
    "get_llvm_bc",
    "get_llvm_module",
    "DEVICE_FUN_PREFIX",
]
