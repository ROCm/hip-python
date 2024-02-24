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

"""Provides access to types and functions in the HIP device library.

Attributes:
    stubs:
        TODO document 'stubs'
    unsupported_stubs:
        TODO document 'unsupported_stubs'
    typing_registry (`numba.core.typing.templates.Registry`):
        A registry of typing declarations. The registry stores such declarations
        for functions, attributes and globals.
    impl_registry (`numba.core.imputils.Registry`):
        A registry of function and attribute implementations.
"""

import rocm.clang.cindex as ci
from rocm.llvm.config.llvm_config import (
    LLVM_VERSION_MAJOR as _LLVM_VERSION_MAJOR,
    LLVM_VERSION_MINOR as _LLVM_VERSION_MINOR,
    LLVM_VERSION_PATCH as _LLVM_VERSION_PATCH,
)

from numba.hip import rocmpaths as _rocmpaths

ci.Config.set_library_path(_rocmpaths.get_rocm_path("llvm", "lib"))

from . import cparser as _cparser

_cparser.CParser.set_clang_res_dir(
    _rocmpaths.get_rocm_path(
        "llvm",
        "lib",
        "clang",
        f"{_LLVM_VERSION_MAJOR}.{_LLVM_VERSION_MINOR}.{_LLVM_VERSION_PATCH}",
    )
)

from .hipdevicelib import HIPDeviceLib as _HIPDeviceLib, DEVICE_FUN_PREFIX

from numba.hip.typing_lowering.registries import (
    typing_registry,
    impl_registry,
)

_all_stubs = _HIPDeviceLib().create_stubs_decls_impls(typing_registry, impl_registry)

unsupported_stubs = {}
thestubs = {}
for name, stub in _all_stubs.items():
    if stub.is_supported():
        thestubs[name] = (
            stub  # allows to easily add them to numba.hip globals() in __init__.py
        )
        globals()[
            name
        ] = stub  # allows to import them via from 'numba.hip.hipdevicelib import ...'
    else:
        unsupported_stubs[name] = stub


def get_llvm_bc(amdgpu_arch):
    """Return a bitcode library for the given AMD GPU architecture.

    Args:
        amdgpu_arch (`str`):
            An AMD GPU arch identifier such as `gfx90a` (MI200 series) or `gfx942` (MI300 series).
            Can also have target features appended that are separated via ":".
            These are stripped away where not needed.
    """
    return _HIPDeviceLib(amdgpu_arch).llvm_bc


__all__ = [
    "thestubs",
    "unsupported_stubs",
    "typing_registry",
    "impl_registry",
    "get_llvm_bc",
    "DEVICE_FUN_PREFIX",
]

# test = get_llvm_bc("gfx90a")
