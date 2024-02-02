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

import rocm.clang.cindex as ci
from rocm.llvm.config.llvm_config import (
    LLVM_VERSION_MAJOR as _LLVM_VERSION_MAJOR,
    LLVM_VERSION_MINOR as _LLVM_VERSION_MINOR,
    LLVM_VERSION_PATCH as _LLVM_VERSION_PATCH,
)

_ROCM_PATH = "/opt/rocm"  # TODO make depend on numba config env var

ci.Config.set_library_path(f"{_ROCM_PATH}/llvm/lib")

from . import cparser as _cparser

_cparser.CParser.set_clang_res_dir(
    f"{_ROCM_PATH}/llvm/lib/clang/"
    + f"{_LLVM_VERSION_MAJOR}.{_LLVM_VERSION_MINOR}.{_LLVM_VERSION_PATCH}"
)

from .hipdevicelib import HIPDeviceLib as _HIPDeviceLib

stubs, typing_registry, impl_registry  = _HIPDeviceLib().create_stubs_decls_impls()

def get_llvm_bc(amdgpu_arch):
    """Return a bitcode library for the given AMD GPU architecture.

    Args:
        amdgpu_arch (`str`):
            An AMD GPU arch identifier such as `gfx90a` (MI200 series) or `gfx942` (MI300 series).
            Can also have target features appended that are separated via ":".
            These are stripped away where not needed.
    """
    return _HIPDeviceLib(amdgpu_arch).llvm_bc


# test = get_llvm_bc("gfx90a")
