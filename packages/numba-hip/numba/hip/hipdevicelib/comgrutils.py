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

__author__ = "Advanced Micro Devices, Inc."

"""AMD COMGR Utilities.
"""

from rocm.amd_comgr import amd_comgr as comgr

import llvmutils


def compile_hip_source_to_llvm(
    source: str,
    amdgpu_arch: str,
    hip_version_tuple: tuple,
    to_llvm_ir: bool = False,
    extra_opts: str = "",
    comgr_logging: bool = False,
):
    """Compiles a HIP C++ source file to LLVM bitcode or human-readable LLVM IR.

    Args:
        source: str: Contents of the HIP C++ source.
        amdgpu_arch (`str`): An AMD GPU arch identifier such as `gfx90a` (MI200 series) or `gfx942` (MI300 series).
        hip_version_tuple (`tuple`): A tuple of `int` values that contains HIP version major, minor, and patch.
        to_llvm_ir (`bool`): If the compilation result should be LLVM IR (versus LLVM BC). Defaults to `False`.
        extra_opts (`str`, optional): Additional opts to append to the compiler command. Defaults to `""`.
        comgr_logging (`bool`, optional): Enable AMD COMGR logging. Defaults to `False`.

    Returns:
        tuple: A triple consisting of LLVM BC/IR, the log or None, diagnostic information or None.
    """
    (
        llvm_bc_or_ir,
        log,
        diagnostic,
    ) = comgr.ext.compile_hip_to_bc(
        source=source,
        isa_name=f"amdgcn-amd-amdhsa--{amdgpu_arch}",
        hip_version_tuple=hip_version_tuple[:3],
        logging=comgr_logging,
        extra_opts=extra_opts,
    )
    if to_llvm_ir:
        llvm_bc_or_ir = llvmutils.convert_llvm_bc_to_ir(
            llvm_bc_or_ir, len(llvm_bc_or_ir)
        )
    return (llvm_bc_or_ir, log, diagnostic)
