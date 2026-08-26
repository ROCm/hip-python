#!/usr/bin/env python3
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""In this example, we generate LLVM IR from HIP C++ code.

This time we use AMD COMGR infrastructure. This requires us to use the
`hiprtc_runtime.h` header file that HIPRTC is using internally, which the
`rocm.comgr.comgr.HIPRTC_RUNTIME_HEADER` variable reads from the ROCm
installation in use.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# Printing the bitcode as IR text goes through the LLVM C API, which the
# bindings load at first call rather than at import, so an absent library would
# surface as a failed call deep in the example. Ask the bindings instead of
# inspecting the platform: has_symbol covers every reason the library may be
# missing -- any build configured with HIP_PYTHON_BUNDLE_LIBLLVM=OFF, which is
# the default on Windows because ROCm ships no shared LLVM there and one has to
# be linked from the static archives.
from rocm.bindings.llvm.c import core as _llvmc_core

if not _llvmc_core.has_symbol("LLVMCreateMemoryBufferWithMemoryRange"):
    raise NotImplementedError(
        "This example needs a loadable shared LLVM behind the "
        "rocm.bindings.llvm.c bindings; none was found. ROCm ships no shared "
        "LLVM on Windows, where rocm-bindings-compiler bundles one only when "
        "built with HIP_PYTHON_BUNDLE_LIBLLVM=ON."
    )

# [literalinclude-begin]
import copy

from rocm import comgr
from rocm.bindings.llvm.c.bitreader import LLVMParseBitcode2
from rocm.bindings.llvm.c.core import (
    LLVMCreateMemoryBufferWithMemoryRange,
    LLVMDisposeMemoryBuffer,
    LLVMDisposeMessage,
    LLVMDisposeModule,
    LLVMPrintModuleToString,
)
from rocm.version import ROCM_VERSION_TUPLE


def llvm_check(status, message):
    if status != 0:
        msg_str = str(message)
        LLVMDisposeMessage(message)
        raise RuntimeError(f"{msg_str}")


class HipProgram:
    def __init__(self, name: str, arch: str, source: str):
        self.hip_source = source
        self.name = name
        self.llvm_bc_or_ir = None
        self.llvm_bc_or_ir_size = None
        self.log = None
        self.diagnostic = None
        self._compile_to_llvm_bc(arch)

    def _compile_to_llvm_bc(self, arch: str):
        # [literalinclude-comgr-compile-hip-to-bc-begin]
        (bc, log, diagnostic) = comgr.compile_hip_to_bc(
            source=self.hip_source,
            isa_name=f"amdgcn-amd-amdhsa--{arch}",
            hip_version_tuple=ROCM_VERSION_TUPLE,  # only same up to last entry
            logging=True,
            extra_opts=["-D__HIPCC_RTC__"],
        )
        # [literalinclude-comgr-compile-hip-to-bc-end]
        self.llvm_bc_or_ir = bc
        self.log = log
        self.diagnostic = diagnostic
        self.llvm_bc_or_ir_size = len(self.llvm_bc_or_ir)

    def get_llvm_ir(self):
        assert (
            self.llvm_bc_or_ir is not None
        ), "run '_compile_to_llvm_bc' first"
        buf = LLVMCreateMemoryBufferWithMemoryRange(
            self.llvm_bc_or_ir,
            self.llvm_bc_or_ir_size,
            "llvm-ir-buffer",
            0,
        )
        (status, mod) = LLVMParseBitcode2(buf)
        llvm_check(status, "failed to parse bitcode")
        ir = LLVMPrintModuleToString(mod)
        result = copy.deepcopy(bytes(ir))  # copies into buffer
        LLVMDisposeMessage(ir)
        LLVMDisposeModule(mod)
        LLVMDisposeMemoryBuffer(buf)
        return result


if __name__ in ("__test__", "__main__"):
    import textwrap

    # [literalinclude-comgr-runtime-header-begin]
    kernel_hip = textwrap.dedent(
        comgr.HIPRTC_RUNTIME_HEADER
        + """\
        __device__ static inline void scale(float arr[], float factor) {
            arr[threadIdx.x] *= fabs(factor);
        }
        """
    )
    # [literalinclude-comgr-runtime-header-end]

    arch = "gfx90a"
    kernel_prog = HipProgram("kernel", arch, kernel_hip)
    print(kernel_prog.get_llvm_ir().decode("utf-8"))
    print("ok")
