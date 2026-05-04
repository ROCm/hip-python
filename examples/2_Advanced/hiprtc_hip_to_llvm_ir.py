#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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

"""In this example, we generate LLVM IR from HIP C++ code.

Note that some necessary includes such as "hip/hip_runtime.h" are
prepended to the kernel by HIPRTC internally. Hence, they
do not appear in the kernel source.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# [literalinclude-begin]
import copy

from rocm.bindings import hip, hiprtc
from rocm.bindings.llvm.c.bitreader import LLVMParseBitcode2
from rocm.bindings.llvm.c.core import (
    LLVMCreateMemoryBufferWithMemoryRange,
    LLVMDisposeMemoryBuffer,
    LLVMDisposeMessage,
    LLVMDisposeModule,
    LLVMPrintModuleToString,
)


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hiprtc.hiprtcResult)
        and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result


def llvm_check(status, message):
    if status != 0:
        msg_str = str(message)
        LLVMDisposeMessage(message)
        raise RuntimeError(f"{msg_str}")


class HipProgram:
    def __init__(self, name: str, arch: bytes, source: bytes):
        self.hip_source = source
        self.name = name.encode("utf-8")
        self.prog = None
        self.llvm_bc_or_ir = None
        self.llvm_bc_or_ir_size = None
        self._compile_to_llvm_bc(arch)

    def _compile_to_llvm_bc(self, arch: bytes):
        self.prog = hip_check(
            hiprtc.hiprtcCreateProgram(self.hip_source, self.name, 0, [], [])
        )
        cflags = [b"--offload-arch=" + arch, b"-fgpu-rdc"]
        (err,) = hiprtc.hiprtcCompileProgram(self.prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(self.prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(self.prog, log))
            raise RuntimeError(log.decode())
        self.llvm_bc_or_ir_size = hip_check(
            hiprtc.hiprtcGetBitcodeSize(self.prog)
        )
        self.llvm_bc_or_ir = bytearray(self.llvm_bc_or_ir_size)
        hip_check(hiprtc.hiprtcGetBitcode(self.prog, self.llvm_bc_or_ir))

    def get_llvm_ir(self):
        assert (
            self.llvm_bc_or_ir is not None
        ), "run '_compile_to_llvm_bc' first"
        buf = LLVMCreateMemoryBufferWithMemoryRange(
            self.llvm_bc_or_ir,
            self.llvm_bc_or_ir_size,
            b"llvm-ir-buffer",
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

    def __del__(self):
        if hasattr(self, 'prog') and self.prog is not None:
            try:
                hip_check(hiprtc.hiprtcDestroyProgram(self.prog.createRef()))
            except Exception:
                pass  # Suppress errors during cleanup


if __name__ in ("__test__", "__main__"):
    import textwrap

    kernel_hip = textwrap.dedent(
        """\
        extern "C" __global__ void scale(float arr[], float factor) {
            arr[threadIdx.x] *= factor;
        }
        """
    ).encode("utf-8")

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName
    kernel_prog = HipProgram("kernel", arch, kernel_hip)
    print(kernel_prog.get_llvm_ir().decode("utf-8"))
