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

"""In this example, we generate AMD HSA assembly from HIP C++ code.

Note that some necessary includes such as "hip/hip_runtime.h" are
prepended to the kernel by hipRTC / AMD COMGR internally. Hence, they
do not appear in the kernel source.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# [literalinclude-begin]

from rocm.version import HIP_VERSION_TUPLE
from rocm.bindings import hip, hiprtc
from rocm import comgr

compile_via_comgr = True

print(f"{compile_via_comgr=}")


def hip_check(call_result):
    if isinstance(call_result, tuple):
        err = call_result[0]
        result = call_result[1:]
        if len(result) == 1:
            result = result[0]
    else:
        # Single-output funcs (e.g. hipMemcpy after the with-nogil
        # codegen refactor) return the bare hipError_t enum, not a
        # 1-tuple. Treat that as an empty-result call.
        err = call_result
        result = ()
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hiprtc.hiprtcResult)
        and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result


class HipProgram:
    def __init__(self, name: str, arch: bytes, source: bytes):
        self.hip_source = source
        self.name = name.encode("utf-8")
        self.prog = None
        self.hsa = None
        self.hsa_size = None
        self.log = None
        self.diagnostic = None
        if compile_via_comgr:
            self._compile_to_hsa_via_comgr(arch)
        else:
            self._compile_to_hsa_via_hiprtc(arch)

    def _compile_to_hsa_via_comgr(self, arch: bytes):
        """Compile HIP C++ to HSA via AMD COMGR."""
        (
            self.hsa,
            self.log,
            self.diagnostic,
        ) = comgr.compile_hip_to_hsa(
            self.hip_source,
            b"amdgcn-amd-amdhsa--" + arch,
            HIP_VERSION_TUPLE,  # type: tuple[int,int,int]
            prepend_hiprtc_runtime_header=True,  # type: bool
        )
        self.hsa_size = len(self.hsa)

    def _compile_to_hsa_via_hiprtc(self, arch: bytes):
        """Compile HIP C++ to HSA via hipRTC.

        Note:
            This is not supported by hipRTC (yet)
            but could technically be supported by adding a
            similar compilation path as for the to-bitcode compilation.
            While the to-bitcode compilation looks
            for a flag `-fgpu-rdc`, the HSA

            Using this method will result in an exception being thrown
            by hip_check (state: ROCm 7.0.0)
        """
        self.prog = hip_check(
            hiprtc.hiprtcCreateProgram(self.hip_source, self.name, 0, [], [])
        )
        cflags = [b"--offload-arch=" + arch, b"-S"]
        (err,) = hiprtc.hiprtcCompileProgram(self.prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(self.prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(self.prog, log))
            raise RuntimeError(log.decode())
        self.hsa_size = hip_check(hiprtc.hiprtcGetBitcodeSize(self.prog))
        self.hsa = bytearray(self.hsa_size)
        hip_check(hiprtc.hiprtcGetBitcode(self.prog, self.hsa))

    def ___del__(self):
        if self.prog is not None:
            hip_check(hiprtc.hiprtcDestroyProgram(self.prog.createRef()))


if __name__ in ("__test__", "__main__"):
    import textwrap

    kernel_hip = textwrap.dedent(
        """\
        extern "C" __global__ void scale(float arr[], float factor) {
            arr[threadIdx.x] *= factor;
        }
        """
    ).encode("utf-8")

    props = hip_check(hip.hipGetDeviceProperties(0))
    arch = props.gcnArchName
    kernel_prog = HipProgram("kernel", arch, kernel_hip)
    print(kernel_prog.hsa.decode())
