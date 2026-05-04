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

"""In this example, we link a device code file that contains
a device kernel "__global__ void print_tid()" that has an unresolved call
to a function "__device__ void foo()" in its body with a second
device code file that contains the definition of device function "foo".

To make this work, both snippets need to be compiled with
the ``-fgpu-rdc`` option and the compilation results needs to
be added as `HIPRTC_JIT_INPUT_LLVM_BITCODE` type input to the link object.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# [literalinclude-begin]
from rocm.bindings import hip, hiprtc


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


class HiprtcProgram:
    def __init__(self, name: str, source: bytes):
        self.source = source
        self.name = name.encode("utf-8")
        self.prog = None
        self.llvm_bitcode = None
        self.llvm_bitcode_size = None

    def _get_arch(self) -> bytes:
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, 0))
        return props.gcnArchName

    def compile_to_llvm_bc(self):
        # [literalinclude-hiprtc-compile-rdc-begin]
        prog = hip_check(
            hiprtc.hiprtcCreateProgram(self.source, self.name, 0, [], [])
        )
        cflags = [b"--offload-arch=" + self._get_arch(), b"-fgpu-rdc"]
        (err,) = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
            raise RuntimeError(log.decode())
        bitcode_size = hip_check(hiprtc.hiprtcGetBitcodeSize(prog))
        bitcode = bytearray(bitcode_size)
        hip_check(hiprtc.hiprtcGetBitcode(prog, bitcode))
        # [literalinclude-hiprtc-compile-rdc-end]
        self.prog = prog
        self.llvm_bitcode_size = bitcode_size
        self.llvm_bitcode = bitcode

    def __del__(self):
        if hasattr(self, 'prog') and self.prog is not None:
            try:
                hip_check(hiprtc.hiprtcDestroyProgram(self.prog.createRef()))
            except Exception:
                pass  # Suppress errors during cleanup


class HiprtcLinker:
    def __init__(self):
        self.link_state = hip_check(hiprtc.hiprtcLinkCreate(0, None, None))
        self.completed = False
        self.code = None
        self.code_size = None

    def add_program(self, hiprtc_program):
        hip_check(
            hiprtc.hiprtcLinkAddData(
                self.link_state,
                hiprtc.hiprtcJITInputType.HIPRTC_JIT_INPUT_LLVM_BITCODE,
                hiprtc_program.llvm_bitcode,
                hiprtc_program.llvm_bitcode_size,
                hiprtc_program.name,
                0,  # size of the options
                None,  # Array of options applied to this input
                None,
            )
        )
        # Array of option values cast to void*

    def complete(self):
        self.code, self.code_size = hip_check(
            hiprtc.hiprtcLinkComplete(self.link_state)
        )

    def __del__(self):
        if hasattr(self, 'link_state') and self.link_state is not None:
            try:
                hip_check(hiprtc.hiprtcLinkDestroy(self.link_state))
            except Exception:
                pass  # Suppress errors during cleanup


if __name__ in ("__test__", "__main__"):
    import textwrap

    # [literalinclude-kernel-sources-begin]
    device_fun_src = textwrap.dedent(
        """\
        __device__ void foo() {
            printf("tid: %d\\n", (int) threadIdx.x);
        }
        """
    ).encode("utf-8")

    kernel_src = textwrap.dedent(
        """\
        __device__ void foo(); // prototype

        extern "C" __global__ void print_tid() {
            foo();
        }
        """
    ).encode("utf-8")
    # [literalinclude-kernel-sources-end]

    # [literalinclude-hiprtc-link-flow-begin]
    linker = HiprtcLinker()
    kernel_prog = HiprtcProgram("kernel", kernel_src)
    device_fun_prog = HiprtcProgram("device_fun", device_fun_src)
    kernel_prog.compile_to_llvm_bc()
    device_fun_prog.compile_to_llvm_bc()
    linker.add_program(kernel_prog)
    linker.add_program(device_fun_prog)
    linker.complete()
    module = hip_check(hip.hipModuleLoadData(linker.code))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"print_tid"))
    # [literalinclude-hiprtc-link-flow-end]
    #
    hip_check(
        hip.hipModuleLaunchKernel(
            kernel,
            *(1, 1, 1),  # grid
            *(32, 1, 1),  # block
            sharedMemBytes=0,
            stream=None,
            kernelParams=None,
            extra=None,
        )
    )

    hip_check(hip.hipDeviceSynchronize())
    hip_check(hip.hipModuleUnload(module))

    print("ok")
