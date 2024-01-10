#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

"""In this example, we link a device code snippet that contains
a HIP C++ device kernel ``__global__ void scale(float[],float)`` that has an unresolved call
to a function ``__device__ void scale_op(float[],float)`` in its body with a
LLVM IR snippet that contains the definition of that device function ``scale_op``.
(Note that we could have also used LLVM BC (bitcode) instead of human-readable LLVM IR.)

To make this work, the HIP C++ snippet needs to be compiled with
the ``-fgpu-rdc`` option and all compilation results needs to
be added as ``HIPRTC_JIT_INPUT_LLVM_BITCODE`` type input to the HIPRTC link object.

Note that the LLVM IR in this example is target dependent.
Therefore, this example can currently only be run with ``gfx90a`` (MI200 series).
The example will quit with an error if other architectures are used.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# [literalinclude-begin]
import array
import ctypes
import math
import sys

from hip import hip, hiprtc


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


class LLVMIRProgram:
    def __init__(self, name: str, source: bytes):
        self.name = name.encode("utf-8")
        self.llvm_bc_or_ir = source
        self.llvm_bc_or_ir_size = len(source)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class HipProgram:
    def __init__(self, name: str, arch: str, source: bytes):
        self.hip_source = source
        self.name = name.encode("utf-8")
        self.prog = None
        self.llvm_bc_or_ir = None
        self.llvm_bc_or_ir_size = None
        self._compile_to_llvm_bc(arch)

    def _compile_to_llvm_bc(self, arch: str):
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
        self.llvm_bc_or_ir_size = hip_check(hiprtc.hiprtcGetBitcodeSize(self.prog))
        self.llvm_bc_or_ir = bytearray(self.llvm_bc_or_ir_size)
        hip_check(hiprtc.hiprtcGetBitcode(self.prog, self.llvm_bc_or_ir))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.prog != None:
            hip_check(hiprtc.hiprtcDestroyProgram(self.prog.createRef()))


class HiprtcLinker:
    def __init__(self):
        self.link_state = hip_check(hiprtc.hiprtcLinkCreate(0, None, None))
        self.completed = False
        self.code = None
        self.code_size = None

    def add_program(self, program):
        hip_check(
            hiprtc.hiprtcLinkAddData(
                self.link_state,
                hiprtc.hiprtcJITInputType.HIPRTC_JIT_INPUT_LLVM_BITCODE,
                program.llvm_bc_or_ir,
                program.llvm_bc_or_ir_size,
                program.name,
                0,
                None,
                None,
            )
        )

    def complete(self):
        self.code, self.code_size = hip_check(
            hiprtc.hiprtcLinkComplete(self.link_state)
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        hip_check(hiprtc.hiprtcLinkDestroy(self.link_state))


if __name__ == "__main__":
    import textwrap

    kernel_hip = textwrap.dedent(
        """\
        // prototypes
        __device__ void scale_op(float arr[], float factor);
        __device__ void print_val(float arr[]);

        extern "C" __global__ void scale(float arr[], float factor) {
            scale_op(arr, factor);
            print_val(arr);
        }
        """
    ).encode("utf-8")

    print_val_hip = textwrap.dedent(
        """\
        __device__ void print_val(float arr[]) {
            printf("%f\\n",arr[threadIdx.x]);
        }
        """
    ).encode("utf-8")

    # warning: below IR contains target dependent information
    scale_op_llvm_ir = {
        "gfx90a": textwrap.dedent(
            """\
        ; ModuleID = 'llvm-ir-buffer'
        source_filename = "scale_op.hip"
        target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
        target triple = "amdgcn-amd-amdhsa"

        ; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
        define hidden void @_Z8scale_opPff(ptr nocapture %0, float %1) local_unnamed_addr #0 {
            %3 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !0, !noundef !1
            %4 = zext i32 %3 to i64
            %5 = getelementptr inbounds float, ptr %0, i64 %4
            %6 = load float, ptr %5, align 4, !tbaa !2
            %7 = fmul contract float %6, %1
            store float %7, ptr %5, align 4, !tbaa !2
            ret void
        }

        ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
        declare i32 @llvm.amdgcn.workitem.id.x() #1

        attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+cumode,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+sramecc,+wavefrontsize64,-xnack" }
        attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

        !0 = !{i32 0, i32 1024}
        !1 = !{}
        !2 = !{!3, !3, i64 0}
        !3 = !{!"float", !4, i64 0}
        !4 = !{!"omnipotent char", !5, i64 0}
        !5 = !{!"Simple C++ TBAA"}
        """
        ).encode("utf-8"),
    }

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName
    gpugen = arch.decode("utf-8").split(":")[0]
    if gpugen not in scale_op_llvm_ir:
        supported_gpugens = ", ".join([f"'{a}'" for a in scale_op_llvm_ir.keys()])
        print(
            f"ERROR: unsupported GPU architecture '{gpugen}' (supported: {supported_gpugens})"
        )
        sys.exit(1)

    with HiprtcLinker() as linker, HipProgram(
        "kernel", arch, kernel_hip
    ) as kernel_prog, HipProgram(
        # "scale_op", arch, scale_op_hip) as scale_op_prog, LLVMIRProgram(
        "print_val",
        arch,
        print_val_hip,
    ) as print_val_prog, LLVMIRProgram(
        "scale_op", scale_op_llvm_ir[gpugen]
    ) as scale_op_prog:
        linker.add_program(kernel_prog)
        linker.add_program(print_val_prog)
        linker.add_program(scale_op_prog)
        linker.complete()
        module = hip_check(hip.hipModuleLoadData(linker.code))
        kernel = hip_check(hip.hipModuleGetFunction(module, b"scale"))

        f32, size = 4, 32
        xh = array.array("f", [1.0] * size)
        xd = hip_check(hip.hipMalloc(f32 * size))
        hip_check(
            hip.hipMemcpy(xd, xh, f32 * size, hip.hipMemcpyKind.hipMemcpyHostToDevice)
        )
        hip_check(
            hip.hipModuleLaunchKernel(
                kernel,
                *(1, 1, 1),  # grid
                *(32, 1, 1),  # block
                sharedMemBytes=0,
                stream=None,
                kernelParams=None,
                extra=(
                    xd,
                    ctypes.c_float(2.0),
                ),
            )
        )
        hip_check(
            hip.hipMemcpy(xh, xd, f32 * size, hip.hipMemcpyKind.hipMemcpyHostToDevice)
        )
        hip_check(hip.hipFree(xd))
        hip_check(hip.hipModuleUnload(module))

        for i in range(0, size):
            assert math.isclose(xh[i], 2.0), f"failed at pos {i}"
        print("ok")
