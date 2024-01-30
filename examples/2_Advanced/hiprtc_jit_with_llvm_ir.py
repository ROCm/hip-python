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

"""In this example, we use HIPRTC's hiprtcLink* APIs to compile 
and launch an AMD GPU kernel that is expressed in LLVM IR.

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


class LLLVMProgram:
    def __init__(self, name: str, source: bytes):
        self.name = name.encode("utf-8")
        self.llvm_bc_or_ir = source
        self.llvm_bc_or_ir_size = len(source)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class HiprtcLinker:
    def __init__(self):
        self.link_state = hip_check(
            hiprtc.ext.hiprtcLinkCreate2(
                HIPRTC_JIT_GENERATE_DEBUG_INFO=1,
                HIPRTC_JIT_GENERATE_LINE_INFO=1,
            )
        )
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


if __name__ in ("__test__","__main__"):
    import textwrap

    # warning: below IR contains target dependent information
    kernel_llvm_ir = {
        "gfx90a": textwrap.dedent(
            """\
        ; ModuleID = 'llvm-ir-buffer'
        source_filename = "scale_op.hip"
        target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
        target triple = "amdgcn-amd-amdhsa"
        
        ; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
        define protected amdgpu_kernel void @scale(ptr addrspace(1) nocapture %0, float %1) local_unnamed_addr #0 {
            %3 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !9, !noundef !10
            %4 = zext i32 %3 to i64
            %5 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %4
            %6 = load float, ptr addrspace(1) %5, align 4, !tbaa !11, !amdgpu.noclobber !10
            %7 = fmul contract float %6, %1
            store float %7, ptr addrspace(1) %5, align 4, !tbaa !11
            ret void
        }

        ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
        declare i32 @llvm.amdgcn.workitem.id.x() #1
        attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "amdgpu-flat-work-group-size"="1,1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+cumode,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+sramecc,+wavefrontsize64,-xnack" "uniform-work-group-size"="true" }
        attributes #1 = { convergent nocallback nofree nounwind willreturn memory(none) }

        !llvm.module.flags = !{!0, !1, !2, !3, !4}
        !opencl.ocl.version = !{!5}
        !llvm.ident = !{!6}

        !0 = !{i32 4, !"amdgpu_hostcall", i32 1}
        !1 = !{i32 1, !"amdgpu_code_object_version", i32 500}
        !2 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
        !3 = !{i32 1, !"wchar_size", i32 4}
        !4 = !{i32 8, !"PIC Level", i32 2}
        !5 = !{i32 2, i32 0}
        !6 = !{!"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.0 23483 7208e8d15fbf218deb74483ea8c549c67ca4985e)"}
        !7 = !{!"omnipotent char", !8, i64 0}
        !8 = !{!"Simple C++ TBAA"}
        !9 = !{i32 0, i32 1024}
        !10 = !{}
        !11 = !{!12, !12, i64 0}
        !12 = !{!"float", !7, i64 0}
        """
        ).encode("utf-8"),
    }

    # kernel_hip = textwrap.dedent(
    #     """\
    #     extern "C" __global__ void scale(float arr[], float factor) {
    #         arr[threadIdx.x] *= factor;
    #     }
    #     """
    # ).encode("utf-8")

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName
    gpugen = arch.decode("utf-8").split(":")[0]
    if gpugen not in kernel_llvm_ir:
        supported_gpugens = ", ".join([f"'{a}'" for a in kernel_llvm_ir.keys()])
        print(
            f"ERROR: unsupported GPU architecture '{gpugen}' (supported: {supported_gpugens})"
        )
        sys.exit(1)

    with HiprtcLinker() as linker, LLLVMProgram(
        "kernel", kernel_llvm_ir[gpugen]
    ) as scale_op_prog:
        linker.add_program(scale_op_prog)
        linker.complete()
        module = hip_check(hip.hipModuleLoadData(linker.code))
        kernel = hip_check(hip.hipModuleGetFunction(module, b"scale"))

        f32, size = 4, 32
        assert size <= 1024
        xh = array.array("f", [1.0] * size)
        xd = hip_check(hip.hipMalloc(f32 * size))
        hip_check(
            hip.hipMemcpy(xd, xh, f32 * size, hip.hipMemcpyKind.hipMemcpyHostToDevice)
        )
        hip_check(
            hip.hipModuleLaunchKernel(
                kernel,
                *(1, 1, 1),  # grid
                *(size, 1, 1),  # block
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