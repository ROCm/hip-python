#!/usr/bin/env -S python3 -m pytest -v -s
# MIT License
#
# Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

import textwrap
import pprint

from numba.hip.amdgcn import AMDGPUTargetMachine

from rocm.amd_comgr import amd_comgr as comgr


def test_00_print_datalayout():
    # pprint.pprint(comgr.ext.get_isa_metadata_all())
    # pprint.pprint(ISA_INFOS)
    machine = AMDGPUTargetMachine(target_cpu="gfx90a")
    assert (
        machine.data_layout in (
          "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7",
          "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8",
          "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", # ROCm 6.2.0
        )
    )


def test_01_verify_module():
    import faulthandler

    faulthandler.enable()
    machine = AMDGPUTargetMachine(target_cpu="gfx90a")

    dep_llvm_ir = textwrap.dedent(
        # LLVM IR extracted from
        # hipcc -S -emit-llvm -fgpu-rdc dep.hip
        # ```c++
        # // dep.hip:
        # include "hip/hip_runtime.h"
        # __device__ void scale(float arr[], float factor) {
        #    arr[threadIdx.x] *= factor;
        # }
        # ```
        """\
        ; ModuleID = 'dep.hip'
        source_filename = "dep.hip"
        target datalayout = "DATA_LAYOUT"
        target triple = "amdgcn-amd-amdhsa"

        ; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
        define hidden void @_Z5scalePff(ptr nocapture %0, float %1) local_unnamed_addr #0 {
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
        attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="gfx90a" }

        !0 = !{i32 0, i32 1024}
        !1 = !{}
        !2 = !{!3, !3, i64 0}
        !3 = !{!"float", !4, i64 0}
        !4 = !{!"omnipotent char", !5, i64 0}
        !5 = !{!"Simple C++ TBAA"}
        """.replace("DATA_LAYOUT",machine.data_layout)
    )
    machine.optimize_module(dep_llvm_ir, passes="default<O3>").decode("utf-8")
    # machine.verify_module(dep_llvm_ir) # TODO get strange error 'Attribute does not match Module context!'


if __name__ == "__main__":
    test_01_verify_module()
