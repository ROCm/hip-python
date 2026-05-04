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

"""This example generates HSA assembly (for gfx942) from LLVM IR (for gfx942).

We use AMD COMGR infrastructure as hipRTC does not yet provide a similar
compilation variant.

Note:
    We derived the LLVM IR input from the original HIP C++ source via
    the below command:

    ```shell
    hipcc -emit-llvm -S --offload-arch=gfx942 vector_add.hip -o - | sed -n "/hip-amdgcn-amd-amdhsa--gfx942/,/hip-amdgcn-amd-amdhsa--gfx942/p"
    ```
"""  # noqa: E501

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# [literalinclude-begin]
from rocm import comgr


class LLVMProgram:
    def __init__(self, name: str, arch: str, source: bytes):
        self.hip_source = source
        self.name = name.encode("utf-8")
        self.hsa = None  # type: bytes
        self.hsa_size = None
        self.log = None
        self.diagnostic = None
        self._compile_to_hsa(arch)

    def _compile_to_hsa(self, arch: str):
        # [literalinclude-comgr-compile-bc-to-hsa-begin]
        (hsa, log, diagnostic) = comgr.compile_bc_to_hsa(
            source=self.hip_source,
            isa_name=f"amdgcn-amd-amdhsa--{arch}",
            logging=True,
        )
        # [literalinclude-comgr-compile-bc-to-hsa-end]
        self.hsa = hsa
        self.log = log
        self.diagnostic = diagnostic
        self.hsa_size = len(self.hsa)


if __name__ in ("__test__", "__main__"):
    import textwrap

    # The original HIP C++ source that was used to generate the below LLVM IR.
    _original_hip_kernel = textwrap.dedent(
        """\
        #include <hip/hip_runtime.h>

        extern "C" __global__ void vector_add(float* output, float* input1,
                                   float* input2, size_t size) {
            int i = threadIdx.x;
            if (i < size) {
                output[i] = input1[i] + input2[i];
            }
        }
        """
    )  # noqa: F401

    # Generated from the original HIP C++ source; see details at the top of
    # this file.
    kernel_llvm_ir = textwrap.dedent(
        """\
        ; __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx942
        ; ModuleID = 'vector_add.hip'
        source_filename = "vector_add.hip"
        target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
        target triple = "amdgcn-amd-amdhsa"

        @__hip_cuid_7c7f9d3655421f20 = addrspace(1) global i8 0
        @llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_7c7f9d3655421f20 to ptr)], section "llvm.metadata"

        ; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
        define protected amdgpu_kernel void @vector_add(ptr addrspace(1) nocapture noundef writeonly %0, ptr addrspace(1) nocapture noundef readonly %1, ptr addrspace(1) nocapture noundef readonly %2, i64 noundef %3) local_unnamed_addr #0 {
        %5 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
        %6 = zext nneg i32 %5 to i64
        %7 = icmp ugt i64 %3, %6
        br i1 %7, label %8, label %15

        8:                                                ; preds = %4
        %9 = getelementptr inbounds nuw float, ptr addrspace(1) %0, i64 %6
        %10 = getelementptr inbounds nuw float, ptr addrspace(1) %2, i64 %6
        %11 = getelementptr inbounds nuw float, ptr addrspace(1) %1, i64 %6
        %12 = load float, ptr addrspace(1) %11, align 4, !tbaa !6
        %13 = load float, ptr addrspace(1) %10, align 4, !tbaa !6
        %14 = fadd contract float %12, %13
        store float %14, ptr addrspace(1) %9, align 4, !tbaa !6
        br label %15

        15:                                               ; preds = %8, %4
        ret void
        }

        ; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
        declare noundef i32 @llvm.amdgcn.workitem.id.x() #1

        attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-conversion-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64,+xf32-insts" "uniform-work-group-size"="true" }
        attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

        !llvm.module.flags = !{!0, !1, !2, !3}
        !opencl.ocl.version = !{!4}
        !llvm.ident = !{!5}

        !0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
        !1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
        !2 = !{i32 1, !"wchar_size", i32 4}
        !3 = !{i32 8, !"PIC Level", i32 2}
        !4 = !{i32 2, i32 0}
        !5 = !{!"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.0 25304 82aed4e69d70bef3c89c38a2ee85c8c41294dfc9)"}
        !6 = !{!7, !7, i64 0}
        !7 = !{!"float", !8, i64 0}
        !8 = !{!"omnipotent char", !9, i64 0}
        !9 = !{!"Simple C++ TBAA"}

        ; __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx942
        """  # noqa: E501
    ).encode(
        "utf-8"
    )  # noqa: E501

    arch = "gfx942"
    kernel_prog = LLVMProgram("kernel", arch, kernel_llvm_ir)
    hsa_result = kernel_prog.hsa.decode()
    print(hsa_result)

    # [literalinclude-end]
    print("ok")
