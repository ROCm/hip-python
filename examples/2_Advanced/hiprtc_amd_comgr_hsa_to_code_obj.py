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

"""In this example, we assemble AMD HSA to an AMD GPU code object.

You can choose to do this via AMD COMGR (default) or via HIPRTC
by toggling the `compile_via_comgr` variable.

If you have an AMD gfx942 GPU in your system, this example
will also launch the compiled AMD GPU kernel and compare
the GPU-computed result with the outcome of the same operation
conducted on the host.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# [literalinclude-begin]
import ctypes

import numpy as np
from rocm import comgr
from rocm.bindings import hip, hiprtc

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


class HsaProgram:
    def __init__(self, name: str, arch: str, source: str):
        global compile_via_comgr
        self.hsa_source = source
        self.name = name
        self.prog = None
        self.code_obj = None
        self.log = None
        self.diagnostic = None
        self.code_obj_size = None
        if compile_via_comgr:
            self._compile_with_comgr(arch)
        else:
            self._compile_with_hiprtc(arch)

    def _compile_with_comgr(self, arch: str):
        """Compile HSA to code object via AMD COMGR."""
        (
            self.code_obj,
            self.log,
            self.diagnostic,
        ) = comgr.compile_hsa(
            self.hsa_source, "amdgcn-amd-amdhsa--" + arch, logging=True
        )  # note: raises RuntimeError and attaches log output
        self.code_obj_size = len(self.code_obj)

    def _compile_with_hiprtc(self, arch: str):
        self.prog = hip_check(
            hiprtc.hiprtcCreateProgram(self.hsa_source, self.name, 0, [], [])
        )
        cflags = [
            "-x",
            "assembler",
            "-target",
            "amdhsa-amd-amdgcn",
            "--offload-arch=gfx942",
            "-Wno-unused-command-line-argument",  # just for nicer COMGR logs
        ]
        (err,) = hiprtc.hiprtcCompileProgram(self.prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(self.prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(self.prog, log))
            raise RuntimeError(log.decode())
        self.code_obj_size = hip_check(hiprtc.hiprtcGetCodeSize(self.prog))
        self.code_obj = bytearray(self.code_obj_size)
        hip_check(hiprtc.hiprtcGetCode(self.prog, self.code_obj))

    def __del__(self):
        if self.prog is not None:
            hip_check(hiprtc.hiprtcDestroyProgram(self.prog.createRef()))


if __name__ in ("__test__", "__main__"):

    __original_kernel_hip = """\
#include <hip/hip_runtime.h>

__global__ void square(float arr[], int n) {
  auto i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < n ) {
    arr[i] = arr[i] * arr[i];
  }
}
"""

    kernel_hsa = """\
        .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
        .amdhsa_code_object_version 6
        .text
        .protected _Z6squarePfi
        .globl _Z6squarePfi
        .p2align 8
        .type _Z6squarePfi,@function
_Z6squarePfi:
        s_load_dword s3, s[0:1], 0x1c
        s_load_dword s4, s[0:1], 0x8
        s_waitcnt lgkmcnt(0)
        s_and_b32 s3, s3, 0xffff
        s_mul_i32 s2, s2, s3
        v_add_u32_e32 v0, s2, v0
        v_cmp_gt_u32_e32 vcc, s4, v0
        s_and_saveexec_b64 s[2:3], vcc
        s_cbranch_execz 12
        s_load_dwordx2 s[0:1], s[0:1], 0x0
        v_mov_b32_e32 v1, 0
        s_waitcnt lgkmcnt(0)
        v_lshl_add_u64 v[0:1], v[0:1], 2, s[0:1]
        global_load_dword v2, v[0:1], off
        s_waitcnt vmcnt(0)
        v_mul_f32_e32 v2, v2, v2
        global_store_dword v[0:1], v2, off
        s_endpgm
        .section .rodata,"a",@progbits
        .p2align 6, 0x0
        .amdhsa_kernel _Z6squarePfi
                .amdhsa_group_segment_fixed_size 0
                .amdhsa_private_segment_fixed_size 0
                .amdhsa_kernarg_size 272
                .amdhsa_user_sgpr_count 2
                .amdhsa_user_sgpr_dispatch_ptr 0
                .amdhsa_user_sgpr_queue_ptr 0
                .amdhsa_user_sgpr_kernarg_segment_ptr 1
                .amdhsa_user_sgpr_dispatch_id 0
                .amdhsa_user_sgpr_private_segment_size 0
                .amdhsa_uses_dynamic_stack 0
                .amdhsa_enable_private_segment 0
                .amdhsa_system_sgpr_workgroup_id_x 1
                .amdhsa_system_sgpr_workgroup_id_y 0
                .amdhsa_system_sgpr_workgroup_id_z 0
                .amdhsa_system_sgpr_workgroup_info 0
                .amdhsa_system_vgpr_workitem_id 0
                .amdhsa_next_free_vgpr 8
                .amdhsa_next_free_sgpr 16
                .amdhsa_accum_offset 4
                .amdhsa_reserve_vcc 0
                .amdhsa_float_round_mode_32 0
                .amdhsa_float_round_mode_16_64 0
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_float_denorm_mode_16_64 3
                .amdhsa_dx10_clamp 1
                .amdhsa_ieee_mode 1
                .amdhsa_fp16_overflow 0
                .amdhsa_exception_fp_ieee_invalid_op 0
                .amdhsa_exception_fp_denorm_src 0
                .amdhsa_exception_fp_ieee_div_zero 0
                .amdhsa_exception_fp_ieee_overflow 0
                .amdhsa_exception_fp_ieee_underflow 0
                .amdhsa_exception_fp_ieee_inexact 0
                .amdhsa_exception_int_div_zero 0
                .amdhsa_user_sgpr_kernarg_preload_length 0
                .amdhsa_user_sgpr_kernarg_preload_offset 0
        .end_amdhsa_kernel
        .amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count: 0
    .args:
      - .address_space: global
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .offset: 8
        .size: 4
        .value_kind: by_value
      - .offset: 16
        .size: 4
        .value_kind: hidden_block_count_x
      - .offset: 20
        .size: 4
        .value_kind: hidden_block_count_y
      - .offset: 24
        .size: 4
        .value_kind: hidden_block_count_z
      - .offset: 28
        .size: 2
        .value_kind: hidden_group_size_x
      - .offset: 30
        .size: 2
        .value_kind: hidden_group_size_y
      - .offset: 32
        .size: 2
        .value_kind: hidden_group_size_z
      - .offset: 34
        .size: 2
        .value_kind: hidden_remainder_x
      - .offset: 36
        .size: 2
        .value_kind: hidden_remainder_y
      - .offset: 38
        .size: 2
        .value_kind: hidden_remainder_z
      - .offset: 56
        .size: 8
        .value_kind: hidden_global_offset_x
      - .offset: 64
        .size: 8
        .value_kind: hidden_global_offset_y
      - .offset: 72
        .size: 8
        .value_kind: hidden_global_offset_z
      - .offset: 80
        .size: 2
        .value_kind: hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language: OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name: _Z6squarePfi
    .private_segment_fixed_size: 0
    .sgpr_count: 11
    .sgpr_spill_count: 0
    .symbol: _Z6squarePfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 3
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target: amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...
      .end_amdgpu_metadata
      """

    arch = "gfx942"
    kernel_prog = HsaProgram("kernel", arch, kernel_hsa)
    _, count = hip.hipGetDeviceCount()
    if count > 0:
        props = hip_check(hip.hipGetDeviceProperties(0))
        if arch == props.gcnArchName.split(":")[0]:
            module = hip_check(hip.hipModuleLoadData(kernel_prog.code_obj))
            kernel = hip_check(
                hip.hipModuleGetFunction(module, "_Z6squarePfi")
            )
            print(
                "Found AMD GPU with matching architecture.\n"
                + "Testing compiled kernel."
            )
            itemsize = ctypes.sizeof(ctypes.c_float)
            num_elements = 20
            h_input = np.array(range(0, num_elements), dtype=np.float32)
            d_inout = hip_check(hip.hipMalloc(itemsize * num_elements))
            h_output = np.zeros(num_elements, dtype=np.float32)
            h_expected = h_input * h_input

            hip_check(
                hip.hipMemcpy(
                    d_inout,
                    h_input,
                    itemsize * num_elements,
                    hip.hipMemcpyKind.hipMemcpyHostToDevice,
                )
            )
            hip_check(
                hip.hipModuleLaunchKernel(
                    kernel,
                    *(1, 1, 1),  # grid
                    *(1024, 1, 1),  # block
                    sharedMemBytes=0,
                    stream=None,
                    kernelParams=None,
                    extra=(
                        d_inout,
                        num_elements,
                    ),
                )
            )
            hip_check(hip.hipDeviceSynchronize())
            hip_check(
                hip.hipMemcpy(
                    h_output,
                    d_inout,
                    itemsize * num_elements,
                    hip.hipMemcpyKind.hipMemcpyDeviceToHost,
                )
            )
            hip_check(hip.hipFree(d_inout))
            hip_check(hip.hipModuleUnload(module))

            assert np.allclose(h_expected, h_output)

print("ok")
