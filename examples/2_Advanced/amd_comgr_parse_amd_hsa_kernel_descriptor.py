# MIT License
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

# [literalinclude-begin]

# note: amd_hsa_kernel_descriptor is a standalone module
from rocm.comgr import amd_hsa_kernel_descriptor

kd_symbol = b"\x00\x00\x00\x00\x00\x00\x00\x00\x10\x01\x00\x00\x00\x00\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\xaf\x00\x84\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00"  # noqa: E501

parse_result = (
    amd_hsa_kernel_descriptor.parse_amdgpu_code_obj_kernel_descriptor(
        kd_symbol, amdgpu_arch="gfx942"
    )
)

assert parse_result.enable_ieee_mode == 1
assert parse_result.float_denorm_mode_32 == 3
assert parse_result.kernarg_size == 272

parse_result_as_dict = parse_result.as_dict()
assert parse_result_as_dict["enable_ieee_mode"] == 1
assert parse_result_as_dict["float_denorm_mode_32"] == 3
assert parse_result_as_dict["kernarg_size"] == 272

amdhsa_kernel_section = parse_result.render_amdhsa_kernel_directive("KERNEL")
print(amdhsa_kernel_section)

assert ".amdhsa_kernarg_size 272" in amdhsa_kernel_section
assert ".amdhsa_next_free_vgpr 8" in amdhsa_kernel_section
assert ".amdhsa_next_free_sgpr 16" in amdhsa_kernel_section

# [literalinclude-end]

expected_meaningful_fields = {
    "accum_offset": 0,
    "bulky": 0,
    "cdbg_user": 0,
    "debug_mode": 0,
    "enable_dx10_clamp": 1,
    "enable_exception_address_watch": 0,
    "enable_exception_fp_denormal_source": 0,
    "enable_exception_ieee_754_fp_division_by_zero": 0,
    "enable_exception_ieee_754_fp_inexact": 0,
    "enable_exception_ieee_754_fp_invalid_operation": 0,
    "enable_exception_ieee_754_fp_overflow": 0,
    "enable_exception_ieee_754_fp_underflow": 0,
    "enable_exception_int_divide_by_zero": 0,
    "enable_exception_memory": 0,
    "enable_ieee_mode": 1,
    "enable_private_segment": 0,
    "enable_sgpr_dispatch_id": 0,
    "enable_sgpr_dispatch_ptr": 0,
    "enable_sgpr_flat_scratch_init": 0,
    "enable_sgpr_kernarg_segment_ptr": 1,
    "enable_sgpr_private_segment_buffer": 0,
    "enable_sgpr_private_segment_size": 0,
    "enable_sgpr_queue_ptr": 0,
    "enable_sgpr_workgroup_id_x": 1,
    "enable_sgpr_workgroup_id_y": 0,
    "enable_sgpr_workgroup_id_z": 0,
    "enable_sgpr_workgroup_info": 0,
    "enable_trap_handler": 0,
    "enable_vgpr_workitem_id": 0,
    "enable_wavefront_size32": 0,
    "float_denorm_mode_16_64": 3,
    "float_denorm_mode_32": 3,
    "float_round_mode_16_64": 0,
    "float_round_mode_32": 0,
    "fp16_ovfl": 0,
    "granulated_lds_size": 0,
    "granulated_wavefront_sgpr_count": 1,
    "granulated_workitem_vgpr_count": 0,
    "group_segment_fixed_size": 0,
    "kernarg_preload_spec_length": 0,
    "kernarg_preload_spec_offset": 0,
    "kernarg_size": 272,
    "kernel_code_entry_byte_offset": 4352,
    "priority": 0,
    "priv": 0,
    "private_segment_fixed_size": 0,
    "tg_split": 0,
    "user_sgpr_count": 2,
    "uses_cu_stores": 0,
    "uses_dynamic_stack": 0,
}


expected_amdhsa_kernel_section = """\
.section .rodata,"a",@progbits
.p2align 6, 0x0
.amdhsa_kernel KERNEL
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
        .amdhsa_next_free_vgpr 8 ; guessed value
        .amdhsa_next_free_sgpr 16 ; guessed value
        .amdhsa_reserve_vcc 0 ; guessed value
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
"""  # noqa: E501


def _clean_asm_snippet(snippet):  # type: (str)->str
    no_comments = ""
    for ln in snippet.splitlines(keepends=True):
        if ";" in ln:
            no_comments += ln.split(";")[0]
    return no_comments.replace(" ", "").replace("\t", "").replace("\n", "")


assert expected_meaningful_fields == parse_result_as_dict

assert _clean_asm_snippet(
    expected_amdhsa_kernel_section
) == _clean_asm_snippet(amdhsa_kernel_section)

print("ok")
