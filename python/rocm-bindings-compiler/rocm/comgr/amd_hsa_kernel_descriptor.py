# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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

import ctypes
import re
import textwrap
import typing  # noqa: F401

from . import amdhsa_kernel_directives

p_amdgpu_arch = re.compile(r"gfx(?P<gen>[0-9]{1,2})(?P<subgen>[0-9a-f]{2})$")


def _get_sgpr_encoding_granule(amdgpu_arch):
    r"""
    Note:
        See the following file for further details:
        <https://github.com/ROCm/llvm-project/blob/release/rocm-rel-7.0/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp>
    """  # noqa: E501
    return 8


def _get_vgpr_encoding_granule(amdgpu_arch, wave32):
    r"""
    Note:
        See the following file for further details:
        <https://github.com/ROCm/llvm-project/blob/release/rocm-rel-7.0/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp>
    """  # noqa: E501

    m = p_amdgpu_arch.match(amdgpu_arch)
    assert m, f"no match for {amdgpu_arch}"
    gfx_gen = int(m.group("gen"), 16)
    gfx_subgen = int(m.group("subgen"), 16)

    if gfx_gen == 9 and gfx_subgen >= 0x0A:
        return 8

    return 8 if wave32 else 4


def _guess_next_free_sgpr(
    granulated_wavefront_sgpr_count,
    amdgpu_arch,
):  # type: (int, str) -> int
    """Guess amdhsa_next_free_sgpr value from granulated wavefront sgpr count.

    Assumes that the following amdhsa_kernel directives are assigned a 0 value:

    * .amdhsa_reserve_vcc
    * .amdhsa_reserve_flat_scratch
    * .amdhsa_reserve_xnack_mask

    Args:
        granulated_wavefront_sgpr_count (`int`):
            GRANULATED_WAVEFRONT_SGPR_COUNT = f(NEXT_FREE_SGPR + VCC +
            FLAT_SCRATCH + XNACK_MASK). We can obtain the left-hand side
            from an AMD HSA kernel descriptor but not the summands
            on the right-hand side.
        amdgpu_arch (`str`):
            AMD GPU architecture. An expression like 'gfx90a' or 'gfx1201'.
            Assumes that any feature flags such as `:xnack+` have been stripped
            off.

    Quote from <https://github.com/ROCm/llvm-project/blob/release/rocm-rel-7.0/llvm/lib/Target/AMDGPU/Disassembler/AMDGPUDisassembler.cpp#L2067>:

        We cannot backward compute values used to calculate
        GRANULATED_WAVEFRONT_SGPR_COUNT. Hence the original values for following
        directives can't be computed:

        .amdhsa_reserve_vcc
        .amdhsa_reserve_flat_scratch
        .amdhsa_reserve_xnack_mask

        They take their respective default values if not specified in the
        assembly.

            GRANULATED_WAVEFRONT_SGPR_COUNT
                = f(NEXT_FREE_SGPR + VCC + FLAT_SCRATCH + XNACK_MASK)

        We compute the inverse as though all directives apart from
        NEXT_FREE_SGPR are set to 0. So while disassembling we consider that:

            GRANULATED_WAVEFRONT_SGPR_COUNT
                = f(NEXT_FREE_SGPR + 0 + 0 + 0)

        The disassembler cannot recover the original values of those 3
        directives.

    Note:
        The GRANULATED_WAVEFRONT_SGPR_COUNT is the only important thing for the
        command processor. The mix of the input values NEXT_FREE_SGPR, ... does
        not matter
    """  # noqa: E501

    # see: https://github.com/ROCm/llvm-project/blob/release/rocm-rel-7.0/llvm/lib/Target/AMDGPU/Disassembler/AMDGPUDisassembler.cpp#L2093  # noqa: E501
    return (granulated_wavefront_sgpr_count + 1) * _get_sgpr_encoding_granule(
        amdgpu_arch
    )


def _guess_next_free_vgpr(
    granulated_workitem_vgpr_count, amdgpu_arch, wave32
):  # type: (int, str, bool) -> int
    """Guess amdhsa_next_free_vgpr value from granulated wavefront vgpr count.

    Guess amdhsa_next_free_vgpr directive value from granulated workitem vgpr
    count.

        We cannot accurately backward compute #VGPRs used from
        GRANULATED_WORKITEM_VGPR_COUNT. But we are concerned with getting the
        same value of GRANULATED_WORKITEM_VGPR_COUNT in the reassembled binary.
        So we simply calculate the inverse of what the assembler does.

    """  # noqa: E501

    # see: https://github.com/ROCm/llvm-project/blob/release/rocm-rel-7.0/llvm/lib/Target/AMDGPU/Disassembler/AMDGPUDisassembler.cpp#L2063  # noqa: E501
    return (granulated_workitem_vgpr_count + 1) * _get_vgpr_encoding_granule(
        amdgpu_arch, wave32
    )


class AMDHSAKernelDescriptor:
    """

    A class for accessing / rendering kernel-related information stored in an
    AMD HSA code object v6.

    Derived from struct `kernel_descript_t` in file:

    <https://github.com/ROCm/llvm-project/blob/a53433cf1b9f533b51b73ea82d69f78041e40f93/llvm/include/llvm/Support/AMDHSAKernelDescriptor.h>

    Note:
        We want to highlight the following comment above the struct:
        `// Kernel descriptor. Must be kept backwards compatible.`,
        which implies that the main layout (`group_types`)
        will likely not change.

    Note:
        We preprocessed ``AMDHSAKernelDescriptor.h`` via

        ```shell
        g++ -E AMDHSAKernelDescriptor.h -o AMDHSAKernelDescriptor.h.i
        ```

        and then collected the enum values and transformed the keys to lower
        case. This yielded the class member `_group_entry_coordinates`.
        The iterator '_iterate_group_entries' has a lot of logic to support
        maintenance via this approach.
    """  # noqa: E501

    _group_types = [
        ("group_segment_fixed_size", ctypes.c_uint32),
        ("private_segment_fixed_size", ctypes.c_uint32),
        ("kernarg_size", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint8 * 4),
        ("kernel_code_entry_byte_offset", ctypes.c_int64),
        ("reserved1", ctypes.c_uint8 * 20),
        ("compute_pgm_rsrc3", ctypes.c_uint32),  # ordering is not by mistake
        ("compute_pgm_rsrc1", ctypes.c_uint32),
        ("compute_pgm_rsrc2", ctypes.c_uint32),
        ("kernel_code_properties", ctypes.c_uint16),
        ("kernarg_preload", ctypes.c_uint16),
        ("reserved3", ctypes.c_uint8 * 4),
    ]

    # We preprocessed ``AMDHSAKernelDescriptor.h`` via
    #
    # ```shell
    # g++ -E AMDHSAKernelDescriptor.h -o AMDHSAKernelDescriptor.h.i
    # ```
    #
    # and then collected the enum values and transformed the keys to lower
    # case. This yielded the class member `_group_entry_coordinates`.
    # The iterator '_iterate_group_entries' has a lot of logic to support
    # maintenance via this approach.

    _group_offsets = dict(
        group_segment_fixed_size_offset=0,
        private_segment_fixed_size_offset=4,
        kernarg_size_offset=8,
        reserved0_offset=12,
        kernel_code_entry_byte_offset_offset=16,
        reserved1_offset=24,
        compute_pgm_rsrc3_offset=44,
        compute_pgm_rsrc1_offset=48,
        compute_pgm_rsrc2_offset=52,
        kernel_code_properties_offset=56,
        kernarg_preload_offset=58,
        reserved3_offset=60,
    )

    _group_entry_coordinates = dict(
        compute_pgm_rsrc1_granulated_workitem_vgpr_count_shift=(0),
        compute_pgm_rsrc1_granulated_workitem_vgpr_count_width=(6),
        compute_pgm_rsrc1_granulated_workitem_vgpr_count=(
            ((1 << (6)) - 1) << (0)
        ),
        compute_pgm_rsrc1_granulated_wavefront_sgpr_count_shift=(6),
        compute_pgm_rsrc1_granulated_wavefront_sgpr_count_width=(4),
        compute_pgm_rsrc1_granulated_wavefront_sgpr_count=(
            ((1 << (4)) - 1) << (6)
        ),
        compute_pgm_rsrc1_priority_shift=(10),
        compute_pgm_rsrc1_priority_width=(2),
        compute_pgm_rsrc1_priority=(((1 << (2)) - 1) << (10)),
        compute_pgm_rsrc1_float_round_mode_32_shift=(12),
        compute_pgm_rsrc1_float_round_mode_32_width=(2),
        compute_pgm_rsrc1_float_round_mode_32=(((1 << (2)) - 1) << (12)),
        compute_pgm_rsrc1_float_round_mode_16_64_shift=(14),
        compute_pgm_rsrc1_float_round_mode_16_64_width=(2),
        compute_pgm_rsrc1_float_round_mode_16_64=(((1 << (2)) - 1) << (14)),
        compute_pgm_rsrc1_float_denorm_mode_32_shift=(16),
        compute_pgm_rsrc1_float_denorm_mode_32_width=(2),
        compute_pgm_rsrc1_float_denorm_mode_32=(((1 << (2)) - 1) << (16)),
        compute_pgm_rsrc1_float_denorm_mode_16_64_shift=(18),
        compute_pgm_rsrc1_float_denorm_mode_16_64_width=(2),
        compute_pgm_rsrc1_float_denorm_mode_16_64=(((1 << (2)) - 1) << (18)),
        compute_pgm_rsrc1_priv_shift=(20),
        compute_pgm_rsrc1_priv_width=(1),
        compute_pgm_rsrc1_priv=(((1 << (1)) - 1) << (20)),
        compute_pgm_rsrc1_gfx6_gfx11_enable_dx10_clamp_shift=(21),
        compute_pgm_rsrc1_gfx6_gfx11_enable_dx10_clamp_width=(1),
        compute_pgm_rsrc1_gfx6_gfx11_enable_dx10_clamp=(
            ((1 << (1)) - 1) << (21)
        ),
        compute_pgm_rsrc1_gfx12_plus_enable_wg_rr_en_shift=(21),
        compute_pgm_rsrc1_gfx12_plus_enable_wg_rr_en_width=(1),
        compute_pgm_rsrc1_gfx12_plus_enable_wg_rr_en=(
            ((1 << (1)) - 1) << (21)
        ),
        compute_pgm_rsrc1_debug_mode_shift=(22),
        compute_pgm_rsrc1_debug_mode_width=(1),
        compute_pgm_rsrc1_debug_mode=(((1 << (1)) - 1) << (22)),
        compute_pgm_rsrc1_gfx6_gfx11_enable_ieee_mode_shift=(23),
        compute_pgm_rsrc1_gfx6_gfx11_enable_ieee_mode_width=(1),
        compute_pgm_rsrc1_gfx6_gfx11_enable_ieee_mode=(
            ((1 << (1)) - 1) << (23)
        ),
        compute_pgm_rsrc1_gfx12_plus_disable_perf_shift=(23),
        compute_pgm_rsrc1_gfx12_plus_disable_perf_width=(1),
        compute_pgm_rsrc1_gfx12_plus_disable_perf=(((1 << (1)) - 1) << (23)),
        compute_pgm_rsrc1_bulky_shift=(24),
        compute_pgm_rsrc1_bulky_width=(1),
        compute_pgm_rsrc1_bulky=(((1 << (1)) - 1) << (24)),
        compute_pgm_rsrc1_cdbg_user_shift=(25),
        compute_pgm_rsrc1_cdbg_user_width=(1),
        compute_pgm_rsrc1_cdbg_user=(((1 << (1)) - 1) << (25)),
        compute_pgm_rsrc1_gfx6_gfx8_reserved0_shift=(26),
        compute_pgm_rsrc1_gfx6_gfx8_reserved0_width=(1),
        compute_pgm_rsrc1_gfx6_gfx8_reserved0=(((1 << (1)) - 1) << (26)),
        compute_pgm_rsrc1_gfx9_plus_fp16_ovfl_shift=(26),
        compute_pgm_rsrc1_gfx9_plus_fp16_ovfl_width=(1),
        compute_pgm_rsrc1_gfx9_plus_fp16_ovfl=(((1 << (1)) - 1) << (26)),
        compute_pgm_rsrc1_reserved1_shift=(27),
        compute_pgm_rsrc1_reserved1_width=(2),
        compute_pgm_rsrc1_reserved1=(((1 << (2)) - 1) << (27)),
        compute_pgm_rsrc1_gfx6_gfx9_reserved2_shift=(29),
        compute_pgm_rsrc1_gfx6_gfx9_reserved2_width=(3),
        compute_pgm_rsrc1_gfx6_gfx9_reserved2=(((1 << (3)) - 1) << (29)),
        compute_pgm_rsrc1_gfx10_plus_wgp_mode_shift=(29),
        compute_pgm_rsrc1_gfx10_plus_wgp_mode_width=(1),
        compute_pgm_rsrc1_gfx10_plus_wgp_mode=(((1 << (1)) - 1) << (29)),
        compute_pgm_rsrc1_gfx10_plus_mem_ordered_shift=(30),
        compute_pgm_rsrc1_gfx10_plus_mem_ordered_width=(1),
        compute_pgm_rsrc1_gfx10_plus_mem_ordered=(((1 << (1)) - 1) << (30)),
        compute_pgm_rsrc1_gfx10_plus_fwd_progress_shift=(31),
        compute_pgm_rsrc1_gfx10_plus_fwd_progress_width=(1),
        compute_pgm_rsrc1_gfx10_plus_fwd_progress=(((1 << (1)) - 1) << (31)),
        compute_pgm_rsrc2_enable_private_segment_shift=(0),
        compute_pgm_rsrc2_enable_private_segment_width=(1),
        compute_pgm_rsrc2_enable_private_segment=(((1 << (1)) - 1) << (0)),
        compute_pgm_rsrc2_user_sgpr_count_shift=(1),
        compute_pgm_rsrc2_user_sgpr_count_width=(5),
        compute_pgm_rsrc2_user_sgpr_count=(((1 << (5)) - 1) << (1)),
        compute_pgm_rsrc2_gfx6_gfx11_enable_trap_handler_shift=(6),
        compute_pgm_rsrc2_gfx6_gfx11_enable_trap_handler_width=(1),
        compute_pgm_rsrc2_gfx6_gfx11_enable_trap_handler=(
            ((1 << (1)) - 1) << (6)
        ),
        compute_pgm_rsrc2_gfx12_plus_reserved1_shift=(6),
        compute_pgm_rsrc2_gfx12_plus_reserved1_width=(1),
        compute_pgm_rsrc2_gfx12_plus_reserved1=(((1 << (1)) - 1) << (6)),
        compute_pgm_rsrc2_enable_sgpr_workgroup_id_x_shift=(7),
        compute_pgm_rsrc2_enable_sgpr_workgroup_id_x_width=(1),
        compute_pgm_rsrc2_enable_sgpr_workgroup_id_x=(((1 << (1)) - 1) << (7)),
        compute_pgm_rsrc2_enable_sgpr_workgroup_id_y_shift=(8),
        compute_pgm_rsrc2_enable_sgpr_workgroup_id_y_width=(1),
        compute_pgm_rsrc2_enable_sgpr_workgroup_id_y=(((1 << (1)) - 1) << (8)),
        compute_pgm_rsrc2_enable_sgpr_workgroup_id_z_shift=(9),
        compute_pgm_rsrc2_enable_sgpr_workgroup_id_z_width=(1),
        compute_pgm_rsrc2_enable_sgpr_workgroup_id_z=(((1 << (1)) - 1) << (9)),
        compute_pgm_rsrc2_enable_sgpr_workgroup_info_shift=(10),
        compute_pgm_rsrc2_enable_sgpr_workgroup_info_width=(1),
        compute_pgm_rsrc2_enable_sgpr_workgroup_info=(
            ((1 << (1)) - 1) << (10)
        ),
        compute_pgm_rsrc2_enable_vgpr_workitem_id_shift=(11),
        compute_pgm_rsrc2_enable_vgpr_workitem_id_width=(2),
        compute_pgm_rsrc2_enable_vgpr_workitem_id=(((1 << (2)) - 1) << (11)),
        compute_pgm_rsrc2_enable_exception_address_watch_shift=(13),
        compute_pgm_rsrc2_enable_exception_address_watch_width=(1),
        compute_pgm_rsrc2_enable_exception_address_watch=(
            ((1 << (1)) - 1) << (13)
        ),
        compute_pgm_rsrc2_enable_exception_memory_shift=(14),
        compute_pgm_rsrc2_enable_exception_memory_width=(1),
        compute_pgm_rsrc2_enable_exception_memory=(((1 << (1)) - 1) << (14)),
        compute_pgm_rsrc2_granulated_lds_size_shift=(15),
        compute_pgm_rsrc2_granulated_lds_size_width=(9),
        compute_pgm_rsrc2_granulated_lds_size=(((1 << (9)) - 1) << (15)),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_invalid_operation_shift=(  # noqa: E501
            24
        ),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_invalid_operation_width=(  # noqa: E501
            1
        ),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_invalid_operation=(
            ((1 << (1)) - 1) << (24)
        ),
        compute_pgm_rsrc2_enable_exception_fp_denormal_source_shift=(25),
        compute_pgm_rsrc2_enable_exception_fp_denormal_source_width=(1),
        compute_pgm_rsrc2_enable_exception_fp_denormal_source=(
            ((1 << (1)) - 1) << (25)
        ),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_division_by_zero_shift=(
            26
        ),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_division_by_zero_width=(
            1
        ),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_division_by_zero=(
            ((1 << (1)) - 1) << (26)
        ),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_overflow_shift=(27),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_overflow_width=(1),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_overflow=(
            ((1 << (1)) - 1) << (27)
        ),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_underflow_shift=(28),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_underflow_width=(1),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_underflow=(
            ((1 << (1)) - 1) << (28)
        ),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_inexact_shift=(29),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_inexact_width=(1),
        compute_pgm_rsrc2_enable_exception_ieee_754_fp_inexact=(
            ((1 << (1)) - 1) << (29)
        ),
        compute_pgm_rsrc2_enable_exception_int_divide_by_zero_shift=(30),
        compute_pgm_rsrc2_enable_exception_int_divide_by_zero_width=(1),
        compute_pgm_rsrc2_enable_exception_int_divide_by_zero=(
            ((1 << (1)) - 1) << (30)
        ),
        compute_pgm_rsrc2_reserved0_shift=(31),
        compute_pgm_rsrc2_reserved0_width=(1),
        compute_pgm_rsrc2_reserved0=(((1 << (1)) - 1) << (31)),
        compute_pgm_rsrc3_gfx90a_accum_offset_shift=(0),
        compute_pgm_rsrc3_gfx90a_accum_offset_width=(6),
        compute_pgm_rsrc3_gfx90a_accum_offset=(((1 << (6)) - 1) << (0)),
        compute_pgm_rsrc3_gfx90a_reserved0_shift=(6),
        compute_pgm_rsrc3_gfx90a_reserved0_width=(10),
        compute_pgm_rsrc3_gfx90a_reserved0=(((1 << (10)) - 1) << (6)),
        compute_pgm_rsrc3_gfx90a_tg_split_shift=(16),
        compute_pgm_rsrc3_gfx90a_tg_split_width=(1),
        compute_pgm_rsrc3_gfx90a_tg_split=(((1 << (1)) - 1) << (16)),
        compute_pgm_rsrc3_gfx90a_reserved1_shift=(17),
        compute_pgm_rsrc3_gfx90a_reserved1_width=(15),
        compute_pgm_rsrc3_gfx90a_reserved1=(((1 << (15)) - 1) << (17)),
        compute_pgm_rsrc3_gfx10_gfx11_shared_vgpr_count_shift=(0),
        compute_pgm_rsrc3_gfx10_gfx11_shared_vgpr_count_width=(4),
        compute_pgm_rsrc3_gfx10_gfx11_shared_vgpr_count=(
            ((1 << (4)) - 1) << (0)
        ),
        compute_pgm_rsrc3_gfx12_plus_reserved0_shift=(0),
        compute_pgm_rsrc3_gfx12_plus_reserved0_width=(4),
        compute_pgm_rsrc3_gfx12_plus_reserved0=(((1 << (4)) - 1) << (0)),
        compute_pgm_rsrc3_gfx10_reserved1_shift=(4),
        compute_pgm_rsrc3_gfx10_reserved1_width=(8),
        compute_pgm_rsrc3_gfx10_reserved1=(((1 << (8)) - 1) << (4)),
        compute_pgm_rsrc3_gfx11_inst_pref_size_shift=(4),
        compute_pgm_rsrc3_gfx11_inst_pref_size_width=(6),
        compute_pgm_rsrc3_gfx11_inst_pref_size=(((1 << (6)) - 1) << (4)),
        compute_pgm_rsrc3_gfx11_trap_on_start_shift=(10),
        compute_pgm_rsrc3_gfx11_trap_on_start_width=(1),
        compute_pgm_rsrc3_gfx11_trap_on_start=(((1 << (1)) - 1) << (10)),
        compute_pgm_rsrc3_gfx11_trap_on_end_shift=(11),
        compute_pgm_rsrc3_gfx11_trap_on_end_width=(1),
        compute_pgm_rsrc3_gfx11_trap_on_end=(((1 << (1)) - 1) << (11)),
        compute_pgm_rsrc3_gfx12_plus_inst_pref_size_shift=(4),
        compute_pgm_rsrc3_gfx12_plus_inst_pref_size_width=(8),
        compute_pgm_rsrc3_gfx12_plus_inst_pref_size=(((1 << (8)) - 1) << (4)),
        compute_pgm_rsrc3_gfx10_plus_reserved2_shift=(12),
        compute_pgm_rsrc3_gfx10_plus_reserved2_width=(1),
        compute_pgm_rsrc3_gfx10_plus_reserved2=(((1 << (1)) - 1) << (12)),
        compute_pgm_rsrc3_gfx10_gfx11_reserved3_shift=(13),
        compute_pgm_rsrc3_gfx10_gfx11_reserved3_width=(1),
        compute_pgm_rsrc3_gfx10_gfx11_reserved3=(((1 << (1)) - 1) << (13)),
        compute_pgm_rsrc3_gfx12_plus_glg_en_shift=(13),
        compute_pgm_rsrc3_gfx12_plus_glg_en_width=(1),
        compute_pgm_rsrc3_gfx12_plus_glg_en=(((1 << (1)) - 1) << (13)),
        compute_pgm_rsrc3_gfx10_plus_reserved4_shift=(14),
        compute_pgm_rsrc3_gfx10_plus_reserved4_width=(17),
        compute_pgm_rsrc3_gfx10_plus_reserved4=(((1 << (17)) - 1) << (14)),
        compute_pgm_rsrc3_gfx10_reserved5_shift=(31),
        compute_pgm_rsrc3_gfx10_reserved5_width=(1),
        compute_pgm_rsrc3_gfx10_reserved5=(((1 << (1)) - 1) << (31)),
        compute_pgm_rsrc3_gfx11_plus_image_op_shift=(31),
        compute_pgm_rsrc3_gfx11_plus_image_op_width=(1),
        compute_pgm_rsrc3_gfx11_plus_image_op=(((1 << (1)) - 1) << (31)),
        kernel_code_property_enable_sgpr_private_segment_buffer_shift=(0),
        kernel_code_property_enable_sgpr_private_segment_buffer_width=(1),
        kernel_code_property_enable_sgpr_private_segment_buffer=(
            ((1 << (1)) - 1) << (0)
        ),
        kernel_code_property_enable_sgpr_dispatch_ptr_shift=(1),
        kernel_code_property_enable_sgpr_dispatch_ptr_width=(1),
        kernel_code_property_enable_sgpr_dispatch_ptr=(
            ((1 << (1)) - 1) << (1)
        ),
        kernel_code_property_enable_sgpr_queue_ptr_shift=(2),
        kernel_code_property_enable_sgpr_queue_ptr_width=(1),
        kernel_code_property_enable_sgpr_queue_ptr=(((1 << (1)) - 1) << (2)),
        kernel_code_property_enable_sgpr_kernarg_segment_ptr_shift=(3),
        kernel_code_property_enable_sgpr_kernarg_segment_ptr_width=(1),
        kernel_code_property_enable_sgpr_kernarg_segment_ptr=(
            ((1 << (1)) - 1) << (3)
        ),
        kernel_code_property_enable_sgpr_dispatch_id_shift=(4),
        kernel_code_property_enable_sgpr_dispatch_id_width=(1),
        kernel_code_property_enable_sgpr_dispatch_id=(((1 << (1)) - 1) << (4)),
        kernel_code_property_enable_sgpr_flat_scratch_init_shift=(5),
        kernel_code_property_enable_sgpr_flat_scratch_init_width=(1),
        kernel_code_property_enable_sgpr_flat_scratch_init=(
            ((1 << (1)) - 1) << (5)
        ),
        kernel_code_property_enable_sgpr_private_segment_size_shift=(6),
        kernel_code_property_enable_sgpr_private_segment_size_width=(1),
        kernel_code_property_enable_sgpr_private_segment_size=(
            ((1 << (1)) - 1) << (6)
        ),
        kernel_code_property_reserved0_shift=(7),
        kernel_code_property_reserved0_width=(2),
        kernel_code_property_reserved0=(((1 << (2)) - 1) << (7)),
        kernel_code_property_uses_cu_stores_shift=(9),
        kernel_code_property_uses_cu_stores_width=(1),
        kernel_code_property_uses_cu_stores=(((1 << (1)) - 1) << (9)),
        kernel_code_property_enable_wavefront_size32_shift=(10),
        kernel_code_property_enable_wavefront_size32_width=(1),
        kernel_code_property_enable_wavefront_size32=(
            ((1 << (1)) - 1) << (10)
        ),
        kernel_code_property_uses_dynamic_stack_shift=(11),
        kernel_code_property_uses_dynamic_stack_width=(1),
        kernel_code_property_uses_dynamic_stack=(((1 << (1)) - 1) << (11)),
        kernel_code_property_reserved1_shift=(12),
        kernel_code_property_reserved1_width=(4),
        kernel_code_property_reserved1=(((1 << (4)) - 1) << (12)),
        kernarg_preload_spec_length_shift=(0),
        kernarg_preload_spec_length_width=(7),
        kernarg_preload_spec_length=(((1 << (7)) - 1) << (0)),
        kernarg_preload_spec_offset_shift=(7),
        kernarg_preload_spec_offset_width=(9),
        kernarg_preload_spec_offset=(((1 << (9)) - 1) << (7)),
    )

    @staticmethod
    def _get_gfx_specific_group_entry_pattern(
        amdgpu_arch,
    ):  # type: (str) -> re.Pattern
        """Get a regex for matching gfx-generation-specific entries."""
        global p_amdgpu_arch
        amdgpu_arch = (
            amdgpu_arch.split(":")[0] if isinstance(amdgpu_arch, str) else ""
        )  # remove :xnack+:..
        m = p_amdgpu_arch.match(amdgpu_arch)
        if m:
            gfx_gen = int(m.group("gen"), 16)
            gfx_subgen = int(m.group("subgen"), 16)
        else:
            raise RuntimeError(
                "'amdgpu_arch' is not a 'str' in expected format : 'gfx<hex>'."
            )
        gfx_specific_group_entry_patterns = [
            f"_{amdgpu_arch}_",
            f"_gfx{gfx_gen}_",
        ]
        gfx_specific_group_entry_patterns += [
            f"_gfx{i}_plus_" for i in range(9, gfx_gen + 1)
        ]
        # NOTE: Keep these updated
        if gfx_gen == 9 and gfx_subgen > 0x0A:
            gfx_specific_group_entry_patterns.append("_gfx90a_")

        # NOTE: Kept for reference
        #       The code below updates the others.
        # if gfx_gen in range(6, 9 + 1):
        #     gfx_specific_group_entry_patterns.append("_gfx6_gfx9_")
        # if gfx_gen in range(6, 11 + 1):
        #     gfx_specific_group_entry_patterns.append("_gfx6_gfx11_")

        last_gen_to_consider = 20  # NOTE: Keep large enough
        lbounds = [str(i) for i in range(6, gfx_gen + 1)]
        ubounds = [str(i) for i in range(gfx_gen + 1, last_gen_to_consider)]
        # example: (gfx(6|7|8|9)_gfx(10|11|12,...))
        lbound_ubound_expr = (
            "(_gfx(" + "|".join(lbounds) + ")_gfx(" + "|".join(ubounds) + "))"
        )
        gfx_specific_group_entry_patterns.append(lbound_ubound_expr)

        return re.compile(
            "|".join(gfx_specific_group_entry_patterns),
        )

    @staticmethod
    def _iterate_possible_fields(remove_prefix=True, filter=lambda x: True):
        """Iterates all the possible fields names.

        What fields are actually utilized by the kernel
        descriptor depends on the architecture.

        Note:
            May yield duplicates due to archicture-specific
            prefixes for some fields, example:

            * compute_pgm_rsrc3_gfx11_inst_pref_size
            * compute_pgm_rsrc3_gfx12_plus_inst_pref_size
        """
        p_gfx_label = re.compile(r"gfx[0-9a-f]+(_plus)?_")

        for group, _ in AMDHSAKernelDescriptor._group_types:
            prefix = group + "_"
            if prefix == "kernel_code_properties_":
                prefix = "kernel_code_property_"

            if not group.startswith("reserved"):
                found_entry = False
                for (
                    entry
                ) in AMDHSAKernelDescriptor._group_entry_coordinates.keys():
                    if entry.startswith(prefix) and entry.endswith("_shift"):
                        if "_reserved" not in entry:
                            found_entry = True
                            name = entry.replace("_shift", "")
                            if filter(entry):
                                if not remove_prefix:
                                    yield name
                                else:
                                    if prefix != "kernarg_preload_":
                                        name = name.replace(prefix, "")
                                    name = p_gfx_label.sub("", name)
                                    yield name
                if not found_entry:
                    yield group

    def get_possible_field_names(remove_prefix=True):
        return sorted(
            set(AMDHSAKernelDescriptor._iterate_possible_fields(remove_prefix))
        )

    @staticmethod
    def _iterate_group_entries(group, group_c_type, amdgpu_arch=None):
        r""" "Yields the entries of a group in order (based on shift).

        Yields the entries of a group in order based on their respective shift
        value. Depending on the architecture, it labels certain entries
        differently: While those fields might have a meaning for some
        architectures, the fields might contain no meaningful data for others;
        more details:
        <https://llvm.org/docs/AMDGPUUsage.html#amdhsa-kernel-name>

        Yields:
            Tuples of size 4, where the entries have the following meaning:

            * 1st entry (`str`): The name of the entry.
            * 2nd entry (`type`): The ctypes type of the entry's group.
            * 3rd entry (`int`): width in bit sof the entry.
            * 4th entry (`bool`): If meaningful data is stored.
        """
        group_c_type_bits = ctypes.sizeof(group_c_type) * 8

        p_gfx_specific_group_entry = (
            AMDHSAKernelDescriptor._get_gfx_specific_group_entry_pattern(
                amdgpu_arch
            )
        )

        p_gfx_label = re.compile(r"gfx[0-9a-f]+(_plus)?_")

        # first collect the most fitting entries for the given architecture
        shift_dict = {}
        for entry in AMDHSAKernelDescriptor._group_entry_coordinates.keys():
            if entry.startswith(group) and entry.endswith("_shift"):
                shift_value = AMDHSAKernelDescriptor._group_entry_coordinates[
                    entry
                ]
                if next(p_gfx_specific_group_entry.finditer(entry), None):
                    shift_dict[shift_value] = entry
                elif not next(p_gfx_label.finditer(entry), None):
                    shift_dict[shift_value] = entry

        _num_unused_bits = group_c_type_bits
        for shift_value in sorted(shift_dict.keys()):
            shift_entry = shift_dict[shift_value]  # type: str
            width = AMDHSAKernelDescriptor._group_entry_coordinates[
                shift_entry.replace("_shift", "_width")
            ]
            name = shift_entry.replace("_shift", "")
            has_meaning = "_reserved" not in name and (
                "_gfx" not in name
                or any(p_gfx_specific_group_entry.finditer(name))
            )
            if has_meaning:
                if group != "kernarg_preload":
                    name = name.replace(group + "_", "")
                name = p_gfx_label.sub("", name)
            else:
                name = "_MEANINGLESS_" + name
            yield (name, group_c_type, width, has_meaning)
            # yield last
            _num_unused_bits = group_c_type_bits - (shift_value + width)
        if _num_unused_bits > 0:
            yield (
                group + "__unused_bits__",
                group_c_type,
                _num_unused_bits,
                False,
            )
        else:
            assert _num_unused_bits == 0, _num_unused_bits

    @staticmethod
    def create_type(
        amdgpu_arch, features=[]
    ):  # type: (str, typing.Iterable [str]) -> ctypes.Structure
        c_types_struct_fields = []
        c_types_struct_properties = []

        sum_bits = 0
        for group, ctype in AMDHSAKernelDescriptor._group_types:
            prefix = group
            if prefix == "kernel_code_properties":
                prefix = "kernel_code_property"

            assert type(group) is str
            iterate_group_entries_with_matching_prefix = (
                entry
                for entry in AMDHSAKernelDescriptor._group_entry_coordinates.keys()  # noqa: E501
                if entry.startswith(prefix)
            )

            if any(iterate_group_entries_with_matching_prefix):
                sum_bits_group = 0
                for (
                    attr,
                    attr_c_type,
                    attr_bits,
                    attr_has_meaning,
                ) in AMDHSAKernelDescriptor._iterate_group_entries(
                    prefix, ctype, amdgpu_arch
                ):
                    if attr_has_meaning:
                        c_types_struct_properties.append(attr)
                    c_types_struct_fields.append(
                        (attr, attr_c_type, attr_bits)
                    )
                    sum_bits_group += attr_bits
                assert (
                    sum_bits_group == ctypes.sizeof(ctype) * 8
                ), f"'{group}' has {sum_bits_group} bits; expected: {ctypes.sizeof(ctype) * 8} bits"  # noqa: E501
                sum_bits += sum_bits_group
            else:
                # put main group data type in fields
                if not group.startswith("reserved"):
                    c_types_struct_properties.append(group)
                c_types_struct_fields.append((group, ctype))
                sum_bits += ctypes.sizeof(ctype) * 8

        assert sum_bits == 64 * 8

        m = p_amdgpu_arch.match(amdgpu_arch)
        amdgpu_arch_major = m.group("gen")
        amdgpu_arch_minor = m.group("subgen")

        class Wrapper(ctypes.Structure):
            _fields_ = c_types_struct_fields

            __doc__ = rf"""Kernel descriptor for {amdgpu_arch}.

            Kernel descriptor `ctypes.Structure` specialized for {amdgpu_arch}.

            Note:
                This wrapper gives access to the data stored in the code
                object.

            Note:
                Use

                ```py
                <this_type>.from_buffer_copy(bytes)
                ```

                to instantiate this class from a AMD HSA
                code object v6 <...>.kd code symbol.

            Note:
                Use the `properties()` class method to get the
                names of fields that store meaningful information.
                Then access via `.` or getattr(instance).

            """

            _amdgpu_arch = amdgpu_arch
            _amdgpu_arch_major = amdgpu_arch_major
            _amdgpu_arch_minor = amdgpu_arch_minor
            _features = features

            _properties = c_types_struct_properties

            @classmethod
            def properties(cls):
                yield from cls._properties

            def as_dict(self):
                """All meaningful fields as dict.

                Meaningful fields are those that are not
                marked as 'reserved' blocks of memory
                for the AMD GPU architecture associated
                with this wrapper class.

                Note:
                    Arrays are stored as `bytes`.
                """
                result = {}
                for attr in self.properties():
                    value = getattr(self, attr)
                    if isinstance(value, ctypes.Array):
                        result[attr] = bytes(value)
                    else:
                        result[attr] = value
                return result

            def render_amdhsa_kernel_directive(
                self, kernel_name, guess_next_free_sgpr_etc=True, indent="\t"
            ):
                """Renders an AMD HSA kernel descriptor.

                The result can be appended to the instructions of
                the kernel.

                Args:
                    kernel_name (`str`):
                        Name of the kernel for which to generate the descriptor
                        .rodata section.
                    guess_next_free_sgpr_vgpr_accum_offset (`bool`, optional):
                        Certain amdhsa_kernel directives are required by the
                        AMD HSA assembler but only derived values can be
                        found in the kernel descriptor code symbol.

                        As these derived value are the important ones
                        that are actually read by the GPU's command processor,
                        we can guess the below directive values:

                        .amdhsa_reserve_vcc
                        .amdhsa_reserve_flat_scratch
                        .amdhsa_reserve_xnack_mask
                        .amdhsa_next_free_sgpr
                        .amdhsa_next_free_vgpr

                        Defaults to `True`. Providing 'False' will likely
                        cause the AMD HSA assembler to fail when attempting
                        to compiling the associated kernel.
                    indent (`str`, optional):
                        Indent chars to prepend to the entries of the
                        ``.amdhsa_kernel`` section.

                Typical output:

                ```asm
                .section .rodata,"a",@progbits
                .p2align 6, 0x0
                .amdhsa_kernel KERNEL_NAME
                    .amdhsa_reserve_vcc 0 ; guessed value
                    ; [other directives...]
                .end_amdhsa_kernel
                ```
                """
                body = ""
                if guess_next_free_sgpr_etc:
                    wave32 = False
                    if "enable_wavefront_size32" in self.properties():
                        wave32 = self.enable_wavefront_size32 > 0

                    next_free_sgpr = _guess_next_free_sgpr(
                        self.granulated_wavefront_sgpr_count, self._amdgpu_arch
                    )
                    next_free_vgpr = _guess_next_free_vgpr(
                        self.granulated_workitem_vgpr_count,
                        self._amdgpu_arch,
                        wave32,
                    )

                    # note: not all directives are supported for all archs
                    #       aside from *vcc, and *free_*gpr.
                    guessed_directive_values = {
                        ".amdhsa_reserve_vcc": 0,
                        ".amdhsa_reserve_flat_scratch": 0,
                        ".amdhsa_reserve_xnack_mask": 0,
                        ".amdhsa_next_free_sgpr": next_free_sgpr,
                        ".amdhsa_next_free_vgpr": next_free_vgpr,
                    }

                    body += textwrap.dedent(
                        """\
                    ; NOTE:
                    ;     The next *reserve_(vcc|flat_scratch|snack_mask)
                    ;     and the *next_free_*gpr values are chosen so that
                    ;     they yield the same GRANULATED_WAVEFRONT_SGPR_COUNT
                    ;     and GRANULATED_WORKITEM_VGPR_COUNT values that have
                    ;     been parsed from the original kernel descriptor.
                    ;     This choice poses no issue as eventually only the
                    ;     GRANULATED_WORKITEM_VGPR_COUNT and
                    ;     GRANULATED_WAVEFRONT_SGPR_COUNT info is passed
                    ;     to the device's command processor (CP).
                    ;     More details: https://github.com/ROCm/llvm-project/blob/release/rocm-rel-7.0/llvm/lib/Target/AMDGPU/Disassembler/AMDGPUDisassembler.cpp
                    """  # noqa: E501
                    )
                    body += "\n"

                for dir in amdhsa_kernel_directives.iter_supported_directives(
                    self._amdgpu_arch, self._features
                ):
                    k = dir["dir"]
                    p = dir["kd_prop"]

                    if k == ".amdhsa_accum_offset":
                        # Offset of a first AccVGPR in the unified register
                        # file. Granularity 4. Value 0-63.
                        # 0 - accum-offset = 4
                        # 1 - accum-offset = 8,
                        # …,
                        # 63 - accum-offset = 256.
                        # More details: https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-compute-pgm-rsrc2-gfx6-gfx12-table  # noqa: E501

                        accum_offset_code = int(getattr(self, "accum_offset"))
                        accum_offset_value = 4 * (1 + accum_offset_code)
                        body += f"{k} {int(accum_offset_value)}\n"
                    elif p in self.properties():
                        body += f"{k} {getattr(self, p)}\n"
                    elif k in guessed_directive_values:
                        body += f"{k} {guessed_directive_values[k]} ; guessed value\n"  # noqa: E501

                return textwrap.dedent(
                    f"""\
                    .section .rodata,"a",@progbits
                    .p2align 6, 0x0
                    .amdhsa_kernel {kernel_name}{{body}}
                    .end_amdhsa_kernel
                    """
                ).format(
                    body="\n" + textwrap.indent(body, indent).rstrip("\n")
                )

        assert ctypes.sizeof(Wrapper) == 64
        return Wrapper


def parse_amdgpu_code_obj_kernel_descriptor(
    code_symbol,  # type: (bytes|bytearray)
    amdgpu_arch,  # type: (str)
):
    """Parse kernel descriptor symbol extracted from v6 code object.

    Args:
        code_symbol (bytes|bytearray):
            The bytes of the kernel descriptor code object.
        amdgpu_arch (`str`):
            AMD GPU architecture. An expression like 'gfx90a' or 'gfx1201'.
            Assumes that any feature flags such as `:xnack+` have been stripped
            off.

    Returns:
        A ctypes.Structure with fields for the given architecture.
    """
    amd_kd_t = AMDHSAKernelDescriptor.create_type(amdgpu_arch)
    return amd_kd_t.from_buffer_copy(code_symbol)
