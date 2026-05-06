#!/usr/bin/env python3

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

# flake8: noqa

"""AMD HSA kernel directives

Note the JavaScript code for generating the main content of this file is shown
below. You can copy and insert the code into your browser's console when
visiting <https://llvm.org/docs/AMDGPUUsage.html> to regenerate or update
the main content of this module.

```js
let directives = []
const p_kd_property = /Controls (?<kd_prop>[A-Z0-9_]+) /
const p_except = /\(except (?<exceptions>([A-Z0-9_]+)+)\)/
const p_tgt_feat = /TargetFeatureSpecific\((?<feat>\w+)\)/
document.body.querySelector("#amdhsa-kernel-directives-table").querySelectorAll("tr").forEach( tr => {
    let cells = []
    tr.querySelectorAll("td p").forEach( td => {
        cells.push(td.textContent.replace(/(\r\n|\n|\r)/gm, " "));
    })
    if (cells.length > 0) {
        // console.log(cells)
        const found = cells[3].match(p_kd_property)
        let kd_prop = ""
        if (found) {
            kd_prop = found.groups["kd_prop"].toLowerCase()
        }

        const supported_raw = cells[2]
        const supported = supported_raw.replace(p_except,"").replaceAll(" ","").split(",")

        const found2 = supported_raw.match(p_except)
        let unsupported = []
        if ( found2 ) {
            unsupported = found2.groups["exceptions"].replaceAll(" ","").split(",")
        }

        let thedefault = cells[1]
        const found3 = thedefault.replaceAll(" ","").match(p_tgt_feat)
        if ( found3 ) {
            thedefault = ":"+found3.groups["feat"]+"+";
        } else if ( thedefault === "Required" ) {
            thedefault = "REQUIRED";
        } else {
            thedefault = parseInt(thedefault);
        }

        // console.log(cells[1])

        directives.push({
            "dir": cells[0],
            "kd_prop": kd_prop,
            "default": thedefault,
            "supported": supported,
            "unsupported": unsupported,
            "default_raw": cells[1],
            "supported_raw": supported_raw,
            "description": cells[3]
        })
    }
})

arch_supported_expressions = new Set()
arch_unsupported_expressions = new Set()

directives.forEach(
    dir => {
       dir["supported"].forEach(arch_supported_expressions .add, arch_supported_expressions)
       dir["unsupported"].forEach(arch_unsupported_expressions.add, arch_unsupported_expressions)
    }
)
let output = `

# The content below was generated from https://llvm.org/docs/AMDGPUUsage.html

# These expressions appear in the directives'"supported" list field.
arch_supported_expressions = ${JSON.stringify(Array.from(arch_supported_expressions), null, 2)}

# These expressions appear in the directives'"unsupported" list field.
arch_unsupported_expressions = ${JSON.stringify(Array.from(arch_unsupported_expressions), null, 2)}

directives = ${JSON.stringify(directives, null, 2)}`

console.log(output)
```
"""

import re
from typing import Generator

p_amdgpu_arch = re.compile(r"gfx(?P<gen>[0-9]{1,2})(?P<subgen>[0-9a-f]{2})$")


def split_amdgpu_arch(amdgpu_arch):
    m = p_amdgpu_arch.match(amdgpu_arch)
    assert m, f"no match for {amdgpu_arch}"
    gfx_gen = int(m.group("gen"), 16)
    gfx_subgen = int(m.group("subgen"), 16)
    return (gfx_gen, gfx_subgen)


def is_amdgpu_arch_supported(
    supported_expression, amdgpu_arch
):  # type: (str,str) -> str
    if not amdgpu_arch.startswith("gfx"):
        return False

    gfx_gen, gfx_subgen = split_amdgpu_arch(amdgpu_arch)

    if supported_expression in (
        "GFX1250+",
        "GFX12.5",
    ):
        return gfx_gen == 12 and gfx_subgen >= 50
    elif supported_expression == "GFX12":
        return gfx_gen == 12
    elif supported_expression in ("GFX942", "GFX90A"):
        return supported_expression.lower() == amdgpu_arch
    else:
        assert "-" in supported_expression
        min_gen, max_gen = supported_expression.replace("GFX", "").split("-")
        return gfx_gen >= int(min_gen, 16) and gfx_gen <= int(max_gen, 16)


def is_amdgpu_arch_unsupported(
    unsupported_expression, amdgpu_arch
):  # type: (str,str) -> str
    if unsupported_expression in ("GFX942"):
        return unsupported_expression.lower() == amdgpu_arch


def iter_supported_directives(
    amdgpu_arch, features
):  # type: (str, list[str]) -> Generator[dict[str,object]]
    for dir in directives:
        if any(
            is_amdgpu_arch_supported(e, amdgpu_arch) for e in dir["supported"]
        ) and not any(
            is_amdgpu_arch_unsupported(e, amdgpu_arch)
            for e in dir["unsupported"]
        ):
            default_value = dir["default"]  # something like: ':xnack+'
            if isinstance(default_value, str) and default_value.startswith(
                ":"
            ):
                if any(
                    f.lstrip(":") == default_value.lstrip(":")
                    for f in features
                ):
                    yield dir
            else:
                yield dir


# The content below was generated from https://llvm.org/docs/AMDGPUUsage.html

# These expressions appear in the directives'"supported" list field.
arch_supported_expressions = [
    "GFX6-GFX12",
    "GFX6-GFX10",
    "GFX12.5",
    "GFX10-GFX12",
    "GFX1250+",
    "GFX942",
    "GFX11-GFX12",
    "GFX90A",
    "GFX7-GFX10",
    "GFX8-GFX10",
    "GFX6-GFX11",
    "GFX12",
    "GFX9-GFX12",
    "GFX10-GFX11",
]

# These expressions appear in the directives'"unsupported" list field.
arch_unsupported_expressions = ["GFX942"]

directives = [
    {
        "dir": ".amdhsa_group_segment_fixed_size",
        "kd_prop": "group_segment_fixed_size",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls GROUP_SEGMENT_FIXED_SIZE in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_private_segment_fixed_size",
        "kd_prop": "private_segment_fixed_size",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls PRIVATE_SEGMENT_FIXED_SIZE in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_kernarg_size",
        "kd_prop": "kernarg_size",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls KERNARG_SIZE in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_user_sgpr_count",
        "kd_prop": "user_sgpr_count",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls USER_SGPR_COUNT in COMPUTE_PGM_RSRC2 compute_pgm_rsrc2 for GFX6-GFX12",
    },
    {
        "dir": ".amdhsa_user_sgpr_private_segment_buffer",
        "kd_prop": "enable_sgpr_private_segment_buffer",
        "default": 0,
        "supported": ["GFX6-GFX10"],
        "unsupported": ["GFX942"],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX10 (except GFX942)",
        "description": "Controls ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_user_sgpr_dispatch_ptr",
        "kd_prop": "enable_sgpr_dispatch_ptr",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_SGPR_DISPATCH_PTR in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_user_sgpr_queue_ptr",
        "kd_prop": "enable_sgpr_queue_ptr",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_SGPR_QUEUE_PTR in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_user_sgpr_kernarg_segment_ptr",
        "kd_prop": "enable_sgpr_kernarg_segment_ptr",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_SGPR_KERNARG_SEGMENT_PTR in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_user_sgpr_dispatch_id",
        "kd_prop": "enable_sgpr_dispatch_id",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_SGPR_DISPATCH_ID in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_user_sgpr_flat_scratch_init",
        "kd_prop": "enable_sgpr_flat_scratch_init",
        "default": 0,
        "supported": ["GFX6-GFX10"],
        "unsupported": ["GFX942"],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX10 (except GFX942)",
        "description": "Controls ENABLE_SGPR_FLAT_SCRATCH_INIT in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_user_sgpr_private_segment_size",
        "kd_prop": "enable_sgpr_private_segment_size",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_SGPR_PRIVATE_SEGMENT_SIZE in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_uses_cu_stores",
        "kd_prop": "uses_cu_stores",
        "default": 0,
        "supported": ["GFX12.5"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX12.5",
        "description": "Controls USES_CU_STORES in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_wavefront_size32",
        "kd_prop": "enable_wavefront_size32",
        "default": ":wavefrontsize64+",
        "supported": ["GFX10-GFX12"],
        "unsupported": [],
        "default_raw": "Target Feature Specific (wavefrontsize64)",
        "supported_raw": "GFX10-GFX12",
        "description": "Controls ENABLE_WAVEFRONT_SIZE32 in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_uses_dynamic_stack",
        "kd_prop": "uses_dynamic_stack",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls USES_DYNAMIC_STACK in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_named_barrier_count",
        "kd_prop": "named_bar_cnt",
        "default": 0,
        "supported": ["GFX1250+"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX1250+",
        "description": "Controls NAMED_BAR_CNT in compute_pgm_rsrc3 for GFX12.",
    },
    {
        "dir": ".amdhsa_system_sgpr_private_segment_wavefront_offset",
        "kd_prop": "enable_private_segment",
        "default": 0,
        "supported": ["GFX6-GFX10"],
        "unsupported": ["GFX942"],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX10 (except GFX942)",
        "description": "Controls ENABLE_PRIVATE_SEGMENT in compute_pgm_rsrc2 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_enable_private_segment",
        "kd_prop": "enable_private_segment",
        "default": 0,
        "supported": ["GFX942", "GFX11-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX942, GFX11-GFX12",
        "description": "Controls ENABLE_PRIVATE_SEGMENT in compute_pgm_rsrc2 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_system_sgpr_workgroup_id_x",
        "kd_prop": "enable_sgpr_workgroup_id_x",
        "default": 1,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "1",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_SGPR_WORKGROUP_ID_X in compute_pgm_rsrc2 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_system_sgpr_workgroup_id_y",
        "kd_prop": "enable_sgpr_workgroup_id_y",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_SGPR_WORKGROUP_ID_Y in compute_pgm_rsrc2 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_system_sgpr_workgroup_id_z",
        "kd_prop": "enable_sgpr_workgroup_id_z",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_SGPR_WORKGROUP_ID_Z in compute_pgm_rsrc2 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_system_sgpr_workgroup_info",
        "kd_prop": "enable_sgpr_workgroup_info",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_SGPR_WORKGROUP_INFO in compute_pgm_rsrc2 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_system_vgpr_workitem_id",
        "kd_prop": "enable_vgpr_workitem_id",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_VGPR_WORKITEM_ID in compute_pgm_rsrc2 for GFX6-GFX12. Possible values are defined in System VGPR Work-Item ID Enumeration Values.",
    },
    {
        "dir": ".amdhsa_next_free_vgpr",
        "kd_prop": "",
        "default": "REQUIRED",
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "Required",
        "supported_raw": "GFX6-GFX12",
        "description": "Maximum VGPR number explicitly referenced, plus one. Used to calculate GRANULATED_WORKITEM_VGPR_COUNT in compute_pgm_rsrc1 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_next_free_sgpr",
        "kd_prop": "",
        "default": "REQUIRED",
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "Required",
        "supported_raw": "GFX6-GFX12",
        "description": "Maximum SGPR number explicitly referenced, plus one. Used to calculate GRANULATED_WAVEFRONT_SGPR_COUNT in compute_pgm_rsrc1 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_accum_offset",
        "kd_prop": "",
        "default": "REQUIRED",
        "supported": ["GFX90A", "GFX942"],
        "unsupported": [],
        "default_raw": "Required",
        "supported_raw": "GFX90A, GFX942",
        "description": "Offset of a first AccVGPR in the unified register file. Used to calculate ACCUM_OFFSET in compute_pgm_rsrc3 for GFX90A, GFX942.",
    },
    {
        "dir": ".amdhsa_reserve_vcc",
        "kd_prop": "",
        "default": 1,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "1",
        "supported_raw": "GFX6-GFX12",
        "description": "Whether the kernel may use the special VCC SGPR. Used to calculate GRANULATED_WAVEFRONT_SGPR_COUNT in compute_pgm_rsrc1 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_reserve_flat_scratch",
        "kd_prop": "",
        "default": 1,
        "supported": ["GFX7-GFX10"],
        "unsupported": ["GFX942"],
        "default_raw": "1",
        "supported_raw": "GFX7-GFX10 (except GFX942)",
        "description": "Whether the kernel may use flat instructions to access scratch memory. Used to calculate GRANULATED_WAVEFRONT_SGPR_COUNT in compute_pgm_rsrc1 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_reserve_xnack_mask",
        "kd_prop": "",
        "default": ":xnack+",
        "supported": ["GFX8-GFX10"],
        "unsupported": [],
        "default_raw": "Target Feature Specific (xnack)",
        "supported_raw": "GFX8-GFX10",
        "description": "Whether the kernel may trigger XNACK replay. Used to calculate GRANULATED_WAVEFRONT_SGPR_COUNT in compute_pgm_rsrc1 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_float_round_mode_32",
        "kd_prop": "float_round_mode_32",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls FLOAT_ROUND_MODE_32 in compute_pgm_rsrc1 for GFX6-GFX12. Possible values are defined in Floating Point Rounding Mode Enumeration Values.",
    },
    {
        "dir": ".amdhsa_float_round_mode_16_64",
        "kd_prop": "float_round_mode_16_64",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls FLOAT_ROUND_MODE_16_64 in compute_pgm_rsrc1 for GFX6-GFX12. Possible values are defined in Floating Point Rounding Mode Enumeration Values.",
    },
    {
        "dir": ".amdhsa_float_denorm_mode_32",
        "kd_prop": "float_denorm_mode_32",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls FLOAT_DENORM_MODE_32 in compute_pgm_rsrc1 for GFX6-GFX12. Possible values are defined in Floating Point Denorm Mode Enumeration Values.",
    },
    {
        "dir": ".amdhsa_float_denorm_mode_16_64",
        "kd_prop": "float_denorm_mode_16_64",
        "default": 3,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "3",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls FLOAT_DENORM_MODE_16_64 in compute_pgm_rsrc1 for GFX6-GFX12. Possible values are defined in Floating Point Denorm Mode Enumeration Values.",
    },
    {
        "dir": ".amdhsa_dx10_clamp",
        "kd_prop": "enable_dx10_clamp",
        "default": 1,
        "supported": ["GFX6-GFX11"],
        "unsupported": [],
        "default_raw": "1",
        "supported_raw": "GFX6-GFX11",
        "description": "Controls ENABLE_DX10_CLAMP in compute_pgm_rsrc1 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_ieee_mode",
        "kd_prop": "enable_ieee_mode",
        "default": 1,
        "supported": ["GFX6-GFX11"],
        "unsupported": [],
        "default_raw": "1",
        "supported_raw": "GFX6-GFX11",
        "description": "Controls ENABLE_IEEE_MODE in compute_pgm_rsrc1 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_round_robin_scheduling",
        "kd_prop": "enable_wg_rr_en",
        "default": 0,
        "supported": ["GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX12",
        "description": "Controls ENABLE_WG_RR_EN in compute_pgm_rsrc1 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_fp16_overflow",
        "kd_prop": "fp16_ovfl",
        "default": 0,
        "supported": ["GFX9-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX9-GFX12",
        "description": "Controls FP16_OVFL in compute_pgm_rsrc1 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_tg_split",
        "kd_prop": "tg_split",
        "default": ":tgsplit+",
        "supported": ["GFX90A", "GFX942", "GFX11-GFX12"],
        "unsupported": [],
        "default_raw": "Target Feature Specific (tgsplit)",
        "supported_raw": "GFX90A, GFX942, GFX11-GFX12",
        "description": "Controls TG_SPLIT in compute_pgm_rsrc3 for GFX90A, GFX942.",
    },
    {
        "dir": ".amdhsa_workgroup_processor_mode",
        "kd_prop": "enable_wgp_mode",
        "default": ":cumode+",
        "supported": ["GFX10-GFX12"],
        "unsupported": [],
        "default_raw": "Target Feature Specific (cumode)",
        "supported_raw": "GFX10-GFX12",
        "description": "Controls ENABLE_WGP_MODE in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_memory_ordered",
        "kd_prop": "mem_ordered",
        "default": 1,
        "supported": ["GFX10-GFX12"],
        "unsupported": [],
        "default_raw": "1",
        "supported_raw": "GFX10-GFX12",
        "description": "Controls MEM_ORDERED in compute_pgm_rsrc1 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_forward_progress",
        "kd_prop": "fwd_progress",
        "default": 1,
        "supported": ["GFX10-GFX12"],
        "unsupported": [],
        "default_raw": "1",
        "supported_raw": "GFX10-GFX12",
        "description": "Controls FWD_PROGRESS in compute_pgm_rsrc1 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_shared_vgpr_count",
        "kd_prop": "shared_vgpr_count",
        "default": 0,
        "supported": ["GFX10-GFX11"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX10-GFX11",
        "description": "Controls SHARED_VGPR_COUNT in compute_pgm_rsrc3 for GFX10-GFX11.",
    },
    {
        "dir": ".amdhsa_inst_pref_size",
        "kd_prop": "inst_pref_size",
        "default": 0,
        "supported": ["GFX11-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX11-GFX12",
        "description": "Controls INST_PREF_SIZE in compute_pgm_rsrc3 for GFX10-GFX11 or compute_pgm_rsrc3 for GFX12",
    },
    {
        "dir": ".amdhsa_exception_fp_ieee_invalid_op",
        "kd_prop": "enable_exception_ieee_754_fp_invalid_operation",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION in compute_pgm_rsrc2 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_exception_fp_denorm_src",
        "kd_prop": "enable_exception_fp_denormal_source",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_EXCEPTION_FP_DENORMAL_SOURCE in compute_pgm_rsrc2 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_exception_fp_ieee_div_zero",
        "kd_prop": "enable_exception_ieee_754_fp_division_by_zero",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO in compute_pgm_rsrc2 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_exception_fp_ieee_overflow",
        "kd_prop": "enable_exception_ieee_754_fp_overflow",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW in compute_pgm_rsrc2 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_exception_fp_ieee_underflow",
        "kd_prop": "enable_exception_ieee_754_fp_underflow",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW in compute_pgm_rsrc2 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_exception_fp_ieee_inexact",
        "kd_prop": "enable_exception_ieee_754_fp_inexact",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_EXCEPTION_IEEE_754_FP_INEXACT in compute_pgm_rsrc2 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_exception_int_div_zero",
        "kd_prop": "enable_exception_int_divide_by_zero",
        "default": 0,
        "supported": ["GFX6-GFX12"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX6-GFX12",
        "description": "Controls ENABLE_EXCEPTION_INT_DIVIDE_BY_ZERO in compute_pgm_rsrc2 for GFX6-GFX12.",
    },
    {
        "dir": ".amdhsa_user_sgpr_kernarg_preload_length",
        "kd_prop": "kernarg_preload_spec_length",
        "default": 0,
        "supported": ["GFX90A", "GFX942"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX90A, GFX942",
        "description": "Controls KERNARG_PRELOAD_SPEC_LENGTH in Code Object V3 Kernel Descriptor.",
    },
    {
        "dir": ".amdhsa_user_sgpr_kernarg_preload_offset",
        "kd_prop": "kernarg_preload_spec_offset",
        "default": 0,
        "supported": ["GFX90A", "GFX942"],
        "unsupported": [],
        "default_raw": "0",
        "supported_raw": "GFX90A, GFX942",
        "description": "Controls KERNARG_PRELOAD_SPEC_OFFSET in Code Object V3 Kernel Descriptor.",
    },
]

# The above code is autogenerated

# if __name__ == "__main__":

# for dir in iter_supported_directives("gfx942", ["xnack"]):
#     name = dir["dir"]
#     supported_raw = dir["supported_raw"]
#     print(f"- {name} - {supported_raw}")
