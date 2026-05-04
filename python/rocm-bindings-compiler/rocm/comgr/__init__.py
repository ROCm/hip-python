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

"""AMD Code Object Manager (Comgr) high-level Python API.

This package provides a high-level Pythonic interface to AMD's Code Object Manager
library for manipulating GPU code objects.

High-level API:
    Data, DataSet, Action classes
    compile_hip_to_bc, compile_bc_to_hsa, compile_hsa functions
    disassemble_* functions
    parse_code_obj_* functions

Low-level bindings:
    amd_comgr_status_s, amd_comgr_data_kind_s, amd_comgr_language_s (enums)
    amd_comgr_* functions (direct C API wrappers)

Kernel descriptor utilities:
    amd_hsa_kernel_descriptor module with parse functions
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# High-level API from comgr.py
from rocm.comgr.comgr import (
    Data,
    DataSet,
    Action,
    Symbol,
    compile_hip_to_bc,
    compile_bc_to_hsa,
    compile_hip_to_hsa,
    compile_bc,
    compile_hsa,
    disassemble_program,
    disassemble_code_obj_function,
    disassemble_amdhsa_code_obj_v6_kernel,
    parse_code_obj_metadata,
    parse_code_obj_kernel_names,
    parse_code_symbols,
    parse_data_symbols,
    parse_data_metadata,
    get_isa_metadata_all,
    get_isa_metadata,
    get_isa_names,
    dump_metadata_yaml,
    HIPRTC_RUNTIME_HEADER,
)

# Re-export low-level bindings from rocm.bindings.amd_comgr
from rocm.bindings.amd_comgr import *

# Re-export kernel descriptor utilities as modules
from rocm.comgr import amd_hsa_kernel_descriptor
from rocm.comgr import amdhsa_kernel_directives
