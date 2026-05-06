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

"""Per-module generators for the **rocm-bindings-systems** wheel.

System-level libraries: RCCL (collective communication), ROCTX
(profiling/tracing), and hipFILE (accelerated file I/O). Each `generate_*`
function returns a configured `CythonModuleGenerator` for the named module.

Generator functions accept explicit `include_dir` and `header_relpath` so the
Cython `extern from` statements use the exact paths intended (no derivation
via os.path.dirname). Optional `header_content` enables in-memory rendering
of .h.in templates without requiring CMake configure.
"""

import textwrap

from interfacegen.cython import CythonModuleGenerator
from interfacegen.support.recipes import rocm as controls


def _make_header_arg(header_relpath: str, header_content: str = None):
    """Build the header argument for CythonModuleGenerator.

    If header_content is provided, returns a tuple (relpath, content) that
    libclang treats as an unsaved (in-memory) file. Otherwise returns the
    relpath string and libclang reads from disk.
    """
    if header_content is not None:
        return (header_relpath, header_content)
    return header_relpath


def generate_rccl(
    *,
    include_dir: str,
    header_relpath: str = "rccl/rccl.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.rccl",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="librccl.so",
        node_filter=controls.rccl.node_filter,
        macro_type=controls.rccl.macro_type,
        ptr_parm_intent=controls.rccl.ptr_parm_intent,
        ptr_rank=controls.rccl.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport hipStream_t
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport ihipStream_t
    """
    )
    return generator


def generate_roctx(
    *,
    include_dir: str,
    header_relpath: str = "roctracer/roctx.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.roctx",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libroctx64.so",
        node_filter=controls.roctx.node_filter,
        macro_type=controls.roctx.macro_type,
        ptr_parm_intent=controls.roctx.ptr_parm_intent,
        ptr_rank=controls.roctx.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    return generator


def generate_hipfile(
    *,
    include_dir: str,
    header_relpath: str = "hipfile.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    """Generator for hipFILE (Accelerated I/O Storage) bindings."""
    generator = CythonModuleGenerator(
        "rocm.bindings.hipfile",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipfile.so",
        node_filter=controls.hipfile.node_filter,
        macro_type=controls.hipfile.macro_type,
        ptr_parm_intent=controls.hipfile.ptr_parm_intent,
        ptr_rank=controls.hipfile.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    return generator


def generate_amdsmi(
    *,
    include_dir: str,
    header_relpath: str = "amd_smi/amdsmi.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    """Generator for AMD SMI (System Management Interface) bindings."""
    generator = CythonModuleGenerator(
        "rocm.bindings.amdsmi",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libamd_smi.so",
        node_filter=controls.amdsmi.node_filter,
        macro_type=controls.amdsmi.macro_type,
        ptr_parm_intent=controls.amdsmi.ptr_parm_intent,
        ptr_rank=controls.amdsmi.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    return generator


def generate_hsa(
    *,
    include_dir: str,
    header_relpath: str = "hsa/hsa_ext_amd.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    """Generator for HSA (Heterogeneous System Architecture) bindings.

    Binds `hsa_ext_amd.h`, which transitively includes `hsa.h`,
    `hsa_ext_image.h`, and `hsa_ven_amd_pc_sampling.h`. The resulting
    `rocm.bindings.hsa` module exposes core HSA + AMD extensions +
    image extensions + AMD vendor PC-sampling in a single namespace
    (all `hsa_*` / `HSA_*` symbols).

    HSA is independent of HIP — no cross-imports needed.
    """
    generator = CythonModuleGenerator(
        "rocm.bindings.hsa",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhsa-runtime64.so.1",
        node_filter=controls.hsa.node_filter,
        ptr_parm_intent=controls.hsa.ptr_parm_intent,
        ptr_rank=controls.hsa.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    return generator
