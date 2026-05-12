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

import interfacegen.tree
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


def _make_status_node_init(prefix, status_type: str):
    """Per-Function override: drop ``except? <STATUS> nogil`` for functions
    whose return type is NOT ``<status_type>``.

    ``prefix`` is either a single string or a tuple of strings (matched
    via ``startswith``); pass a tuple for libs with multiple name
    prefixes (e.g. RCCL has ``ncclX`` AND ``pncclX`` profiling variants).

    See ``generators_libraries._make_status_node_init`` for the full
    rationale (handles non-enum returns, different-enum returns, and
    void returns; ``noexcept nogil`` is the right modifier for all of
    them).
    """
    prefixes = (prefix,) if isinstance(prefix, str) else tuple(prefix)

    def _init(node):
        if isinstance(node, interfacegen.tree.Function):
            if not node.name.startswith(prefixes):
                return
            try:
                return_typename = node.cython_global_typename
            except Exception:
                return_typename = ""
            if return_typename != status_type:
                node.error_return_value_lazy_loader = None
                node.modifiers_lazy_loader = " noexcept nogil"
    return _init


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
        # rccl functions return ``ncclResult_t``; ``ncclInternalError``
        # is the sentinel hijacked for Python-exception propagation.
        modifiers_lazy_loader=" except? ncclInternalError nogil",
        error_return_value_lazy_loader="ncclInternalError",
        # ``pnccl*`` profiling-mirror helpers are filtered out by
        # ``rccl.node_filter`` (see rocm.py) — only ``ncclX`` symbols
        # appear here. Non-status-returning helpers like
        # ``ncclGetErrorString`` (returns ``const char *``) and
        # ``ncclResetDebugInit`` (returns ``void``) need ``noexcept
        # nogil`` instead of ``except? ncclInternalError``.
        node_init=_make_status_node_init("nccl", "ncclResult_t"),
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
        # roctx is a profiling-annotation API: functions return
        # ``roctx_range_id_t`` (uint64) or void — no status enum to
        # translate into Python exceptions. Module-wide ``noexcept
        # nogil`` lets call sites be wrapped in ``with nogil:``.
        modifiers_lazy_loader=" noexcept nogil",
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
        # hipfile functions return a ``hipFileError`` struct (by value),
        # not an enum — no single-value sentinel for ``except?`` to
        # match. Use module-wide ``noexcept nogil`` so call sites can
        # still drop the GIL.
        modifiers_lazy_loader=" noexcept nogil",
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
    # Define ENABLE_ESMI_LIB so the 75 `amdsmi_*_cpu_*` functions
    # gated by `#ifdef ENABLE_ESMI_LIB` (CPU energy/freq/boost-limit
    # monitoring) get parsed and bound. Without this define libclang
    # silently skips the entire ESMI block in `amd_smi/amdsmi.h`
    # (lines 7082-8550 in the 7.13 header). The backing
    # `libamd_smi.so` resolves these symbols at runtime via the
    # lazy-loader; if the installed library was compiled without
    # ESMI support, the first call surfaces a clear "symbol not
    # found" rather than the symbol being silently absent at the
    # Python layer.
    amdsmi_cflags = list(generator_args) + ["-DENABLE_ESMI_LIB"]
    generator = CythonModuleGenerator(
        "rocm.bindings.amdsmi",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libamd_smi.so",
        modifiers_lazy_loader=" except? AMDSMI_STATUS_INTERNAL_EXCEPTION nogil",
        error_return_value_lazy_loader="AMDSMI_STATUS_INTERNAL_EXCEPTION",
        node_init=_make_status_node_init("amdsmi", "amdsmi_status_t"),
        node_filter=controls.amdsmi.node_filter,
        macro_type=controls.amdsmi.macro_type,
        ptr_parm_intent=controls.amdsmi.ptr_parm_intent,
        ptr_rank=controls.amdsmi.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=amdsmi_cflags,
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

    Additionally pre-includes `hsa_ext_finalize.h` (the BRIG/HSAIL
    finalizer extension: `hsa_ext_program_create` /
    `hsa_ext_program_finalize` / etc.) via a `-include` cflag so
    libclang's parse surfaces those decls. `hsa_ext_amd.h` does not
    pull the finalize header in transitively, and there is no
    upstream `#define` switch that would do so — only
    `hsa_api_trace.h` includes it, gated behind
    `AMD_INTERNAL_BUILD` plus a tooling-internal `inc/` path that
    isn't shipped in the install tree. The matching gcc-side
    `-include` (so the Cython-generated `.c` compile sees the same
    typedefs) lives in
    `packages/rocm-bindings-systems/CMakeLists.txt` keyed on
    `_lib STREQUAL "hsa"` — same codegen-side ↔ wheel-build-side
    parity pattern as amdsmi's `ENABLE_ESMI_LIB`.

    HSA is independent of HIP — no cross-imports needed.
    """
    hsa_cflags = list(generator_args) + ["-include", "hsa/hsa_ext_finalize.h"]
    generator = CythonModuleGenerator(
        "rocm.bindings.hsa",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhsa-runtime64.so.1",
        # hsa functions return ``hsa_status_t``; ``HSA_STATUS_ERROR`` is
        # the generic failure sentinel hijacked for Python-exception
        # propagation.
        modifiers_lazy_loader=" except? HSA_STATUS_ERROR nogil",
        error_return_value_lazy_loader="HSA_STATUS_ERROR",
        node_init=_make_status_node_init("hsa", "hsa_status_t"),
        node_filter=controls.hsa.node_filter,
        ptr_parm_intent=controls.hsa.ptr_parm_intent,
        ptr_rank=controls.hsa.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=hsa_cflags,
    )
    return generator
