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

"""Per-module generators for the **rocm-bindings-libraries** wheel.

Math / FFT / random / sparse libraries: hipBLAS, hipSOLVER, hipRAND,
hipFFT, hipSPARSE. Each `generate_*` function returns a configured
`CythonModuleGenerator` for the named module.

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
    """Build the header argument for CythonModuleGenerator."""
    if header_content is not None:
        return (header_relpath, header_content)
    return header_relpath


def _make_status_node_init(prefix, status_type: str):
    """Per-Function override: drop ``except? <STATUS> nogil`` for functions
    whose return type is NOT ``<status_type>``.

    ``prefix`` is either a single string or a tuple of strings (matched
    via ``startswith``); pass a tuple when the library uses several name
    prefixes (e.g. RCCL has ``ncclX`` AND ``pncclX`` profiling variants).

    The module-level ``modifiers_lazy_loader=" except? <STATUS_INTERNAL_ERROR>
    nogil"`` only type-checks for functions that actually return the status
    enum. Three families of functions need the override:

    1. **Non-enum returns** — ``hipblasStatusToString`` returns ``const char
       *``, ``hipfftMakePlan`` helpers may return ``int`` / ``size_t``, etc.
       Cython rejects ``except?`` because the sentinel value type doesn't
       match the return type.
    2. **Different-enum returns** — ``hipsparseGetMatType`` returns
       ``hipsparseMatrixType_t`` (NOT ``hipsparseStatus_t``). Same
       type-mismatch problem; ``is_enum`` alone wouldn't catch this.
    3. **void returns** — ``noexcept`` is the only valid modifier.

    For all of these, ``noexcept nogil`` is the right modifier — the
    ``nogil`` declaration still applies (so ``with nogil:`` blocks at call
    sites are valid) but no exception-translation watcher is inserted.

    Same overall pattern as ``hip_node_init`` / ``hiprtc_node_init`` in
    ``generators_hip.py``; those don't need the ``status_type`` check
    because hip's own non-status-returning functions are all non-enum
    (struct/void/char*).
    """
    prefixes = (prefix,) if isinstance(prefix, str) else tuple(prefix)

    def _init(node):
        if isinstance(node, interfacegen.tree.Function):
            if not node.name.startswith(prefixes):
                return
            # Check the function's return-type cython spelling; if it is
            # not the status enum we need ``noexcept`` instead of
            # ``except? <STATUS>``.
            try:
                return_typename = node.cython_global_typename
            except Exception:
                return_typename = ""
            if return_typename != status_type:
                node.error_return_value_lazy_loader = None
                node.modifiers_lazy_loader = " noexcept nogil"
    return _init


def generate_hipblas(
    *,
    include_dir: str,
    header_relpath: str = "hipblas/hipblas.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hipblas",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipblas.so",
        modifiers_lazy_loader=" except? HIPBLAS_STATUS_INTERNAL_ERROR nogil",
        error_return_value_lazy_loader="HIPBLAS_STATUS_INTERNAL_ERROR",
        node_init=_make_status_node_init("hipblas", "hipblasStatus_t"),
        node_filter=controls.hipblas.node_filter,
        ptr_parm_intent=controls.hipblas.ptr_parm_intent,
        ptr_rank=controls.hipblas.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        raw_comment_cleaner=controls.hipblas.raw_comment_cleaner,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport *
    """
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip import _hipDataType__Base
    """
    )
    return generator


def generate_hipsolver(
    *,
    include_dir: str,
    header_relpath: str = "hipsolver/hipsolver.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hipsolver",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipsolver.so",
        modifiers_lazy_loader=" except? HIPSOLVER_STATUS_INTERNAL_ERROR nogil",
        error_return_value_lazy_loader="HIPSOLVER_STATUS_INTERNAL_ERROR",
        node_init=_make_status_node_init("hipsolver", "hipsolverStatus_t"),
        node_filter=controls.hipsolver.node_filter,
        ptr_parm_intent=controls.hipsolver.ptr_parm_intent,
        ptr_rank=controls.hipsolver.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        raw_comment_cleaner=controls.hipsolver.raw_comment_cleaner,
        cflags=generator_args,
    )
    # hipsolver-types.h aliases hipblas typedefs:
    #   typedef hipblasOperation_t hipsolverOperation_t;
    #   typedef hipblasFillMode_t  hipsolverFillMode_t;
    #   typedef hipblasSideMode_t  hipsolverSideMode_t;
    # The codegen emits these as `ctypedef hipblas<X> hipsolver<X>` in
    # cyhipsolver.pxd, so the Cython compiler needs hipblas types in
    # scope via cyhipblas. The high-level hipsolver.pyx similarly
    # references the hipblas Python wrappers.
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    from rocm.bindings.cyhipblas cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport *
    from rocm.bindings.hipblas cimport *
    """
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip import _hipDataType__Base
    from rocm.bindings.hipblas import *
    """
    )
    return generator


def generate_hiprand(
    *,
    include_dir: str,
    header_relpath: str = "hiprand/hiprand.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hiprand",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhiprand.so",
        modifiers_lazy_loader=" except? HIPRAND_STATUS_INTERNAL_ERROR nogil",
        error_return_value_lazy_loader="HIPRAND_STATUS_INTERNAL_ERROR",
        node_init=_make_status_node_init("hiprand", "hiprandStatus_t"),
        node_filter=controls.hiprand.node_filter,
        macro_type=controls.hiprand.macro_type,
        ptr_parm_intent=controls.hiprand.ptr_parm_intent,
        ptr_rank=controls.hiprand.ptr_rank,
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


def generate_hipfft(
    *,
    include_dir: str,
    header_relpath: str = "hipfft/hipfft.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    # NOTE: `hipfft.h` does not `#include` the eXtended-API header
    # `hipfftXt.h`, so the 17 `hipfftXt*` entry points are not part
    # of this binding. Folding both headers into a synthetic TU
    # surfaced multiple downstream codegen issues (OUT-pointer
    # hoisting for `hipfftXtExecDescriptor*`, missing
    # `hipLibXtDesc_t` resolution, missing imports for `hipDataType`).
    # Adding the Xt API is tracked separately; see
    # share/design/CODEGEN.md for the gap list.
    generator = CythonModuleGenerator(
        "rocm.bindings.hipfft",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipfft.so",
        modifiers_lazy_loader=" except? HIPFFT_INTERNAL_ERROR nogil",
        error_return_value_lazy_loader="HIPFFT_INTERNAL_ERROR",
        node_init=_make_status_node_init("hipfft", "hipfftResult"),
        node_filter=controls.hipfft.node_filter,
        macro_type=controls.hipfft.macro_type,
        ptr_parm_intent=controls.hipfft.ptr_parm_intent,
        ptr_rank=controls.hipfft.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport hipStream_t, float2, double2
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport ihipStream_t, float2, double2
    """
    )
    return generator


def generate_hipsparse(
    *,
    include_dir: str,
    header_relpath: str = "hipsparse/hipsparse.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hipsparse",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipsparse.so",
        modifiers_lazy_loader=" except? HIPSPARSE_STATUS_INTERNAL_ERROR nogil",
        error_return_value_lazy_loader="HIPSPARSE_STATUS_INTERNAL_ERROR",
        node_init=_make_status_node_init("hipsparse", "hipsparseStatus_t"),
        node_filter=controls.hipsparse.node_filter,
        macro_type=controls.hipsparse.macro_type,
        ptr_parm_intent=controls.hipsparse.ptr_parm_intent,
        ptr_rank=controls.hipsparse.ptr_rank,
        raw_comment_cleaner=controls.hipsparse.raw_comment_cleaner,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport ihipStream_t, float2, double2 # C import structs/union types
    """
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip import hipError_t, _hipDataType__Base # PY import enums
    """
    )
    return generator


def generate_hipblaslt(
    *,
    include_dir: str,
    header_relpath: str = "hipblaslt/hipblaslt.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hipblaslt",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipblaslt.so",
        # hipblaslt functions return ``hipblasStatus_t`` (the hipblas
        # enum, NOT a separate hipblasLt enum) — same sentinel.
        modifiers_lazy_loader=" except? HIPBLAS_STATUS_INTERNAL_ERROR nogil",
        error_return_value_lazy_loader="HIPBLAS_STATUS_INTERNAL_ERROR",
        # NOTE: prefix is camelCase "hipblasLt" — the C functions
        # spell it that way (hipblasLtCreate, hipblasLtInitialize,
        # ...) and the prefix match in `_make_status_node_init` is
        # case-sensitive. A lowercase "hipblaslt" prefix never
        # matches and the void-return / non-status-return fixup
        # silently doesn't fire.
        node_init=_make_status_node_init("hipblasLt", "hipblasStatus_t"),
        node_filter=controls.hipblaslt.node_filter,
        ptr_parm_intent=controls.hipblaslt.ptr_parm_intent,
        ptr_rank=controls.hipblaslt.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        raw_comment_cleaner=controls.hipblaslt.raw_comment_cleaner,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    from rocm.bindings.cyhipblas cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport *
    from rocm.bindings.hipblas cimport *
    """
    )
    # The hipblaslt API surface uses several hipblas enums directly
    # (`hipblasComputeType_t`, `hipblasOperation_t`, etc.) — the
    # generated `.pyx` does runtime `isinstance(arg,
    # _<EnumName>__Base)` checks against those enums' Python wrapper
    # base classes. Those wrapper classes live in `rocm.bindings.hipblas`
    # at module scope; without the star-import they aren't visible to
    # hipblaslt's `.pyx` and Cython compile fails with `undeclared
    # name not builtin: _hipblasComputeType_t__Base`. Same shape as
    # hipsolver's prolog above.
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip import _hipDataType__Base
    from rocm.bindings.hipblas import *
    """
    )
    return generator


def generate_hiptensor(
    *,
    include_dir: str,
    header_relpath: str = "hiptensor/hiptensor.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hiptensor",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhiptensor.so",
        modifiers_lazy_loader=" except? HIPTENSOR_STATUS_INTERNAL_ERROR nogil",
        error_return_value_lazy_loader="HIPTENSOR_STATUS_INTERNAL_ERROR",
        node_init=_make_status_node_init("hiptensor", "hiptensorStatus_t"),
        node_filter=controls.hiptensor.node_filter,
        ptr_parm_intent=controls.hiptensor.ptr_parm_intent,
        ptr_rank=controls.hiptensor.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    # `hiptensorLoggerSetFile(FILE * file)` is the lone consumer of a
    # libc-defined struct in hiptensor's public surface. Cython ships
    # an opaque `ctypedef struct FILE` in
    # `Cython/Includes/libc/stdio.pxd`. cimport it twice — once
    # under its public `FILE` name and once aliased to `_IO_FILE`
    # (glibc's canonical struct tag, which is what the codegen
    # renderer emits for the function-decl parameter type because
    # libclang reports the canonical spelling). The alias lets a
    # single libc-defined opaque type back BOTH spellings the
    # binding references. The high-level wrapper rendering for this
    # parm routes through `rocm.bindings.util.types.Pointer` via the
    # foreign-record fallback in
    # `_function.py:handle_in_inout_ptr_`.
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    from libc.stdio cimport FILE
    from libc.stdio cimport FILE as _IO_FILE
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport *
    """
    )
    return generator


def generate_hipdnn(
    *,
    include_dir: str,
    header_relpath: str = "hipdnn/backend/hipdnn_backend.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    # Upstream hipdnn ships C-API headers that contain C++ `constexpr`
    # constants (e.g. `constexpr hipdnnPluginLoadingMode_ext_t
    # HIPDNN_DEFAULT_PLUGIN_LOADING_MODE = ...` in
    # `HipdnnBackendPluginLoadingMode.h`). Without this define, both
    # libclang's parse and the generated Cython .c's gcc compile fail
    # on `unknown type name 'constexpr'`. Mapping the keyword to
    # `const` is semantically equivalent for these declarations
    # (compile-time constant of an enum type) and lets the bindings
    # build with a stock C compiler. Filed-against-upstream as
    # ROCm/rocm-libraries hipdnn issue (link in
    # share/design/UPSTREAM_BUGS).
    hipdnn_cflags = list(generator_args) + ["-Dconstexpr=const"]
    generator = CythonModuleGenerator(
        "rocm.bindings.hipdnn",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipdnn.so",
        modifiers_lazy_loader=" except? HIPDNN_STATUS_INTERNAL_ERROR nogil",
        error_return_value_lazy_loader="HIPDNN_STATUS_INTERNAL_ERROR",
        node_init=_make_status_node_init("hipdnn", "hipdnnStatus_t"),
        node_filter=controls.hipdnn.node_filter,
        ptr_parm_intent=controls.hipdnn.ptr_parm_intent,
        ptr_rank=controls.hipdnn.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=hipdnn_cflags,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport *
    """
    )
    return generator


def generate_hipsparselt(
    *,
    include_dir: str,
    header_relpath: str = "hipsparselt/hipsparselt.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hipsparselt",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipsparselt.so",
        # hipsparselt functions return ``hipsparseStatus_t`` (the
        # hipsparse enum, NOT a separate hipsparseLt enum).
        modifiers_lazy_loader=" except? HIPSPARSE_STATUS_INTERNAL_ERROR nogil",
        error_return_value_lazy_loader="HIPSPARSE_STATUS_INTERNAL_ERROR",
        # NOTE: prefix is camelCase "hipsparseLt" — same case-
        # sensitivity caveat as `generate_hipblaslt` above. The C
        # functions spell it that way (hipsparseLtInitialize,
        # hipsparseLtMatmulPlanInit, ...) and a lowercase prefix
        # would silently skip the void-return fixup.
        node_init=_make_status_node_init("hipsparseLt", "hipsparseStatus_t"),
        node_filter=controls.hipsparselt.node_filter,
        macro_type=controls.hipsparselt.macro_type,
        ptr_parm_intent=controls.hipsparselt.ptr_parm_intent,
        ptr_rank=controls.hipsparselt.ptr_rank,
        raw_comment_cleaner=controls.hipsparselt.raw_comment_cleaner,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    from rocm.bindings.cyhipsparse cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport ihipStream_t, float2, double2
    from rocm.bindings.hipsparse cimport *
    """
    )
    # hipsparselt's API takes a small set of hipsparse enum
    # arguments directly (`hipsparseOperation_t`,
    # `hipsparseOrder_t`) — the generated `.pyx` runtime-checks
    # them via `isinstance(arg, _<EnumName>__Base)`. Those base
    # classes live in `rocm.bindings.hipsparse`; without the
    # explicit Python-level import Cython compile fails with
    # `undeclared name not builtin: _hipsparseOperation_t__Base`.
    # Import only the names actually needed (rather than `import *`)
    # — broader star-imports pull in `cdef class` types that
    # collide with the ones cimport'd above and trigger
    # `Cannot overwrite C type` at module init time.
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip import hipError_t, _hipDataType__Base, _hipLibraryPropertyType__Base
    from rocm.bindings.hipsparse import (
        hipsparseStatus_t,
        _hipsparseOperation_t__Base,
        _hipsparseOrder_t__Base,
    )
    """
    )
    return generator
