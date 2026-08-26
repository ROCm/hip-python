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
from interfacegen.support.recipes.control import ParmIntent


def _make_header_arg(header_relpath: str, header_content: str = None):
    """Build the header argument for CythonModuleGenerator.

    If header_content is provided, returns a tuple (relpath, content) that
    libclang treats as an unsaved (in-memory) file. Otherwise returns the
    relpath string and libclang reads from disk.
    """
    if header_content is not None:
        return (header_relpath, header_content)
    return header_relpath


def _make_status_node_init(prefix, status_type: str, success_const: str):
    """Per-Function override: drop ``except? <STATUS> nogil`` for functions
    whose return type is NOT ``<status_type>``, and prepend the status
    enum's success value as their first Python return value.

    ``prefix`` is either a single string or a tuple of strings (matched
    via ``startswith``); pass a tuple for libs with multiple name
    prefixes (e.g. RCCL has ``ncclX`` AND ``pncclX`` profiling variants).

    See ``generators_libraries._make_status_node_init`` for the full
    rationale (handles non-enum returns, different-enum returns, and
    void returns; ``noexcept nogil`` is the right modifier for all of
    them, and the ``<status_type>.<success_const>`` prepend keeps the
    status-first-tuple contract uniform under
    ``python_interface_always_return_tuple``).
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
                node.prepend_python_return_value(
                    f"{status_type}.{success_const}",
                    status_type,
                    f"Always returns `~.{status_type}.{success_const}`.",
                )

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
        module_opts={"python_interface_always_return_tuple": True},
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
        node_init=_make_status_node_init(
            "nccl", "ncclResult_t", "ncclSuccess"
        ),
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
        # nogil`` lets call sites be wrapped in ``with nogil:``. There
        # is no status to prepend, so the always-return-tuple flag just
        # wraps bare returns in a 1-tuple for shape uniformity.
        module_opts={"python_interface_always_return_tuple": True},
        modifiers_lazy_loader=" noexcept nogil",
        node_filter=controls.roctx.node_filter,
        macro_type=controls.roctx.macro_type,
        ptr_parm_intent=controls.roctx.ptr_parm_intent,
        ptr_rank=controls.roctx.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    return generator


# Plain POSIX structs that hipFILE references BY POINTER ONLY:
#   * ``struct sockaddr`` (<sys/socket.h>) — userspace-RDMA fs-op callbacks
#     ``getRDMADeviceList`` / ``getRDMADevicePriority``.
#   * ``struct timespec`` (<time.h>)       — batch-I/O poll timeout
#     (``hipFileIOEvents`` query).
# Both carry no ``hipFile`` prefix, so the shared recipe node_filter rejects
# them (correctly — they are not part of the library surface). Rather than
# admit them and emit their full platform-specific field layout from the AST,
# we hand-declare them as OPAQUE structs (``cdef struct X: pass``) in the
# module prolog below. hipFILE only ever passes them by pointer, so no layout
# is needed; an opaque struct is the standard Cython idiom for a pointer-only
# type (same shape as the ``ihipStream_t`` / ``hipArray`` opaque handles in
# cyhip.pxd). This keeps the platform-dependent provisioning in one place a
# future Windows hipFILE build would edit, bakes no Linux ABI (``sa_family``
# widths, ``time_t`` size) into the bindings, and avoids Cython's Linux-only
# ``posix.time`` cimport. The real definitions come from hipfile.h's
# transitive system includes at C-compile time.
_HIPFILE_OPAQUE_PTR_TYPES_DECL = """\
cdef extern from "hipfile.h":
    # Pointer-only POSIX structs (see generators_systems.py). Opaque: hipFILE
    # never dereferences them, so no field layout is bound.
    cdef struct sockaddr:
        pass
    cdef struct timespec:
        pass
"""


# hipfile.h omits the ``@param[out]`` direction tag on several config getters,
# so the doxygen intent classifier leaves their output pointers at the INOUT
# fallback (caller-allocated ``ListOf*`` buffer). We force
# ``OUT_CALLEE_ALLOCATED`` so the codegen returns the value instead:
#
#   * ``_HIPFILE_CSTR_OUT_BUFFERS`` — ``char*`` OUT buffers the binding
#     allocates (``CStr.malloc``) and RETURNS as a decoded ``CStr``, sized by a
#     by-value length input. This is the same mechanism ``generators_hip``
#     uses for ``hipDeviceGetName`` et al. (``_CSTR_OUT_BUFFERS``): pair the
#     ``OUT_CALLEE_ALLOCATED`` intent with a ``desc_str.malloc(len)`` prepend
#     (see ``_hipfile_node_init``) and the CStr return + docstring fall out of
#     the mechanical codegen — no verbatim body/docstring override needed.
#     Maps ``(func, buffer_parm) -> size_parm``.
#   * ``_HIPFILE_SCALAR_OUT_PARMS`` — scalar ``value`` out-pointers on the
#     numeric/bool getters; paired with ``hipfile.ptr_rank`` rank 0 they come
#     back as a plain Python number.
_HIPFILE_CSTR_OUT_BUFFERS = {
    ("hipFileGetParameterString", "desc_str"): "len",
}

_HIPFILE_SCALAR_OUT_PARMS = frozenset(
    (
        ("hipFileGetParameterSizeT", "value"),
        ("hipFileGetParameterBool", "value"),
    )
)


# hipfile.h spells its file/buffer offsets ``hoff_t``, which is ``off_t`` on
# POSIX and ``__int64`` on Windows. Neither is part of <stdint.h>, so the
# renderer would canonicalize it to the codegen host's type -- ``long`` on LP64,
# ``long long`` on LLP64 -- and the generated tree, produced once and compiled
# everywhere, would carry a 32-bit declaration for a 64-bit offset on Windows.
# It also leaves the hand-written consumers (the overrides below, the
# cuda-interop cufile module) with no spelling they can use on both platforms,
# since Cython compares pointer types by identity. ``int64_t`` denotes the very
# same C type as ``hoff_t`` on every platform hip-python supports, and Cython
# already knows it through the stdint cimport in every generated prolog.
_HIPFILE_TYPEDEF_ALIASES = {"hoff_t": "int64_t"}


def _hipfile_ptr_parm_intent(parm):
    """Force ``OUT_CALLEE_ALLOCATED`` on the untagged hipFILE getter outputs
    (see ``_HIPFILE_CSTR_OUT_BUFFERS`` / ``_HIPFILE_SCALAR_OUT_PARMS``);
    everything else defers to the shared ``controls.hipfile`` chain.
    """
    parent = parm.parent
    if parent is not None:
        key = (parent.name, parm.name)
        if (
            key in _HIPFILE_CSTR_OUT_BUFFERS
            or key in _HIPFILE_SCALAR_OUT_PARMS
        ):
            return ParmIntent.OUT_CALLEE_ALLOCATED
    return controls.hipfile.ptr_parm_intent(parm)


# ---------------------------------------------------------------------------
# Robust hipFileRead / hipFileWrite overrides.
#
# These two sync I/O calls return an ``ssize_t`` whose negative values point at
# a *transient thread-local* side-channel: ``-1`` means "see POSIX ``errno``"
# and ``-hipFileHipDriverError`` means "see ``hipPeekAtLastError()``". Both are
# clobbered by the GIL re-acquire + tuple allocation the mechanical wrapper
# performs on the way out, so the auto-generated 1-tuple wrapper drops them
# irrecoverably (see share/design/HIPFILE.md section 6).
#
# We override the rendered Python body (via ``_hipfile_node_init`` below, which
# inlines each verbatim ``def`` onto the Function node through the
# ``python_interface_impl_override`` / ``python_docstring_override`` hooks) so
# the C call and BOTH side-channel reads happen
# in the SAME ``with nogil`` block, before the GIL is re-acquired, and are
# returned as ``(retval, errno, hip_drv_err)``. The high-level
# ``rocm.hipfile.file`` consumer raises ``OSError`` / ``HipFileException`` from
# that richer tuple. Scope is the two SYNC functions only; the async variants
# already return ``hipFileError`` by value.
# ---------------------------------------------------------------------------
def _hipfile_node_init(node):
    """Install robust body/docstring overrides on the ``hipFileRead`` /
    ``hipFileWrite`` Function nodes (see the module comment above).

    Each function's docstring and body are written out inline here (rather than
    via a shared parametrized helper): the C call and the ``errno`` /
    ``hipPeekAtLastError()`` snapshots run in one ``with nogil`` block and the
    result is returned as ``(retval, errno, hip_drv_err)``. ``textwrap.indent``
    lays the verbatim docstring + body into the ``def`` scope.

    The ``char*`` OUT buffer of ``hipFileGetParameterString`` is handled here
    too: its ``OUT_CALLEE_ALLOCATED`` intent (from ``_hipfile_ptr_parm_intent``)
    makes the codegen emit a returned ``CStr``, and this hook injects the
    ``desc_str.malloc(len)`` call the binding needs before the C call so the
    CStr owns a ``len``-byte scratch buffer — the same mechanism
    ``generators_hip`` uses for ``hipDeviceGetName`` (see
    ``_HIPFILE_CSTR_OUT_BUFFERS``). No verbatim body/docstring override is
    needed for it.
    """
    if isinstance(node, interfacegen.tree.Parm):
        parent = node.parent
        if (
            parent is not None
            and (parent.name, node.name) in _HIPFILE_CSTR_OUT_BUFFERS
        ):
            size_name = _HIPFILE_CSTR_OUT_BUFFERS[(parent.name, node.name)]
            parent.python_body_prepend_before_c_interface_call(
                f"{node.name}.malloc({size_name})"
            )
        return
    if not isinstance(node, interfacegen.tree.Function):
        return
    ind = "    "
    if node.name == "hipFileRead":
        docstring = textwrap.dedent(
            '''\
            r"""Synchronously read data from a file into a GPU buffer.

            Args:
                fh (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
                    hipFile handle for the target file.

                buffer_base (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
                    Base pointer of the registered GPU buffer.

                size (:py:obj:`~.int`) -- *IN*:
                    Number of bytes to read.

                file_offset (:py:obj:`~.int`) -- *IN*:
                    Offset into the file.

                buffer_offset (:py:obj:`~.int`) -- *IN*:
                    Offset into the GPU buffer.

            Returns:
                A :py:obj:`~.tuple` of size 3 that contains (in that order):

                * :py:obj:`~.int`: The raw ``ssize_t`` result. One of:
                        - if >= 0: Number of bytes transferred
                        - if -1:   POSIX system error (see the ``errno`` element)
                        - else:    Negated :py:obj:`~.hipFileOpError_t`; when it equals
                                   ``-hipFileHipDriverError`` the HIP driver error is
                                   carried in the ``hip_drv_err`` element

                * :py:obj:`~.int`: ``errno``, snapshotted inside the ``with nogil`` block
                        right after the call (meaningful only when the result is -1).

                * :py:obj:`~.int`: the :py:obj:`~.hipError_t` value from
                        ``hipPeekAtLastError()``, snapshotted inside the same
                        ``with nogil`` block (meaningful only when the result is
                        ``-hipFileHipDriverError``).
            """'''
        )
        body = textwrap.dedent(
            """\
            cdef rocm.bindings.util.types.Pointer _cy_hipFileRead__arg_0_obj = rocm.bindings.util.types.Pointer.fromPyobj(fh)
            cdef void * _cy_hipFileRead__arg_0 = <void *>_cy_hipFileRead__arg_0_obj.getPtr()
            cdef rocm.bindings.util.types.Pointer _cy_hipFileRead__arg_1_obj = rocm.bindings.util.types.Pointer.fromPyobj(buffer_base)
            cdef void * _cy_hipFileRead__arg_1 = <void *>_cy_hipFileRead__arg_1_obj.getPtr()
            cdef ssize_t _cy_hipFileRead__retval
            cdef int _cy_hipFileRead__err
            cdef int _cy_hipFileRead__hip_drv_err
            with nogil:
                _cy_hipFileRead__retval = cyhipfile.hipFileRead(_cy_hipFileRead__arg_0,_cy_hipFileRead__arg_1,size,file_offset,buffer_offset)
                _cy_hipFileRead__err = errno
                _cy_hipFileRead__hip_drv_err = <int>hipPeekAtLastError()
            return (_cy_hipFileRead__retval,_cy_hipFileRead__err,_cy_hipFileRead__hip_drv_err)"""
        )
        node.python_docstring_override = docstring
        node.python_interface_impl_override = (
            "@cython.embedsignature(True)\n"
            "def hipFileRead(object fh, object buffer_base, size_t size, int64_t file_offset, int64_t buffer_offset):\n"
            + textwrap.indent(docstring, ind).rstrip()
            + "\n"
            + textwrap.indent(body, ind).rstrip()
            + "\n"
        )
    elif node.name == "hipFileWrite":
        docstring = textwrap.dedent(
            '''\
            r"""Synchronously write data from a GPU buffer to a file.

            Args:
                fh (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
                    hipFile handle for the target file.

                buffer_base (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
                    Base pointer of the registered GPU buffer.

                size (:py:obj:`~.int`) -- *IN*:
                    Number of bytes to write.

                file_offset (:py:obj:`~.int`) -- *IN*:
                    Offset into the file.

                buffer_offset (:py:obj:`~.int`) -- *IN*:
                    Offset into the GPU buffer.

            Returns:
                A :py:obj:`~.tuple` of size 3 that contains (in that order):

                * :py:obj:`~.int`: The raw ``ssize_t`` result. One of:
                        - if >= 0: Number of bytes transferred
                        - if -1:   POSIX system error (see the ``errno`` element)
                        - else:    Negated :py:obj:`~.hipFileOpError_t`; when it equals
                                   ``-hipFileHipDriverError`` the HIP driver error is
                                   carried in the ``hip_drv_err`` element

                * :py:obj:`~.int`: ``errno``, snapshotted inside the ``with nogil`` block
                        right after the call (meaningful only when the result is -1).

                * :py:obj:`~.int`: the :py:obj:`~.hipError_t` value from
                        ``hipPeekAtLastError()``, snapshotted inside the same
                        ``with nogil`` block (meaningful only when the result is
                        ``-hipFileHipDriverError``).
            """'''
        )
        body = textwrap.dedent(
            """\
            cdef rocm.bindings.util.types.Pointer _cy_hipFileWrite__arg_0_obj = rocm.bindings.util.types.Pointer.fromPyobj(fh)
            cdef void * _cy_hipFileWrite__arg_0 = <void *>_cy_hipFileWrite__arg_0_obj.getPtr()
            cdef rocm.bindings.util.types.Pointer _cy_hipFileWrite__arg_1_obj = rocm.bindings.util.types.Pointer.fromPyobj(buffer_base)
            cdef const void * _cy_hipFileWrite__arg_1 = <const void *>_cy_hipFileWrite__arg_1_obj.getPtr()
            cdef ssize_t _cy_hipFileWrite__retval
            cdef int _cy_hipFileWrite__err
            cdef int _cy_hipFileWrite__hip_drv_err
            with nogil:
                _cy_hipFileWrite__retval = cyhipfile.hipFileWrite(_cy_hipFileWrite__arg_0,_cy_hipFileWrite__arg_1,size,file_offset,buffer_offset)
                _cy_hipFileWrite__err = errno
                _cy_hipFileWrite__hip_drv_err = <int>hipPeekAtLastError()
            return (_cy_hipFileWrite__retval,_cy_hipFileWrite__err,_cy_hipFileWrite__hip_drv_err)"""
        )
        node.python_docstring_override = docstring
        node.python_interface_impl_override = (
            "@cython.embedsignature(True)\n"
            "def hipFileWrite(object fh, object buffer_base, size_t size, int64_t file_offset, int64_t buffer_offset):\n"
            + textwrap.indent(docstring, ind).rstrip()
            + "\n"
            + textwrap.indent(body, ind).rstrip()
            + "\n"
        )


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
        # still drop the GIL. No status enum to prepend; the
        # always-return-tuple flag only wraps bare returns in a 1-tuple.
        module_opts={"python_interface_always_return_tuple": True},
        modifiers_lazy_loader=" noexcept nogil",
        node_filter=controls.hipfile.node_filter,
        node_init=_hipfile_node_init,
        macro_type=controls.hipfile.macro_type,
        ptr_parm_intent=_hipfile_ptr_parm_intent,
        ptr_rank=controls.hipfile.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        typedef_aliases=_HIPFILE_TYPEDEF_ALIASES,
        cflags=generator_args,
    )
    # hipfile.h uses ``hipStream_t`` / ``hipError_t`` from the HIP runtime
    # (async I/O APIs + the ``hip_drv_err`` field of ``hipFileError``), which
    # are not defined in hipfile.h itself, so the generated cy-module cimports
    # them from cyhip (mirrors the ``hipStream_t`` handling in generate_rccl).
    # ``struct sockaddr`` / ``struct timespec`` are hand-declared here as
    # opaque, pointer-only structs (see ``_HIPFILE_OPAQUE_PTR_TYPES_DECL``).
    generator.c_interface_decl_prolog += (
        textwrap.dedent(
            """\
    from rocm.bindings.cyhip cimport hipStream_t, hipError_t
    """
        )
        + _HIPFILE_OPAQUE_PTR_TYPES_DECL
    )
    # The high-level module constructs the ``hipError_t`` Python enum to wrap
    # ``hipFileError.hip_drv_err`` (and type-checks against it), so hipfile.pyx
    # needs the Python-level enum via a runtime ``import`` (impl prolog → .pyx),
    # NOT a ``cimport`` of the C typedef (which would shadow the enum and make
    # ``hipError_t(...)`` a non-callable C type). ``hipStream_t`` is only ever
    # referenced C-qualified (``cyhipfile.hipStream_t``) in the high-level
    # module, so no extra Python-side import is required for it.
    #
    # ``errno`` (ISO C ``<errno.h>``, nogil-safe, cross-platform) and
    # ``hipPeekAtLastError`` are cimported for the robust hipFileRead/hipFileWrite
    # overrides (see _hipfile_node_init), which snapshot both in-nogil.
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from libc.errno cimport errno
    from rocm.bindings.cyhip cimport hipPeekAtLastError
    from rocm.bindings.hip import hipError_t
    """
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
        module_opts={"python_interface_always_return_tuple": True},
        modifiers_lazy_loader=" except? AMDSMI_STATUS_INTERNAL_EXCEPTION nogil",
        error_return_value_lazy_loader="AMDSMI_STATUS_INTERNAL_EXCEPTION",
        node_init=_make_status_node_init(
            "amdsmi", "amdsmi_status_t", "AMDSMI_STATUS_SUCCESS"
        ),
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
        module_opts={"python_interface_always_return_tuple": True},
        # hsa functions return ``hsa_status_t``; ``HSA_STATUS_ERROR`` is
        # the generic failure sentinel hijacked for Python-exception
        # propagation.
        modifiers_lazy_loader=" except? HSA_STATUS_ERROR nogil",
        error_return_value_lazy_loader="HSA_STATUS_ERROR",
        node_init=_make_status_node_init(
            "hsa", "hsa_status_t", "HSA_STATUS_SUCCESS"
        ),
        node_filter=controls.hsa.node_filter,
        ptr_parm_intent=controls.hsa.ptr_parm_intent,
        ptr_rank=controls.hsa.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=hsa_cflags,
    )
    return generator
