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

"""Per-module generators for the **rocm-bindings-compiler** wheel.

Hosts the AMD COMGR single-header generator and the LLVM multi-module
generator. Both mirror the orchestration pattern in `generators_systems.py`
/ `generators_libraries.py` so that `binding_generator.py` can dispatch
through one `AVAILABLE_GENERATORS` registry.
"""

import os

import clang.cindex
from interfacegen.cparser import TypeHandler
from interfacegen.cython import CythonModuleGenerator
from interfacegen.support.recipes import rocm as controls
from interfacegen.tree import Function, Node

from .node_init import make_status_node_init


def _make_header_arg(header_relpath: str, header_content: str = None):
    """Build the header argument for CythonModuleGenerator."""
    if header_content is not None:
        return (header_relpath, header_content)
    return header_relpath


def generate_amd_comgr(
    *,
    include_dir: str,
    header_relpath: str = "amd_comgr/amd_comgr.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    """Generator for AMD COMGR (`amd_comgr/amd_comgr.h`) bindings."""

    # Wrap the comgr-specific ListOfBytes mapping with the project's
    # util_types prefix; everything else delegates to the default handler.
    util_types_prefix = "rocm.bindings.util.types."

    def ptr_complicated_type_handler(node):
        if controls.comgr.is_listofbytes_pointer(node):
            return f"{util_types_prefix}ListOfBytes"
        return default_ptr_handler(node)

    generator = CythonModuleGenerator(
        "rocm.bindings.amd_comgr",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libamd_comgr.so",
        # Replaces the original generate_comgr.py global mutation:
        #   `Function.python_interface_always_return_tuple = True`.
        # Now scoped to this generator; no cross-recipe interference.
        module_opts={"python_interface_always_return_tuple": True},
        node_filter=controls.comgr.node_filter,
        # ``amd_comgr_get_version`` returns void, so it cannot carry the
        # module-level ``except? AMD_COMGR_STATUS_ERROR``.
        node_init=make_status_node_init(
            "amd_comgr", "amd_comgr_status_s", "AMD_COMGR_STATUS_SUCCESS"
        ),
        ptr_rank=controls.comgr.ptr_rank,
        ptr_parm_intent=controls.comgr.ptr_parm_intent,
        ptr_complicated_type_handler=ptr_complicated_type_handler,
        modifiers_lazy_loader=" except? AMD_COMGR_STATUS_ERROR nogil",
        error_return_value_lazy_loader="AMD_COMGR_STATUS_ERROR",
        cflags=generator_args,
    )
    generator.python_interface_decl_prolog += (
        "cimport rocm.bindings.util.types\n"
    )
    return generator


def _make_llvm_ptr_handler(default_ptr_handler, util_types_prefix):
    """Build the ptr_complicated_type_handler for an LLVM module.

    Wraps the `is_listofpointer_param` / `is_ndbuffer_return` predicates
    from controls.llvm_c with the project's util_types prefix; falls back
    to default_ptr_handler for everything else. Mirrors the closure body
    in the legacy generate_llvm.create_generator (lines 117-135).
    """

    def _handler(node):
        if controls.llvm_c.is_listofpointer_param(node):
            return f"{util_types_prefix}ListOfPointer"
        if controls.llvm_c.is_ndbuffer_return(node):
            return f"{util_types_prefix}NDBuffer"
        return default_ptr_handler(node)

    return _handler


def is_nogil_header(header_relpath: str) -> bool:
    """Whether `header_relpath` (e.g. `llvm-c/Transforms/PassBuilder.h`)
    is one of `controls.llvm_c.nogil_headers`."""
    return os.path.basename(header_relpath) in controls.llvm_c.nogil_headers


def _enum_sentinel(node: Function):
    """Returns the `except?` sentinel for an enum-returning function,
    or `None` if the enum leaves no value out of band.

    The sentinel is -1 cast to the enum type. Cython rejects a bare
    `-1` against an enum return ("Exception value incompatible with
    function return type"), hence the cast, and -1 is out of band for
    every enum in these headers: not one of the 43 named `llvm-c`
    enums declares a negative enumerator, and the 21 that functions
    return sit in ranges like `LLVMIntPredicate` 32..41 and
    `lto_symbol_attributes` 31..32768.

    Naming a constant the C API itself defines as out-of-band would
    read better, but `llvm-c` has none. The spellings that look the
    part are all values their getter returns in normal operation:
    `LLVMDSError` is a diagnostic *severity*,
    `LLVMModuleFlagBehaviorError` a module-flag behaviour, and
    `LLVMCodeGenLevelNone` / `LLVMTailCallKindNone` /
    `LTO_DEBUG_MODEL_NONE` ordinary settings. Using one of those
    would stay correct -- `except?` re-checks `PyErr_Occurred` -- but
    would put a GIL-taking check on the common return path.

    A future header that does declare -1 gets `None`, which drops the
    function to `noexcept nogil` rather than silently mistaking a
    valid return for a failed symbol load.
    """
    enum_node = node.lookup_innermost_type()
    cursor = getattr(enum_node, "cursor", None)
    if cursor is not None:
        for child in cursor.get_children():
            if (
                child.kind == clang.cindex.CursorKind.ENUM_CONSTANT_DECL
                and child.enum_value == -1
            ):
                return None
    return f"<{node.cython_global_typename}>-1"


def _nogil_modifiers(node: Function):
    """Returns the `(modifiers, error_return_value)` pair that makes
    `node`'s lazy-loader shim `nogil`-callable.

    A shim raises when the symbol cannot be resolved -- libLLVM is
    optional at runtime, and the LLVM chapter of the user guide
    documents that first-call failure as the way a missing library
    surfaces. Keeping that behaviour under `nogil` needs an exception
    sentinel the caller can test without holding the GIL, chosen from
    the return type:

    * pointers get `NULL`,
    * integral and boolean returns get `-1` (Cython's `bint` is a C
      `int`, so -1 survives the return),
    * enums get -1 cast to the enum type (see
      :py:func:`_enum_sentinel`).

    `void` returns, by-value records and floating-point returns have
    no free sentinel. The alternative there, `except *`, forces the
    caller to take the GIL after *every* call just to poll
    `PyErr_Occurred`, so those fall back to `noexcept nogil` as the
    hip and comgr bindings do. Cython still aborts such a shim at the
    raising `__init_symbol` -- the call through the NULL function
    pointer is not reached -- but reports the failure as an unraisable
    exception instead of propagating it.
    """
    if node.is_any_pointer:
        return " except? NULL nogil", "NULL"
    if node.is_enum:
        sentinel = _enum_sentinel(node)
        if sentinel is not None:
            return f" except? {sentinel} nogil", sentinel
    if node.is_basic_type:
        kind = next(node.typehandler.clang_type_layer_kinds(canonical=True))
        if not TypeHandler.match_float_type(kind):
            return " except? -1 nogil", "-1"
    return " noexcept nogil", None


def nogil_node_init(header_relpath: str):
    """Returns a node_init closure that opts `header_relpath`'s
    functions into the with-nogil emitter.

    The emitter is selected by the presence of `nogil` in a function's
    `modifiers_lazy_loader` (see `interfacegen.cython._function`), so
    marking the declaration is all it takes to move the C call into a
    `with nogil:` block; argument conversion and return-value wrapping
    stay under the GIL either way.

    Functions taking a callback are left alone. Registering one is
    cheap, so there is nothing to gain, and the parameter is the only
    route by which LLVM could re-enter the caller's code mid-call.
    (Handlers registered elsewhere -- a diagnostic handler firing
    during a parse, say -- are unaffected: the bindings only accept
    raw C function pointers, and a `ctypes` callback reacquires the
    GIL itself.)
    """
    in_scope = controls.llvm_c.location_filter(header_relpath)

    def _node_init(node: Node):
        if not isinstance(node, Function):
            return
        if not in_scope(node):
            return
        if node._has_funptr_parm:
            return
        modifiers, error_return_value = _nogil_modifiers(node)
        node.modifiers_lazy_loader = modifiers
        node.error_return_value_lazy_loader = error_return_value

    return _node_init


def write_llvm_modules(
    *,
    output_dir: str,  # = .../rocm-bindings-compiler/rocm/bindings
    include_dir: str,  # = <rocm_path>/llvm/include
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
    **_ignored,  # absorbs header_relpath/header_content (always None for llvm)
):
    """Build every rocm.bindings.llvm.* module and write them to disk.

    Runs inside the LLVM pool worker — all libclang state (the inctree,
    the per-module CythonModuleGenerator instances) lives entirely in
    the worker process.

    Returns the list of dotted module names emitted (e.g.
    "rocm.bindings.llvm.c.core").
    """
    from interfacegen.support import includetree as it

    util_types_prefix = "rocm.bindings.util.types."

    # Filter mirrors generate_llvm.py:282-287.
    def filter(filepath):
        if "llvm-c" in filepath:
            return not filepath.endswith("ExternC.h")
        if filepath.endswith(os.path.join("llvm", "Config", "llvm-config.h")):
            return True
        return False

    inctree = it.build_include_tree(
        include_dir,
        py_namespace="rocm.bindings",
        filter=filter,
    )
    inctree.find_node(name="llvm-c").py_split_at_char("-")

    # lljitutils was introduced in ROCm 6.2.0 — register cross-deps so
    # cy_resolve_internal_dependencies emits the cimports.
    lljitutils = inctree.find_node(py_name="lljitutils")
    if lljitutils:
        lljitutils.includes += [
            inctree.find_node(py_name="types"),
            inctree.find_node(py_name="error"),
        ]

    # Per-File CythonModuleGenerator construction.
    for node in inctree.walk_files():
        relpath = node.relpath
        kwargs = dict(
            module_opts={"python_interface_always_return_tuple": False},
            ptr_rank=controls.llvm_c.ptr_rank,
            ptr_parm_intent=controls.llvm_c.ptr_parm_intent,
            ptr_complicated_type_handler=_make_llvm_ptr_handler(
                default_ptr_handler,
                util_types_prefix,
            ),
            cflags=generator_args,
        )
        if relpath.endswith("DataTypes.h"):
            # FIXME marker preserved from generate_llvm.py:163-168.
            kwargs["node_filter"] = lambda n: False
        elif relpath.endswith("llvm-config.h"):
            kwargs["node_filter"] = controls.llvm_config.node_filter
            kwargs["macro_type"] = controls.llvm_config.macro_type
        else:
            kwargs["node_filter"] = controls.llvm_c.location_filter(relpath)
            if is_nogil_header(relpath):
                # Compilation, linking, parsing and JIT materialization
                # run long enough to hand the GIL to other threads.
                kwargs["node_init"] = nogil_node_init(relpath)
        gen = CythonModuleGenerator(
            node.py_global_name,
            include_dir,
            relpath,
            runtime_linking=runtime_linking,
            util_pkg="rocm.bindings.util",
            dll="libLLVM.so",
            **kwargs,
        )
        gen.python_interface_decl_prolog += (
            "cimport rocm.bindings.util.types\n"
        )
        node.codegen = gen

    # Resolve cross-package cimports now that every node has its codegen.
    for node in inctree.walk_files():
        node.cy_resolve_internal_dependencies()

    # Emit every discovered LLVM module — no per-module subset filter
    # (the wheel-level --include / --exclude is handled in
    # binding_generator.generate; users either get all of LLVM or none).
    #
    # `output_dir` is the per-package source dir (e.g.
    # `.../rocm-bindings-compiler/src/rocm/bindings`), and
    # `node.parent.py_global_path` is the dotted-name path
    # (e.g. `rocm/bindings/llvm/c`). Naively joining the two would
    # duplicate the `rocm/bindings/` prefix into the on-disk path
    # (-> `.../src/rocm/bindings/rocm/bindings/llvm/c/...`). Strip
    # the leading `rocm/bindings/` so the target ends up as
    # `.../src/rocm/bindings/llvm/c/...` as expected.
    _PREFIX_TO_STRIP = "rocm/bindings/"
    module_names = []
    for node in inctree.walk_files():
        if not isinstance(node, it.File):
            continue
        sub_path = node.parent.py_global_path
        if sub_path.startswith(_PREFIX_TO_STRIP):
            sub_path = sub_path[len(_PREFIX_TO_STRIP) :]
        target = os.path.join(output_dir, sub_path)
        os.makedirs(target, exist_ok=True)
        node.codegen.write_module_files(output_dir=target)
        module_names.append(node.py_global_name)

    return module_names
