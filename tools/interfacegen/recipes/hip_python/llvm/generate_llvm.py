# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

"""This is the project's code generation script.

After pointing this setup script to an ROCm LLVM installation,
it generates Cython files.
"""

__author__ = "Advanced Micro Devices, Inc."

import logging
import os
import textwrap
from pathlib import Path

import interfacegen
from interfacegen.cython import (
    CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
    CythonModuleGenerator,
)
from interfacegen.support import cython as support
from interfacegen.support import includetree as it
from interfacegen.support.recipes.control import ParmIntent
from interfacegen.tree import (
    MacroDefinition,
    Node,
    Parm,
)

interfacegen.enable_logging(logging.INFO)
_log = logging.getLogger("interfacegen")

# configure codegen
# see: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#role-py-obj
interfacegen.cython.python_interface_pyobj_role_template = (
    r"`~.{name}`"  # ~: removes the qualifier from the link text
)


def create_generator(
    global_module_name: str, header_file: str, **opts
) -> CythonModuleGenerator:
    global pkg_opts

    def ptr_rank(node: Node):
        # TODO: LLVMGetParamTypes(FunctionTy, Dest/**out**/)
        # LLVMFunctionType(ReturnType, ParamTypes, unsigned int ParamCount, int IsVarArg)
        if (node.parent.cursor.spelling, node.cursor.spelling) in (
            ("LLVMFunctionType", "ParamTypes"),
            ("LLVMGetParams", "Params"),
            ("LLVMGetParamTypes", "Dest"),
        ):
            return 1
        elif node.is_pointer_to_char(degree=-1):
            return 1
        return 0

    def ptr_parm_intent(node: Parm):
        fn_name: str = node.parent.cursor.spelling
        parm_name: str = node.cursor.spelling
        # order is important
        # trailing commas are important with single entry tuples
        if (fn_name, parm_name) in (
            ("LLVMGetParams", "Params"),
            ("LLVMGetParamTypes", "Dest"),
            ("LLVMTargetMachineEmitToMemoryBuffer", "OutMemBuf"),
            ("LLVMDisasmInstruction", "OutString"),
        ):
            return ParmIntent.INOUT
        if (
            fn_name in ("LLVMGetVersion",)
            or parm_name
            in (
                "OutEE",
                "OutError",
                "OutFn",
                "OutInterp",
                "OutJIT",
                "OutM",
                "OutMemBuf",
                "OutMessage",
                "OutMod",
                "OutModule",
                # "OutString", INOUT buffer
                # "OutStringSize", IN
            )
            or (fn_name, parm_name)
            in (
                ("LLVMGetValueName2", "Length"),
                ("LLVMGetTargetFromTriple", "T"),
                ("LLVMGetTargetFromTriple", "ErrorMessage"),
            )
        ):
            return ParmIntent.OUT
        return ParmIntent.IN

    def ptr_complicated_type_handler(node: Node):
        if (
            node.parent.cursor.spelling,
            node.cursor.spelling,
        ) in (  # parameters
            ("LLVMFunctionType", "ParamTypes"),
            ("LLVMGetParams", "Params"),
            ("LLVMGetParamTypes", "Dest"),
            ("LLVMRunFunction", "Args"),
            ("LLVMGetBufferStart"),
        ):
            return f"{pkg_opts.util_types_prefix}ListOfPointer"
        elif node.cursor.spelling in (  # function return values
            "LLVMGetBufferStart",
        ):
            return f"{pkg_opts.util_types_prefix}NDBuffer"
        return CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(
            pkg_opts.util_types_prefix
        )(node)

    generator = CythonModuleGenerator(
        global_module_name,
        pkg_opts.abs_inc_dir,
        header_file,
        runtime_linking=pkg_opts.runtime_linking,
        util_pkg=pkg_opts.util_pkg,
        dll=pkg_opts.dll,
        cflags=pkg_opts.generator_args,
        ptr_rank=ptr_rank,
        ptr_parm_intent=ptr_parm_intent,
        ptr_complicated_type_handler=ptr_complicated_type_handler,
        **opts,
    )
    # generator.c_interface_decl_prolog += cython_c_preamble
    generator.python_interface_decl_prolog += (
        f"cimport {pkg_opts.util_pkg}.types\n"
    )

    return generator


def create_generators(root: it.Root):
    for node in root.walk_files():
        h = node.relpath
        global_module_name = node.py_global_name
        opts = dict()
        if h == "llvm-c/DataTypes.h":

            def node_filter(node: Node):
                return False  # FIXME add the most important types and routines

            opts.update(node_filter=node_filter)
        elif h == "llvm/Config/llvm-config.h":

            def macro_type(node: Node):
                name = node.cursor.spelling
                if name in (
                    "LLVM_NATIVE_ARCH",
                    "LLVM_NATIVE_ASMPARSER",
                    "LLVM_NATIVE_ASMPRINTER",
                    "LLVM_NATIVE_DISASSEMBLER",
                    "LLVM_NATIVE_TARGET",
                    "LLVM_NATIVE_TARGETINFO",
                    "LLVM_NATIVE_TARGETMC",
                ):
                    return None  # these are target/platform specific, don't want to hardcode them.
                    # return str # means: interpret RHS tokens as str
                elif name in (
                    "LLVM_DEFAULT_TARGET_TRIPLE",
                    "LLVM_HOST_TRIPLE",
                    "LLVM_VERSION_STRING",
                ):
                    return "const char *"
                elif name in (
                    "LLVM_VERSION_MAJOR",
                    "LLVM_VERSION_MINOR",
                    "LLVM_VERSION_PATCH",
                ):
                    return "int"
                elif name in (
                    "LLVM_ENABLE_THREADS",
                    "LLVM_HAS_ATOMICS",
                    "LLVM_ON_UNIX",
                    "LLVM_USE_INTEL_JITEVENTS",
                    "LLVM_USE_OPROFILE",
                    "LLVM_USE_PERF",
                    "LLVM_FORCE_ENABLE_STATS",
                    "LLVM_ENABLE_ZLIB",
                    "LLVM_ENABLE_ZSTD",
                    "HAVE_SYSEXITS_H",
                    "LLVM_UNREACHABLE_OPTIMIZE",
                    "LLVM_ENABLE_DIA_SDK",
                ):
                    return "bint"
                elif name in ("LLVM_ENABLE_PLUGINS",):  # existence means True
                    return True
                return None

            def node_filter(node: Node):
                if isinstance(node, MacroDefinition):
                    return macro_type(node) is not None
                return False

            opts.update(macro_type=macro_type, node_filter=node_filter)
        else:

            def create_node_filter(
                header: str,
            ):  # we need to value-capture 'header'
                def inner(node: Node):
                    # print(f"{header}")
                    if not isinstance(node, MacroDefinition):
                        return header in node.render_location()
                    return False

                return inner

            opts.update(node_filter=create_node_filter(h))

        generator: CythonModuleGenerator = create_generator(
            global_module_name, h, **opts
        )

        node.codegen = generator

    for node in root.walk_files():
        node.cy_resolve_internal_dependencies()


def _make_pkg_opts(opts):
    """Build RocmPackageOpts from the unified-CLI options object."""
    pkg_opts = support.RocmPackageOpts()
    pkg_opts.env_var_prefix = "ROCM_LLVM_PYTHON"
    pkg_opts.author = "Advanced Micro Devices, Inc."
    pkg_opts.email = "hip-python.maintainer@amd.com"
    pkg_opts.dll = "libLLVM.so"
    pkg_opts.util_pkg = "rocm.bindings.util"
    pkg_opts.rel_inc_dir = os.path.join("llvm", "include")
    pkg_opts.abs_inc_dir = os.path.join(opts.rocm_path, "llvm", "include")
    pkg_opts.rocm_path = opts.rocm_path
    pkg_opts.rocm_version = tuple(int(p) for p in opts.rocm_version.split("."))
    pkg_opts.libs_user_spec = opts.llvm_libs
    pkg_opts.runtime_linking = opts.runtime_linking
    pkg_opts.generator_args = list(opts.generator_args or [])
    if opts.clang_resource_dir:
        pkg_opts.generator_args += ["-resource-dir", opts.clang_resource_dir]
    pkg_opts.clang_res_dir = opts.clang_resource_dir
    pkg_opts.output_dir = opts.output_dir
    pkg_opts.package_dir = os.path.join(
        opts.output_dir, "python", "rocm-bindings-compiler"
    )
    return pkg_opts


def generate(opts):
    """Run LLVM-C subgenerator against the modern hip-python layout.

    Writes only `.pxd`/`.pyx` files into
    `<output_dir>/python/rocm-bindings-compiler/rocm/bindings/llvm/`. Does NOT
    produce any `__init__.py`, `_version.py.in`, or other Python-packaging
    files (plan §B.7).
    """
    global pkg_opts
    pkg_opts = _make_pkg_opts(opts)

    def filter(filepath: str):
        if "llvm-c" in filepath:
            return not filepath.endswith("ExternC.h")
        if filepath.endswith(os.path.join("llvm", "Config", "llvm-config.h")):
            return True
        return False

    # Modern namespace: rocm.bindings.llvm (not legacy rocm.llvm).
    inctree = it.build_include_tree(
        pkg_opts.abs_inc_dir, py_namespace="rocm.bindings", filter=filter
    )
    inctree.find_node(name="llvm-c").py_split_at_char("-")
    interfacegen.cython.Function.python_interface_always_return_tuple = False

    # lljitutils was introduced in ROCm 6.2.0
    lljitutils = inctree.find_node(py_name="lljitutils")
    if lljitutils:
        lljitutils.includes += [
            inctree.find_node(py_name="types"),
            inctree.find_node(py_name="error"),
        ]

    create_generators(inctree)

    # Cython-source-only emission. This skips the helpers that write
    # __init__.py, _version.py.in, etc. (kept handcoded in hip-python repo).
    pkg_opts.package_dir = os.path.join(opts.output_dir, "python", "rocm-bindings-compiler")
    Path(pkg_opts.package_dir).mkdir(parents=True, exist_ok=True)
    inctree.py_create_package_dirs(pkg_opts.package_dir, output_dir_parents=True)
    selected_modules = support.write_autogenerated_module_files(
        output_dir=pkg_opts.package_dir,
        root=inctree,
        libs_user_spec=opts.llvm_libs,
    )
    return dict(llvm_modules=selected_modules)


if __name__ == "__main__":
    # Standalone entry point — adapts the legacy CLI to the unified opts.
    pkg_opts = support.create_rocm_package_opts_from_cli(
        project="ROCm LLVM Python",
        env_var_prefix="ROCM_LLVM_PYTHON",
        libs_example="cores,types",
        package="rocm-bindings-compiler",
        rel_inc_dir=os.path.join("llvm", "include"),
        util_pkg="rocm.bindings.util",
        dll="libLLVM.so",
        author="Advanced Micro Devices, Inc.",
        email="hip-python.maintainer@amd.com",
    )

    class _Opts:
        pass
    opts = _Opts()
    opts.output_dir = os.path.dirname(pkg_opts.package_dir)
    opts.rocm_path = pkg_opts.rocm_path
    opts.rocm_version = pkg_opts.rocm_version
    opts.llvm_libs = pkg_opts.libs_user_spec
    opts.runtime_linking = pkg_opts.runtime_linking
    generate(opts)
