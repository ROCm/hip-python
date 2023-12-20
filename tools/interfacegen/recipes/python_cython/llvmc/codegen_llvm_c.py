# MIT License
# 
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

"""This is the project's setup script.

After pointing this setup script to an ROCm LLVM installation,
it generates Cython files and compiles them to Python
modules. The generated Cython declaration files can be used
by Cython users of this project.
"""

__author__ = "Advanced Micro Devices, Inc. <rocm-llvm-python.maintainer@amd.com>"

import os
import re
import textwrap
import argparse

from pathlib import Path

import interfacegen.gitversion
import interfacegen.cython

from _include_graph import build_include_graph

import logging
interfacegen.enable_logging(logging.INFO)
_log = logging.getLogger("interfacegen")

# configure codegen
# see: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#role-py-obj
interfacegen.cython.python_interface_pyobj_role_template = r"`~.{name}`" # ~: removes the qualifier from the link text

from interfacegen.cython import (
    CythonModuleGenerator,
    CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
)

from interfacegen.cparser import TypeHandler

TypeCategory = TypeHandler.TypeCategory

from interfacegen.tree import (
    Node,
    MacroDefinition,
    Parm,
)

from interfacegen.control import ParmIntent

def parse_options():
    global OUTPUT_DIR
    global ROCM_LLVM_INC
    global RUNTIME_LINKING
    global GENERATOR_ARGS
    global LIBS
    global ROCM_VERSION_MAJOR
    global ROCM_VERSION_MINOR
    global ROCM_VERSION_PATCH

    def get_bool_environ_var(env_var, default):
        yes_vals = ("true", "1", "t", "y", "yes")
        no_vals = ("false", "0", "f", "n", "no")
        value = os.environ.get(env_var, default).lower()
        if value in yes_vals:
            return True
        elif value in no_vals:
            return False
        else:
            allowed_vals = ", ".join(
                [f"'{a}'" for a in (list(yes_vals) + list(no_vals))]
            )
            raise RuntimeError(
                f"value of '{env_var}' must be one of (case-insensitive): {allowed_vals}"
            )

    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
        Generator for ROCm LLVM Python package 'rocm-llvm'.
    
        NOTE:
            You can also use the environment variables 'ROCM_PATH' (or 'ROCM_HOME'),
            'ROCM_LLVM_PYTHON_CLANG_RES_DIR', 'ROCM_LLVM_PYTHON_LIBS',
            'ROCM_LLVM_PYTHON_RUNTIME_LINKING' instead of the command line interface.
        """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    def dir_path(arg):
        if not os.path.isdir(arg):
            raise NotADirectoryError(arg)
        return arg

    parser.add_argument(
        "output_dir",
        type=dir_path,
        help="The output directory to which the files should be written to. Must contain `rocm-llvm` subfolder.",
    )
    parser.add_argument(
        "--rocm-path",
        type=str,
        required=False,
        dest="rocm_path",
        help="The ROCm installation directory. Can be set via environment variables 'ROCM_PATH', 'ROCM_HOME' too.",
    )

    def rocm_version(arg):
        if not re.match(r"[0-9]+\.[0-9]+\.[0-9]+",arg):
            raise ValueError("Value of required argument `--rocm-version` must be a dot-separated number triple.")
        return [int(p) for p in arg.split(".")]

    parser.add_argument(
        "--rocm-version",
        type=rocm_version,
        required=True,
        dest="rocm_version",
        help="The ROCm version.",
    )
    parser.add_argument(
        "--clang-resource-dir",
        required=False,
        dest="clang_resource_dir",
        help="The clang resource directory. Can also be set via environment variable 'ROCM_LLVM_PYTHON_CLANG_RES_DIR'.",
    )
    parser.add_argument(
        "--libs",
        type=str,
        required=False,
        dest="libs",
        help="The libaries to generate interfaces for, as comma-separated list, e.g. 'core,types'. Pass '*' to generate all, pass '' to generate none. Add a prefix '^' to NOT generate code for the comma-separated list of libraries that follows but all other libraries.",
    )
    parser.add_argument(
        "--no-rt-linking",
        required=False,
        action="store_false",
        dest="runtime_linking",
        help="If HIP libraries should not be linked at runtime by the HIP Python modules.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        action="store_true",
        dest="verbose",
        default=False,
        help="Verbose output.",
    )
    parser.set_defaults(
        rocm_path=os.environ.get("ROCM_PATH", os.environ.get("ROCM_HOME", None)),
        clang_resource_dir=os.environ.get("ROCM_LLVM_PYTHON_CLANG_RES_DIR", None),
        libs=os.environ.get("ROCM_LLVM_PYTHON_LIBS", "*"),
        runtime_linking=get_bool_environ_var("ROCM_LLVM_PYTHON_RUNTIME_LINKING", "true"),
        verbose=False,
    )
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    RUNTIME_LINKING = args.runtime_linking
    LIBS = args.libs

    ( ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR, 
     ROCM_VERSION_PATCH ) = args.rocm_version

    if not args.rocm_path:
        raise RuntimeError("ROCm path is not set")
    ROCM_LLVM_INC = os.path.join(args.rocm_path,"llvm","include")

    GENERATOR_ARGS = [f"-I{ROCM_LLVM_INC}"]
    if not args.clang_resource_dir:
        raise RuntimeError(
            textwrap.dedent(
                """\
            Clang resource directory is not set.
            
            Hint: If `clang` is in the PATH, you can 
            run `clang -print-resource-dir` to obtain the path to
            the resource directory.

            Hint: If you have the HIP SDK installed, you have `amdclang` installed in
            `ROCM_PATH/bin/`. You can use it to run the above command too.
            
            Hint: If you have the HIP SDK installed, the last include folder listed in ``hipconfig --cpp_config``
            points to the `amdclang` compiler's resource dir too.
            """
            )
        )
    GENERATOR_ARGS += ["-resource-dir", args.clang_resource_dir]

def header_file_to_module_name(header_file: str):
    """Given a header file, returns output directory plus global and local module name.
    """
    global OUTPUT_DIR
    parts = header_file.lower().split("/")
    module_name = parts.pop(-1)[:-2].replace("-","_")
    package = ["rocm"]
    if parts[0] == "llvm":
        package += parts
    else:
        package += ["llvm","c"]+parts[1:]
    return (os.path.join(OUTPUT_DIR,*package), (".".join(package)+"."+module_name), module_name)

def create_llvm_c_default_generator(
    global_module_name: str,
    header_file: str,
    **opts
) -> CythonModuleGenerator:
    global ROCM_LLVM_INC
    global RUNTIME_LINKING
    global GENERATOR_ARGS

    util_pkg = "rocm.llvm._util"
    util_types_prefix = util_pkg + ".types."

    def ptr_rank(node: Node):
        # TODO: LLVMGetParamTypes(FunctionTy, Dest/**out**/)
        # LLVMFunctionType(ReturnType, ParamTypes, unsigned int ParamCount, int IsVarArg)
        if (node.parent.cursor.spelling, node.cursor.spelling) in (
            ("LLVMFunctionType","ParamTypes"),
            ("LLVMGetParams","Params"),
            ("LLVMGetParamTypes","Dest"),
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
            ("LLVMGetParams","Params"),
            ("LLVMGetParamTypes","Dest"),
            ("LLVMTargetMachineEmitToMemoryBuffer","OutMemBuf"),
            ("LLVMDisasmInstruction","OutString"),
        ):
            return ParmIntent.INOUT
        if fn_name in (
            "LLVMGetVersion",
        ) or parm_name in (
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
        ) or (fn_name,parm_name) in (
            ("LLVMGetValueName2","Length"),
            ("LLVMGetTargetFromTriple", "T"),
            ("LLVMGetTargetFromTriple", "ErrorMessage"),
        ):
            return ParmIntent.OUT
        return ParmIntent.IN

    def ptr_complicated_type_handler(node: Node):
        if (node.parent.cursor.spelling, node.cursor.spelling) in (
            ("LLVMFunctionType","ParamTypes"),
            ("LLVMGetParams","Params"),
            ("LLVMGetParamTypes","Dest"),
            ("LLVMRunFunction","Args"),
        ):
            return f"{util_types_prefix}ListOfPointer"
        return CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(util_types_prefix)(node)

    generator = CythonModuleGenerator(
        global_module_name,
        ROCM_LLVM_INC,
        header_file,
        runtime_linking=RUNTIME_LINKING,
        util_pkg=util_pkg,
        dll="librocmllvm.so",
        cflags=GENERATOR_ARGS,
        record_can_wrap_device_data=lambda _: False,
        ptr_rank = ptr_rank,
        ptr_parm_intent = ptr_parm_intent,
        ptr_complicated_type_handler = ptr_complicated_type_handler,
        **opts,
    )
    # generator.c_interface_decl_prolog += cython_c_preamble
    generator.python_interface_decl_prolog += f"cimport {util_pkg}.types\n"

    return generator

def parent_pkg_name(global_module_name: str):
    return ".".join(global_module_name.split(".")[:-1])

def resolve_internal_dependencies(generators):
    for h,incs in LLVM_C_INCLUDES.items():
        _, _, module_name = header_file_to_module_name(h)
        (generator, _, _) = generators[module_name]
        assert isinstance(generator,CythonModuleGenerator)
        for inc in incs:
            _, dep_global_name, dep_name = header_file_to_module_name(inc)
            dep_pkg_prefix=parent_pkg_name(dep_global_name)
            logging.getLogger("interfacegen").info(f" {h}: handle dep: {inc} ({dep_global_name})")
            (dep_generator, _, _) = generators[dep_name]
            assert isinstance(dep_generator,CythonModuleGenerator)
            generator.c_interface_decl_prolog += f"from {dep_pkg_prefix}.c{dep_name} cimport *\n"
            generator.python_interface_decl_prolog += "\n"
            generator.python_interface_impl_prolog += "\n"
            for node in dep_generator.backend.walk_entities_to_cimport(False):
                generator.python_interface_decl_prolog += f"from {dep_global_name} cimport {node.cython_global_name}\n"
            for node in dep_generator.backend.walk_entities_to_import(False):
                generator.python_interface_impl_prolog += f"from {dep_global_name} import {node.cython_global_name}\n"
            generator.python_interface_decl_prolog += "\n"
            generator.python_interface_impl_prolog += "\n"

def create_generators():
    global LLVM_C_INCLUDES
    generators = dict()
    for h, _ in LLVM_C_INCLUDES.items():
        output_dir, global_module_name, module_name = header_file_to_module_name(h)
        opts = dict()
        if h == "llvm-c/DataTypes.h":
            def node_filter(node: Node):
                return False # FIXME add the most important types and routines
            
            opts.update(
                node_filter=node_filter
            )
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
                    return None # these are target/platform specific, don't want to hardcode them.
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
                elif name in (
                    "LLVM_ENABLE_PLUGINS",
                ): # existence means True
                    return True
                return None
            
            def node_filter(node: Node):
                if isinstance(node, MacroDefinition):
                    return macro_type(node) != None
                return False

            opts.update(
                macro_type = macro_type,
                node_filter = node_filter
            )
        else:
            def create_node_filter(header: str): # we need to value-capture 'header'
                def inner(node: Node):
                    #print(f"{header}")
                    if not isinstance(node, MacroDefinition):
                        return header in node.render_location()
                    return False
                return inner

            opts.update(
                node_filter = create_node_filter(h)
            )

        generator: CythonModuleGenerator = create_llvm_c_default_generator(
            global_module_name, h, **opts
        )

        generators[module_name] = (
            generator,
            output_dir,
            global_module_name, 
        )

    resolve_internal_dependencies(generators)
    return generators

def lstrip_all_lines(text: str,lstrip_chars: str=" "):
    return "".join([line.lstrip(lstrip_chars) for line in text.splitlines(keepends=True)])

if __name__ == "__main__":
    OUTPUT_DIR = None

    ROCM_LLVM_INC = None
    RUNTIME_LINKING = None
    GENERATOR_ARGS = None
    LIBS = None
    ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR, ROCM_VERSION_PATCH = (
        0, 0, 0)
    parse_options() # sets the globals
    LLVM_C_INCLUDES = build_include_graph(ROCM_LLVM_INC)
    interfacegen.cython.FunctionMixin.python_interface_always_return_tuple = False # same for all modules, unlike callbacks
    AVAILABLE_GENERATORS = create_generators()

    # print(AVAILABLE_GENERATORS)

    # process and check user-provided library names
    avail_lib_names = AVAILABLE_GENERATORS.keys()
    processed_libs = LIBS.replace(" ","")
    if processed_libs == "*":
        lib_names = avail_lib_names
    else:
        if processed_libs.startswith("^"):
            processed_libs = processed_libs[1:].split(",")
            lib_names = [name for name in avail_lib_names if name not in processed_libs]
        else:
            processed_libs = processed_libs.split(",")
            lib_names = processed_libs
        for name in processed_libs:
            if name not in avail_lib_names:
                raise ValueError(f"library name '{name}' is not valid, use one of: {', '.join(avail_lib_names)}")

    Path(os.path.join(OUTPUT_DIR, "rocm-llvm-python", "rocm")).mkdir(parents=False, exist_ok=True) # throw error if it does not exist
    rocm_llvm_output_dir = os.path.join(OUTPUT_DIR, "rocm-llvm", "rocm", "llvm")
    Path(rocm_llvm_output_dir).mkdir(parents=False, exist_ok=True) # throw error if it does not exist
    # FIXME catch error
    global_module_names = []
    for entry in avail_lib_names:
        libname = entry.strip()
        if libname not in AVAILABLE_GENERATORS:
            available_libs = ", ".join([f"'{a}'" for a in AVAILABLE_GENERATORS.keys()])
            msg = f"no codegenerator found for library '{libname}'; please choose from: {available_libs}, or '*', which implies that all code generators will be used."
            raise KeyError(msg)
        generator, output_dir, global_module_name = AVAILABLE_GENERATORS[libname]
        global_module_names.append(global_module_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        generator.write_module_files(output_dir=output_dir)

    ROCM_VERSION_NAME = f"{ROCM_VERSION_MAJOR}.{ROCM_VERSION_MINOR}.{ROCM_VERSION_PATCH}"
    ROCM_VERSION = (
        ROCM_VERSION_MAJOR * 10000000 + ROCM_VERSION_MINOR * 100000 + ROCM_VERSION_PATCH
    )

    VERSION = f"{ROCM_VERSION_MAJOR}.{ROCM_VERSION_MINOR}.{ROCM_VERSION_PATCH}.{interfacegen.gitversion.version()}"
    LONG_VERSION = (
        f"{ROCM_VERSION_NAME}.{interfacegen.gitversion.version(append_hash=True,append_date=True)}"
    )
    with open(os.path.join("..","LICENSE"),"r") as licensefile:
        LICENSE_TEXT = "".join([f"# {ln}\n" for ln in licensefile.read().rstrip().splitlines()])

    # rocm/llvm/c/_version.py
    module_list_as_str = "\n".join(f'\t"{n}",'for n in global_module_names)
    with open(os.path.join(rocm_llvm_output_dir, "_version.py.in"), "w") as f:
        f.write(lstrip_all_lines(
            f"""\
            {LICENSE_TEXT}
            
            # This file has been autogenerated, do not modify.
            
            __author__ = "Advanced Micro Devices, Inc. <rocm-llvm-python.maintainer@amd.com>"

            VERSION = __version__ = "{VERSION}.{{ROCM_VERSION_SHORT}}"
            LONG_VERSION = __long_version__ = "{LONG_VERSION}.{{ROCM_VERSION}}"
            ROCM_LLVM_PYTHON_CODEGEN_BRANCH = "{interfacegen.gitversion.git_current_branch()}"
            ROCM_LLVM_PYTHON_CODEGEN_VERSION = "{interfacegen.gitversion.version(append_hash=True,append_date=True)}"
            ROCM_LLVM_PYTHON_CODEGEN_REV = "{interfacegen.gitversion.git_rev()}"
            ROCM_LLVM_PYTHON_BRANCH = "{{ROCM_LLVM_PYTHON_BRANCH}}"
            ROCM_VERSION = "{{ROCM_VERSION}}"
            ROCM_LLVM_PYTHON_REV = "{{ROCM_LLVM_PYTHON_REV}}"

            ROCM_LLVM_PYTHON_MODULE_LIST = [
            {module_list_as_str}
            ]
            """
            ).replace("\t","  ")
        )
    # rocm/llvm/c/__init__.py
    # TODO make option to use all generators or only specified ones
    ROCM_LLVM_PYTHON_LIB_NAMES = AVAILABLE_GENERATORS.keys()

    with open(os.path.join(rocm_llvm_output_dir, "__init__.py"), "w") as f:
        init_content = (
            lstrip_all_lines(f"""\
                {LICENSE_TEXT}
            
                # This file has been autogenerated, do not modify.
                
                __author__ = "Advanced Micro Devices, Inc. <rocm-llvm-python.maintainer@amd.com>"

                from ._version import *
                ROCM_VERSION = {ROCM_VERSION}
                ROCM_VERSION_NAME = rocm_llvm_version_name = "{ROCM_VERSION_NAME}"
                ROCM_VERSION_TUPLE = rocm_llvm_version_tuple = ({ROCM_VERSION_MAJOR},{ROCM_VERSION_MINOR},{ROCM_VERSION_PATCH})

                from . import _util
                from . import c
                from . import config

                import sys
                import os
            
                for module_name, module in sys.modules.items():
                \t\tif module_name.startswith("rocm.llvm.c."):
                \t\t\t\tif "DLL" in vars(module):
                \t\t\t\t\t\tmodule.DLL = os.path.join(os.path.dirname(__file__),module.DLL.decode("utf-8")).encode("utf-8")
                del sys
                del os
                """
            ).replace("\t"," "*2)
        )
        f.write(init_content)
    
    # TODO derive these dependencies automatically from LLVM_C_INCLUDES
    # subdir init files
    for subpkg in (
        ("c",),
        ("c","transforms",),
        ("config",),
    ):
        with open(os.path.join(rocm_llvm_output_dir, *subpkg, "__init__.py"), "w") as f:
            f.write(lstrip_all_lines(f"""\
                {LICENSE_TEXT}

                # This file has been autogenerated, do not modify.

                __author__ = "Advanced Micro Devices, Inc. <rocm-llvm-python.maintainer@amd.com>"
                
                import os
                """))
            for h in LLVM_C_INCLUDES.keys():
                _, global_module_name, module_name = header_file_to_module_name(h)
                if f"rocm.llvm.{'.'.join(subpkg)}.{module_name}" == global_module_name:
                    f.write(textwrap.dedent(f"""\
                        try:
                            from . import {module_name}
                        except ImportError:
                            pass # may have been excluded from build
                        """)
                    )
            if subpkg == ("c",):
                f.write("from . import transforms\n")
