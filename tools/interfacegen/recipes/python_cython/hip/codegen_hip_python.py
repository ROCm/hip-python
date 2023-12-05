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

After pointing this setup script to an HIP installation,
it generates Cython files and compiles them to Python
modules. The generated Cython declaration files can be used
by Cython users of this project.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

import os
from pathlib import Path
import enum
import textwrap
import argparse
import logging

import interfacegen
interfacegen.enable_logging(logging.INFO)
_log = logging.getLogger("interfacegen")

import controls
import cuda_interop_layer_gen
import interfacegen.gitversion
import interfacegen.cython

# configure codegen
# see: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#role-py-obj
interfacegen.cython.python_interface_pyobj_role_template = r"`~.{name}`" # ~: removes the qualifier from the link text
cuda_interop_layer_gen.python_interface_pyobj_role_template = r"`.{name}`" # note: here we want to keep the qualifier

from interfacegen.cython import (
    CythonModuleGenerator,
    CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
)

HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER = CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER("hip._util.types.")

from interfacegen.cparser import TypeHandler

TypeCategory = TypeHandler.TypeCategory

from interfacegen.tree import (
    Node,
    MacroDefinition,
    Parm,
)

from parse_hipify_perl import parse_hipify_perl

def parse_options():
    global OUTPUT_DIR
    global ROCM_INC
    global RUNTIME_LINKING
    global GENERATOR_ARGS
    global LIBS
    global HIP_2_CUDA

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
        Generator for HIP Python packages 'hip-python' and 'hip-python-as-cuda'.
    
        NOTE:
            You can also use the environment variables 'ROCM_PATH' (or 'ROCM_HOME'),
            'HIP_PLATFORM', 'HIP_PYTHON_CLANG_RES_DIR', 'HIP_PYTHON_LIBS',
            'HIP_PYTHON_RUNTIME_LINKING' instead of the command line interface.
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
        help="The output directory to which the files should be written to. Must contain `hip-python` and `hip-python-as-cuda` subfolders.",
    )
    parser.add_argument(
        "--rocm-path",
        type=str,
        required=False,
        dest="rocm_path",
        help="The ROCm installation directory. Can be set via environment variables 'ROCM_PATH', 'ROCM_HOME' too.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        required=False,
        dest="platform",
        help="The HIP platform, 'amd' or 'nvidia'. Can be set via environment variable HIP_PLATFORM too.",
    )
    parser.add_argument(
        "--clang-resource-dir",
        required=False,
        dest="clang_resource_dir",
        help="The clang resource directory. Can also be set via environment variable 'HIP_PYTHON_CLANG_RES_DIR'.",
    )
    parser.add_argument(
        "--libs",
        type=str,
        required=False,
        dest="libs",
        help="The ROCm libaries to generate interfaces for, as comma-separated list, e.g. 'hip,hiprtc'. Pass '*' to generate all, pass '' to generate none. Add a prefix '^' to NOT generate code for the comma-separated list of libraries that follows but all other libraries.",
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
        platform=os.environ.get("HIP_PLATFORM", "amd"),
        clang_resource_dir=os.environ.get("HIP_PYTHON_CLANG_RES_DIR", None),
        libs=os.environ.get("HIP_PYTHON_LIBS", "*"),
        runtime_linking=get_bool_environ_var("HIP_PYTHON_RUNTIME_LINKING", "true"),
        verbose=False,
    )
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    RUNTIME_LINKING = args.runtime_linking
    LIBS = args.libs

    if not args.rocm_path:
        raise RuntimeError("ROCm path is not set")
    ROCM_INC = os.path.join(args.rocm_path, "include")

    hipify_perl_path = os.path.join(args.rocm_path, "bin", "hipify-perl")
    (_, HIP_2_CUDA) = parse_hipify_perl(hipify_perl_path)

    if args.platform not in ("amd", "hcc"):
        raise RuntimeError("Currently only platform 'amd' is supported")

    class HipPlatform(enum.IntEnum):
        AMD = 0
        NVIDIA = 1

        @staticmethod
        def from_string(key: str):
            valid_inputs = ("amd", "hcc", "nvidia", "nvcc")
            key = key.lower()
            if key in valid_inputs[0:2]:
                return HipPlatform.AMD
            elif key in valid_inputs[2:4]:
                return HipPlatform.NVIDIA
            else:
                raise ValueError(
                    f"Input must be one of: {','.join(valid_inputs)} (any case)"
                )

        @property
        def cflags(self):
            return ["-D", f"__HIP_PLATFORM_{self.name}__"]

    hip_platform = HipPlatform.from_string(args.platform)

    GENERATOR_ARGS = hip_platform.cflags + [f"-I{ROCM_INC}"]
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


# hip
def generate_hip_module_files():
    global OUTPUT_DIR
    global ROCM_INC
    global RUNTIME_LINKING
    global GENERATOR_ARGS
    global HIP_GENERATOR

    global HIP_VERSION_MAJOR
    global HIP_VERSION_MINOR
    global HIP_VERSION_PATCH
    global HIP_VERSION_GITHASH

    def toclassname(name: str):
        return name[0].upper() + name[1:]

    def hip_ptr_complicated_type_handler(parm: Node):
        if (parm.parent.name, parm.name) == ("hipModuleLaunchKernel", "extra"):
            return f"hip._hip_helpers.{toclassname(parm.parent.name)}_{parm.name}"
        if (parm.parent.name, parm.name) in (
            ("hipMalloc", "ptr"),
            ("hipExtMallocWithFlags", "ptr"),
            ("hipMallocManaged", "dev_ptr"),
            ("hipMallocAsync", "dev_ptr"),
            ("hipMallocFromPoolAsync", "dev_ptr"),
        ):
            if parm.parent.name == "hipExtMallocWithFlags":
                size = "sizeBytes"
            else:
                size = "size"
            parm.parent.python_body_prepend_before_return(
                f"{parm.name}.configure(_force=True,shape=(cpython.long.PyLong_FromUnsignedLong({size}),))"
            )
            return "hip._util.types.DeviceArray"
        
        return HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(parm)

    def hip_node_init(node: Node):
        # node modifications
        if isinstance(node,interfacegen.tree.Function):
            if not node.is_enum and node.name.startswith("hip"):
                node.prepend_python_return_value(
                    "hipError_t.hipSuccess",
                    "hipError_t",
                    "Always returns `~.hipError_t.hipSuccess`.")
        elif isinstance(node,interfacegen.tree.Parm):
            func_name, parm_idx = node.parent.name, node.parm_index
            if (func_name, parm_idx) in (
                ("hipDeviceGetName", 0),
                ("hipDeviceGetPCIBusId", 0),
            ):
                func = node.parent
                assert isinstance(func,interfacegen.cython.FunctionMixin)
                len_param: interfacegen.tree.Parm = func.get_parm(1)
                func.python_body_prepend_before_c_interface_call(
                    f"{node.name}.malloc({len_param.name})"
                )

    generator = CythonModuleGenerator(
        "hip",
        ROCM_INC,
        "hip/hip_runtime.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="hip._util",
        dll="libamdhip64.so",
        node_init = hip_node_init,
        node_filter=controls.hip.node_filter,
        ptr_parm_intent=controls.hip.ptr_parm_intent,
        ptr_rank=controls.hip.ptr_rank,
        ptr_complicated_type_handler=hip_ptr_complicated_type_handler,
        macro_type=controls.hip.macro_type,
        raw_comment_cleaner=controls.hip.raw_comment_cleaner,
        cflags=GENERATOR_ARGS,
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    cimport hip._hip_helpers
    """
    )

    HIP_VERSION_MAJOR = 0
    HIP_VERSION_MINOR = 0
    HIP_VERSION_PATCH = 0
    HIP_VERSION_GITHASH = ""
    for node in generator.backend.root.walk():
        if isinstance(node, MacroDefinition):
            last_token = list(node.cursor.get_tokens())[-1].spelling
            if node.name == "HIP_VERSION_MAJOR":
                HIP_VERSION_MAJOR = int(last_token)
            elif node.name == "HIP_VERSION_MINOR":
                HIP_VERSION_MINOR = int(last_token)
            elif node.name == "HIP_VERSION_PATCH":
                HIP_VERSION_PATCH = int(last_token)
            elif node.name == "HIP_VERSION_GITHASH":
                HIP_VERSION_GITHASH = last_token.strip('"')
    HIP_GENERATOR = generator
    return generator


# hiprtc
def generate_hiprtc_module_files():
    global OUTPUT_DIR
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING
    global HIPRTC_GENERATOR

    def hiprtc_ptr_complicated_type_handler(parm: Parm):
        list_of_str_parms = (
            ("hiprtcCompileProgram", "options"),
            ("hiprtcCreateProgram", "headers"),
            ("hiprtcCreateProgram", "includeNames"),
        )
        if (parm.parent.name, parm.name) in list_of_str_parms:
            return "hip._util.types.ListOfBytes"
        return HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(parm)

    generator = CythonModuleGenerator(
        "hiprtc",
        ROCM_INC,
        "hip/hiprtc.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="hip._util",
        dll="libhiprtc.so",
        node_filter=controls.hiprtc.node_filter,
        ptr_parm_intent=controls.hiprtc.ptr_parm_intent,
        ptr_rank=controls.hiprtc.ptr_rank,
        ptr_complicated_type_handler=hiprtc_ptr_complicated_type_handler,
        cflags=GENERATOR_ARGS,
    )
    HIPRTC_GENERATOR = generator
    return generator


# hipblas
def generate_hipblas_module_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonModuleGenerator(
        "hipblas",
        ROCM_INC,
        "hipblas/hipblas.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="hip._util",
        dll="libhipblas.so",
        node_filter=controls.hipblas.node_filter,
        ptr_parm_intent=controls.hipblas.ptr_parm_intent,
        ptr_rank=controls.hipblas.ptr_rank,
        ptr_complicated_type_handler=HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
        raw_comment_cleaner=controls.hipblas.raw_comment_cleaner,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from .chip cimport hipStream_t
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    cimport hip._util.types
    from .hip cimport ihipStream_t
    """
    )
    return generator


# rccl
def generate_rccl_module_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonModuleGenerator(
        "rccl",
        ROCM_INC,
        "rccl/rccl.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="hip._util",
        dll="librccl.so",
        node_filter=controls.rccl.node_filter,
        macro_type=controls.rccl.macro_type,
        ptr_parm_intent=controls.rccl.ptr_parm_intent,
        ptr_rank=controls.rccl.ptr_rank,
        ptr_complicated_type_handler=HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from .chip cimport hipStream_t
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    cimport hip._util.types
    from .hip cimport ihipStream_t
    """
    )
    return generator


# hiprand
def generate_hiprand_module_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonModuleGenerator(
        "hiprand",
        ROCM_INC,
        "hiprand/hiprand.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="hip._util",
        dll="libhiprand.so",
        node_filter=controls.hiprand.node_filter,
        macro_type=controls.hiprand.macro_type,
        ptr_parm_intent=controls.hiprand.ptr_parm_intent,
        ptr_rank=controls.hiprand.ptr_rank,
        ptr_complicated_type_handler=HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from .chip cimport hipStream_t
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    cimport hip._util.types
    from .hip cimport ihipStream_t
    """
    )
    return generator


# hipfft
def generate_hipfft_module_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonModuleGenerator(
        "hipfft",
        ROCM_INC,
        "hipfft/hipfft.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="hip._util",
        dll="libhipfft.so",
        node_filter=controls.hipfft.node_filter,
        macro_type=controls.hipfft.macro_type,
        ptr_parm_intent=controls.hipfft.ptr_parm_intent,
        ptr_rank=controls.hipfft.ptr_rank,
        ptr_complicated_type_handler=HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from .chip cimport hipStream_t, float2, double2
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    cimport hip._util.types
    from .hip cimport ihipStream_t, float2, double2
    """
    )
    return generator


# hipsparse
def generate_hipsparse_module_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonModuleGenerator(
        "hipsparse",
        ROCM_INC,
        "hipsparse/hipsparse.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="hip._util",
        dll="libhipsparse.so",
        node_filter=controls.hipsparse.node_filter,
        macro_type=controls.hipsparse.macro_type,
        ptr_parm_intent=controls.hipsparse.ptr_parm_intent,
        ptr_rank=controls.hipsparse.ptr_rank,
        raw_comment_cleaner=controls.hipsparse.raw_comment_cleaner,
        ptr_complicated_type_handler=HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
        from .chip cimport *
        """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
        cimport hip._util.types
        from .hip import hipError_t, _hipDataType__Base # PY import enums
        from .hip cimport ihipStream_t, float2, double2 # C import structs/union types
        """
    )
    return generator

def write_version_file(
    output_dir: str,
    LICENSE_TEXT: str,
    VERSION,
    LONG_VERSION,
):
    with open(os.path.join(output_dir, "_version.py.in"), "w") as f:
        f.write(
            LICENSE_TEXT
            + textwrap.dedent(
            f"""\
            
            # This file has been autogenerated, do not modify.
            
            __author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

            VERSION = __version__ = "{VERSION}.{{HIP_PYTHON_VERSION_SHORT}}"
            LONG_VERSION = __long_version__ = "{LONG_VERSION}.{{HIP_PYTHON_VERSION}}"
            HIP_PYTHON_CODEGEN_BRANCH = "{interfacegen.gitversion.git_current_branch()}"
            HIP_PYTHON_CODEGEN_VERSION = "{interfacegen.gitversion.version(append_hash=True,append_date=True)}"
            HIP_PYTHON_CODEGEN_REV = "{interfacegen.gitversion.git_rev()}"
            HIP_PYTHON_BRANCH = "{{HIP_PYTHON_BRANCH}}"
            HIP_PYTHON_VERSION = "{{HIP_PYTHON_VERSION}}"
            HIP_PYTHON_REV = "{{HIP_PYTHON_REV}}"\
            """
            ).strip()
        )

def write_package_init_file(
    output_dir: str,
    license_text: str,
    for_hip_python_package: str,
    lib_names,
    hip_version: str,
    hip_version_name: str,
    hip_version_major: str,
    hip_version_minor: str,
    hip_version_patch: str,
    hip_version_githash: str,
):
    with open(os.path.join(output_dir, "__init__.py"), "w") as f:
        init_content = (
            license_text
            + textwrap.dedent(
                f"""\
            
                # This file has been autogenerated, do not modify.
                
                __author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

                from ._version import *
                HIP_VERSION = {hip_version}
                HIP_VERSION_NAME = hip_version_name = "{hip_version_name}"
                HIP_VERSION_TUPLE = hip_version_tuple = ({hip_version_major},{hip_version_minor},{hip_version_patch},"{hip_version_githash}")

                """
            )
        )
        if for_hip_python_package:
            init_content += "\nfrom . import _util"
        
        for module_name in lib_names:
            init_content += textwrap.dedent(f"""
            try:
                from . import {module_name}
            except ImportError:
                pass # may have been excluded from build""")
        f.write(init_content)

def write_cuda_python_requirements_file(requirements_file: str, license_text: str,version: str):
    with open(requirements_file, "w") as outfile:
        outfile.write(
            license_text
            + textwrap.dedent(
                f"""\
            
                # This file has been autogenerated, do not modify.
                
                # Python dependencies required for development
                setuptools>=42
                cython
                wheel
                build
                hip-python=={version}.{{HIP_PYTHON_VERSION_SHORT}}"""
            )
        )

def write_docs_page_per_module(hip_python_docs_dir: str, hip_python_lib_names, cuda_python_lib_names):
    
    def write_module_markdown_file_(module,lib,extra=""):
        nonlocal hip_python_docs_dir

        with open(os.path.join(hip_python_docs_dir, "python_api", f"{lib}.md"),"w") as outfile:
            outfile.write(textwrap.dedent(
                f"""\
                # {module}.{lib}
                
                <!-- This file has been autogenerated, do not modify. -->

                <!-- global automodule options are set in conf.py -->
                ```{{eval-rst}}
                .. automodule:: {module}.{lib}
                {extra}

                ```"""
            ))

    for lib in hip_python_lib_names:
        write_module_markdown_file_("hip",lib)
    for lib in cuda_python_lib_names:
        write_module_markdown_file_("cuda",lib,extra="   :noindex:") # noindex, prevents ambiguity issues with enum constants

def render_index_md(
    hip_python_docs_dir: str,
    hip_python_lib_names,
    cuda_python_lib_names,
    hip_version_name
):
    index_md = os.path.join(
        hip_python_docs_dir, "index.md"
    )
    python_api_doc_names = [f"- {{doc}}`python_api/{lib}`" for lib in hip_python_lib_names]
    python_api_doc_names_cuda = [f"- {{doc}}`python_api/{lib}`" for lib in cuda_python_lib_names]
    with open(index_md + ".in","r"
         ) as infile, open(index_md, "w") as outfile:
        
        rendered = infile.read()
        rendered = rendered.replace("{PYTHON_API_DOC_NAMES}","\n".join(python_api_doc_names))
        rendered = rendered.replace("{PYTHON_API_DOC_NAMES_CUDA}","\n".join(python_api_doc_names_cuda))
        rendered = rendered.replace("{HIP_VERSION_NAME}", hip_version_name)
        outfile.write(rendered)

def render_toc_yml_in(
    hip_python_docs_dir: str,
    hip_python_lib_names,
    cuda_python_lib_names,
):
    toc_yml_in = os.path.join(
        hip_python_docs_dir, ".sphinx", "_toc.yml.in"
    )
    python_api_file_names = [f"      - file: python_api/{lib}" for lib in hip_python_lib_names]
    python_api_file_names_cuda = [f"      - file: python_api/{lib}" for lib in cuda_python_lib_names]
    with open(toc_yml_in + ".in","r"
         ) as infile, open(toc_yml_in, "w") as outfile:
        
        rendered = infile.read()
        rendered = rendered.replace("{PYTHON_API_FILE_NAMES}","\n".join(python_api_file_names))
        rendered = rendered.replace("{PYTHON_API_FILE_NAMES_CUDA}","\n".join(python_api_file_names_cuda))
        outfile.write(rendered)

def generate_cuda_interop_layer_files():
    """Generate the CUDA interoperability layer.

    Note:
        Some CUDA Driver and Runtime routines, namely cuLink*, have been mapped to HIPRTC instead of the HIP runtime.
    """
    global HIPRTC_GENERATOR
    global HIP_GENERATOR

    if HIPRTC_GENERATOR==None or HIP_GENERATOR==None:
        _log.warn("No CUDA runtime layer generated as 'hip' and/or 'hiprtc' have not been specified as libraries to parse.")
        return

    cuda_interop_layer_gen.generate_cuda_interop_module_files(
        OUTPUT_DIR,"nvrtc", HIPRTC_GENERATOR, HIP_2_CUDA
    )

    def collect_imports_(import_stmt: str,py_generator):
        contribs = ""
        for node in py_generator:
            hip_name = node.cython_global_name
            if hip_name in HIP_2_CUDA:
                for cuda_name in HIP_2_CUDA[hip_name]:
                    contribs += f"{import_stmt} {cuda_name}\n"
        return contribs
    extra_imports = collect_imports_("from cuda.nvrtc import",HIPRTC_GENERATOR.backend.walk_entities_to_import(False))
    extra_cimports = collect_imports_("from cuda.nvrtc cimport",HIPRTC_GENERATOR.backend.walk_entities_to_cimport(False))
    extra_cmodule_cimports = collect_imports_("from cuda.cnvrtc cimport",HIPRTC_GENERATOR.backend.walk_entities_to_cimport(True))

    cuda_interop_layer_gen.generate_cuda_interop_module_files(
        OUTPUT_DIR, "cuda", HIP_GENERATOR, HIP_2_CUDA,
        extra_imports=extra_imports,
        extra_cimports=extra_cimports,
        extra_cmodule_cimports=extra_cmodule_cimports,
    )
    cuda_interop_layer_gen.generate_cuda_interop_module_files(
        OUTPUT_DIR, "cudart", HIP_GENERATOR, HIP_2_CUDA, warn=False,
        extra_imports=extra_imports,
        extra_cimports=extra_cimports,
        extra_cmodule_cimports=extra_cmodule_cimports,
    )  # NOTE: cudart is the same as cuda, but we generate it to have also the corresponding pxd/pyx files. Could be solved via symlinks & __init__.py mod too.

if __name__ == "__main__":
    OUTPUT_DIR = None

    ROCM_INC = None
    RUNTIME_LINKING = None
    GENERATOR_ARGS = None
    LIBS = None
    HIP_2_CUDA = None

    HIP_VERSION_MAJOR = 0
    HIP_VERSION_MINOR = 0
    HIP_VERSION_PATCH = 0
    HIP_VERSION_GITHASH = ""

    parse_options()

    AVAILABLE_GENERATORS = dict(
        hip=generate_hip_module_files, # produces the versions
        hiprtc=generate_hiprtc_module_files,
        hipblas=generate_hipblas_module_files,
        rccl=generate_rccl_module_files,
        hiprand=generate_hiprand_module_files,
        hipfft=generate_hipfft_module_files,
        hipsparse=generate_hipsparse_module_files,
    )

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

    Path(os.path.join(OUTPUT_DIR, "hip-python")).mkdir(parents=False, exist_ok=True)
    Path(os.path.join(OUTPUT_DIR, "hip-python-as-cuda")).mkdir(parents=False, exist_ok=True)
    hip_output_dir = os.path.join(OUTPUT_DIR, "hip-python", "hip")
    cuda_output_dir = os.path.join(OUTPUT_DIR, "hip-python-as-cuda", "cuda") # must be here because of cuda interop codegen
    Path(hip_output_dir).mkdir(parents=False, exist_ok=True)
    Path(cuda_output_dir).mkdir(parents=False, exist_ok=True)
    for entry in lib_names:
        libname = entry.strip()
        if libname not in AVAILABLE_GENERATORS:
            available_libs = ", ".join([f"'{a}'" for a in AVAILABLE_GENERATORS.keys()])
            msg = f"no codegenerator found for library '{libname}'; please choose from: {available_libs}, or '*', which implies that all code generators will be used."
            raise KeyError(msg)
        generator = AVAILABLE_GENERATORS[libname]()
        generator.write_module_files(output_dir=hip_output_dir)
    generate_cuda_interop_layer_files()

    hip_version_name = f"{HIP_VERSION_MAJOR}.{HIP_VERSION_MINOR}.{HIP_VERSION_PATCH}-{HIP_VERSION_GITHASH}"
    hip_version = (
        HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH
    )

    version = f"{HIP_VERSION_MAJOR}.{HIP_VERSION_MINOR}.{HIP_VERSION_PATCH}.{interfacegen.gitversion.version()}"
    long_version = (
        f"{hip_version_name}.{interfacegen.gitversion.version(append_hash=True,append_date=True)}"
    )
    
    hip_python_lib_names = AVAILABLE_GENERATORS.keys()
    cuda_python_lib_names = ["cuda","cudart","nvrtc"]

    with open("LICENSE","r") as licensefile:
        license_text = "".join([f"# {ln}\n" for ln in licensefile.read().rstrip().splitlines()])
    for output_dir in (hip_output_dir, cuda_output_dir):
        # hip|cuda/_version.py
        write_version_file(output_dir,license_text,version,long_version)
        # hip|cuda/__init__.py
        # TODO make option to use all generators or only specified ones
        for_hip_python_package = output_dir == hip_output_dir
        write_package_init_file(
            output_dir,
            license_text,
            for_hip_python_package,
            hip_python_lib_names if for_hip_python_package else cuda_python_lib_names,
            hip_version,
            hip_version_name,
            HIP_VERSION_MAJOR,
            HIP_VERSION_MINOR,
            HIP_VERSION_PATCH,
            HIP_VERSION_GITHASH,
        )

    # hip-python-as-cuda/requirements.txt
    requirements_file = os.path.join(
        OUTPUT_DIR, "hip-python-as-cuda", "requirements.txt.in"
    )
    write_cuda_python_requirements_file(requirements_file,license_text,version)
    
    # hip-python docs
    hip_python_docs_dir = os.path.join(OUTPUT_DIR,"hip-python","docs")
    Path(hip_python_docs_dir).mkdir(parents=False, exist_ok=True)
    # files per api
    Path(os.path.join(hip_python_docs_dir, "python_api")).mkdir(parents=False, exist_ok=True)
    write_docs_page_per_module(hip_python_docs_dir, hip_python_lib_names, cuda_python_lib_names)
    # index.md from index.md.in
    render_index_md(hip_python_docs_dir,hip_python_lib_names,cuda_python_lib_names,hip_version_name)
    # _toc.yml.in from _toc.yml.in.in
    Path(os.path.join(hip_python_docs_dir, ".sphinx")).mkdir(parents=False, exist_ok=True)
    render_toc_yml_in(hip_python_docs_dir,hip_python_lib_names,cuda_python_lib_names)
