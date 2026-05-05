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

"""This is the project's setup script.

After pointing this setup script to an HIP installation,
it generates Cython files and compiles them to Python
modules. The generated Cython declaration files can be used
by Cython users of this project.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

import argparse
import ctypes
import enum
import logging
import os
import re
import textwrap
from pathlib import Path

from . import cuda_interop as cuda_interop_layer_gen
from .hipify import parse_hipify_perl

import interfacegen
from interfacegen.cparser import TypeHandler
from interfacegen.cython import (
    CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
    CythonModuleGenerator,
)
from interfacegen.support import gitversion
from interfacegen.support.recipes import hip as controls
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
cuda_interop_layer_gen.python_interface_pyobj_role_template = (
    r"`.{name}`"  # note: here we want to keep the qualifier
)

HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER = (
    CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER("rocm.bindings.util.types.")
)

TypeCategory = TypeHandler.TypeCategory


def parse_options():
    global OUTPUT_DIR
    global ROCM_INC
    global RUNTIME_LINKING
    global GENERATOR_ARGS
    global LIBS
    global HIP_2_CUDA
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

    def rocm_version(arg):
        if not re.match(r"[0-9]+\.[0-9]+\.[0-9]+", arg):
            raise ValueError(
                "Value of required argument `--rocm-version` must be a dot-separated number triple."
            )
        return [int(p) for p in arg.split(".")]

    parser.add_argument(
        "--rocm-version",
        type=rocm_version,
        required=True,
        dest="rocm_version",
        help="The ROCm version.",
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
        rocm_path=os.environ.get(
            "ROCM_PATH", os.environ.get("ROCM_HOME", None)
        ),
        platform=os.environ.get("HIP_PLATFORM", "amd"),
        clang_resource_dir=os.environ.get("HIP_PYTHON_CLANG_RES_DIR", None),
        libs=os.environ.get("HIP_PYTHON_LIBS", "*"),
        runtime_linking=get_bool_environ_var(
            "HIP_PYTHON_RUNTIME_LINKING", "true"
        ),
        verbose=False,
    )
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    RUNTIME_LINKING = args.runtime_linking
    LIBS = args.libs

    (ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR, ROCM_VERSION_PATCH) = (
        args.rocm_version
    )

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
# TODO C901 function is too complex
def generate_hip_module_files():  # noqa: C901
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
            return (
                f"rocm.bindings._hip_helpers.{toclassname(parm.parent.name)}_{parm.name}"
            )
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
            return "rocm.bindings.util.types.DeviceArray"

        return HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(parm)

    def hip_node_init(node: Node):
        # node modifications
        if isinstance(node, interfacegen.tree.Function):
            if not node.is_enum and node.name.startswith("hip"):
                # hip routines without hipError_t return status
                # we force them to not throw exceptions
                node.error_return_value_lazy_loader = None
                node.modifiers_lazy_loader = " noexcept nogil"
                # we force them to have always return hipSuccess as first return value
                node.prepend_python_return_value(
                    "hipError_t.hipSuccess",
                    "hipError_t",
                    "Always returns `~.hipError_t.hipSuccess`.",
                )
        elif isinstance(node, interfacegen.tree.Parm):
            func_name, parm_idx = node.parent.name, node.parm_index
            if (func_name, parm_idx) in (
                ("hipDeviceGetName", 0),
                ("hipDeviceGetPCIBusId", 0),
            ):
                func = node.parent
                assert isinstance(func, interfacegen.cython.Function)
                len_param: interfacegen.tree.Parm = func.get_parm(1)
                func.python_body_prepend_before_c_interface_call(
                    f"{node.name}.malloc({len_param.name})"
                )

    def renamer(name: str):
        return interfacegen.cython.DEFAULT_RENAMER(controls.hip.renamer(name))

    def macro_type(node: MacroDefinition):
        macro_name = node.name
        if macro_name in controls.hip.void_p_macros:
            return ctypes.c_ulonglong(controls.hip.void_p_macros[macro_name])
        return controls.hip.macro_type(node)

    generator = CythonModuleGenerator(
        "rocm.bindings.hip",
        ROCM_INC,
        "hip/hip_runtime.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="rocm.bindings.util",
        dll="libamdhip64.so",
        modifiers_lazy_loader=" except? hipErrorInitializationError nogil",
        error_return_value_lazy_loader="hipErrorInitializationError",
        # we hijack hipError_t constant hipErrorInitializationError for propagating exceptions
        # more details: https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#error-return-values
        node_init=hip_node_init,
        renamer=renamer,
        node_filter=controls.hip.node_filter,
        ptr_parm_intent=controls.hip.ptr_parm_intent,
        ptr_rank=controls.hip.ptr_rank,
        ptr_complicated_type_handler=hip_ptr_complicated_type_handler,
        macro_type=macro_type,
        raw_comment_cleaner=controls.hip.raw_comment_cleaner,
        cflags=GENERATOR_ARGS,
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    cimport rocm.bindings._hip_helpers
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
    global ROCM_VERSION_MAJOR
    global ROCM_VERSION_MAJOR
    global GENERATOR_ARGS
    global RUNTIME_LINKING
    global HIPRTC_GENERATOR

    def hiprtc_ptr_complicated_type_handler(node: Node):
        if isinstance(node, Parm):
            if (node.parent.name, node.name) in (
                ("hiprtcCompileProgram", "options"),
                ("hiprtcCreateProgram", "headers"),
                ("hiprtcCreateProgram", "includeNames"),
            ):
                return "rocm.bindings.util.types.ListOfBytes"
            if (node.parent.name, node.name) == (
                "hiprtcLinkCreate",
                "option_ptr",
            ):
                return "rocm.bindings._hiprtc_helpers.HiprtcLinkCreate_option_ptr"
            if (node.parent.name, node.name) == (
                "hiprtcLinkCreate",
                "option_vals_pptr",
            ):
                return "rocm.bindings.util.types.ListOfPointer"
            if (node.parent.name, node.parm_index) in (
                ("hiprtcLinkComplete", 1),
                ("hiprtcGetCode", 1),
                ("hiprtcGetBitcode", 1),
            ):
                return "rocm.bindings.util.types.NDBuffer"
        return HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(node)

    def hiprtc_node_init(node: Node):
        # node modifications
        if isinstance(node, interfacegen.tree.Function):
            if not node.is_enum and node.name.startswith("hiprtc"):
                # hip routines without hipError_t return status
                # we force them to not throw exceptions
                node.error_return_value_lazy_loader = None
                node.modifiers_lazy_loader = " noexcept nogil"

    generator = CythonModuleGenerator(
        "rocm.bindings.hiprtc",
        ROCM_INC,
        "hip/hiprtc.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="rocm.bindings.util",
        dll="libhiprtc.so",
        # we hijack hiprtcResult constant HIPRTC_ERROR_INTERNAL_ERROR for propagating exceptions
        # more details: https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#error-return-values
        modifiers_lazy_loader=" except? HIPRTC_ERROR_INTERNAL_ERROR nogil",
        error_return_value_lazy_loader="HIPRTC_ERROR_INTERNAL_ERROR",
        node_init=hiprtc_node_init,
        node_filter=controls.hiprtc.node_filter,
        ptr_parm_intent=controls.hiprtc.ptr_parm_intent,
        ptr_rank=controls.hiprtc.ptr_rank,
        ptr_complicated_type_handler=hiprtc_ptr_complicated_type_handler,
        cflags=GENERATOR_ARGS,
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
        cimport rocm.bindings._hiprtc_helpers
        """
    )

    if (ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR) >= (6, 4):
        generator.c_interface_decl_prolog += textwrap.dedent(
            """\
            from rocm.bindings.cyhip cimport hipJitOption
            from rocm.bindings.cyhip cimport hipJitOption as hiprtcJIT_option
            from rocm.bindings.cyhip cimport hipJitInputType
            from rocm.bindings.cyhip cimport hipJitInputType as hiprtcJITInputType
            """
        )

        generator.python_interface_impl_prolog += textwrap.dedent(
            """\
            from rocm.bindings.hip import _hipJitOption__Base
            from rocm.bindings.hip import hipJitOption
            from rocm.bindings.hip import hipJitOption as hiprtcJIT_option
            from rocm.bindings.hip import _hipJitInputType__Base
            from rocm.bindings.hip import hipJitInputType
            from rocm.bindings.hip import hipJitInputType as hiprtcJITInputType
            """
        )

    HIPRTC_GENERATOR = generator
    return generator


# hipblas
def generate_hipblas_module_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonModuleGenerator(
        "rocm.bindings.hipblas",
        ROCM_INC,
        "hipblas/hipblas.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="rocm.bindings.util",
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


# hipsolver
def generate_hipsolver_module_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonModuleGenerator(
        "rocm.bindings.hipsolver",
        ROCM_INC,
        "hipsolver/hipsolver.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="rocm.bindings.util",
        dll="libhipsolver.so",
        node_filter=controls.hipsolver.node_filter,
        ptr_parm_intent=controls.hipsolver.ptr_parm_intent,
        ptr_rank=controls.hipsolver.ptr_rank,
        ptr_complicated_type_handler=HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
        raw_comment_cleaner=controls.hipsolver.raw_comment_cleaner,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    # from rocm.bindings.cyhip cimport * # via chipblas
    from rocm.bindings.cyhipblas cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    # from rocm.bindings.hip cimport * # via chipblas
    from rocm.bindings.hipblas cimport *
    """
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hipblas import _hipblasSideMode_t__Base
    from rocm.bindings.hipblas import _hipblasFillMode_t__Base
    from rocm.bindings.hipblas import _hipblasOperation_t__Base
    from rocm.bindings.hip import _hipDataType__Base
    """
    )
    return generator


# rccl
def generate_rccl_module_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonModuleGenerator(
        "rocm.bindings.rccl",
        ROCM_INC,
        "rccl/rccl.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="rocm.bindings.util",
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
    from rocm.bindings.cyhip cimport hipStream_t
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport ihipStream_t
    """
    )
    return generator


# hiprand
def generate_hiprand_module_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonModuleGenerator(
        "rocm.bindings.hiprand",
        ROCM_INC,
        "hiprand/hiprand.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="rocm.bindings.util",
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
    from rocm.bindings.cyhip cimport hipStream_t
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport ihipStream_t
    """
    )
    return generator


# hipfft
def generate_hipfft_module_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonModuleGenerator(
        "rocm.bindings.hipfft",
        ROCM_INC,
        "hipfft/hipfft.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="rocm.bindings.util",
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
    from rocm.bindings.cyhip cimport hipStream_t, float2, double2
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport ihipStream_t, float2, double2
    """
    )
    return generator


# hipsparse
def generate_hipsparse_module_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonModuleGenerator(
        "rocm.bindings.hipsparse",
        ROCM_INC,
        "hipsparse/hipsparse.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="rocm.bindings.util",
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


# roctx
def generate_roctx_module_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonModuleGenerator(
        "rocm.bindings.roctx",
        ROCM_INC,
        "roctracer/roctx.h",
        runtime_linking=RUNTIME_LINKING,
        util_pkg="rocm.bindings.util",
        dll="libroctx64.so",
        node_filter=controls.roctx.node_filter,
        macro_type=controls.roctx.macro_type,
        ptr_parm_intent=controls.roctx.ptr_parm_intent,
        ptr_rank=controls.roctx.ptr_rank,
        ptr_complicated_type_handler=HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
        cflags=GENERATOR_ARGS,
    )
    return generator


# NOTE: helpers that previously wrote `_version.py.in`, `__init__.py`,
# `requirements.txt.in`, and Sphinx docs files were removed during the
# generator-output scope cleanup (plan §B.7). The hip-python repo owns
# those files as handcoded sources; the generator only emits Cython
# bindings and CMake module-list includes.


def _version_as_str(major, minor, patch, githash=None):
    if githash:
        return f"{major}.{minor}.{patch}-{githash}"
    else:
        return f"{major}.{minor}.{patch}"


def _version_as_int(major, minor, patch):
    """Return ROCm / HIP version as integer.
    Note:
        This is the typical way to compute the version.
    """
    return major * 10000000 + minor * 100000 + patch


def generate_cuda_interop_layer_files(license_text: str):
    """Generate the CUDA interoperability layer.

    Note:
        Some CUDA Driver and Runtime routines, namely cuLink*, have been mapped to HIPRTC instead of the HIP runtime.

    Note:
        CUDA Python's `cudaRuntimeGetVersion(...)` returns the version of the
        CUDA version that has been used to generate the bindings. This might
        differ (at least in the patch version) from the version of the CUDA
        runtime that a user might use the bindings for; more details:
        <https://github.com/NVIDIA/cuda-python/issues/16>
        HIP Python's `hipRuntimeGetVersion` has always been calling into the
        loaded runtime so there is no need for a `getLocalRuntimeVersion`.
        In the CUDA compatibility layer's modules, we make
        `getLocalRuntimeVersion` an alias of `hipRuntimeGetVersion`.
    """
    global OUTPUT_DIR
    global HIP_2_CUDA
    global HIPRTC_GENERATOR
    global HIP_GENERATOR
    global ROCM_VERSION_MAJOR
    global ROCM_VERSION_MINOR

    if HIPRTC_GENERATOR is None or HIP_GENERATOR is None:
        _log.warning(
            "No CUDA runtime layer generated as 'hip' and/or 'hiprtc' have not been specified as libraries to parse."
        )
        return

    # See: https://github.com/NVIDIA/cuda-python/issues/16
    assert "hipRuntimeGetVersion" in HIP_2_CUDA
    HIP_2_CUDA["hipRuntimeGetVersion"].append("getLocalRuntimeVersion")

    if (ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR) >= (6, 4):
        # NOTE: Hipify may lag behind the header files.
        #       So we remove outdated keys and ensure
        #       the values are equal to the new names.
        for bad_name in ("hiprtcJITInputType", "hiprtcJIT_option"):
            try:
                del HIP_2_CUDA[bad_name]
            except KeyError:
                pass
        HIP_2_CUDA["hipJitInputType"] = [
            "CUjitInputType",
            "CUjitInputType_enum",
        ]
        HIP_2_CUDA["hipJitOption"] = ["CUjit_option", "CUjit_option_enum"]

    def collect_imports_(import_stmt: str, py_generator):
        contribs = ""
        for node in py_generator:
            hip_name = node.cython_global_name
            if hip_name in HIP_2_CUDA:
                for cuda_name in HIP_2_CUDA[hip_name]:
                    contribs += f"{import_stmt} {cuda_name}\n"
        return contribs

    # Modern layout: only emit cuda.bindings.{driver,runtime,nvrtc} with the
    # `cy` prefix on C-level wrappers. The legacy cuda.{cuda,cudart} flat
    # modules are no longer produced (plan §B.6 / §B.5).
    for config in (
        dict(
            parent_package="cuda.bindings",
            driver_name="driver",
            runtime_name="runtime",
            cmodule_prefix="cy",
        ),
    ):
        # write nvrtcm module files
        parent_package = config["parent_package"]
        cmodule_prefix = config["cmodule_prefix"]
        cuda_interop_layer_gen.generate_cuda_interop_module_files(
            OUTPUT_DIR,
            f"{parent_package}.nvrtc",
            HIPRTC_GENERATOR,
            HIP_2_CUDA,
            license_text,
            cuda_cmodule_prefix=cmodule_prefix,
        )

        # NOTE: hiprtc functions such as hiprtcLink* correspond to cuLink*
        #       functions associated with the driver/runtime and not with nvrtc.
        #       With the below trick we ensure that cuLink* symbols are present
        #       in both driver/runtime and the nvrtc interop layer.
        cuda_extra_args = dict(
            extra_imports=collect_imports_(
                f"from {parent_package}.nvrtc import",
                HIPRTC_GENERATOR.backend.walk_entities_to_import(False),
            ),
            extra_cimports=collect_imports_(
                f"from {parent_package}.nvrtc cimport",
                HIPRTC_GENERATOR.backend.walk_entities_to_cimport(False),
            ),
            extra_cmodule_cimports=collect_imports_(
                f"from {parent_package}.{cmodule_prefix}nvrtc cimport",
                HIPRTC_GENERATOR.backend.walk_entities_to_cimport(True),
            ),
            cuda_cmodule_prefix=cmodule_prefix,
        )

        # writer driver module files
        cuda_interop_layer_gen.generate_cuda_interop_module_files(
            OUTPUT_DIR,
            f"{parent_package}.{config['driver_name']}",
            HIP_GENERATOR,
            HIP_2_CUDA,
            license_text,
            **cuda_extra_args,
        )

        # write runtime module files
        cuda_interop_layer_gen.generate_cuda_interop_module_files(
            OUTPUT_DIR,
            f"{parent_package}.{config['runtime_name']}",
            HIP_GENERATOR,
            HIP_2_CUDA,
            license_text,
            warn=False,
            **cuda_extra_args,
        )  # NOTE: cudart is the same as cuda, but we generate it to have also the corresponding pxd/pyx files. Could be solved via symlinks & __init__.py mod too.


AVAILABLE_GENERATORS = dict(
    hip=generate_hip_module_files,  # produces the versions
    hiprtc=generate_hiprtc_module_files,
    hipblas=generate_hipblas_module_files,
    rccl=generate_rccl_module_files,
    hiprand=generate_hiprand_module_files,
    hipfft=generate_hipfft_module_files,
    hipsparse=generate_hipsparse_module_files,
    roctx=generate_roctx_module_files,
    hipsolver=generate_hipsolver_module_files,
)


def generate(opts):  # noqa: C901
    """Run HIP + interop subgenerators against the modern hip-python layout.

    Writes only `.pxd`/`.pyx` files. Python-packaging artifacts (`__init__.py`,
    `_version.py.in`, `setup.py`, `requirements.txt`, docs) are NOT produced
    — those are handcoded in the hip-python repo (plan §B.7).

    `opts` is the unified-CLI options object. Required attributes:
      output_dir, rocm_path, rocm_version, platform, clang_resource_dir,
      hip_libs, runtime_linking.
    """
    global OUTPUT_DIR, ROCM_INC, RUNTIME_LINKING, GENERATOR_ARGS, LIBS, HIP_2_CUDA
    global ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR, ROCM_VERSION_PATCH
    global HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH, HIP_VERSION_GITHASH
    global HIPRTC_GENERATOR, HIP_GENERATOR

    OUTPUT_DIR = opts.output_dir
    ROCM_INC = os.path.join(opts.rocm_path, "include")
    RUNTIME_LINKING = opts.runtime_linking
    LIBS = opts.hip_libs
    hipify_perl_path = os.path.join(opts.rocm_path, "bin", "hipify-perl")
    (_, HIP_2_CUDA) = parse_hipify_perl(hipify_perl_path)
    HIPRTC_GENERATOR = None
    HIP_GENERATOR = None

    rocm_v = opts.rocm_version.split(".")
    ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR, ROCM_VERSION_PATCH = (
        int(rocm_v[0]), int(rocm_v[1]), int(rocm_v[2])
    )
    HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH = (0, 0, 0)
    HIP_VERSION_GITHASH = ""

    # Match the legacy parse_options() construction so libclang gets
    # __HIP_PLATFORM_AMD__/__HIP_PLATFORM_NVIDIA__, the ROCm include path,
    # and the clang resource dir. Without these the headers parse as a much
    # smaller subset and the output is missing most constants/comments.
    GENERATOR_ARGS = list(opts.generator_args or [])
    GENERATOR_ARGS += ["-D", f"__HIP_PLATFORM_{opts.platform.upper()}__"]
    GENERATOR_ARGS += [f"-I{ROCM_INC}"]
    if not opts.clang_resource_dir:
        raise RuntimeError(
            "Clang resource directory is not set. Pass --clang-resource-dir "
            "(e.g. `$($ROCM_PATH/llvm/bin/clang -print-resource-dir)`)."
        )
    GENERATOR_ARGS += ["-resource-dir", opts.clang_resource_dir]

    # Resolve the requested library set
    avail_lib_names = AVAILABLE_GENERATORS.keys()
    processed_libs = LIBS.replace(" ", "")
    if processed_libs == "*":
        lib_names = list(avail_lib_names)
    elif processed_libs.startswith("^"):
        excludes = processed_libs[1:].split(",")
        lib_names = [n for n in avail_lib_names if n not in excludes]
    else:
        lib_names = processed_libs.split(",")
        for name in lib_names:
            if name not in avail_lib_names:
                raise ValueError(
                    f"library name '{name}' is not valid, use one of: {', '.join(avail_lib_names)}"
                )

    # Modern layout:
    #   hip, hiprtc      -> rocm-bindings-hip
    #   rccl, roctx      -> rocm-bindings-systems  (collective comm + tracing)
    #   everything else  -> rocm-bindings-libraries  (math / FFT / random / sparse)
    HIP_CORE_LIBS = {"hip", "hiprtc"}
    SYSTEMS_LIBS = {"rccl", "roctx"}
    hip_pkg_dir = os.path.join(
        OUTPUT_DIR, "python", "rocm-bindings-hip", "rocm", "bindings"
    )
    libraries_pkg_dir = os.path.join(
        OUTPUT_DIR, "python", "rocm-bindings-libraries", "rocm", "bindings"
    )
    systems_pkg_dir = os.path.join(
        OUTPUT_DIR, "python", "rocm-bindings-systems", "rocm", "bindings"
    )
    cuda_output_dir = os.path.join(
        OUTPUT_DIR, "python", "hip-python-interop", "cuda"
    )
    Path(hip_pkg_dir).mkdir(parents=True, exist_ok=True)
    Path(libraries_pkg_dir).mkdir(parents=True, exist_ok=True)
    Path(systems_pkg_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(cuda_output_dir, "bindings")).mkdir(parents=True, exist_ok=True)

    for libname in (entry.strip() for entry in lib_names):
        if libname not in AVAILABLE_GENERATORS:
            available_libs = ", ".join(f"'{a}'" for a in AVAILABLE_GENERATORS.keys())
            raise KeyError(
                f"no codegenerator found for library '{libname}'; please choose "
                f"from: {available_libs}, or '*'."
            )
        generator = AVAILABLE_GENERATORS[libname]()
        if libname in HIP_CORE_LIBS:
            target_dir = hip_pkg_dir
        elif libname in SYSTEMS_LIBS:
            target_dir = systems_pkg_dir
        else:
            target_dir = libraries_pkg_dir
        generator.write_module_files(output_dir=target_dir)

    # CUDA interop layer (writes into <output_dir>/python/hip-python-interop/cuda/...)
    license_path = opts.license_path or os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "LICENSE"
    )
    with open(license_path, "r") as licensefile:
        license_text = "".join(
            f"# {ln}\n" for ln in licensefile.read().rstrip().splitlines()
        )
    generate_cuda_interop_layer_files(license_text)

    # Return data the orchestrator may want (e.g. version metadata for
    # cmake/generated_versions.cmake; per-package module lists).
    libraries_modules = [
        n for n in lib_names
        if n not in HIP_CORE_LIBS and n not in SYSTEMS_LIBS
    ]
    systems_modules = [n for n in lib_names if n in SYSTEMS_LIBS]
    return dict(
        rocm_version=(ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR, ROCM_VERSION_PATCH),
        hip_version=(
            HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH,
            HIP_VERSION_GITHASH,
        ),
        hip_modules=lib_names,
        libraries_modules=libraries_modules,
        systems_modules=systems_modules,
    )


if __name__ == "__main__":
    OUTPUT_DIR = None
    ROCM_INC = None
    RUNTIME_LINKING = None
    GENERATOR_ARGS = None
    LIBS = None
    HIP_2_CUDA = None
    ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR, ROCM_VERSION_PATCH = (0, 0, 0)
    HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH = (0, 0, 0)
    HIP_VERSION_GITHASH = ""
    HIPRTC_GENERATOR = None
    HIP_GENERATOR = None

    parse_options()  # populates the module-level globals from CLI

    # Adapt populated globals into the generate(opts) entry point.
    class _Opts:
        pass
    opts = _Opts()
    opts.output_dir = OUTPUT_DIR
    opts.rocm_path = os.path.dirname(ROCM_INC)
    opts.rocm_version = f"{ROCM_VERSION_MAJOR}.{ROCM_VERSION_MINOR}.{ROCM_VERSION_PATCH}"
    opts.platform = "amd"
    opts.clang_resource_dir = None
    opts.hip_libs = LIBS
    opts.runtime_linking = RUNTIME_LINKING
    opts.generator_args = GENERATOR_ARGS
    opts.license_path = None
    generate(opts)
