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
it generates Fortran module files.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

import os
import textwrap
import logging

import interfacegen

interfacegen.enable_logging(logging.INFO)
_log = logging.getLogger("interfacegen")
from interfacegen.support import includetree as it

from interfacegen.support import fortran as support
from interfacegen.support.recipes import hip as controls
from interfacegen.support.recipes.hipify import parse_hipify_perl

from interfacegen.fortran import FortranModuleGenerator

from interfacegen.cparser import TypeHandler

TypeCategory = TypeHandler.TypeCategory

from interfacegen.tree import (
    Node,
    MacroDefinition,
    Parm,
)


# hip
def generate_hip_module_files():
    global pkg_opts
    global GENERATOR_ARGS
    global HIP_GENERATOR

    global HIP_VERSION_MAJOR
    global HIP_VERSION_MINOR
    global HIP_VERSION_PATCH
    global HIP_VERSION_GITHASH

    # def toclassname(name: str):
    #     return name[0].upper() + name[1:]

    # def hip_ptr_complicated_type_handler(parm: Node):
    #     if (parm.parent.name, parm.name) == ("hipModuleLaunchKernel", "extra"):
    #         return f"hip._hip_helpers.{toclassname(parm.parent.name)}_{parm.name}"
    #     if (parm.parent.name, parm.name) in (
    #         ("hipMalloc", "ptr"),
    #         ("hipExtMallocWithFlags", "ptr"),
    #         ("hipMallocManaged", "dev_ptr"),
    #         ("hipMallocAsync", "dev_ptr"),
    #         ("hipMallocFromPoolAsync", "dev_ptr"),
    #     ):
    #         if parm.parent.name == "hipExtMallocWithFlags":
    #             size = "sizeBytes"
    #         else:
    #             size = "size"
    #         parm.parent.python_body_prepend_before_return(
    #             f"{parm.name}.configure(_force=True,shape=(cpython.long.PyLong_FromUnsignedLong({size}),))"
    #         )
    #         return "hip._util.types.DeviceArray"

    #     return HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(parm)

    def hip_node_init(node: Node):
        pass
        # # node modifications
        # if isinstance(node, interfacegen.tree.Function):
        #     if not node.is_enum and node.name.startswith("hip"):
        #         # hip routines without hipError_t return status
        #         # we force them to not throw exceptions
        #         node.error_return_value_lazy_loader = None
        #         node.modifiers_lazy_loader = " noexcept nogil"
        #         # we force them to have always return hipSuccess as first return value
        #         node.prepend_python_return_value(
        #             "hipError_t.hipSuccess",
        #             "hipError_t",
        #             "Always returns `~.hipError_t.hipSuccess`.",
        #         )
        # elif isinstance(node, interfacegen.tree.Parm):
        #     func_name, parm_idx = node.parent.name, node.parm_index
        #     if (func_name, parm_idx) in (
        #         ("hipDeviceGetName", 0),
        #         ("hipDeviceGetPCIBusId", 0),
        #     ):
        #         func = node.parent
        #         assert isinstance(func, interfacegen.cython.Function)
        #         len_param: interfacegen.tree.Parm = func.get_parm(1)
        #         func.python_body_prepend_before_c_interface_call(
        #             f"{node.name}.malloc({len_param.name})"
        #         )

    def renamer(name: str):
        return interfacegen.cython.DEFAULT_RENAMER(controls.hip.renamer(name))

    generator = FortranModuleGenerator(
        "hipfort_hip",
        pkg_opts.abs_inc_dir,
        "hip/hip_runtime.h",
        node_init=hip_node_init,
        renamer=renamer,
        node_filter=controls.hip.node_filter,
        ptr_parm_intent=controls.hip.ptr_parm_intent,
        ptr_rank=controls.hip.ptr_rank,
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
    global pkg_opts
    global GENERATOR_ARGS
    global HIPRTC_GENERATOR

    # def hiprtc_ptr_complicated_type_handler(node: Node):
    #     if isinstance(node, Parm):
    #         if (node.parent.name, node.name) in (
    #             ("hiprtcCompileProgram", "options"),
    #             ("hiprtcCreateProgram", "headers"),
    #             ("hiprtcCreateProgram", "includeNames"),
    #         ):
    #             return "hip._util.types.ListOfBytes"
    #         if (node.parent.name, node.name) == ("hiprtcLinkCreate", "option_ptr"):
    #             return "hip._hiprtc_helpers.HiprtcLinkCreate_option_ptr"
    #         if (node.parent.name, node.name) == ("hiprtcLinkCreate", "option_vals_pptr"):
    #             return "hip._util.types.ListOfPointer"
    #         if (node.parent.name, node.parm_index) in (
    #             ("hiprtcLinkComplete", 1),
    #             ("hiprtcGetCode", 1),
    #             ("hiprtcGetBitcode", 1),
    #         ):
    #             return "hip._util.types.NDBuffer"
    #     return HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(node)

    # def hiprtc_node_init(node: Node):
    #     # node modifications
    #     if isinstance(node, interfacegen.tree.Function):
    #         if not node.is_enum and node.name.startswith("hiprtc"):
    #             # hip routines without hipError_t return status
    #             # we force them to not throw exceptions
    #             node.error_return_value_lazy_loader = None
    #             node.modifiers_lazy_loader = " noexcept nogil"

    generator = FortranModuleGenerator(
        "hipfort_hiprtc",
        pkg_opts.abs_inc_dir,
        "hip/hiprtc.h",
        # node_init=hiprtc_node_init,
        node_filter=controls.hiprtc.node_filter,
        ptr_parm_intent=controls.hiprtc.ptr_parm_intent,
        ptr_rank=controls.hiprtc.ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
        cimport hip._hiprtc_helpers
        """
    )

    HIPRTC_GENERATOR = generator
    return generator


# hipblas
def generate_hipblas_module_files():
    global pkg_opts
    global GENERATOR_ARGS

    generator = FortranModuleGenerator(
        "hipfort_hipblas",
        pkg_opts.abs_inc_dir,
        "hipblas/hipblas.h",
        node_filter=controls.hipblas.node_filter,
        ptr_parm_intent=controls.hipblas.ptr_parm_intent,
        ptr_rank=controls.hipblas.ptr_rank,
        raw_comment_cleaner=controls.hipblas.raw_comment_cleaner,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_prolog += textwrap.dedent(
        """\
    from .chip cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from .hip cimport *
    """
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from .hip import _hipDataType__Base
    """
    )
    return generator


# hipsolver
def generate_hipsolver_module_files():
    global pkg_opts
    global GENERATOR_ARGS

    generator = FortranModuleGenerator(
        "hipfort_hipsolver",
        pkg_opts.abs_inc_dir,
        "hipsolver/hipsolver.h",
        node_filter=controls.hipsolver.node_filter,
        ptr_parm_intent=controls.hipsolver.ptr_parm_intent,
        ptr_rank=controls.hipsolver.ptr_rank,
        raw_comment_cleaner=controls.hipsolver.raw_comment_cleaner,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_prolog += textwrap.dedent(
        """\
    # from .chip cimport * # via chipblas
    from .chipblas cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    # from .hip cimport * # via chipblas
    from .hipblas cimport *
    """
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from .hipblas import _hipblasSideMode_t__Base
    from .hipblas import _hipblasFillMode_t__Base
    from .hipblas import _hipblasOperation_t__Base
    """
    )
    return generator


# rccl
def generate_rccl_module_files():
    global pkg_opts
    global GENERATOR_ARGS

    generator = FortranModuleGenerator(
        "hipfort_rccl",
        pkg_opts.abs_inc_dir,
        "rccl/rccl.h",
        dll="librccl.so",
        node_filter=controls.rccl.node_filter,
        macro_type=controls.rccl.macro_type,
        ptr_parm_intent=controls.rccl.ptr_parm_intent,
        ptr_rank=controls.rccl.ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_prolog += textwrap.dedent(
        """\
    from .chip cimport hipStream_t
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from .hip cimport ihipStream_t
    """
    )
    return generator


# hiprand
def generate_hiprand_module_files():
    global pkg_opts
    global GENERATOR_ARGS

    generator = FortranModuleGenerator(
        "hipfort_hiprand",
        pkg_opts.abs_inc_dir,
        "hiprand/hiprand.h",
        node_filter=controls.hiprand.node_filter,
        macro_type=controls.hiprand.macro_type,
        ptr_parm_intent=controls.hiprand.ptr_parm_intent,
        ptr_rank=controls.hiprand.ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_prolog += textwrap.dedent(
        """\
    from .chip cimport hipStream_t
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from .hip cimport ihipStream_t
    """
    )
    return generator


# hipfft
def generate_hipfft_module_files():
    global pkg_opts
    global GENERATOR_ARGS

    generator = FortranModuleGenerator(
        "hipfort_hipfft",
        pkg_opts.abs_inc_dir,
        "hipfft/hipfft.h",
        node_filter=controls.hipfft.node_filter,
        macro_type=controls.hipfft.macro_type,
        ptr_parm_intent=controls.hipfft.ptr_parm_intent,
        ptr_rank=controls.hipfft.ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_prolog += textwrap.dedent(
        """\
    from .chip cimport hipStream_t, float2, double2
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from .hip cimport ihipStream_t, float2, double2
    """
    )
    return generator


# hipsparse
def generate_hipsparse_module_files():
    global pkg_opts
    global GENERATOR_ARGS

    generator = FortranModuleGenerator(
        "hipfort_hipsparse",
        pkg_opts.abs_inc_dir,
        "hipsparse/hipsparse.h",
        node_filter=controls.hipsparse.node_filter,
        macro_type=controls.hipsparse.macro_type,
        ptr_parm_intent=controls.hipsparse.ptr_parm_intent,
        ptr_rank=controls.hipsparse.ptr_rank,
        raw_comment_cleaner=controls.hipsparse.raw_comment_cleaner,
        cflags=GENERATOR_ARGS,
    )
    return generator


# roctx
def generate_roctx_module_files():
    global pkg_opts

    generator = FortranModuleGenerator(
        "hipfort_roctx",
        ROCM_INC,
        "roctracer/roctx.h",
        node_filter=controls.roctx.node_filter,
        macro_type=controls.roctx.macro_type,
        ptr_parm_intent=controls.roctx.ptr_parm_intent,
        ptr_rank=controls.roctx.ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    return generator


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

if __name__ == "__main__":
    GENERATOR_ARGS = None
    HIP_2_CUDA = None

    pkg_opts: support.RocmPackageOpts = support.create_rocm_package_opts_from_cli(
        project="HIPFORT",
        env_var_prefix="HIPFORT_",
        libs_example="hip,hiprtc",
        package="hipfort",
        rel_inc_dir="include",
        author="Advanced Micro Devices, Inc.",
        email="hipfort.maintainer@amd.com",
    )

    def filter(filepath: str):
        filename = os.path.basename(filepath)
        # print(filename)
        if filename in ("rccl.h", "miopen.h", "hip_runtime_api.h"):
            return True
        if filename[:3] in ("hip", "roc", "amd"):
            # TODO fortran cory are those needed?
            for temporarily_excluded in (
                "rocrandapi.h",
                "hiplibxt.h",
                "v2/rocprofiler.h",
            ):
                if temporarily_excluded in filepath:
                    return False
            if "-" in filename:
                return False
            if "_" in filename:
                if filename not in ("rocm_smi.h",):
                    return False
            for key in ("detail", "internal", "version"):
                if key in filepath:
                    return False
            return True
        return False

    root = it.build_include_tree(incdir=pkg_opts.abs_inc_dir, filter=filter)
    # create_generators(INCTREE)
    print(root.file_tree_to_str())
    print(root.py_module_tree_to_str())
    print(root.py_imports_to_str())
