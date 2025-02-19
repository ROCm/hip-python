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

import logging
import os

import interfacegen
from interfacegen.cparser import TypeHandler
from interfacegen.fortran import FortranModuleGenerator
from interfacegen.support import fortran as support
from interfacegen.support import includetree as it
from interfacegen.support.recipes import hip as controls
from interfacegen.tree import (
    MacroDefinition,
)

# from interfacegen.support.recipes.hipify import parse_hipify_perl


interfacegen.enable_logging(logging.INFO)

_log = logging.getLogger("interfacegen")

TypeCategory = TypeHandler.TypeCategory

HIPFORT_FILE_EXT = "f"


# hip
def generate_hip_module_files():
    global pkg_opts
    global HIPFORT_FILE_EXT

    global HIP_VERSION_MAJOR
    global HIP_VERSION_MINOR
    global HIP_VERSION_PATCH
    global HIP_VERSION_GITHASH

    def renamer(name: str):
        return interfacegen.cython.DEFAULT_RENAMER(controls.hip.renamer(name))

    generator = FortranModuleGenerator(
        "hipfort",  # note: no "_hip" suffix used here
        HIPFORT_FILE_EXT,
        pkg_opts.abs_inc_dir,
        "hip/hip_runtime.h",
        renamer=renamer,
        node_filter=controls.hip.node_filter,
        ptr_parm_intent=controls.hip.ptr_parm_intent,
        ptr_rank=controls.hip.ptr_rank,
        macro_type=controls.hip.macro_type,
        raw_comment_cleaner=controls.hip.raw_comment_cleaner,
        cflags=pkg_opts.generator_args,
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
    return generator


# hiprtc
def generate_hiprtc_module_files():
    global pkg_opts
    global HIPFORT_FILE_EXT

    generator = FortranModuleGenerator(
        "hipfort_hiprtc",
        HIPFORT_FILE_EXT,
        pkg_opts.abs_inc_dir,
        "hip/hiprtc.h",
        # node_init=hiprtc_node_init,
        node_filter=controls.hiprtc.node_filter,
        ptr_parm_intent=controls.hiprtc.ptr_parm_intent,
        ptr_rank=controls.hiprtc.ptr_rank,
        cflags=pkg_opts.generator_args,
    )

    return generator


# hipblas
def generate_hipblas_module_files():
    global pkg_opts

    generator = FortranModuleGenerator(
        "hipfort_hipblas",
        HIPFORT_FILE_EXT,
        pkg_opts.abs_inc_dir,
        "hipblas/hipblas.h",
        node_filter=controls.hipblas.node_filter,
        ptr_parm_intent=controls.hipblas.ptr_parm_intent,
        ptr_rank=controls.hipblas.ptr_rank,
        raw_comment_cleaner=controls.hipblas.raw_comment_cleaner,
        cflags=pkg_opts.generator_args,
    )
    return generator


# hipsolver
def generate_hipsolver_module_files():
    global pkg_opts

    generator = FortranModuleGenerator(
        "hipfort_hipsolver",
        HIPFORT_FILE_EXT,
        pkg_opts.abs_inc_dir,
        "hipsolver/hipsolver.h",
        node_filter=controls.hipsolver.node_filter,
        ptr_parm_intent=controls.hipsolver.ptr_parm_intent,
        ptr_rank=controls.hipsolver.ptr_rank,
        raw_comment_cleaner=controls.hipsolver.raw_comment_cleaner,
        cflags=pkg_opts.generator_args,
    )
    return generator


# rccl
def generate_rccl_module_files():
    global pkg_opts

    generator = FortranModuleGenerator(
        "hipfort_rccl",
        HIPFORT_FILE_EXT,
        pkg_opts.abs_inc_dir,
        "rccl/rccl.h",
        node_filter=controls.rccl.node_filter,
        macro_type=controls.rccl.macro_type,
        ptr_parm_intent=controls.rccl.ptr_parm_intent,
        ptr_rank=controls.rccl.ptr_rank,
        cflags=pkg_opts.generator_args,
    )
    return generator


# hiprand
def generate_hiprand_module_files():
    global pkg_opts

    generator = FortranModuleGenerator(
        "hipfort_hiprand",
        HIPFORT_FILE_EXT,
        pkg_opts.abs_inc_dir,
        "hiprand/hiprand.h",
        node_filter=controls.hiprand.node_filter,
        macro_type=controls.hiprand.macro_type,
        ptr_parm_intent=controls.hiprand.ptr_parm_intent,
        ptr_rank=controls.hiprand.ptr_rank,
        cflags=pkg_opts.generator_args,
    )
    return generator


# hipfft
def generate_hipfft_module_files():
    global pkg_opts

    generator = FortranModuleGenerator(
        "hipfort_hipfft",
        HIPFORT_FILE_EXT,
        pkg_opts.abs_inc_dir,
        "hipfft/hipfft.h",
        node_filter=controls.hipfft.node_filter,
        macro_type=controls.hipfft.macro_type,
        ptr_parm_intent=controls.hipfft.ptr_parm_intent,
        ptr_rank=controls.hipfft.ptr_rank,
        cflags=pkg_opts.generator_args,
    )
    return generator


# hipsparse
def generate_hipsparse_module_files():
    global pkg_opts

    generator = FortranModuleGenerator(
        "hipfort_hipsparse",
        HIPFORT_FILE_EXT,
        pkg_opts.abs_inc_dir,
        "hipsparse/hipsparse.h",
        node_filter=controls.hipsparse.node_filter,
        macro_type=controls.hipsparse.macro_type,
        ptr_parm_intent=controls.hipsparse.ptr_parm_intent,
        ptr_rank=controls.hipsparse.ptr_rank,
        raw_comment_cleaner=controls.hipsparse.raw_comment_cleaner,
        cflags=pkg_opts.generator_args,
    )
    return generator


# roctx
def generate_roctx_module_files():
    global pkg_opts

    generator = FortranModuleGenerator(
        "hipfort_roctx",
        HIPFORT_FILE_EXT,
        pkg_opts.abs_inc_dir,
        "roctracer/roctx.h",
        node_filter=controls.roctx.node_filter,
        macro_type=controls.roctx.macro_type,
        ptr_parm_intent=controls.roctx.ptr_parm_intent,
        ptr_rank=controls.roctx.ptr_rank,
        cflags=pkg_opts.generator_args,
    )
    return generator


SPECIALIZED_GENERATORS = dict(
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


def resolve_dependencies(node: it.File):
    assert node.codegen is not None, "no codegenerator set"
    codegen: FortranModuleGenerator = node.codegen
    for dep in node.includes:
        assert isinstance(dep, it.File)
        codegen.backend.module_preamble += f"use hipfort_{dep.name}\n"
        codegen.backend.function_preamble += f"use hipfort_{dep.name}\n"
    # Reinitialize the nodes with the new information
    node.codegen.backend.initialize_nodes()


# roctx
def create_default_generator(node: it.File):
    global pkg_opts

    generator = FortranModuleGenerator(
        module_name=f"hipfort_{node.basename_no_ext}",
        module_ext=HIPFORT_FILE_EXT,
        include_dir=pkg_opts.abs_inc_dir,
        header=node.relpath,
        cflags=pkg_opts.generator_args,
    )
    return generator


def create_generators(root: it.Root):
    global SPECIALIZED_GENERATORS
    for node in root.walk_files():
        key = node.basename_no_ext.replace("hip_runtime", "hip")
        if key in SPECIALIZED_GENERATORS:
            node.codegen = SPECIALIZED_GENERATORS[key]()
        else:
            node.codegen = create_default_generator(node)

    for node in root.walk_files():
        resolve_dependencies(node)


if __name__ == "__main__":
    HIP_2_CUDA = None

    pkg_opts: support.RocmPackageOpts = (
        support.create_rocm_package_opts_from_cli(
            project="HIPFORT",
            env_var_prefix="HIPFORT_",
            libs_example="hip,hiprtc",
            package="hipfort",
            rel_inc_dir="include",
            author="Advanced Micro Devices, Inc.",
            email="hipfort.maintainer@amd.com",
        )
    )

    def filter(filepath: str):
        _log.debug(f"touch header file {filepath}")
        filename = os.path.basename(filepath)
        # print(filename)
        if filename in ("rccl.h", "miopen.h", "hip_runtime.h"):
            _log.info(f"accept header file {filepath}")
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
            _log.info(f"accept header file {filepath}")
            return True
        return False

    root = it.build_include_tree(incdir=pkg_opts.abs_inc_dir, filter=filter)
    # patch some of the includes
    root.find_node(name="hipsolver.h").includes.append(
        root.find_node(name="hipblas.h")
    )
    root.find_node(name="rocsolver.h").includes.append(
        root.find_node(name="rocblas.h")
    )
    # create_generators(INCTREE)
    # _log.info(root.file_tree_to_str())
    _log.info(
        "\nBEGIN INCLUDES\n"
        + root.includes_to_str().rstrip()
        + "\nEND INCLUDES"
    )

    create_generators(root)

    for node in root.walk_files():
        node.codegen.write_module_files(pkg_opts.output_dir)
