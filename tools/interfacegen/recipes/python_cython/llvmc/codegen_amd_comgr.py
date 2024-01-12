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

After pointing this setup script to an AMD COMGR installation,
it generates Cython files.
"""

__author__ = "Advanced Micro Devices, Inc."

import os

import logging

import interfacegen

interfacegen.enable_logging(logging.INFO)
_log = logging.getLogger("interfacegen")
from interfacegen.support import includetree as it

# configure codegen
# see: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#role-py-obj
interfacegen.cython.python_interface_pyobj_role_template = (
    r"`~.{name}`"  # ~: removes the qualifier from the link text
)

from interfacegen.cython import (
    CythonModuleGenerator,
    CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
)

from interfacegen.tree import (
    Node,
    Parm,
    Typed
)

from interfacegen.control import ParmIntent

from interfacegen.support import cython as support


def create_generator(
    global_module_name: str, header_file: str, **opts
) -> CythonModuleGenerator:
    global pkg_opts

    def ptr_rank(node: Typed):
        if node.is_pointer_to_char(degree=-1):
            return 1
        return 0

    def ptr_parm_intent(parm: Parm):
        func_name, parm_index = parm.parent.name, parm.index
        if func_name in (
            "amd_comgr_get_isa_count",
            "amd_comgr_get_isa_name", # char** isa_name
            # "amd_comgr_action_data_count",
            "amd_comgr_get_version",
        ): 
            return ParmIntent.OUT
        if (func_name, parm_index) in (
            ("amd_comgr_status_string",1),
            ("amd_comgr_action_info_get_option_list_count",1),
        ):
            return ParmIntent.OUT
        if parm.is_pointer_to_enum(degree=1):
            return ParmIntent.OUT
        return ParmIntent.INOUT

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
        **opts,
    )
    # generator.c_interface_decl_prolog += cython_c_preamble
    generator.python_interface_decl_prolog += f"cimport {pkg_opts.util_pkg}.types\n"

    return generator


def create_generators(root: it.Root):
    for node in root.walk_files():

        def node_filter(node: Node):
            return (
                node.global_name("_").startswith(
                    "amd_comgr"
                )  # use global_name because of anonymous funptrs
                or node.name.startswith("AMD_COMGR_INTERFACE_VERSION")
                or node.name == "code_object_info_s"
            )

        generator: CythonModuleGenerator = create_generator(
            node.py_global_name, os.path.join("amd_comgr",node.relpath), node_filter=node_filter
        )

        node.codegen = generator


if __name__ == "__main__":
    interfacegen.cython.FunctionMixin.python_interface_always_return_tuple = (
        False  # same for all modules
    )

    pkg_opts: support.RocmPackageOpts = support.create_rocm_package_opts_from_cli(
        project="AMD Code Object Manager (Comgr) Python",
        env_var_prefix="AMD_COMGR_PYTHON",
        libs_example="amd_comgr",
        package="rocm-llvm-python",
        rel_inc_dir="include",
        util_pkg="rocm.llvm._util",  # is part of rocm-llvm-python
        dll="libamd_comgr.so",
        author="Advanced Micro Devices, Inc.",
        email="hip-python.maintainer@amd.com",
    )

    INCTREE = it.build_include_tree(
        pkg_opts.abs_inc_dir,
        py_namespace="rocm",  # namespace influences py package dirs
        glob_expr=os.path.join("**", "amd_comgr", "*.h"),
    ).find_node(py_global_name="rocm.amd_comgr").create_root( # rocm.amd_comgr: corresponds to /opt/rocm/include/amd_comgr
        py_namespace="rocm.amd_comgr" # py output will be generated into "<package_dir>/rocm/amd_comgr/<py_mod_or_pkg_path>"
    ) # make the node corresponding to "/opt/rocm/include/amd_comgr" the new root
    create_generators(INCTREE)

    support.generate_all_rocm_package_files(
        INCTREE, pkg_opts, main_dir=os.path.join(pkg_opts.package_dir, "rocm","amd_comgr")
    )
