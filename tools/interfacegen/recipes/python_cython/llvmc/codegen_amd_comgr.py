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

import textwrap

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

from interfacegen.tree import Node, Parm, Typed

from interfacegen.support.recipes.control import ParmIntent

from interfacegen.support import cython as support


def create_generator(
    global_module_name: str, header_file: str
) -> CythonModuleGenerator:
    global pkg_opts

    def ptr_rank(node: Typed):
        if node.is_pointer_to_char(degree=-1):
            return 1
        return 0

    def ptr_parm_intent(parm: Parm):
        func_name, parm_index = parm.parent.name, parm.parm_index
        if (func_name, parm_index) in (("amd_comgr_action_info_set_option_list", 1),):
            return ParmIntent.IN
        if func_name in (
            "amd_comgr_get_isa_count",
            "amd_comgr_get_version",
            "amd_comgr_create_data_set",
            "amd_comgr_create_action_info",
        ):
            return ParmIntent.OUT
        if (func_name, parm_index) in (
            ("comgr.amd_comgr_create_data", 1),
            ("amd_comgr_status_string", 1),
            ("amd_comgr_action_info_get_option_list_count", 1),
            ("amd_comgr_create_data", 1),
            ("amd_comgr_get_data_kind", 1),
            ("amd_comgr_action_data_count", 2),
            ("amd_comgr_action_data_get_data", 3),
            ("amd_comgr_get_isa_name", 1),
            ("amd_comgr_get_isa_metadata", 1),
            ("amd_comgr_get_data_metadata", 1),
            ("amd_comgr_get_metadata_kind", 1),
            ("amd_comgr_get_metadata_map_size", 1),
            ("amd_comgr_metadata_lookup", 2),
            ("amd_comgr_get_metadata_list_size", 1),
            ("amd_comgr_index_list_metadata", 2),
            ("amd_comgr_create_symbolizer_info", 2),
            ("amd_comgr_create_disassembly_info", 4),
        ):
            return ParmIntent.OUT
        # INOUT:
        # amd_comgr_get_data
        # amd_comgr_get_data_name
        # amd_comgr_get_isa_name
        # amd_comgr_get_metadata_string
        # amd_comgr_iterate_map_metadata
        return ParmIntent.INOUT

    def ptr_complicated_type_handler(node: Node):
        if isinstance(node, Parm):
            func_name, parm_index = node.parent.name, node.parm_index
            if (func_name, parm_index) in (
                ("amd_comgr_action_info_set_option_list", 1),
            ):
                return f"{pkg_opts.util_types_prefix}ListOfBytes"
        return CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(pkg_opts.util_types_prefix)(
            node
        )

    def node_filter(node: Node):
        return (
            node.global_name("_").startswith(
                "amd_comgr"
            )  # use global_name because of anonymous funptrs
            or node.name.startswith("AMD_COMGR_INTERFACE_VERSION")
            or node.name == "code_object_info_s"
        )

    def node_init(node: Node):
        # node modifications
        if isinstance(node, interfacegen.tree.Function):
            if not node.is_enum and node.name.startswith("amd_comgr"):
                # hip routines without hipError_t return status
                # we force them to not throw exceptions
                node.error_return_value_lazy_loader = None
                node.modifiers_lazy_loader = " noexcept nogil"
                # we force them to have always return hipSuccess as first return value
                node.prepend_python_return_value(
                    "amd_comgr_status_s.AMD_COMGR_STATUS_SUCCESS",
                    "amd_comgr_status_s",
                    "Always returns `~.amd_comgr_status_s.AMD_COMGR_STATUS_SUCCESS`.",
                )

    generator = CythonModuleGenerator(
        global_module_name,
        pkg_opts.abs_inc_dir,
        header_file,
        runtime_linking=pkg_opts.runtime_linking,
        util_pkg=pkg_opts.util_pkg,
        dll=pkg_opts.dll,
        cflags=pkg_opts.generator_args,
        node_filter=node_filter,
        node_init=node_init,
        ptr_rank=ptr_rank,
        ptr_parm_intent=ptr_parm_intent,
        ptr_complicated_type_handler=ptr_complicated_type_handler,
        modifiers_lazy_loader=" except? AMD_COMGR_STATUS_ERROR nogil",
        error_return_value_lazy_loader="AMD_COMGR_STATUS_ERROR",
    )
    # generator.c_interface_decl_prolog += cython_c_preamble
    generator.python_interface_decl_prolog += f"cimport {pkg_opts.util_pkg}.types\n"

    return generator


def create_generators(root: it.Root):
    for node in root.walk_files():
        generator: CythonModuleGenerator = create_generator(
            node.py_global_name,
            os.path.join("amd_comgr", node.relpath),
        )
        node.codegen = generator


if __name__ == "__main__":
    interfacegen.cython.Function.python_interface_always_return_tuple = True

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

    INCTREE = (
        it.build_include_tree(
            pkg_opts.abs_inc_dir,
            py_namespace="rocm",  # namespace influences py package dirs
            glob_expr=os.path.join("**", "amd_comgr", "*.h"),
        )
        .find_node(py_global_name="rocm.amd_comgr")
        .create_root(  # rocm.amd_comgr: corresponds to /opt/rocm/include/amd_comgr
            py_namespace="rocm.amd_comgr"  # py output will be generated into "<package_dir>/rocm/amd_comgr/<py_mod_or_pkg_path>"
        )
    )  # make the node corresponding to "/opt/rocm/include/amd_comgr" the new root
    create_generators(INCTREE)

    support.generate_all_rocm_package_files(
        INCTREE,
        pkg_opts,
        main_dir=os.path.join(pkg_opts.package_dir, "rocm", "amd_comgr"),
        main_child_modules=[],
        main_init_file_epilog=textwrap.dedent(
            """
        from . import amd_comgr
        from . import amd_comgr_pyext
        setattr(amd_comgr,"ext",amd_comgr_pyext)
        """
        ),
        year_start="2023",
    )
