#!/usr/bin/env python3
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

import os
import glob
import argparse
from pathlib import Path
import re
import textwrap
import copy


def _collect_files(
    incdir: str,
    glob_expr=os.path.join("**", "*.h"),
    filter: callable = lambda fp: True,
):
    final_glob_expr = os.path.join(incdir, glob_expr)
    result = [fp for fp in glob.glob(final_glob_expr, recursive=True) if filter(fp)]
    return result


include_pattern = r'\s*#\s*include\s+["<](?P<file>[A-Za-z0-9_\-/]+(\.h\w*)?)[">]'


def _collect_includes_per_file(filepath: str):
    global include_pattern
    includes = []
    include_expr = re.compile(include_pattern)
    with open(filepath, "r") as infile:
        for line in infile.readlines():
            match = include_expr.match(line)
            if match:
                includes.append(match.group("file"))
    return includes


class Node:
    def __init__(self, parent, name: str):
        self.parent = parent
        self.name = name

    def _get_path_parts(self, py_path=False):
        result = [self.py_name if py_path else self.name]
        cur = self.parent
        while cur != None:
            if py_path:
                result.insert(0, cur.py_name)
            else:
                if isinstance(cur, Root):
                    result.insert(0, cur.incdir)
                else:
                    result.insert(0, cur.name)
            cur = cur.parent
        return result

    @property
    def level(self):
        return len(self._get_path_parts())

    @property
    def relpath(self):
        return os.path.join(*self._get_path_parts()[1:])

    @property
    def abspath(self):
        return os.path.join(*self._get_path_parts())

    def get_root(self):
        cur = self
        while not isinstance(cur, Root):
            cur = cur.parent
        return cur

    @property
    def py_global_name(self):
        return ".".join(self._get_path_parts(True))

    def file_tree_to_str(self):
        result = ""
        for node in self.walk():
            result += "  " * (node.level - self.level)
            if isinstance(node, Root):
                result += f"{node.incdir}/ ; {node.py_global_name:30}, {node.py_global_path:30}\n"
            elif isinstance(node, Directory):
                result += f"{node.name}/ ; {node.py_global_name:30}, {node.py_global_path:30}\n"
            else:
                result += f"{node.name:30}; {node.py_global_name:30}, {node.py_global_path:30}, {node.relpath:30}, {node.abspath}\n"
        return result

    def py_module_tree_to_str(self):
        result = ""
        for node in self.walk():
            result += "  " * (node.level - self.level)
            if isinstance(node, Directory):
                result += f"{node.py_name}.\n"
            else:
                result += f"{node.py_name}\n"
        return result

    def py_imports_to_str(self):
        result = ""
        for file in root.walk_files():
            result += f"{file.py_global_name}\n"
            for incfile in file.includes:
                assert isinstance(incfile, File)
                result += f"  -> {incfile.py_global_name}\n"
        return result


class File(Node):
    def __init__(self, parent: Node, name: str):
        Node.__init__(self, parent, name)
        self.includes = []
        self.codegen = None

    @staticmethod
    def DEFAULT_PY_MODULE_NAMER(name: str):
        result: str = name.split(os.extsep, maxsplit=1)[0]
        result = result.replace("-", "_")
        return result.lower()

    py_namer = DEFAULT_PY_MODULE_NAMER

    @property
    def py_name(self):
        return File.py_namer(self.name)

    @property
    def py_global_path(self):
        return self.py_global_name.replace(".", os.path.sep) + ".py"

    def walk(self):
        """Yields self."""
        yield self

    def cy_resolve_internal_dependencies(self):
        assert self.codegen != None, "no codegenerator set"
        for dep in self.includes:
            dep_global_name = dep.py_global_name
            dep_pkg_prefix = dep.parent.py_global_name
            dep_name = dep.py_name

            self.codegen.c_interface_decl_prolog += (
                f"from {dep_pkg_prefix}.c{dep_name} cimport *\n"
            )
            self.codegen.python_interface_decl_prolog += "\n"
            self.codegen.python_interface_impl_prolog += "\n"
            for entity in dep.codegen.backend.walk_entities_to_cimport(False):
                self.codegen.python_interface_decl_prolog += (
                    f"from {dep_global_name} cimport {entity.cython_global_name}\n"
                )
            for entity in dep.codegen.backend.walk_entities_to_import(False):
                self.codegen.python_interface_impl_prolog += (
                    f"from {dep_global_name} import {entity.cython_global_name}\n"
                )
            self.codegen.python_interface_decl_prolog += "\n"
            self.codegen.python_interface_impl_prolog += "\n"


class Directory(Node):
    def __init__(self, parent: Node, name: str, /, py_name=None):
        Node.__init__(self, parent, name)
        self._py_name = py_name
        self.children = []

    @staticmethod
    def DEFAULT_PY_MODULE_NAMER(name: str):
        result = name.replace("-", "_")
        result = result.replace(".", "_")
        result = result.replace("-", "_")
        return result.lower()

    py_namer = DEFAULT_PY_MODULE_NAMER

    @property
    def py_name(self):
        if self._py_name:
            return Directory.py_namer(self._py_name)
        else:
            return Directory.py_namer(self.name)

    @property
    def py_global_path(self):
        return self.py_global_name.replace(".", os.path.sep)

    def walk(self):
        """Preorder walk, yield itself before the children."""
        yield self
        for child in self.children:
            yield from child.walk()

    def get_or_add_directory(self, dirname: str):
        for child in self.children:
            if child.name == dirname:
                assert isinstance(child, Directory)
                return child
        new = Directory(self, dirname)
        self.children.append(new)
        return new

    def get_or_add_file(self, filename: str):
        for child in self.children:
            if child.name == filename:
                assert isinstance(child, File)
                return child
        new = File(self, filename)
        self.children.append(new)
        return new

    def walk_directories(self):
        for node in self.walk():
            if isinstance(node, Directory):
                if not isinstance(node, Root):
                    yield node

    def walk_files(self):
        for node in self.walk():
            if isinstance(node, File):
                yield node

    def py_render_init_file(self, license_text: str, author: str):
        result = license_text
        result += textwrap.dedent(
            f"""
            
            # This file has been autogenerated, do not modify.

            __author__ = "{author}"
            
            """
        )
        for module in sorted(
            set(file.py_name for file in self.children)
        ):  # removes duplicates due to splits etc.
            result += textwrap.dedent(
                f"""\
                try:
                    from . import {module}
                except ImportError:
                    pass # may have been excluded from build
                """
            )
        return result

    def py_split_at_char(self, sep="-"):
        """Inserts a fake directory at the separator.

        Example:
            Dir('llvm-c')

            becomes

            Dir('llvm-c',py_name:'llvm') -> Dir('',py_name:'c')

        Note:
            May introduce duplicate nodes when there is already a directory node with the same.
        """
        parts = self.name.split(sep, maxsplit=1)
        new = Directory(self, "", py_name=parts[1])
        new.children = self.children  # contains itself
        for child in new.children:
            child.parent = new
        self._py_name = parts[0]
        self.children = [new]

    def create_root(self, py_namespace: str, deepcopy=False):
        """Create a root node from this directory.

        Args:
            deepcopy(bool):
                Perform deep copy of children.
        """
        root = Root(self.abspath, py_namespace)
        if deepcopy:
            root.children = copy.deepcopy(self.children)
        else:
            root.children = self.children
        for child in self.children:
            child.parent = root
        return root


class Root(Directory):
    def __init__(self, incdir: str, py_namespace: str = ""):
        Directory.__init__(self, None, "<root>", py_name=None)
        self.incdir = incdir
        self._py_name = py_namespace

    @property
    def py_name(self):
        if self._py_name:
            return self._py_name
        else:
            return self.name

    def find_nodes(
        self,
        /,
        name: str = None,
        abspath: str = None,
        relpath: str = None,
        py_name: str = None,
        py_global_name: str = None,
    ) -> Node:
        """Find nodes with one of the specified properties."""
        assert (
            name != None
            or abspath != None
            or relpath != None
            or py_name != None
            or py_global_name != None
        )
        for node in self.walk():
            if not isinstance(node, Root):
                if name and node.name == name:
                    yield node
                elif relpath and node.relpath == relpath:
                    yield node
                elif abspath and node.abspath == abspath:
                    yield node
                elif py_name and node.py_name == py_name:
                    yield node
                elif py_global_name and node.py_global_name == py_global_name:
                    yield node

    def find_node(self, /, **kwargs) -> Node:
        return next(self.find_nodes(**kwargs), None)

    def py_create_package_dirs(self, output_dir: str, output_dir_parents: bool = False):
        """Creates package directory structure for python projects.

        Args:
            output_dir:
                The output directory root directory (must exist if `output_dir_parents` is `False`).
        """
        Path(output_dir).mkdir(
            parents=output_dir_parents, exist_ok=True
        )  # make package dir
        namespace_dir = os.path.join(output_dir, self.py_global_path)
        Path(namespace_dir).mkdir(parents=True, exist_ok=True)  # make
        for dir in self.walk_directories():
            pkg_dir = os.path.join(output_dir, dir.py_global_path)
            Path(pkg_dir).mkdir(parents=True, exist_ok=True)


def build_include_tree(
    incdir: str,
    glob_expr=os.path.join("**", "*.h"),
    filter: callable = lambda fp: True,
    py_namespace: str = "",
) -> Root:
    """Build include graph.

    Args:
        incdir:
            Top-level directory.
        glob_expr:
            A glob expression for pre-selecting the files to consider.
            Defaults to "**/*.h".
        filter:
            An expression for finely selecting the files to consider.
            Defaults to a function that returns always ``True``.
    """
    root = Root(incdir=incdir, py_namespace=py_namespace)
    for abspath in _collect_files(incdir, glob_expr, filter):
        relpath = abspath.replace(incdir, "").lstrip("/")
        parts = relpath.split(os.path.sep)
        # create/update directories
        cur = root
        for part in parts[:-1]:
            cur = cur.get_or_add_directory(part)
        cur = cur.get_or_add_file(parts[-1])
        assert isinstance(cur, File)
    # resolve dependencies
    for file in root.walk_files():
        for fp in _collect_includes_per_file(file.abspath):
            existing_node = root.find_node(name=fp, relpath=fp, abspath=fp)
            if existing_node:
                assert isinstance(
                    existing_node, File
                ), f"{abspath}: include: {fp}, found path: {existing_node.abspath}"
                file.includes.append(existing_node)
    return root


if __name__ == "__main__":

    def build_cli():
        parser = argparse.ArgumentParser("_include_graph.py")
        parser.add_argument("rocm_dir", type=str)  #
        return parser

    args = build_cli().parse_args()
    ROCM_DIR = args.rocm_dir

    # LLVM
    ROCM_LLVM_INC_DIR = os.path.join(ROCM_DIR, "llvm", "include")

    def filter(filepath: str):
        if "llvm-c" in filepath:
            return not filepath.endswith("ExternC.h")
        if filepath.endswith(os.path.join("llvm", "Config", "llvm-config.h")):
            return True
        return False

    root = build_include_tree(
        incdir=ROCM_LLVM_INC_DIR, py_namespace="rocm", filter=filter
    )
    root.find_node(name="llvm-c").py_split_at_char("-")
    # root = root.find_node(py_name="c").create_root("TEST")
    #print(root.file_tree_to_str())
    #print(root.py_module_tree_to_str())
    print(root.py_imports_to_str())

    # # HSA
    # HSA_INC_DIR = os.path.join(ROCM_DIR, "include", "hsa")
    # root = build_include_tree(incdir=HSA_INC_DIR, py_namespace="rocm.hsa")
    # print(root.file_tree_to_str())
    # print(root.py_module_tree_to_str())
    # print(root.py_imports_to_str())

    # # HIP
    # ROCM_DIR = os.path.join(ROCM_DIR, "include")
    # def filter(filepath: str):
    #     filename = os.path.basename(filepath)
    #     if filename.startswith("hip"):
    #         if "detail" in filepath:
    #             return False
    #         if "internal" in filepath:
    #             return False
    #         if "version" in filepath:
    #             return False
    #         return True
    #     elif filename in (
    #         "rccl.h",
    #         "roctx.h"
    #     ):
    #         return True
    #     return False

    # root = build_include_tree(incdir=ROCM_DIR, py_namespace="rocm",filter=filter)
    # print(root.file_tree_to_str())
    # print(root.py_module_tree_to_str())
    # print(root.py_imports_to_str())

    # AMD_COMGR
    # INC_DIR = os.path.join(ROCM_DIR, "include", "amd_comgr")
    # root = build_include_tree(incdir=INC_DIR, py_namespace="rocm.amd_comgr")
    root = build_include_tree(
        os.path.join(ROCM_DIR, "include"),
        py_namespace="rocm",  # namespace influences py package dirs
        glob_expr=os.path.join("**", "amd_comgr", "*.h"),
    ).find_node(py_global_name="rocm.amd_comgr").create_root( # rocm.amd_comgr: corresponds to /opt/rocm/include/amd_comgr
        py_namespace="rocm.amd_comgr" # py output will be generated into "<package_dir>/rocm/amd_comgr/<py_mod_or_pkg_path>"
    ) # make the node corresponding to "/opt/rocm/include/amd_comgr" the new root
    print(root.file_tree_to_str())
    print(root.py_module_tree_to_str())
    print(root.py_imports_to_str())
