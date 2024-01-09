#!/usr/bin/env python3
import os
import glob
import argparse
import re
import textwrap


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
    py_namespace = ""

    def __init__(self, parent, name: str):
        self.parent = parent
        self.name = name

    def _get_path_parts(self, py_path=False):
        result = [self.py_module_name if py_path else self.name]
        cur = self.parent
        while cur != None:
            if py_path:
                result.insert(0, cur.py_module_name)
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

    @property
    def py_global_module_name(self):
        parts = self._get_path_parts(True)[1:]
        parts[-1] = self.py_module_name
        if len(Node.py_namespace):
            parts.insert(0, Node.py_namespace.rstrip("."))
        return ".".join(parts)

    def file_tree_to_str(self):
        result = ""
        for node in self.walk():
            result += "  " * (node.level - self.level)
            if isinstance(node, Root):
                result += f"{node.incdir}/\n"
            elif isinstance(node, Directory):
                result += f"{node.name}/\n"
            else:
                result += f"{node.name:30}, {node.py_global_module_name:30}, {node.relpath:30}, {node.abspath}\n"
        return result

    def py_module_tree_to_str(self):
        result = ""
        for node in self.walk():
            result += "  " * (node.level - self.level - 1)
            if isinstance(node, Directory):
                result += f"{node.py_module_name}.\n"
            else:
                result += f"{node.py_module_name}\n"
        return result

    def py_imports_to_str(self):
        result = ""
        for file in root.walk_files():
            result += f"{file.py_global_module_name}\n"
            for incfile in file.includes:
                assert isinstance(incfile, File)
                result += f"  -> {incfile.py_global_module_name}\n"
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

    py_module_namer = DEFAULT_PY_MODULE_NAMER

    @property
    def py_module_name(self):
        return File.py_module_namer(self.name)

    def walk(self):
        """Yields self."""
        yield self


class Directory(Node):
    def __init__(self, parent: Node, name: str, /, py_name=None):
        Node.__init__(self, parent, name)
        self.py_name = py_name
        self.children = []

    @staticmethod
    def DEFAULT_PY_MODULE_NAMER(name: str):
        result = name.replace("-", "_")
        result = result.replace(".", "_")
        result = result.replace("-", "_")
        return result.lower()

    py_module_namer = DEFAULT_PY_MODULE_NAMER

    @property
    def py_module_name(self):
        if self.py_name:
            return Directory.py_module_namer(self.py_name)
        else:
            return Directory.py_module_namer(self.name)

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

    def render_py_init_file(self, license_text: str, email: str):
        result = textwrap.dedent(
            f"""\
            {license_text}

            # This file has been autogenerated, do not modify.

            __author__ = "Advanced Micro Devices, Inc. <{email}>"
            
            import os
            """
        )
        for file in self.children:
            result += textwrap.dedent(
                f"""\
                try:
                    from . import {file.py_module_name}
                except ImportError:
                    pass # may have been excluded from build
                """
            )
        return result

    def py_init_file_abspath(self):
        return os.path.join(self.abspath, "__init__.py")

    def py_init_file_relpath(self):
        return os.path.join(self.relpath, "__init__.py")

    def split(self, sep="-"):
        """Inserts a fake directory at the separator.

        Example:

        Dir('llvm-c')

        becomes

        Dir('llvm-c',py_name:'llvm') -> Dir('',py_name:'c')
        """
        parts = self.name.split(sep, maxsplit=1)
        new = Directory(self, "", py_name=parts[1])
        new.children = self.children
        for child in new.children:
            child.parent = new
        self.py_name = parts[0]
        self.children = [new]


class Root(Directory):
    def __init__(self, incdir: str):
        Directory.__init__(self, None, "<root>", py_name=None)
        self.incdir = incdir
        self.py_name = Node.py_namespace

    @property
    def py_module_name(self):
        if self.py_name:
            return self.py_name
        else:
            return self.name

    def find_node(
        self,
        /,
        name: str = None,
        abspath: str = None,
        relpath: str = None,
        py_global_module_name: str = None,
    ):
        """Find a node with one of the specified properties."""
        assert (
            name != None
            or abspath != None
            or relpath != None
            or py_global_module_name != None
        )
        for node in self.walk():
            if not isinstance(node, Root):
                if name and node.name == name:
                    return node
                elif relpath and node.relpath == relpath:
                    return node
                elif abspath and node.abspath == abspath:
                    return node
                elif (
                    py_global_module_name
                    and node.py_global_module_name == py_global_module_name
                ):
                    return node
        return None

    def walk_directories(self):
        for node in self.walk():
            if isinstance(node, Directory):
                if not isinstance(node, Root):
                    yield node

    def walk_files(self):
        for node in self.walk():
            if isinstance(node, File):
                yield node


def build_include_graph(
    incdir: str,
    glob_expr=os.path.join("**", "*.h"),
    filter: callable = lambda fp: True,
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
    root = Root(incdir)
    for abspath in _collect_files(incdir, glob_expr, filter):
        relpath = abspath.replace(incdir, "").lstrip("/")
        parts = relpath.split(os.path.sep)
        # create/update directories
        cur = root
        for part in parts[:-1]:
            cur = cur.get_or_add_directory(part)
        cur = cur.get_or_add_file(parts[-1])
        assert isinstance(cur, File)
        for fp in _collect_includes_per_file(abspath):
            existing_node = root.find_node(name=fp, relpath=fp, abspath=fp)
            if existing_node:
                assert isinstance(existing_node, File)
                cur.includes.append(existing_node)
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

    Node.py_namespace = "rocm"
    root = build_include_graph(ROCM_LLVM_INC_DIR, filter=filter)
    root.find_node(name="llvm-c").split("-")
    print(root.file_tree_to_str())
    print(root.py_module_tree_to_str())
    print(root.py_imports_to_str())

    # HSA
    HSA_INC_DIR = os.path.join(ROCM_DIR, "include", "hsa")
    Node.py_namespace = "rocm.hsa"
    root = build_include_graph(HSA_INC_DIR)
    # root.find_node(name="llvm-c").split("-")
    print(root.file_tree_to_str())
    print(root.py_module_tree_to_str())
    print(root.py_imports_to_str())
