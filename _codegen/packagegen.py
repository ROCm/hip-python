# AMD_COPYRIGHT

import os
import enum
import textwrap

import clang.cindex

from . import cparser
from . import nodes

__author__ = "AMD_AUTHOR"

indent = " "*4

class HipPlatform(enum.IntEnum):
    AMD = 0
    NVIDIA = 1

    @property
    def cflags(self):
        return ["-D", f"__HIP_PLATFORM_{self.name}__"]

class PackageGenerator:
    """Generate Python/Cython packages for a HIP C interface.

    Generates Python/Cython packages for a HIP C interface
    based on a list of header file names and the name of
    a library to link.
    """

    def __init__(
        self,
        pkg_name: str,
        include_dir: str,
        headers: list,
        dll: str,
        node_filter: callable = lambda x: True,
        platform=HipPlatform.AMD,
    ):
        """Constructor.

        Args:
            pkg_name (str): Name of the package that should be generated. Influences filesnames.
            include_dir (str): Name of the main include dir.
            headers (list): Name of the header files. Absolute paths or w.r.t. to include dir.
            dll (str): Name of the DLL/shared object to link.
            node_filter (callable, optional): Filter for selecting the nodes to include in generated output. Defaults to lambdax:True.
            platform (HipPlatform, optional): The hip platform to use. Defaults to HipPlatform.AMD.
        """
        self.pkg_name = pkg_name
        if not len(headers):
            raise RuntimeError("Argument 'headers' must not be empty")
        self.include_dir = include_dir
        self.headers = headers
        self.platform = platform
        self.apis = {}
        for h in self.headers:
            print(h) # todo logging
            self.apis[h] = []
            if include_dir != None:
                abspath = os.path.join(include_dir, h)
            else:
                abspath = h
            cflags = platform.cflags + ["-I", f"{include_dir}"]
            parser = cparser.CParser(abspath, append_cflags=cflags)
            parser.parse()
            #print(parser.render_cursors())
            self.apis[h] = nodes.create_nodes(parser,node_filter)
        self._dll = dll
        # self._node_renamer = lambda name: name # unused, symbol renamer might be better name

    def apis_from_all_files(self):
        """Yields all APIs from all specified files."""
        for apis_per_file in self.apis.values():
            yield from apis_per_file

    def create_cython_bindings(self):
        """Returns the content of a Cython bindings file.

        Creates the content of a Cython bindings file.
        Contains Cython declarations per C declaration that we want to use
        plus helper types that have been introduced for nested enum/struct/union types.
        """
        global indent
        result = []
        for filename, nodelist in self.apis.items():
            is_extern = True
            prev_is_extern = False
            for node in nodelist:
                is_extern = (
                    not isinstance(node, nodes.UserTypeDeclBase)
                    or not node.is_helper_type
                )
                if is_extern:
                    if not prev_is_extern:
                        result.append(f'cdef extern from "{filename}":')
                    indent = " "*4
                else:
                    indent = ""
                result.append(
                    textwrap.indent(node.render_cython_binding(), indent)
                )
                prev_is_extern = is_extern
        return result

    def render_cython_bindings(self):
        """Returns the Cython bindings file content for the given headers."""
        result = self.create_cython_bindings()
        nl = "\n\n"
        return f"""\
{nl.join(result)}"""

    def create_python_interfaces(self,cython_bindings_module):
        """Returns the content of a Cython bindings file.

        Creates the content of a Cython bindings file.
        Contains Cython declarations per C declaration that we want to use
        plus helper types that have been introduced for nested enum/struct/union types.
        """
        global indent
        result = []
        for _, nodelist in self.apis.items():
            for node in nodelist:
                if isinstance(node,nodes.MacroDefinition):
                    result.append(f"from {cython_bindings_module} cimport {node.name}")
                elif isinstance(node,nodes.EnumDecl):
                    if not node.is_helper_type:
                        result.append(node.render_python_interface())
                elif isinstance(node,nodes.TypedefDecl):
                    if node.is_aliasing_enum_decl:
                        result.append(node.render_python_interface())
                elif isinstance(node,nodes.FunctionDecl):
                    result.append(node.render_python_interface())
        return result

    def render_python_interfaces(self,cython_bindings_module):
        """Returns the Python interface file content for the given headers."""
        result = self.create_python_interfaces(cython_bindings_module)
        nl = "\n\n"
        return f"""\
{nl.join(result)}"""