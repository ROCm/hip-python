# AMD_COPYRIGHT

import os
import enum
import textwrap

from . import cparser
from . import nodes

__author__ = "AMD_AUTHOR"

indent = " "*4

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
        node_filter: callable = lambda node: True,
        macro_type: callable = lambda node: "int",
        param_intent: callable = lambda node: nodes.ParmDecl.Intent.IN,
        param_rank: callable = lambda node: nodes.ParmDecl.Rank.SCALAR,
        runtime_linking = True,
        cflags=[],
    ):
        """Constructor.

        Args:
            pkg_name (str): Name of the package that should be generated. Influences filesnames.
            include_dir (str): Name of the main include dir.
            headers (list): Name of the header files. Absolute paths or w.r.t. to include dir.
            dll (str): Name of the DLL/shared object to link.
            node_filter (callable, optional): Filter for selecting the nodes to include in generated output. Defaults to lambdax:True.
            macro_type (callable, optional): Assigns a type to a macro node.
            param_intent (callable, optional): Assigns the intent (in,out,inout) to a function parameter declaration node. 
            param_rank (callable, optional): Assigns the "rank" (scalar,buffer) to a function parameter declaration node.
            runtime_linking (bool, optional): If runtime-linking code should be generated, defaults to True.
            cflags (list(str), optional): Flags to pass to the C parser.
        """
        self.pkg_name = pkg_name
        if not len(headers):
            raise RuntimeError("Argument 'headers' must not be empty")
        self.include_dir = include_dir
        self.headers = headers
        self.macro_type = macro_type
        self.param_intent = param_intent
        self.param_rank = param_rank
        self.runtime_linking = runtime_linking
        self.cflags = cflags

        self.apis = {}
        for h in self.headers:
            print(h) # todo logging
            self.apis[h] = []
            if include_dir != None:
                abspath = os.path.join(include_dir, h)
            else:
                abspath = h
            cflags = self.cflags + ["-I", f"{include_dir}"]
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

    def create_cython_declaration_part(self):
        """Returns the content of a Cython bindings file.

        Creates the content of a Cython bindings file.
        Contains Cython declarations per C declaration that we want to use
        plus helper types that have been introduced for nested enum/struct/union types.
        """
        global indent
        curr_indent = ""
        result = []
        for filename, nodelist in self.apis.items():
            is_extern = True
            prev_is_extern = False
            for node in nodelist:
                if self.runtime_linking and isinstance(node,nodes.FunctionDecl):
                    result.append(node.render_cython_lazy_loader_decl())
                    curr_indent = ""
                    is_extern = False
                else:
                    contrib = node.render_cython_c_binding()
                    if contrib != None:
                        is_extern = (
                            not isinstance(node, nodes.ElaboratedTypeDeclBase)
                            or not node.is_helper_type
                        )
                        if is_extern:
                            if not prev_is_extern:
                                result.append(f'cdef extern from "{filename}":')
                            curr_indent = indent
                        else:
                            curr_indent = ""
                        result.append(
                            textwrap.indent(contrib, curr_indent)
                        )
                prev_is_extern = is_extern
        return result

    def create_cython_lazy_loader_decls(self):
        result = []
        for node in self.apis_from_all_files():
            if isinstance(node,nodes.FunctionDecl):
                result.append(node.render_cython_lazy_loader_decl())
        return result

    def create_cython_lazy_loader_defs(self):
        # TODO: Add compiler? switch to switch between MS and Linux loaders
        # TODO: Add compiler? switch to switch between HIP and CUDA backends?
        lib_handle = "_lib_handle"
        result = f"""\
cimport hip._util.posixloader as loader
cdef void* {lib_handle} = loader.open_library("{self._dll}")
""".splitlines(keepends=True)
        for node in self.apis_from_all_files():
            if isinstance(node,nodes.FunctionDecl):
                result.append(node.render_cython_lazy_loader_def(lib_handle=lib_handle))
        return result

    def render_cython_declaration_part(self):
        """Returns the Cython bindings file content for the given headers."""
        nl = "\n\n"
        return nl.join(self.create_cython_declaration_part())
    
    def render_cython_definition_part(self):
        """Returns the Cython bindings file content for the given headers."""
        nl = "\n\n"
        if self.runtime_linking:
            return nl.join(self.create_cython_lazy_loader_defs())
        else:
            return ""
    
    def create_python_interfaces(self,cython_c_bindings_module):
        """Renders Python interfaces in Cython.
        """
        result = []
        for node in self.apis_from_all_files():
            if isinstance(node,nodes.MacroDefinition):
                result.append(f"from {cython_c_bindings_module} cimport {node.name}")
            elif isinstance(node,nodes.EnumDecl):
                if not node.is_helper_type:
                    result.append(node.render_python_interface())
            elif isinstance(node,nodes.TypedefDecl):
                if node.is_aliasing_enum_decl:
                    result.append(node.render_python_interface())
            elif isinstance(node,nodes.FunctionDecl):
                result.append(node.render_python_interface())
        return result

    def render_python_interfaces(self,cython_c_bindings_module):
        """Returns the Python interface file content for the given headers."""
        result = self.create_python_interfaces(cython_c_bindings_module)
        nl = "\n\n"
        return f"""\
{nl.join(result)}"""