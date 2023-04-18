#AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import sys
import os
import keyword
import textwrap

import clang.cindex

from . import cparser
from . import control

indent = " "*4

funptr_name_template = "{name}_funptr"

restricted_names =  keyword.kwlist + [
     "cdef",
     "cpdef",    # TODO extend
     ]

def DEFAULT_RENAMER(name): # backend-specific
     result = name
     while result in restricted_names:
         result += "_"
     return result

def DEFAULT_MACRO_TYPE(node): # backend-specific
    return "int"

# Mixins
class CythonMixin:

    def __init__(self): # Will not be called, attribs specified for type hinting
        self.renamer = DEFAULT_RENAMER
        self.sep = "_"
    
    def _cython_and_c_name(self,orig_name: str):
        """Returns `<orig_name> "<renamed>"` if `renamer` had an effect, else returns `orig_name`.

        Note:
            For more details see https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html#resolving-naming-conflicts-c-name-specifications
        """
        renamed = self.renamer(orig_name)
        if orig_name == renamed:
            return orig_name
        else:
            return f'{renamed} "{orig_name}"'

class MacroDefinitionMixin(CythonMixin):

    def __init__(self):
        CythonMixin.__init__(self)
        self.macro_type = DEFAULT_MACRO_TYPE

    def render_cython_c_binding(self):
        from . import tree
        assert isinstance(self,tree.MacroDefinition)
        return f"cdef {self.macro_type(self)} {self._cython_and_c_name(self.name)}"

class FieldMixin(CythonMixin):

    def __init__(self):
        CythonMixin.__init__(self)
        self.ptr_rank = control.DEFAULT_PTR_RANK
    
    def cython_repr(self):
        from . import tree
        assert isinstance(self,tree.Field)
        typename = self.global_typename(self.sep,self.renamer)
        name = self._cython_and_c_name(self.name)
        return f"{typename} {name}"

class RecordMixin(CythonMixin):
    
    @property
    def c_record_kind(self) -> str:
        if self.cursor.kind == clang.cindex.CursorKind.STRUCT_DECL:
            return "struct"
        else:
            return "union"

    def _render_cython_c_binding_head(self) -> str:
        from . import tree
        assert isinstance(self,tree.Record)
        name = self._cython_and_c_name(self.global_name(self.sep))
        cython_def_kind = "ctypedef" if self._from_typedef_with_anon_child else "cdef"
        
        return f"{cython_def_kind} {self.c_record_kind} {name}:\n"
    
    def render_cython_c_binding(self):
        """Render Cython binding for this struct/union declaration.

        Renders a Cython binding for this struct/union declaration, does
        not render declarations for nested types.

        Returns:
            str: Cython C-binding representation of this struct declaration.
        """
        from . import tree
        assert isinstance(self,tree.Record)
        global indent
        result = self._render_cython_c_binding_head()
        fields = list(self.fields)
        if len(fields):
            result += textwrap.indent(
                "\n".join([field.cython_repr() for field in fields]), indent
            )
        else:
            result += f"{indent}pass"
        return result

class StructMixin(RecordMixin):
    pass

class UnionMixin(RecordMixin):
    pass

class EnumMixin(CythonMixin):
    
    def _render_cython_enums(self):
        """Yields the enum constants' names."""
        from . import tree
        assert isinstance(self,tree.Enum)
        for child_cursor in self.cursor.get_children():
            name = self._cython_and_c_name(child_cursor.spelling)
            yield name

    def _render_cython_c_binding_head(self) -> str:
        from . import tree
        assert isinstance(self,tree.Enum)
        cython_def_kind = "ctypedef" if self._from_typedef_with_anon_child else "cdef"
        name = self._cython_and_c_name(self.global_name(self.sep))
        return f"{cython_def_kind} enum{'' if self.is_cursor_anonymous else ' '+name}:\n"

    def render_cython_c_binding(self):
        from . import tree
        # assert isinstance(self,tree.Enum)
        global indent
        return self._render_cython_c_binding_head() + textwrap.indent(
            "\n".join(self._render_cython_enums()), indent
        )
    
    def _render_python_enums(self,prefix: str):
        from . import tree
        # assert isinstance(self,tree.Enum)
        """Yields the enum constants' names."""
        for child_cursor in self.cursor.get_children():
            name = self.renamer(child_cursor.spelling)
            yield f"{name} = {prefix}{name}"
    
    def render_python_interface(self,prefix: str,
                                renamer: callable = DEFAULT_RENAMER):
        """Renders an enum.IntEnum class.

        Note:
            Does not create an enum.IntEnum class but only exposes the enum constants 
            from the Cython package corresponding to the prefix if the 
            Enum is anonymous.
        """
        from . import tree
        assert isinstance(self,tree.Enum)
        global indent
        if self.is_cursor_anonymous:
            return "\n".join(self._render_python_enums(prefix))
        else:
            name = self._cython_and_c_name(self.global_name(self.sep))
            return f"class {name}(enum.IntEnum):\n" + textwrap.indent(
                "\n".join(self._render_python_enums(prefix)), indent
            )

# TODO check if we need a differen treatment for the Cython part
# class NestedEnumMixin(EnumMixin):
# pass 
# def _render_cython_enums(self):
#    """Yields the enum constants' names."""
#    assert isinstance(self,tree.NestedEnum)
#    for child_cursor in self.cursor.get_children():
#        yield f"{child_cursor.spelling} = {child_cursor.enum_value}"

class TypedefMixin(CythonMixin):
    
    def render_cython_c_binding(self):
        from . import tree
        assert isinstance(self,tree.Typedef)
        """Returns a Cython binding for this Typedef.
        """
        underlying_type_name = self.global_typename(self.sep,self.renamer)
        name = self._cython_and_c_name(self.name)
        
        return f"ctypedef {underlying_type_name} {name}"

class FunctionPointerMixin(CythonMixin):
    
    def render_cython_c_binding(self):
        """Returns a Cython binding for this Typedef.
        """
        from . import tree
        assert isinstance(self,tree.FunctionPointer)
        parm_types = ",".join(self.global_parm_types(self.sep,self.renamer))
        underlying_type_name = self.renamer(self.canonical_result_typename)
        typename = self.renamer(self.global_name(self.sep)) # might be AnonymousFunctionPointer
        return f"ctypedef {underlying_type_name} (*{typename}) ({parm_types})"

class TypedefedFunctionPointerMixin(FunctionPointerMixin):
    pass

class AnonymousFunctionPointerMixin(FunctionPointerMixin):
    pass

class ParmMixin(CythonMixin):

    def __init__(self):
        CythonMixin.__init__(self)
        self.ptr_rank = control.DEFAULT_PTR_RANK
        self.ptr_intent = control.DEFAULT_PTR_PARM_INTENT

    def cython_repr(self):
        from . import tree
        assert isinstance(self,tree.Parm)
        typename = self.global_typename(self.sep,self.renamer)
        name = self.renamer(self.name)
        return f"{typename} {name}"
    
class FunctionMixin(CythonMixin):
    
    def _raw_comment_as_python_comment(self):
        from . import tree
        assert isinstance(self,tree.Function)
        if self.raw_comment != None:
            comment = self._raw_comment_stripped()
            return "".join(["# " + l for l in comment.splitlines(keepends=True)])
        else:
            return ""
        
    def _raw_comment_as_docstring(self):
        from . import tree
        assert isinstance(self,tree.Function)
        return f'"""{"".join(self._raw_comment_stripped()).rstrip()}\n"""'

    def render_cython_c_binding(self, modifiers="nogil"):
        from . import tree
        assert isinstance(self,tree.Function)
        typename = self.global_typename(self.sep,self.renamer)
        name = self.renamer(self.name)
        parm_decls = ",".join([arg.cython_repr() for arg in self.parms])
        return f"""\
{self._raw_comment_as_python_comment().rstrip()}
{typename} {name}({parm_decls}) {modifiers}
"""

    def cython_funptr_name(self):
        from . import tree
        assert isinstance(self,tree.Function)
        name = self.renamer(self.name)
        return funptr_name_template.format(name=name)

    def render_cython_lazy_loader_decl(self,
                                       modifiers="nogil"):
        from . import tree
        assert isinstance(self,tree.Function)
        parm_decls = ",".join([parm.cython_repr() for parm in self.parms])
        typename = self.global_typename(self.sep,self.renamer)
        name = self._cython_and_c_name(self.name)
        return f"""\
{self._raw_comment_as_python_comment().rstrip()}
cdef {typename} {name}({parm_decls}) {modifiers}
    """

    def render_cython_lazy_loader_def(self,
                                      lib_handle: str="__lib_handle", 
                                      modifiers="nogil"):
        from . import tree
        assert isinstance(self,tree.Function)
        funptr_name = {self.cython_funptr_name()}
        parm_types = ",".join(self.global_parm_types(self.sep,self.renamer))
        parm_names = ",".join(self.parm_names(self.renamer))
        typename = self.global_typename(self.sep,self.renamer)
        return f"""\
cdef void* {funptr_name}
{self.render_cython_lazy_loader_decl(modifiers).strip()}:
global {lib_handle}
global {funptr_name}
if {funptr_name} == NULL:
    with gil:
        {funptr_name} = loader.load_symbol({lib_handle}, "{self.name}")
return (<{typename} (*)({parm_types})> {funptr_name})({parm_names})
"""

# TODO render_python_interfaces

class CythonBackend:

    def from_libclang_translation_unit(
        translation_unit: clang.cindex.TranslationUnit,
        filename: str,
        node_filter: callable = control.DEFAULT_NODE_FILTER,
        macro_type: callable = DEFAULT_MACRO_TYPE,
        ptr_parm_intent: callable = control.DEFAULT_PTR_PARM_INTENT,
        ptr_rank: callable = control.DEFAULT_PTR_RANK,
        renamer: callable = DEFAULT_RENAMER,
        warnings: control.Warnings = control.Warnings.IGNORE
    ):
        from . import tree
        root = tree.from_libclang_translation_unit(translation_unit,warnings)
        return CythonBackend(root,
                             filename,
                             node_filter,
                             macro_type, 
                             ptr_parm_intent,
                             ptr_rank,
                             renamer)

    def __init__(self,root,
                 filename: str,
                 node_filter: callable = control.DEFAULT_NODE_FILTER,
                 macro_type: callable = DEFAULT_MACRO_TYPE,
                 ptr_parm_intent: callable = control.DEFAULT_PTR_PARM_INTENT,
                 ptr_rank: callable = control.DEFAULT_PTR_RANK,
                 renamer: callable = DEFAULT_RENAMER):
        """
        Note:
            Argument 'root' has no type hint in order to prevent a circular inclusion error.
            Instead an assertion is used in the body that checks if the type is `tree.Root`.
        """
        from . import tree
        assert isinstance(root,tree.Root)
        self.root = root
        self.filename = filename
        self.node_filter = node_filter
        self.macro_type = macro_type
        self.ptr_parm_intent = ptr_parm_intent # TODO use for FunctionMixin.render_python_interface
        self.ptr_rank = ptr_rank     # TODO use for FunctionMixin.render_python_interface
        self.renamer = renamer

    def _walk_filtered_nodes(self):
        """Walks the filtered nodes in post-order and sets the renamer of each node.

        Note:
            Post-order yields nested struct/union/enum declarations before their
            parent.
        """
        for node in self.root.walk(postorder=True):
            if isinstance(node,CythonMixin):
                if self.node_filter(node):
                    # set defaults
                    setattr(node,"sep","_")
                    # set user callbacks
                    setattr(node,"renamer",self.renamer)
                    if isinstance(node,MacroDefinitionMixin):
                        setattr(node,"macro_type",self.macro_type)
                    elif isinstance(node,(FieldMixin)):
                        setattr(node,"ptr_rank",self.ptr_rank)
                    elif isinstance(node,(ParmMixin)):
                        setattr(node,"ptr_rank",self.ptr_rank)
                        setattr(node,"ptr_intent",self.ptr_parm_intent)
                    # yield relevant nodes
                    if not isinstance(node,(FieldMixin,ParmMixin)):
                        yield node

    def create_cython_declaration_part(self,runtime_linking: bool = False):
        """Returns the content of a Cython bindings file.

        Creates the content of a Cython bindings file.
        Contains Cython declarations per C declaration 
        plus helper types that have been introduced for nested enum/struct/union types.
        """
        global indent
        curr_indent = ""
        result = []
        
        prev_is_extern = False
        for node in self._walk_filtered_nodes():
            is_extern = True
            #prev_is_extern = False
            if runtime_linking and isinstance(node,FunctionMixin):
                result.append(node.render_cython_lazy_loader_decl())
                curr_indent = ""
                #is_extern = False
            else:
                is_extern = True # check simple way first, old: not isinstance(node,tree.Nested) or not node.is_anonymous
                if is_extern:
                    if not prev_is_extern:
                        result.append(f'cdef extern from "{self.filename}":')
                    curr_indent = indent
                else:
                    curr_indent = ""
                contrib = node.render_cython_c_binding()
                result.append(
                    textwrap.indent(contrib, curr_indent)
                )
                prev_is_extern = is_extern
        return result
    
    def create_cython_lazy_loader_decls(self):
        result = []
        for node in self._walk_filtered_nodes():
            if isinstance(node,FunctionMixin):
                result.append(node.render_cython_lazy_loader_decl(self.renamer))
        return result

    def create_cython_lazy_loader_defs(self,dll: str):
        # TODO: Add compiler? switch to switch between MS and Linux loaders
        # TODO: Add compiler? switch to switch between HIP and CUDA backends?
          # Should be possible to implement this via the renamer and generating multiple modules
        lib_handle = "_lib_handle"
        result = f"""\
cimport hip._util.posixloader as loader
cdef void* {lib_handle} = loader.open_library("{dll}")
""".splitlines(keepends=True)
        for node in self._walk_filtered_nodes():
            if isinstance(node,FunctionMixin):
                result.append(node.render_cython_lazy_loader_def(lib_handle=lib_handle))
        return result
    
    def create_python_interfaces(self,c_interface_module):
        """Renders Python interfaces in Cython.
        """
        result = []
        for node in self._walk_filtered_nodes():
            if isinstance(node,MacroDefinitionMixin):
                result.append(f"from {c_interface_module} cimport {node.name}")
            elif isinstance(node,EnumMixin):
                result.append(node.render_python_interface(
                    prefix=f"{c_interface_module}."
                ))
            elif isinstance(node,TypedefMixin):
                pass#result.append(node.render_python_interface())
            elif isinstance(node,FunctionMixin):
                pass#result.append(node.render_python_interface())
            # TODO ignore nested typs on the top-level
        return result

    def render_python_interfaces(self,cython_c_bindings_module: str):
        """Returns the Python interface file content for the given headers."""
        result = self.create_python_interfaces(cython_c_bindings_module)
        nl = "\n\n"
        return f"""\
{nl.join(result)}"""

    def render_cython_declaration_part(self, runtime_linking: bool = False):
        """Returns the Cython bindings file content for the given headers."""
        nl = "\n\n"
        return nl.join(self.create_cython_declaration_part(runtime_linking))
    
    def render_cython_definition_part(self,runtime_linking: bool = False):
        """Returns the Cython bindings file content for the given headers."""
        nl = "\n\n"
        if runtime_linking:
            return nl.join(self.create_cython_lazy_loader_defs())
        else:
            return ""

class CythonPackageGenerator:
    """Generate Python/Cython packages for a HIP C interface.

    Generates Python/Cython packages for a HIP C interface
    based on a list of header file names and the name of
    a library to link.
    """

    def __init__(
        self,
        pkg_name: str,
        include_dir: str,
        header: str,
        runtime_linking = False,
        dll: str = None,
        node_filter: callable = control.DEFAULT_NODE_FILTER,
        macro_type: callable = lambda macro: "int",
        ptr_parm_intent: callable = lambda parm: control.Intent.INOUT,
        ptr_rank: callable = lambda parm: control.Rank.ANY,
        renamer: callable = DEFAULT_RENAMER,
        warnings = control.Warnings.WARN,
        cflags=[],
    ):
        """Constructor.

        Args:
            pkg_name (str): Name of the package that should be generated. Influences filesnames.
            include_dir (str): Name of the main include dir.
            header (str|tuple): Name of the header file. Absolute paths or w.r.t. to include dir.
            runtime_linking (bool, optional): If runtime-linking code should be generated, defaults to False.
            dll (str): Name of the DLL/shared object to link. Must not be none if  
                       `runtime_linking` is specified. Defaults to None.
            node_filter (callable, optional): Filter for selecting the nodes to include in generated output. Defaults to `lambda x: True`.
            macro_type (callable, optional): Assigns a type to a macro node. Defaults to `lambda x: "int"`.
            ptr_parm_intent (callable, optional): Assigns the intent (in,out,inout,create) to a pointer-type function parameter/struct field node.
                                                  Defaults to `lambda node: cython.Intent.IN`.
            ptr_rank (callable, optional): Assigns the "rank" (scalar,buffer) to a function parameter node.
                                            Defaults to `lambda parm: cython.Intent.ANY`.
            cflags (list(str), optional): Flags to pass to the C parser.
        """
        self.pkg_name = pkg_name
        self.include_dir = include_dir
        self.header = header
        self.runtime_linking = runtime_linking
        self.dll = dll
        self.cflags = cflags
        self.c_interface_preamble = """\
# AMD_COPYRIGHT
from libc.stdint cimport *
"""
        self.python_interface_preamble = """\
# AMD_COPYRIGHT
from libc.stdint cimport *
"""

        if isinstance(header,str):
            filename = header
            unsaved_files = None
        elif isinstance(h,tuple):
            filename = header[0]
            unsaved_files = [header]
        else:
            raise ValueError("type of 'headers' must be str or tuple")
        print(filename,file=sys.stderr) # TODO logging
        if include_dir != None:
            abspath = os.path.join(include_dir, filename)
        else:
            abspath = filename
        cflags = self.cflags + ["-I", f"{include_dir}"]
        parser = cparser.CParser(abspath, append_cflags=cflags, unsaved_files=unsaved_files)
        parser.parse()

        self.backend = CythonBackend.from_libclang_translation_unit(
            parser.translation_unit,
            header,
            node_filter,
            macro_type,
            ptr_parm_intent,
            ptr_rank,
            renamer,
            warnings)

    def write_package_files(self,output_dir: str = None):
        """Write all files required to build this Cython/Python package.

        Args:
            pkg_name (str): Name of the package that should be generated. Influences filesnames.
        """
        c_interface_preamble = self.c_interface_preamble + "\n"
        python_interface_preamble = ( 
            self.python_interface_preamble 
            + f"\nfrom . cimport c{self.pkg_name}\n"
        )

        with open(f"{output_dir}/c{self.pkg_name}.pxd", "w") as outfile:
            outfile.write(c_interface_preamble)
            outfile.write(self.backend.render_cython_declaration_part(
                runtime_linking=self.runtime_linking
            ))
        with open(f"{output_dir}/c{self.pkg_name}.pyx", "w") as outfile:
            outfile.write(c_interface_preamble)
            outfile.write(self.backend.render_cython_definition_part(
                runtime_linking=self.runtime_linking
            ))
        with open(f"{output_dir}/{self.pkg_name}.pyx", "w") as outfile:
            outfile.write(python_interface_preamble)
            outfile.write(self.backend.render_python_interfaces(f"c{self.pkg_name}"))