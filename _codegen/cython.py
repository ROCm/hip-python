# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import sys
import os
import keyword
import textwrap

import clang.cindex

import Cython.Tempita

from . import cparser
from . import control

indent = " " * 4

funptr_name_template = "{name}_funptr"

restricted_names = keyword.kwlist + [
    "cdef",
    "cpdef",  # TODO extend
]


def DEFAULT_RENAMER(name):  # backend-specific
    result = name
    while result in restricted_names:
        result += "_"
    return result


def DEFAULT_MACRO_TYPE(node):  # backend-specific
    return "int"

wrapper_class_base_template = """
cdef class {{name}}:
    cdef {{cname}}* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef {{name}} from_ptr({{cname}} *_ptr, bint owner=False):
        \"""Factory function to create ``{{name}}`` objects from
        given ``{{cname}}`` pointer.
{{if has_new}}

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
{{endif}}
        \"""
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef {{name}} wrapper = {{name}}.__new__({{name}})
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
{{if has_new}}
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
{{endif}}
{{if has_new}}
    @staticmethod
    cdef {{name}} new():
        \"""Factory function to create {{name}} objects with
        newly allocated {{cname}}\"""
        cdef {{cname}} *_ptr = <{{cname}} *>stdlib.malloc(sizeof({{cname}}))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return {{name}}.from_ptr(_ptr, owner=True)
{{endif}}
"""

wrapper_class_property_template = """\
{{if is_basic}}
def get_{{attr}}(self,i):
    \"""Get ``{{attr}}`` value of element ``i``.
    \"""
    return self._ptr[i].{{attr}}
def set_{{attr}}(self,i,{{typename}} value):
    \"""Set ``{{attr}}`` value of element ``i``.
    \"""
    self._ptr[i].{{attr}} = value
@property
def {{attr}}(self):
    \"""Getter for ``{{attr}}``.\"""
    return self.get_{{attr}}(0)
@{{attr}}.setter
def {{attr}}(self,{{typename}} value):
    \"""Setter for ``{{attr}}``.\"""
    self.set_{{attr}}(0,value)
{{endif}}
"""


# Mixins
class CythonMixin:
    def __init__(self):  # Will not be called, attribs specified for type hinting
        self.renamer = DEFAULT_RENAMER
        self.sep = "_"

    def _cython_and_c_name(self, orig_name: str):
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

    def render_c_interface(self):
        from . import tree

        assert isinstance(self, tree.MacroDefinition)
        return f"cdef {self.macro_type(self)} {self._cython_and_c_name(self.name)}"

    def render_python_interface(self, cprefix: str):
        """Renders '{self.name} = {prefix}{self.name}'."""
        from . import tree

        assert isinstance(self, tree.MacroDefinition)
        name = self.renamer(self.name)
        return f"{name} = {cprefix}{name}"


class FieldMixin(CythonMixin):
    def __init__(self):
        CythonMixin.__init__(self)
        self.ptr_rank = control.DEFAULT_PTR_RANK

    def cython_repr(self):
        from . import tree

        assert isinstance(self, tree.Field)
        typename = self.global_typename(self.sep, self.renamer)
        name = self._cython_and_c_name(self.name)
        return f"{typename} {name}"

    def render_python_property(self,cprefix: str):
        from . import tree

        assert isinstance(self, tree.Field)
        attr = self.renamer(self.name)
        template = Cython.Tempita.Template(wrapper_class_property_template)
        return template.substitute(
          typename = self.global_typename(self.sep, self.renamer),
          attr = attr,
          is_basic = self.is_basic_type,
        )

class RecordMixin(CythonMixin):
    @property
    def c_record_kind(self) -> str:
        if self.cursor.kind == clang.cindex.CursorKind.STRUCT_DECL:
            return "struct"
        else:
            return "union"

    def _render_c_interface_head(self) -> str:
        from . import tree

        assert isinstance(self, tree.Record)
        name = self._cython_and_c_name(self.global_name(self.sep))
        cython_def_kind = "ctypedef" if self._from_typedef_with_anon_child else "cdef"
        return f"{cython_def_kind} {self.c_record_kind} {name}:\n"

    def render_c_interface(self) -> str:
        """Render Cython binding for this struct/union declaration.

        Renders a Cython binding for this struct/union declaration, does
        not render declarations for nested types.

        Returns:
            str: Cython C-binding representation of this struct declaration.
        """
        from . import tree

        assert isinstance(self, tree.Record)
        global indent
        result = self._render_c_interface_head()
        fields = list(self.fields)
        if len(fields):
            result += textwrap.indent(
                "\n".join([field.cython_repr() for field in fields]), indent
            )
        else:
            result += f"{indent}pass"
        return result

    def _render_python_interface_head(self,cprefix: str) -> str:
        from . import tree

        assert isinstance(self, tree.Record)
        global wrapper_class_base_template
        name = self.renamer(self.global_name(self.sep))
        template = Cython.Tempita.Template(wrapper_class_base_template)
        return template.substitute(
          name = name,
          cname = cprefix + name,
          has_new = not self.is_incomplete,
        )

    def render_python_interface(self,cprefix: str) -> str:
        """Render Cython binding for this struct/union declaration.

        Renders a Cython binding for this struct/union declaration, does
        not render declarations for nested types.

        Returns:
            str: Cython C-binding representation of this struct declaration.
        """
        from . import tree

        assert isinstance(self, tree.Record)
        global indent

        result = self._render_python_interface_head(cprefix)
        for field in self.fields:
            result += textwrap.indent(field.render_python_property(cprefix),indent)
        # fields = list(self.fields)
        #result += f"{indent}pass"
        return result


class StructMixin(RecordMixin):
    pass


class UnionMixin(RecordMixin):
    pass


class EnumMixin(CythonMixin):
    def _render_cython_enums(self):
        """Yields the enum constants' names."""
        from . import tree

        assert isinstance(self, tree.Enum)
        for child_cursor in self.cursor.get_children():
            name = self._cython_and_c_name(child_cursor.spelling)
            yield name

    def _render_c_interface_head(self) -> str:
        from . import tree

        assert isinstance(self, tree.Enum)
        cython_def_kind = "ctypedef" if self._from_typedef_with_anon_child else "cdef"
        name = self._cython_and_c_name(self.global_name(self.sep))
        return f"{cython_def_kind} enum{'' if self.is_anonymous else ' '+name}:\n"

    def render_c_interface(self):
        from . import tree

        # assert isinstance(self,tree.Enum)
        global indent
        return self._render_c_interface_head() + textwrap.indent(
            "\n".join(self._render_cython_enums()), indent
        )

    def _render_python_enums(self, cprefix: str):
        from . import tree

        # assert isinstance(self,tree.Enum)
        """Yields the enum constants' names."""
        for child_cursor in self.cursor.get_children():
            name = self.renamer(child_cursor.spelling)
            yield f"{name} = {cprefix}{name}"

    def render_python_interface(self, cprefix: str):
        """Renders an enum.IntEnum class.

        Note:
            Does not create an enum.IntEnum class but only exposes the enum constants
            from the Cython package corresponding to the cprefix if the
            Enum is anonymous.
        """
        from . import tree

        assert isinstance(self, tree.Enum)
        global indent
        if self.is_anonymous:
            return "\n".join(self._render_python_enums(cprefix))
        else:
            name = self._cython_and_c_name(self.global_name(self.sep))
            return f"class {name}(enum.IntEnum):\n" + textwrap.indent(
                "\n".join(self._render_python_enums(cprefix)), indent
            )

class TypedefMixin(CythonMixin):
   
    def render_c_interface(self):
        from . import tree

        assert isinstance(self, tree.Typedef)
        """Returns a Cython binding for this Typedef.
        """
        underlying_type_name = self.global_typename(self.sep, self.renamer)
        name = self._cython_and_c_name(self.name)

        return f"ctypedef {underlying_type_name} {name}"

    def render_python_interface(self,cprefix: str) -> str:
        from . import tree
        assert isinstance(self, tree.Typedef)
        name = self.renamer(self.global_name(self.sep))
        if self.is_pointer_to_record_or_enum:
            return f"{name} = {self.renamer(self.typeref.global_name(self.sep))}"
        else:
            return None

class FunctionPointerMixin(CythonMixin):
    def render_c_interface(self):
        """Returns a Cython binding for this Typedef."""
        from . import tree

        assert isinstance(self, tree.FunctionPointer)
        parm_types = ",".join(self.global_parm_types(self.sep, self.renamer))
        underlying_type_name = self.renamer(self.canonical_result_typename)
        typename = self.renamer(
            self.global_name(self.sep)
        )  # might be AnonymousFunctionPointer
        return f"ctypedef {underlying_type_name} (*{typename}) ({parm_types})"
    
    def render_python_interface(self,cprefix: str) -> str:
        from . import tree

        assert isinstance(self, tree.FunctionPointer)
        global wrapper_class_base_template
        name = self.renamer(self.global_name(self.sep))
        template = Cython.Tempita.Template(wrapper_class_base_template)
        return template.substitute(
          name = name,
          cname = cprefix + name,
          has_new = False,
        )

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

        assert isinstance(self, tree.Parm)
        typename = self.global_typename(self.sep, self.renamer)
        name = self.renamer(self.name)
        return f"{typename} {name}"


class FunctionMixin(CythonMixin):
    def _raw_comment_as_python_comment(self):
        from . import tree

        assert isinstance(self, tree.Function)
        if self.raw_comment != None:
            comment = self._raw_comment_stripped()
            return "".join(["# " + l for l in comment.splitlines(keepends=True)])
        else:
            return ""

    def _raw_comment_as_docstring(self):
        from . import tree

        assert isinstance(self, tree.Function)
        return f'"""{"".join(self._raw_comment_stripped()).rstrip()}\n"""'

    def render_c_interface(self, modifiers=" nogil",modifiers_front=""):
        from . import tree

        assert isinstance(self, tree.Function)
        typename = self.global_typename(self.sep, self.renamer)
        name = self.renamer(self.name)
        parm_decls = ",".join([arg.cython_repr() for arg in self.parms])
        return f"""\
{self._raw_comment_as_python_comment().rstrip()}
{modifiers_front}{typename} {name}({parm_decls}){modifiers}
"""

    def render_cython_lazy_loader_decl(self, modifiers=" nogil"):
        return self.render_c_interface(modifiers_front="cdef ")

    def cython_funptr_name(self):
        from . import tree

        assert isinstance(self, tree.Function)
        name = self.renamer(self.name)
        return funptr_name_template.format(name=name)

    def render_cython_lazy_loader_def(
        self, lib_handle: str = "__lib_handle", modifiers="nogil"
    ):
        from . import tree

        assert isinstance(self, tree.Function)
        funptr_name = self.cython_funptr_name()
        parm_types = ",".join(self.global_parm_types(self.sep, self.renamer))
        parm_names = ",".join(self.parm_names(self.renamer))
        typename = self.global_typename(self.sep, self.renamer)
        return f"""\
cdef void* {funptr_name} = NULL
{self.render_cython_lazy_loader_decl(modifiers).strip()}:
    global {lib_handle}
    global {funptr_name}
    if {funptr_name} == NULL:
        with gil:
            {funptr_name} = loader.load_symbol({lib_handle}, "{self.name}")
    return (<{typename} (*)({parm_types}) nogil> {funptr_name})({parm_names})
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
        warnings: control.Warnings = control.Warnings.IGNORE,
    ):
        from . import tree

        root = tree.from_libclang_translation_unit(translation_unit, warnings)
        return CythonBackend(
            root, filename, node_filter, macro_type, ptr_parm_intent, ptr_rank, renamer
        )

    def __init__(
        self,
        root,
        filename: str,
        node_filter: callable = control.DEFAULT_NODE_FILTER,
        macro_type: callable = DEFAULT_MACRO_TYPE,
        ptr_parm_intent: callable = control.DEFAULT_PTR_PARM_INTENT,
        ptr_rank: callable = control.DEFAULT_PTR_RANK,
        renamer: callable = DEFAULT_RENAMER,
    ):
        """
        Note:
            Argument 'root' has no type hint in order to prevent a circular inclusion error.
            Instead an assertion is used in the body that checks if the type is `tree.Root`.
        """
        from . import tree

        assert isinstance(root, tree.Root)
        self.root = root
        self.filename = filename
        self.node_filter = node_filter
        self.macro_type = macro_type
        self.ptr_parm_intent = (
            ptr_parm_intent  # TODO use for FunctionMixin.render_python_interface
        )
        self.ptr_rank = ptr_rank  # TODO use for FunctionMixin.render_python_interface
        self.renamer = renamer

    def _walk_filtered_nodes(self):
        """Walks the filtered nodes in post-order and sets the renamer of each node.

        Note:
            Post-order yields nested struct/union/enum declarations before their
            parent.
        """
        for node in self.root.walk(postorder=True):
            if isinstance(node, CythonMixin):
                # set defaults
                setattr(node, "sep", "_")
                # set user callbacks
                setattr(node, "renamer", self.renamer)
                if isinstance(node, MacroDefinitionMixin):
                    setattr(node, "macro_type", self.macro_type)
                elif isinstance(node, (FieldMixin)):
                    setattr(node, "ptr_rank", self.ptr_rank)
                elif isinstance(node, (ParmMixin)):
                    setattr(node, "ptr_rank", self.ptr_rank)
                    setattr(node, "ptr_intent", self.ptr_parm_intent)
                # yield relevant nodes
                if not isinstance(node, (FieldMixin, ParmMixin)):
                    if self.node_filter(node):
                        yield node

    def create_cython_declaration_part(self, runtime_linking: bool = False):
        """Returns the content of a Cython bindings file.

        Creates the content of a Cython bindings file.
        Contains Cython declarations per C declaration
        plus helper types that have been introduced for nested enum/struct/union types.

        Note:
            Nested anonymous types for which we have a tree node with
            autogenerated name must be excluded from the `extern from "<header_name.h>`
            block as entities listed within the body of the construct,
            are assumed by Cython to be present in C code whenever
            the respective header is included.

            Moving those entities out of the `extern from` block
            ensures that Cython creates a proper C type on its own.
        """
        from . import tree

        global indent
        curr_indent = ""
        result = []

        last_was_extern = False
        for node in self._walk_filtered_nodes():
            if  (  (runtime_linking and isinstance(node, FunctionMixin))
                   or isinstance(node,AnonymousFunctionPointerMixin)
                   or (isinstance(node,(tree.NestedEnum,tree.NestedStruct,tree.NestedUnion))
                       and node.is_cursor_anonymous)
            ):
                if isinstance(node,FunctionMixin):
                    contrib = node.render_cython_lazy_loader_decl()
                else:
                    contrib = node.render_c_interface()
                curr_indent = ""
                last_was_extern = False
            else:
                if not last_was_extern:
                    result.append(f'cdef extern from "{self.filename}":')
                curr_indent = indent
                contrib = node.render_c_interface()
                last_was_extern = True
            result.append(textwrap.indent(contrib, curr_indent))
        return result

    def create_cython_lazy_loader_decls(self):
        result = []
        for node in self._walk_filtered_nodes():
            if isinstance(node, FunctionMixin):
                result.append(node.render_cython_lazy_loader_decl(self.renamer))
        return result

    def create_cython_lazy_loader_defs(self, dll: str):
        # TODO: Add compiler? switch to switch between MS and Linux loaders
        # TODO: Add compiler? switch to switch between HIP and CUDA backends?
        # Should be possible to implement this via the renamer and generating multiple modules
        lib_handle = "_lib_handle"
        result = f"""\
cimport hip._util.posixloader as loader
cdef void* {lib_handle} = loader.open_library(\"{dll}\")
""".splitlines(
            keepends=True
        )
        for node in self._walk_filtered_nodes():
            if isinstance(node, FunctionMixin):
                result.append(node.render_cython_lazy_loader_def(lib_handle=lib_handle))
        return result

    def create_python_interfaces(self, cmodule):
        """Renders Python interfaces in Cython."""
        from . import tree
        result = []
        cprefix=f"{cmodule}."
        for node in self._walk_filtered_nodes():
            contrib = None
            if isinstance(node, (
                MacroDefinitionMixin,
                EnumMixin,
                StructMixin,UnionMixin,
                TypedefedFunctionPointerMixin,
                AnonymousFunctionPointerMixin,
                TypedefMixin,
              )
            ):
                contrib = node.render_python_interface(cprefix=cprefix)
            elif isinstance(node, FunctionMixin):
                pass  # result.append(node.render_python_interface())
            # TODO ignore nested typs on the top-level
            if contrib != None:
                result.append(contrib)
        return result

    def render_python_interfaces(self, cython_c_bindings_module: str):
        """Returns the Python interface file content for the given headers."""
        result = self.create_python_interfaces(cython_c_bindings_module)
        nl = "\n\n"
        return f"""\
{nl.join(result)}"""

    def render_cython_declaration_part(self, runtime_linking: bool = False):
        """Returns the Cython bindings file content for the given headers."""
        nl = "\n\n"
        return nl.join(self.create_cython_declaration_part(runtime_linking))

    def render_cython_definition_part(
        self, runtime_linking: bool = False, dll: str = None
    ):
        """Returns the Cython bindings file content for the given headers."""
        nl = "\n\n"
        if runtime_linking:
            if dll is None:
                raise ValueError(
                    "argument 'dll' must not be 'None' if 'runtime_linking' is set to 'True'"
                )
            return nl.join(self.create_cython_lazy_loader_defs(dll))
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
        runtime_linking=False,
        dll: str = None,
        node_filter: callable = control.DEFAULT_NODE_FILTER,
        macro_type: callable = lambda macro: "int",
        ptr_parm_intent: callable = lambda parm: control.Intent.INOUT,
        ptr_rank: callable = lambda parm: control.Rank.ANY,
        renamer: callable = DEFAULT_RENAMER,
        warnings=control.Warnings.WARN,
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
from libc cimport stdlib
from libc.stdint cimport *
import enum
"""

        if isinstance(header, str):
            filename = header
            unsaved_files = None
        elif isinstance(h, tuple):
            filename = header[0]
            unsaved_files = [header]
        else:
            raise ValueError("type of 'headers' must be str or tuple")
        print(filename, file=sys.stderr)  # TODO logging
        if include_dir != None:
            abspath = os.path.join(include_dir, filename)
        else:
            abspath = filename
        cflags = self.cflags + ["-I", f"{include_dir}"]
        parser = cparser.CParser(
            abspath, append_cflags=cflags, unsaved_files=unsaved_files
        )
        parser.parse()

        self.backend = CythonBackend.from_libclang_translation_unit(
            parser.translation_unit,
            header,
            node_filter,
            macro_type,
            ptr_parm_intent,
            ptr_rank,
            renamer,
            warnings,
        )

    def write_package_files(self, output_dir: str = None):
        """Write all files required to build this Cython/Python package.

        Args:
            pkg_name (str): Name of the package that should be generated. Influences filesnames.
        """
        c_interface_preamble = self.c_interface_preamble + "\n"
        python_interface_preamble = (
            self.python_interface_preamble + f"\nfrom . cimport c{self.pkg_name}\n"
        )

        with open(f"{output_dir}/c{self.pkg_name}.pxd", "w") as outfile:
            outfile.write(c_interface_preamble)
            outfile.write(
                self.backend.render_cython_declaration_part(
                    runtime_linking=self.runtime_linking
                )
            )
        with open(f"{output_dir}/c{self.pkg_name}.pyx", "w") as outfile:
            outfile.write(c_interface_preamble)
            outfile.write(
                self.backend.render_cython_definition_part(
                    runtime_linking=self.runtime_linking, dll=self.dll
                )
            )
        with open(f"{output_dir}/{self.pkg_name}.pyx", "w") as outfile:
            outfile.write(python_interface_preamble)
            outfile.write(self.backend.render_python_interfaces(f"c{self.pkg_name}"))
