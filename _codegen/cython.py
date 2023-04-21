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

c_interface_funptr_name_template = "{name}_funptr"

python_interface_retval_template = "{name}_____retval"

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
{{if is_basic_type}}
def get_{{attr}}(self, i):
    \"""Get value ``{{attr}}`` of ``self._ptr[i]``.
    \"""
    return self._ptr[i].{{attr}}
def set_{{attr}}(self, i, {{typename}} value):
    \"""Set value ``{{attr}}`` of ``self._ptr[i]``.
    \"""
    self._ptr[i].{{attr}} = value
@property
def {{attr}}(self):
    return self.get_{{attr}}(0)
@{{attr}}.setter
def {{attr}}(self, {{typename}} value):
    self.set_{{attr}}(0,value)
{{elif is_basic_type_constantarray}}
def get_{{attr}}(self, i):
    \"""Get value of ``{{attr}}`` of ``self._ptr[i]``.
    \"""
    return self._ptr[i].{{attr}}
@property
def {{attr}}(self):
    return self.get_{{attr}}(0)
# TODO is_basic_type_constantarray: add setters
{{elif is_enum}}
def get_{{attr}}(self, i):
    \"""Get value of ``{{attr}}`` of ``self._ptr[i]``.
    \"""
    return {{typename}}(self._ptr[i].{{attr}})
def set_{{attr}}(self, i, value):
    \"""Set value ``{{attr}}`` of ``self._ptr[i]``.
    \"""
    if not isinstance(value, {{typename}}):
        raise TypeError("'value' must be of type '{{typename}}'")
    self._ptr[i].{{attr}} = value.value
@property
def {{attr}}(self):
    return self.get_{{attr}}(0)
@{{attr}}.setter
def {{attr}}(self, value):
    self.set_{{attr}}(0,value)
{{elif is_enum_constantarray}}
# TODO is_enum_constantarray: add
{{elif is_record}}
def get_{{attr}}(self, i):
    \"""Get value of ``{{attr}}`` of ``self._ptr[i]``.
    \"""
    return {{typename}}.from_ptr(&self._ptr[i].{{attr}})
@property
def {{attr}}(self):
    return self.get_{{attr}}(0)
{{endif}}
"""


# Mixins
class CythonMixin:
    def __init__(self):  # Will not be called, attribs specified for type hinting
        self.renamer = DEFAULT_RENAMER
        self.sep = "_"

    @property
    def cython_name(self):
        return self.renamer(self.name)

    @property
    def cython_global_name(self):
        return self.renamer(self.global_name(self.sep))

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


class Typed:
    @property
    def cython_global_typename(self):
        from . import tree

        assert isinstance(self, tree.Typed)
        return self.global_typename(self.sep, self.renamer)

    @property
    def has_array_rank(self):
        from . import tree

        assert isinstance(self, tree.Typed)
        if self.is_any_array:
            return True
        else:
            return self.ptr_rank(self) > 0

    @property
    def is_autoconverted_by_cython(self):
        return (
            self.is_basic_type
            or self.is_basic_type_constarray
            or self.is_pointer_to_char()
            or self.is_char_incompletearray
        )


class FieldMixin(CythonMixin, Typed):
    def __init__(self):
        CythonMixin.__init__(self)
        self.ptr_rank = control.DEFAULT_PTR_RANK

    def cython_repr(self):
        from . import tree

        assert isinstance(self, tree.Field)
        typename = self.global_typename(self.sep, self.renamer)
        name = self._cython_and_c_name(self.name)
        return f"{typename} {name}"

    def render_python_property(self, cprefix: str):
        from . import tree

        assert isinstance(self, tree.Field)
        attr = self.renamer(self.name)
        template = Cython.Tempita.Template(wrapper_class_property_template)
        return template.substitute(
            typename=self.global_typename(self.sep, self.renamer),
            attr=attr,
            is_basic_type=(
                self.is_basic_type
                or self.is_pointer_to_char()  # TODO user should be consulted if char pointer is a string
            ),
            is_basic_type_constantarray=self.is_basic_type_constarray,
            is_record=self.is_record,
            is_enum=self.is_enum,
            is_enum_constantarray=self.is_enum_constantarray,
            is_record_constantarray=self.is_record_constantarray,
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

    def _render_python_interface_head(self, cprefix: str) -> str:
        from . import tree

        assert isinstance(self, tree.Record)
        global wrapper_class_base_template
        name = self.renamer(self.global_name(self.sep))
        template = Cython.Tempita.Template(wrapper_class_base_template)
        return template.substitute(
            name=name,
            cname=cprefix + name,
            has_new=not self.is_incomplete,
        )

    def render_python_interface(self, cprefix: str) -> str:
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
            result += textwrap.indent(field.render_python_property(cprefix), indent)
        # fields = list(self.fields)
        # result += f"{indent}pass"
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


class TypedefMixin(CythonMixin, Typed):
    def render_c_interface(self):
        from . import tree

        assert isinstance(self, tree.Typedef)
        """Returns a Cython binding for this Typedef.
        """
        underlying_type_name = self.global_typename(self.sep, self.renamer)
        name = self._cython_and_c_name(self.name)

        return f"ctypedef {underlying_type_name} {name}"

    def render_python_interface(self, cprefix: str) -> str:
        from . import tree

        assert isinstance(self, tree.Typedef)
        name = self.cython_global_name
        if self.is_pointer_to_record() or self.is_pointer_to_enum():
            return f"{name} = {self.renamer(self.typeref.global_name(self.sep))}"
        elif self.is_pointer_to_void:
            template = Cython.Tempita.Template(wrapper_class_base_template)
            return template.substitute(
                name=name,
                cname="void",  # hardcode as canonical type is `void *`, template already uses pointer
                has_new=False,
            )
        elif self.is_autoconverted_by_cython:
            return self.render_c_interface()
        else:
            return None


class FunctionPointerMixin(CythonMixin):
    def render_c_interface(self):
        """Returns a Cython binding for this Typedef."""
        from . import tree

        assert isinstance(self, tree.FunctionPointer)
        parm_types = ",".join(self.global_parm_types(self.sep, self.renamer))
        underlying_type_name = self.renamer(self.canonical_result_typename)
        typename = self.cython_global_name  # might be AnonymousFunctionPointer
        return f"ctypedef {underlying_type_name} (*{typename}) ({parm_types})"

    def render_python_interface(self, cprefix: str) -> str:
        from . import tree

        assert isinstance(self, tree.FunctionPointer)
        global wrapper_class_base_template
        name = self.cython_global_name
        template = Cython.Tempita.Template(wrapper_class_base_template)
        return template.substitute(
            name=name,
            cname=cprefix + name,
            has_new=False,
        )


class TypedefedFunctionPointerMixin(FunctionPointerMixin):
    pass


class AnonymousFunctionPointerMixin(FunctionPointerMixin):
    pass


class ParmMixin(CythonMixin, Typed):
    def __init__(self):
        CythonMixin.__init__(self)
        self.ptr_rank = control.DEFAULT_PTR_RANK
        self.ptr_create = control.DEFAULT_PTR_PARAM_INTENT

    @property
    def cython_repr(self):
        from . import tree

        assert isinstance(self, tree.Parm)
        typename = self.cython_global_typename
        name = self.cython_name
        return f"{typename} {name}"

    @property
    def is_c_style_reference(self):
        from . import tree

        actual_rank = self.ptr_rank(self)
        assert isinstance(self, tree.Parm)
        return self.get_pointer_degree() == actual_rank + 1

    @property
    def is_created_by_function(self):
        return (
            self.is_c_style_reference
            and self.ptr_create(self) == control.PointerParamIntent.OUT
        )


class FunctionMixin(CythonMixin, Typed):
    def _raw_comment_as_python_comment(self):
        from . import tree

        assert isinstance(self, tree.Function)
        if self.raw_comment != None:
            comment = self._raw_comment_stripped()
            return "".join(["# " + l for l in comment.splitlines(keepends=True)])
        else:
            return ""

    # TODO Identify and extract doxygen params and other sections to create higher quality docstring
    # doxygen param is terminated by blank line or new section/paragraph
    # Can be used to build parser for args
    # More details https://doxygen.nl/manual/commands.html#cmdparam
    def _raw_comment_as_docstring(self):
        from . import tree

        assert isinstance(self, tree.Function)
        return f'"""{"".join(self._raw_comment_stripped()).rstrip()}\n"""'

    def render_c_interface(self, modifiers=" nogil", modifiers_front=""):
        from . import tree

        assert isinstance(self, tree.Function)
        typename = self.cython_global_typename
        name = self.cython_name
        parm_decls = ",".join([arg.cython_repr for arg in self.parms])
        return f"""\
{self._raw_comment_as_python_comment().rstrip()}
{modifiers_front}{typename} {name}({parm_decls}){modifiers}
"""

    def render_cython_lazy_loader_decl(self, modifiers=" nogil"):
        return self.render_c_interface(modifiers_front="cdef ")

    @property
    def cython_funptr_name(self):
        global c_interface_funptr_name_template
        return c_interface_funptr_name_template.format(name=self.cython_name)

    def render_cython_lazy_loader_def(
        self, lib_handle: str = "__lib_handle", modifiers="nogil"
    ):
        from . import tree

        assert isinstance(self, tree.Function)
        funptr_name = self.cython_funptr_name
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

    def _render_python_signature(self):
        from . import tree

        assert isinstance(self, tree.Function)
        global indent
        args = []
        for parm in self.parms:
            parm_name = parm.cython_name
            assert isinstance(parm, tree.Parm)
            if parm.is_autoconverted_by_cython:
                args.append(parm.cython_repr)
            elif not parm.is_created_by_function:  # out arg
                args.append(parm_name)
        name = self.renamer(self.name)
        return f"def {name}({', '.join(args)}):\n" + textwrap.indent(
            self._raw_comment_as_docstring(), indent
        )

    def _render_python_interface_head(self):
        from . import tree

        out_args = []
        c_interface_call_args = []
        py2c_conversions = []
        for parm in self.parms:
            parm_name = parm.cython_name
            assert isinstance(parm, tree.Parm)
            if parm.is_created_by_function:  # out arg
                if parm.is_pointer_to_basic_type(degree=1):
                    typehandler = parm._type_handler.create_from_layer(
                        1, canonical=True
                    )
                    parm_typename = typehandler.clang_type.spelling
                    py2c_conversions.append(f"cdef {parm_typename} {parm_name}")
                    out_args.append(parm_name)
                    c_interface_call_args.append(f"&{parm_name}")
                elif parm.is_pointer_to_record(degree=2):
                    parm_typename = parm.lookup_innermost_type().cython_name
                    py2c_conversions.append(f"{parm_name} = {parm_typename}.from_ptr(NULL,owner=True)")
                    c_interface_call_args.append(f"&{parm_name}._ptr")
                    out_args.append(parm_name)
            elif parm.is_autoconverted_by_cython:
                c_interface_call_args.append(f"{parm_name}")
            elif parm.is_pointer_to_record(degree=1):
                #py2c_conversions.append(parm_name)
                pass
        
        fully_specified = len(list(self.parms)) == len(c_interface_call_args)
        setattr(self,"is_python_code_complete",fully_specified)
        
        return (
            fully_specified,
            out_args,
            c_interface_call_args,
            textwrap.indent("\n".join(py2c_conversions), indent) + "\n" if len(py2c_conversions) else "",
        )
    
    @property
    def _python_interface_retval(self):
        global python_interface_retval_template
        return python_interface_retval_template.format(name=self.cython_name)

    def _render_python_interface_c_interface_call(self, 
                                                  cprefix: str,
                                                  call_args: list,
                                                  out_args: list):
        from . import tree

        typename = self.cython_global_typename
        retvalname = self._python_interface_retval
        comma = ","
        c_interface_call =  f"{cprefix}{self.cython_name}({comma.join(call_args)})"
        assert isinstance(self, tree.Function)
        if self.is_void:
            return c_interface_call
        elif self.is_basic_type:
            out_args.insert(0,retvalname)
            return f"cdef {retvalname} = {c_interface_call}"
        elif self.is_enum:
            out_args.insert(0,retvalname)
            return f"{retvalname} = {typename}({c_interface_call})"
        else:
            return ""

    def render_python_interface(self, cprefix: str) -> str:
        """Render the Python interface for this function.

        Args:
            cprefix (str): A prefix for referenced C interface types.
        Returns:
            str: Rendered Python interface.
        """
        result = self._render_python_signature().rstrip() + "\n"
        (fully_specified, out_args, call_args, partial_result) = self._render_python_interface_head()
        result += partial_result
        if fully_specified:
            result += f"{indent}{self._render_python_interface_c_interface_call(cprefix,call_args,out_args)}"
            result += f"{indent}# fully specified\n"
            if len(out_args) > 1:
                comma = ","
                result += f"{indent}return ({comma.join(out_args)})\n"
            elif len(out_args):
                result += f"{indent}return {out_args[0]}\n"
        else:
            result += f"{indent}pass"
        return result


class CythonBackend:
    def from_libclang_translation_unit(
        translation_unit: clang.cindex.TranslationUnit,
        filename: str,
        node_filter: callable = control.DEFAULT_NODE_FILTER,
        macro_type: callable = DEFAULT_MACRO_TYPE,
        ptr_parm_intent: callable = control.DEFAULT_PTR_PARAM_INTENT,
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
        ptr_parm_intent: callable = control.DEFAULT_PTR_PARAM_INTENT,
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
                    setattr(node, "ptr_create", self.ptr_parm_intent)
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
            if (
                (runtime_linking and isinstance(node, FunctionMixin))
                or isinstance(node, AnonymousFunctionPointerMixin)
                or (
                    isinstance(
                        node, (tree.NestedEnum, tree.NestedStruct, tree.NestedUnion)
                    )
                    and node.is_cursor_anonymous
                )
            ):
                if isinstance(node, FunctionMixin):
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
        cprefix = f"{cmodule}."
        for node in self._walk_filtered_nodes():
            contrib = None
            if isinstance(
                node,
                (
                    MacroDefinitionMixin,
                    EnumMixin,
                    StructMixin,
                    UnionMixin,
                    TypedefedFunctionPointerMixin,
                    AnonymousFunctionPointerMixin,
                    TypedefMixin,
                    FunctionMixin,
                ),
            ):
                contrib = node.render_python_interface(cprefix=cprefix)
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
        macro_type: callable = DEFAULT_MACRO_TYPE,
        ptr_parm_intent: callable = control.DEFAULT_PTR_PARAM_INTENT,
        ptr_rank: callable = control.DEFAULT_PTR_RANK,
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
        elif isinstance(header, tuple):
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