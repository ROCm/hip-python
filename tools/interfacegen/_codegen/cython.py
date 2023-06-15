# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import re
import sys
import os
import keyword
import textwrap
import warnings

import clang.cindex

import Cython.Tempita

from . import cparser
from . import control

from . import doxyparser

indent = " " * 4

restricted_names = keyword.kwlist + [
    "cdef",
    "cpdef",  # TODO extend
]

c_interface_funptr_name_template = "_{name}__funptr"

python_interface_retval_template = "_{name}__retval"

python_interface_int_enum_base_class = "enum.IntEnum"

python_interface_int_enum_base_class_name_template = "_{name}__Base"

# Always return a tuple if there is at least one return value
python_interface_always_return_tuple = True

python_interface_record_properties_name = "PROPERTIES"

python_interface_pyobj_role_template = r":py:obj:`.{name}`"

def CYTHON_AUTOCONV_FROM_PYTHON_TYPES(canonical_ctype: str):
    """Convert a canonical C type to the Python types from which
    it is converted automatically by Cython.

    Returns:
        tuple(str): The Python types that Cython autoconverts to the C type.

    Note:
        For implementation details, see 
        https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#automatic-type-conversions
    """
    tokens = [tk for tk in canonical_ctype.split(" ") if tk not in ("const","unsigned")]
    if tokens in [
        ["char", "*"],
        ["char", "[]"],
    ]:
        return ("bytes",)
    elif tokens in (
        ["char"],["short"],["int"],["long"],["long","long"]
    ):
        return ("int",) # no long in Python 3 anymore
    elif tokens in [
        ["float"],["double"],["long","double"],
    ]:
        return ("float","int") # no long in Python 3, int can be converted to float too
    elif len(tokens) == 2 and tokens[0] in ("union","struct","enum"):
        raise KeyError("Cython cannot autoconvert to C structs, unions, and enums from Python types.")
    else:
        # C array and struct union are not handled yet
        # requires
        raise NotImplementedError(f"not implemented for type '{canonical_ctype}'")

def CYTHON_AUTOCONV_TO_PYTHON_TYPES(canonical_ctype: str):
    """Convert a canonical C type to the Python type to which
    it is converted automatically by Cython.

    Returns:
        str: The Python type that Cython autoconverts to from the C type.

    Note:
        For implementation details, see 
        https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#automatic-type-conversions
    """
    tokens = [tk for tk in canonical_ctype.split(" ") if tk not in ("const","unsigned")]
    if tokens in [
        ["char", "*"],
        ["char", "[]"],
    ]:
        return "bytes"
    elif tokens in (
        ["char"],["short"],["int"],["long"],["long","long"]
    ):
        return "int" # no long in Python 3 anymore
    elif tokens in [
        ["float"],["double"],["long","double"],
    ]:
        return "float"
    elif len(tokens) == 2 and tokens[0] in ("union","struct","enum"):
        raise NotImplementedError("struct, union, enum types are not handled")
    else:
        # C array and struct union are not handled yet
        # requires
        raise NotImplementedError(f"not implemented for type '{canonical_ctype}'")


def DEFAULT_RENAMER(name):  # backend-specific
    result = name
    while result in restricted_names:
        result += "_"
    if "[]" in result: # Cython does not like this in certain signatures
        result = result.replace("[]","*")
    return result


def DEFAULT_RAW_COMMENT_CLEANER(raw_comment: str):
    return raw_comment

def DEFAULT_DOCSTRING_CLEANER(docstring: str):
    return docstring

def DEFAULT_MACRO_TYPE(node):  # backend-specific
    return "int"


def DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(parm_or_field):
    from . import tree

    assert isinstance(parm_or_field,tree.Typed)
    if parm_or_field.actual_rank == 1:
        innermost_type_kind = next(parm_or_field.clang_type_layer_kinds(postorder=-1,canonical=True))
        if innermost_type_kind == clang.cindex.TypeKind.INT:
            return "hip._util.types.ListOfInt"
        elif innermost_type_kind == clang.cindex.TypeKind.UINT:
            return "hip._util.types.ListOfUnsigned"
        elif innermost_type_kind == clang.cindex.TypeKind.ULONG:
            return "hip._util.types.ListOfUnsignedLong"
    if parm_or_field.actual_rank == 2:
        return "hip._util.types.ListOfDataHandle"
    return "hip._util.types.DataHandle"


default_c_interface_decl_preamble = """\
# AMD_COPYRIGHT
from libc.stdint cimport *
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
"""

default_c_interface_impl_preamble = """\
# AMD_COPYRIGHT
"""

default_python_interface_decl_preamble = """\
# AMD_COPYRIGHT
from libc cimport stdlib
from libc cimport string
from libc.stdint cimport *
cimport cpython.long
cimport cpython.buffer
cimport hip._util.types
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
"""

default_python_interface_impl_preamble = """\
# AMD_COPYRIGHT

\"""
[ATTRIBUTES]
\"""

import cython
import ctypes
import enum
"""

# Note: wrapper_class_decl_template must declare all ``@staticmethod`` ``cdef`` functions
# Note: Syntax ``bint owner=*`` is necessary to specify default value in implementation part

wrapper_class_decl_template = """
{{default cptr_type = cname + "*"}}
{{default has_new = True}}
{{default has_from_pyobj = True}}
cdef class {{name}}:
    cdef {{cptr_type}} _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef {{name}} from_ptr({{cptr_type}} ptr, bint owner=*)
    {{if has_from_pyobj}}
    @staticmethod
    cdef {{name}} from_pyobj(object pyobj)
    {{endif}}
    {{if has_new}}
    @staticmethod
    cdef __allocate({{cptr_type}}* ptr)
    @staticmethod
    cdef {{name}} new()
    @staticmethod
    cdef {{name}} from_value({{cname}} other)
    {{endif}}
"""

wrapper_class_impl_base_template = """
{{default cptr_type = cname + "*"}}
{{default is_funptr = False}}
{{default is_union = False}}
{{default has_new = True}}
{{default has_from_pyobj = True}}
{{default defaults = dict()}}
{{default properties_name = None}}
{{default all_properties_rendered = False}}
cdef class {{name}}:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef {{name}} from_ptr({{cptr_type}} ptr, bint owner=False):
        \"""Factory function to create ``{{name}}`` objects from
        given ``{{cname}}`` pointer.
        {{if has_new}}

        Setting ``owner`` flag to ``True`` causes
        the extension type to free the structure pointed to by ``ptr``
        when the wrapper object is deallocated.
        {{endif}}
        \"""
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef {{name}} wrapper = {{name}}.__new__({{name}})
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    {{if has_from_pyobj}}
    @staticmethod
    cdef {{name}} from_pyobj(object pyobj):
        \"""Derives a {{name}} from a Python object.

        Derives a {{name}} from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``{{name}}`` reference, this method
        returns it directly. No new ``{{name}}`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `{{name}}`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of {{name}}!
        \"""
        cdef {{name}} wrapper = {{name}}.__new__({{name}})
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,{{name}}):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <{{cptr_type}}>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <{{cptr_type}}>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        {{if is_funptr}}
        elif str(type(pyobj)).startswith("<class 'ctypes.CFUNCTYPE.") and str(type(pyobj)).endswith(".CFunctionType'>" ):
            wrapper._ptr = <{{cptr_type}}>cpython.long.PyLong_AsVoidPtr(ctypes.cast(pyobj, ctypes.c_void_p).value)
        {{else}}
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <{{cptr_type}}>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <{{cptr_type}}>wrapper._py_buffer.buf
        {{endif}}
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")
        return wrapper
    {{endif}}
    def __dealloc__(self):
        # Release the buffer handle
        {{if has_from_pyobj}}
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        {{endif}}
        {{if has_new}}
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
        {{endif}}
    {{if has_new}}

    @staticmethod
    cdef __allocate({{cptr_type}}* ptr):
        ptr[0] = <{{cptr_type}}>stdlib.malloc(sizeof({{cname}}))

        if ptr[0] is NULL:
            raise MemoryError
        # TODO init values, if present

    @staticmethod
    cdef {{name}} new():
        \"""Factory function to create {{name}} objects with
        newly allocated {{cname}}\"""
        cdef {{cptr_type}} ptr
        {{name}}.__allocate(&ptr)
        return {{name}}.from_ptr(ptr, owner=True)

    @staticmethod
    cdef {{name}} from_value({{cname}} other):
        \"""Allocate new C type and copy from ``other``.
        \"""
        wrapper = {{name}}.new()
        string.memcpy(wrapper._ptr, &other, sizeof({{cname}}))
        return wrapper
   
    {{py: all_properties_and_is_no_union = all_properties_rendered and not is_union}}
    {{if all_properties_and_is_no_union}}
    def __init__(self,*args,**kwargs):
    {{else}}
    # {{all_properties_rendered}}
    # {{is_union}}
    def __init__(self,**kwargs):
    {{endif}}
        {{name}}.__allocate(&self._ptr)
        self.ptr_owner = True
        {{for k,v in defaults.items()}}
        self.{{k}} = {{v}}
        {{endfor}}
        attribs = self.{{properties_name}}()
        used_attribs = set()
        {{if all_properties_and_is_no_union}}
        if len(args) > len(attribs):
            raise ValueError("More positional arguments specified than this type has properties.")
        for i,v in enumerate(args):
            setattr(self,attribs[i],v)
            used_attribs.add(attribs[i])
        {{endif}}
        {{if is_union}}
        if len(kwargs) > 1:
            raise ValueError("Not more than one attribute might specified for Python types derived from C union types.")
        {{endif}}
        valid_names = ", ".join(["'"+p+"'" for p in attribs])
        for k,v in kwargs.items():
            if k in used_attribs:
                raise KeyError(f"argument '{k}' has already been specified as positional argument.")
            elif k not in attribs:
                raise KeyError(f"'{k}' is no valid property name. Valid names: {valid_names}")
            setattr(self,k,v)
    {{endif}}
    
    def __int__(self):
        \"""Returns the data's address as long integer.
        \"""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __repr__(self):
        return f"<{{name}} object, self.ptr={int(self)}>"
    def as_c_void_p(self):
        \"""Returns the data's address as `ctypes.c_void_p`
        \"""
        return ctypes.c_void_p(int(self))
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
{{elif is_pointer_to_basic_type_or_void}}
def get_{{attr}}(self, i):
    \"""Get value ``{{attr}}`` of ``self._ptr[i]``.
    \"""
    return {{handler}}.from_ptr(self._ptr[i].{{attr}})
def set_{{attr}}(self, i, object value):
    \"""Set value ``{{attr}}`` of ``self._ptr[i]``.

    Note:
        This can be dangerous if the pointer is from a python object
        that is later on garbage collected.
    \"""
    self._ptr[i].{{attr}} = <{{typename}}>cpython.long.PyLong_AsVoidPtr(int({{handler}}.from_pyobj(value)))
@property
def {{attr}}(self):
    \"""
    Note:
        Setting this {{attr}} can be dangerous if the underlying pointer is from a python object that
        is later on garbage collected.
    \"""
    return self.get_{{attr}}(0)
@{{attr}}.setter
def {{attr}}(self, object value):
    self.set_{{attr}}(0,value)
{{elif is_basic_type_constantarray}}
def get_{{attr}}(self, i):
    \"""Get value of ``{{attr}}`` of ``self._ptr[i]``.
    \"""
    return self._ptr[i].{{attr}}
# TODO add setters
#def set_{{attr}}(self, i, {{typename}} value):
#    \"""Set value ``{{attr}}`` of ``self._ptr[i]``.
#    \"""
#    self._ptr[i].{{attr}} = value
@property
def {{attr}}(self):
    return self.get_{{attr}}(0)
# TODO add setters
#@{{attr}}.setter
#def {{attr}}(self, {{typename}} value):
#    self.set_{{attr}}(0,value)
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

    def render_c_interface(self):
        """Render a Cython interface for external C code."""
        return None

    def render_python_interface_decl(self, cprefix: str):
        """Render the declaration part for the Python interface."""
        return None

    def render_python_interface_impl(self, cprefix: str):
        """Render the implementation part for the Python interface."""
        return None
    
    @staticmethod
    def to_sphinx_pyobj(expr: str):
        return python_interface_pyobj_role_template.format(name=expr)

class MacroDefinitionMixin(CythonMixin):
    def __init__(self):
        CythonMixin.__init__(self)
        self.macro_type = DEFAULT_MACRO_TYPE

    def render_c_interface(self):
        from . import tree

        assert isinstance(self, tree.MacroDefinition)
        return f"cdef {self.macro_type(self)} {self._cython_and_c_name(self.name)}"

    def render_python_interface_impl(self, cprefix: str):
        """Returns '{self.name} = {prefix}{self.name}'."""
        from . import tree

        assert isinstance(self, tree.MacroDefinition)
        name = self.renamer(self.name)
        self.docstring_attributes.append(
                textwrap.dedent(
                        f"""\
                        {name} ({self.to_sphinx_pyobj(CYTHON_AUTOCONV_TO_PYTHON_TYPES(self.macro_type(self)))}):
                            Macro constant.
                        """
                )
            )
        return f"{name} = {cprefix}{name}"


class Typed:
    @property
    def cython_global_typename(self):
        from . import tree

        assert isinstance(self, tree.Typed)
        result = self.global_typename(
            self.sep, self.renamer, prefer_canonical=True
        )
        #if "[]" in result: # Cython does not like this in signatures
        #    result = result.replace("[]", "*")
        return result

    @property
    def actual_rank(self):
        """The actual rank of the parameter, if this is an indirection.
        """
        return self.ptr_rank(self)

    @property
    def has_array_rank(self):
        from . import tree

        assert isinstance(self, tree.Typed)
        if self.is_any_array:
            return True
        else:
            return self.actual_rank(self)
    
    @property
    def is_ptr(self):
        from . import tree

        assert isinstance(self, tree.Parm)
        return self.get_pointer_degree(incomplete_array=True) > 0

    @property
    def is_indirection(self):
        """If this is not the actual value but an indirection.

        Returns:
            bool: If this is not the actual value but an indirection.
        """
        from . import tree

        actual_rank = self.ptr_rank(self)
        assert isinstance(self, tree.Parm)
        return self.get_pointer_degree() > actual_rank

    @property
    def actual_rank(self):
        """The actual rank of the parameter, if this is an indirection.
        """
        return self.ptr_rank(self)

    @property
    def is_out_ptr(self):
        """If this parameter has been specified as out parameter."""
        assert self.is_ptr
        return self.ptr_intent(self) == control.ParmIntent.OUT

    @property
    def is_inout_ptr(self):
        """If this is an inout parameter."""
        assert self.is_ptr
        return self.ptr_intent(self) == control.ParmIntent.INOUT

    @property
    def is_in_ptr(self):
        """If this is an inout parameter."""
        assert self.is_ptr
        return self.ptr_intent(self) == control.ParmIntent.IN


    @property
    def is_autoconverted_by_cython(self):
        from . import tree

        assert isinstance(self,tree.Typed)

        return (
            self.is_basic_type
            or self.is_basic_type_constarray
            or self.is_pointer_to_char(incomplete_array=True)
        )


class FieldMixin(CythonMixin, Typed):
    def __init__(self):
        CythonMixin.__init__(self)
        self.ptr_rank = control.DEFAULT_PTR_RANK
        self.ptr_complicated_type_handler = DEFAULT_PTR_COMPLICATED_TYPE_HANDLER

    @property
    def cython_repr(self):
        from . import tree

        assert isinstance(self, tree.Field)
        typename = self.global_typename(self.sep, self.renamer, prefer_canonical=True)
        name = self._cython_and_c_name(self.name)
        return f"{typename} {name}"

    def render_python_property(self, cprefix: str):
        from . import tree

        assert isinstance(self, tree.Field)
        attr = self.renamer(self.name)
        template = Cython.Tempita.Template(wrapper_class_property_template)
        return template.substitute(
            handler=self.ptr_complicated_type_handler(self),
            typename=self.global_typename(
                self.sep, self.renamer, prefer_canonical=True
            ),
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
            is_pointer_to_basic_type_or_void=(
                self.is_pointer_to_basic_type(degree=-1)
                or self.is_pointer_to_void(degree=-1)
            ),
            # is_pointer_to_record ... # TODO
            # is_pointer_to_function_proto ...
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
                "\n".join([field.cython_repr for field in fields]), indent
            )
        else:
            result += f"{indent}pass"
        return result

    def render_python_interface_decl(self, cprefix: str) -> str:
        from . import tree

        assert isinstance(self, tree.Record)
        global wrapper_class_decl_template
        name = self.renamer(self.global_name(self.sep))
        template = Cython.Tempita.Template(wrapper_class_decl_template)
        return template.substitute(
            name=name,
            cname=cprefix + name,
            has_new=not self.is_incomplete,
        )

    def _render_python_interface_head(
        self, cprefix: str, all_propertys_rendered: bool = False
    ) -> str:
        from . import tree

        assert isinstance(self, tree.Record)
        global wrapper_class_impl_base_template
        global python_interface_record_properties_name
        name = self.renamer(self.global_name(self.sep))
        template = Cython.Tempita.Template(wrapper_class_impl_base_template)
        return template.substitute(
            name=name,
            cname=cprefix + name,
            has_new=not self.is_incomplete,
            defaults=self._defaults if self.has_defaults else {},
            properties_name=python_interface_record_properties_name,
            all_properties_rendered=all_propertys_rendered,
            is_union=self.c_record_kind == "union",
        )

    def set_defaults(self, **kwargs):
        """Set the defaults for certain variables."""
        setattr(self, "_defaults", kwargs)

    @property
    def has_defaults(self):
        return hasattr(self, "_defaults")

    @property
    def has_python_body_epilog(self):
        return hasattr(self, "_python_body_epilog")

    def append_to_python_body(self, code: str):
        """Append additional code to the generated Python type's body.

        Append additional code to the Python type's body
        Provide dedented input, the correct indent is added by this routine.
        """
        if not self.has_python_body_epilog:
            setattr(self, "_python_body_epilog", [])
        self._python_body_epilog.append(code)

    def render_python_interface_impl(self, cprefix: str) -> str:
        from . import tree

        assert isinstance(self, tree.Record)
        global python_interface_record_properties_name
        global indent

        rendered_property_names = []
        all_properties_rendered = True
        for field in self.fields:
            prop = field.render_python_property(cprefix)
            if len(prop.strip()):
                rendered_property_names.append(field.cython_name)
                self.append_to_python_body(prop)
            else:
                all_properties_rendered = False
        self.append_to_python_body(
            textwrap.dedent(
                f"""\
        @staticmethod
        def {python_interface_record_properties_name}():
            return [{','.join(['"'+a+'"' for a in rendered_property_names])}]
        """
            )
        )
        if self.c_record_kind == "struct":
            self.append_to_python_body(
                textwrap.dedent(
                    f"""\
            def __contains__(self,item):
                properties = self.{python_interface_record_properties_name}()
                return item in properties
                
            def __getitem__(self,item):
                properties = self.{python_interface_record_properties_name}()
                if isinstance(item,int):
                    if item < 0 or item >= len(properties):
                        raise IndexError()
                    return getattr(self,properties[item])
                raise ValueError("'item' type must be 'int'")
            """
                )
            )
        result = self._render_python_interface_head(cprefix, all_properties_rendered)
        result += textwrap.indent("\n".join(self._python_body_epilog), indent)
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

    @property
    def python_base_class_name(self):
        global python_interface_int_enum_base_class_name_template
        return python_interface_int_enum_base_class_name_template.format(name=self.cython_global_name)

    def render_python_interface_impl(self, cprefix: str):
        """Renders an enum.IntEnum class.

        Note:
            Does not create an enum.IntEnum class but only exposes the enum constants
            from the Cython package corresponding to the cprefix if the
            Enum is anonymous.
        """
        from . import tree

        assert isinstance(self, tree.Enum)
        global indent
        global python_interface_int_enum_base_class

        if self.is_anonymous:
            for child_cursor in self.cursor.get_children():
                name = self.renamer(child_cursor.spelling)
                self.docstring_attributes.append(
                        textwrap.dedent(
                                f"""\
                                {name}:
                                    Enum constant.
                                """
                        )
                    )
            return "\n".join(self._render_python_enums(cprefix))
        else:
            name = self.cython_global_name
            base_class_name = self.python_base_class_name
            
            result = textwrap.dedent(f"""\
               class {base_class_name}({python_interface_int_enum_base_class}):
                   \"""Empty enum base class that allows subclassing.
                   \"""
                   pass
               class {name}({base_class_name}):
               """
            )
            result += textwrap.indent(
                "\n".join(self._render_python_enums(cprefix)), indent
            )
            # add methods
            enum_type = self.cursor.enum_type.get_canonical().spelling
            ctypes_map = {
                "short": "ctypes.c_short",
                "unsigned short": "ctypes.c_ushort",
                "int": "ctypes.c_int",
                "unsigned int": "ctypes.c_uint",
            }
            result += textwrap.indent(
                textwrap.dedent(
                    f"""\
                
                @staticmethod
                def ctypes_type():
                    \"""The type of the enum constants as ctypes type.\"""
                    return {ctypes_map[enum_type]} 
                """
                ),
                indent,
            )
            return result


class TypedefMixin(CythonMixin, Typed):
    def render_c_interface(self):
        from . import tree

        assert isinstance(self, tree.Typedef)
        """Returns a Cython binding for this Typedef.
        """
        underlying_type_name = self.global_typename(self.sep, self.renamer)
        name = self._cython_and_c_name(self.name)

        return f"ctypedef {underlying_type_name} {name}"

    def render_python_interface_decl(self, cprefix: str) -> str:
        from . import tree

        assert isinstance(self, tree.Typedef)
        return None

    def render_python_interface_impl(self, cprefix: str) -> str:
        from . import tree

        assert isinstance(self, tree.Typedef)
        name = self.cython_global_name
        if self.is_pointer_to_record(degree=(0,-1)) or self.is_pointer_to_enum(degree=(0,-1)):
            aliased = self.renamer(self.typeref.global_name(self.sep))
            self.docstring_attributes.append(
                textwrap.dedent(
                        f"""\
                        {name}:
                            alias of {self.to_sphinx_pyobj(aliased)}
                        """
                )
            )
            return f"{name} = {aliased}"
        return None


class FunctionPointerMixin(CythonMixin):
    def render_c_interface(self):
        """Returns a Cython binding for this Typedef."""
        from . import tree

        assert isinstance(self, tree.FunctionPointer)
        parm_types = ",".join([parm.cython_global_typename for parm in self.parms])
        underlying_type_name = self.renamer(self.canonical_result_typename)
        typename = self.cython_global_name  # might be AnonymousFunctionPointer
        return f"ctypedef {underlying_type_name} (*{typename}) ({parm_types})"

    def render_python_interface_decl(self, cprefix: str) -> str:
        from . import tree

        assert isinstance(self, tree.FunctionPointer)
        global wrapper_class_decl_template
        name = self.cython_global_name
        cname = cprefix + name
        template = Cython.Tempita.Template(wrapper_class_decl_template)
        return template.substitute(
            name=name,
            cname=cname,
            cptr_type=cname,  # type is already a pointer
            has_new=False,
        )

    def render_python_interface_impl(self, cprefix: str) -> str:
        from . import tree

        assert isinstance(self, tree.FunctionPointer)
        global wrapper_class_impl_base_template
        name = self.cython_global_name
        cname = cprefix + name
        template = Cython.Tempita.Template(wrapper_class_impl_base_template)
        return template.substitute(
            name=name,
            cname=cname,
            cptr_type=cname,  # type is already a pointer
            is_funptr=True,
            has_new=False,
        )


class TypedefedFunctionPointerMixin(FunctionPointerMixin):
    pass


class AnonymousFunctionPointerMixin(FunctionPointerMixin):
    pass


class ParmMixin(CythonMixin, Typed):
    def __init__(self):
        global DEFAULT_PTR_COMPLICATED_TYPE_HANDLER
        CythonMixin.__init__(self)
        self.ptr_rank = control.DEFAULT_PTR_RANK
        self.ptr_intent = control.DEFAULT_PTR_PARM_INTENT
        self.ptr_complicated_type_handler = DEFAULT_PTR_COMPLICATED_TYPE_HANDLER

    @property
    def cython_repr(self):
        from . import tree

        assert isinstance(self, tree.Parm)
        typename = self.cython_global_typename
        name = self.cython_name
        return f"{typename} {name}"

class FunctionMixin(CythonMixin, Typed):
    
    def _raw_comment_cleaned(self):
        from . import tree

        assert isinstance(self, tree.Function)
        if self.raw_comment != None:
            cleaned_raw_comment = self.raw_comment_cleaner(self.raw_comment)
            return doxyparser.remove_doxygen_cpp_comments(cleaned_raw_comment)
        else:
            return ""

    def _raw_comment_as_python_comment(self):
        from . import tree

        assert isinstance(self, tree.Function)
        if self.raw_comment != None:
            comment = self._raw_comment_cleaned()
            return "".join(["# " + l for l in comment.splitlines(keepends=True)])
        else:
            return ""

    @property
    def has_python_body_prolog(self):
        return hasattr(self, "_python_body_prolog")
    
    @property
    def has_python_body_epilog(self):
        return hasattr(self, "_python_body_epilog")

    def python_body_prepend_before_c_interface_call(self,code: str):
        if not self.has_python_body_prolog:
            setattr(self, "_python_body_prolog", [])
        self._python_body_prolog.append(code)

    def python_body_prepend_before_return(self,code: str):
        if not self.has_python_body_epilog:
            setattr(self, "_python_body_epilog", [])
        self._python_body_epilog.append(code)

    @property
    def _has_funptr_parm(self):
        from . import tree

        assert isinstance(self, tree.Function)
        for node in self.walk():
            if isinstance(node, tree.Parm):
                if isinstance(node.typeref, tree.FunctionPointer):
                    return True
        return False

    def render_c_interface(self, modifiers_front=""):
        from . import tree

        assert isinstance(self, tree.Function)
        typename = self.cython_global_typename
        name = self.cython_name
        parm_decls = ",".join([parm.cython_repr for parm in self.parms])
        modifiers = "" if self._has_funptr_parm else " nogil"
        return f"""\
{self._raw_comment_as_python_comment().rstrip()}
{modifiers_front}{typename} {name}({parm_decls}){modifiers}
"""

    def render_cython_lazy_loader_decl(self):
        return self.render_c_interface(modifiers_front="cdef ")

    @property
    def cython_funptr_name(self):
        global c_interface_funptr_name_template
        return c_interface_funptr_name_template.format(name=self.cython_name)

    def render_cython_lazy_loader_def(self):
        from . import tree

        assert isinstance(self, tree.Function)
        funptr_name = self.cython_funptr_name

        parm_types = ",".join([parm.cython_global_typename for parm in self.parms])
        parm_names = ",".join(self.parm_names(self.renamer))
        typename = self.global_typename(self.sep, self.renamer, prefer_canonical=True)
        modifiers = "" if self._has_funptr_parm else " nogil"
        return f"""\
cdef void* {funptr_name} = NULL
{self.render_cython_lazy_loader_decl().strip()}:
    global {funptr_name}
    __init_symbol(&{funptr_name},"{self.name}")
    return (<{typename} (*)({parm_types}){modifiers}> {funptr_name})({parm_names})
"""

    @staticmethod
    def _render_doxygen_section_body(section,single_level_indent,outer_indent) -> str:
        """Renders the body of a doxygen section.
        """
        result = ""
        for block in section.blocks:
            if isinstance(block,doxyparser.TextBlock):
                # variants we've seen
                # \note: texttext => firstline == ": texttext"
                # \note texttext
                # \note texttext
                #    texttext
                lines = block.transformed_text.lstrip(":\n\t ").rstrip().splitlines()
                if len(lines):
                    firstline = lines[0]
                    other_lines = lines[1:]
                    if len(other_lines):
                        transformed_text = (
                            firstline + "\n"
                            + textwrap.dedent("\n".join(other_lines))
                        )
                    else:
                        transformed_text = firstline
                    result += textwrap.indent(transformed_text,outer_indent) + "\n"
            elif isinstance(block,doxyparser.VerbatimBlock):
                result += f"\n{outer_indent}.. code-block::"
                if block.kind == "code": # \code { lang } TEXT \endcode
                    if block.tokens == 6:
                        lang = block.tokens[2][1:]
                        result += lang
                result += "\n\n"
                inner_indent = outer_indent+" "*3
                code = textwrap.dedent(block.code)
                result += textwrap.indent(code,inner_indent) + "\n\n"
            elif isinstance(block,doxyparser.MathBlock):
                inner_indent = outer_indent+" "*3
                result += f"\n{outer_indent}.. math::\n"
                if block.env != None:
                    result += "{inner_indent}:nowrap:"
                    result += rf"{inner_indent}\begin{{{block.env}}}\n"
                result += "\n"
                code = textwrap.dedent(block.code)
                result += textwrap.indent(code,inner_indent).rstrip() + "\n"
                if block.env != None:
                    result += rf"{inner_indent}\end{{{block.env}}}\n"
                result += "\n"
        return result
    
    @staticmethod
    def _dedent_first_line(text: str) -> str:
        lines = text.splitlines(keepends=True)
        result = lines[0].strip(" \t")
        if len(lines) > 1:
            result += "".join(lines[1:])
        return result

    def _python_interface_retval_typename(self):
        """Returns a docstring expression for the return value type.
        """
        
        typename = self.cython_global_typename
        if self.is_void:
            return "None"
        elif ( self.is_basic_type or self.is_pointer_to_char(degree=1) ):
            return CYTHON_AUTOCONV_TO_PYTHON_TYPES(typename)
        elif self.is_enum or self.is_record or self.is_union:
            return typename
        else:
            return None

    def _create_python_docstring(self,out_arg_names,parm_python_types):
        """Converts doxygen comment to a Python docstring using the doxyparser API.
        """
        # TODO handle groups; issue detecting addgroup; detecting ingroup is easier
        from . import tree

        assert isinstance(self, tree.Function)

        # doxygen parser
        translater = doxyparser.DoxygenGrammar()
        translater.escaped.setParseAction(doxyparser.format.PythonDocstrings.escaped)
        translater.with_word.setParseAction(doxyparser.format.PythonDocstrings.with_word)
        translater.fdollar.setParseAction(doxyparser.format.PythonDocstrings.fdollar)
        translater.frnd.setParseAction(doxyparser.format.PythonDocstrings.frnd)
        
        def reference_(tokens):
            global python_interface_pyobj_role_template
            reference: str = tokens[0].replace("#",".")
            reference = reference.replace("::",".")
            return python_interface_pyobj_role_template.format(name=reference.lstrip('.'))
        translater.see_reference.setParseAction(reference_)
        translater.in_text_reference.setParseAction(reference_)

        def other_parse_action(tokens):
            cmd = tokens[0][1:]
            if cmd == "ref":
                return f"``{tokens[1]}`` "
            return [] # suppress all others
        translater.other.setParseAction(other_parse_action)

        tree = translater.parse_structure(self._raw_comment_cleaned())
        sections = list(tree.children)
        # brief
        doxygen_brief = next((sec for sec in sections if sec.kind in ("brief","short")),None)
        if doxygen_brief != None:
            # clip other sections before the brief, TODO make option
            sections = sections[sections.index(doxygen_brief)+1:]
            if len(doxygen_brief[0]) > 1:
                warnings.warn(f"function {self.name}: doxygen: more than one text/verbatim/math block in section 'brief'. Ignore others.")
            if not isinstance(doxygen_brief.first_block,doxyparser.TextBlock):
                raise RuntimeError(f"function {self.name}: doxygen: expected single text block in section 'brief'")
            docstring_body = doxygen_brief.first_block.transformed_text.strip() +"\n\n"
        else:
            docstring_body = "(No short description)\n\n"
        
        # other sections
        single_level_indent = " "*4
        docstring_returns = []
        docstring_args = []
        docstring_out_arg_returns = []
        undocumented_parms = [parm.name for parm in self.parms]
        for section in sections:
            if section.kind in (
              "result",
              "return",
              "returns",
            ):
                descr = self._render_doxygen_section_body(section,single_level_indent,outer_indent=single_level_indent).lstrip("-* \t")
                docstring_returns.append(descr)
            elif section.kind == "param":
                # ['\\param', '[in]', 'param1', 'Description text is here.']
                names = section.tokens[2]
                # Args:
                #    <arg>: line1
                #       line2
                # ^ hence, 2x indent for descr
                descr = self._dedent_first_line(self._render_doxygen_section_body(section,single_level_indent,outer_indent=single_level_indent*2).rstrip()+"\n")
                descr = descr.lstrip("-*")
                dir = (f"**{section.tokens[1].replace(' ','')}** ") if section.tokens[1] != None else ""
                for name in names:
                    if not len(descr.strip()):
                        warnings.warn(f"function {self.name}: doxygen: doxygen param '{name}' has empty documentation")
                    #
                    if not name in undocumented_parms:
                        type_info = ""
                        warnings.warn(f"function {self.name}: doxygen: doxygen param '{name}' is not part of function signature")
                    else:
                        type_info = "/".join([CythonMixin.to_sphinx_pyobj(p) for p in parm_python_types[name].split("/")])
                        undocumented_parms.remove(name)
                    if name in out_arg_names:
                        docstring_out_arg_returns.append(f"{single_level_indent}{type_info}: {descr}")
                    else:
                        if len(type_info):
                            type_info = f" ({type_info})"
                        docstring_args.append((name+type_info,dir,descr))
            else:
                docstring_body += "\n"
                if section.kind in ("details","details*"):
                    outer_indent = ""
                    if self.name == "hipsparseScsrmm2":
                        print(section.kind)
                        print(section.first_block.text)
                else:
                    docstring_body += f"\n{section.kind[0].upper() + section.kind[1:]}:\n"
                    outer_indent = single_level_indent
                body = self._render_doxygen_section_body(section,single_level_indent,outer_indent)
                if section.kind in ("see","sa"):
                    docstring_body += translater.see_reference.transformString(body)
                else:
                    docstring_body += body
        # Args
        if len(undocumented_parms):
            for name in undocumented_parms:
                warnings.warn(f"function {self.name}: doxygen: function arg '{name}' is not documented")

        if len(docstring_args):
            docstring_body += "\nArgs:\n"

            for (name, dir, descr) in docstring_args:
                docstring_body += f"{single_level_indent}{name}: {dir}{descr}\n"
        
        # Return values
        retval_typename = self._python_interface_retval_typename()
        if not len(docstring_returns) and not self.is_void:
            warnings.warn(f"function {self.name}: doxygen: undocumented return value")
            if retval_typename != None:
                docstring_returns.append(
                    CythonMixin.to_sphinx_pyobj(retval_typename)
                )
        elif len(docstring_returns):
            first_entry = docstring_returns[0].lstrip(" \t\n*-")
            docstring_returns[0] = f"{CythonMixin.to_sphinx_pyobj(retval_typename)}: {first_entry}"
        docstring_returns += docstring_out_arg_returns # add the additional return parameter
       
        if len(docstring_returns):
            docstring_body += "\nReturns:\n"
            if len(docstring_returns) > 1 or python_interface_always_return_tuple:
                docstring_body += f"{single_level_indent}A {self.to_sphinx_pyobj('tuple')} of size {len(docstring_returns)} that contains (in that order):\n\n"
                prefix = "* "
            else:
                prefix = ""
            for descr in docstring_returns:
                docstring_body += textwrap.indent(
                    prefix + descr.lstrip(" \t\n*-"),
                    single_level_indent
                ).rstrip() + "\n"
        # remove multiple blank lines
        docstring_body = self.docstring_cleaner(docstring_body)
        # remove multiple blank lines
        docstring_body = re.sub(r"(\n\s*)+\n+", "\n\n", docstring_body).rstrip()
        return f'r"""{docstring_body}\n"""' # r required if verbatim/code is in body

    def _analyze_parms(self, cprefix: str):
        from . import tree

        parm_python_types = {} # Python type names of signature and out args, always use original typename as key
        sig_args = []  # argument definitions that appear in the signature
        out_args = []  # return values, might include conversions
        out_parms = (
            []
        )  # names of the return values, required for identifying doxygen parameters
        c_interface_call_args = []  # arguments that are passed to the C interface
        prolog = []  # additional code before the C interface call

        def emit_datahandle_(parm_typename: str, parm: tree.Parm, cprefix: str = ""):
            global indent
            nonlocal sig_args
            nonlocal c_interface_call_args
            nonlocal parm_python_types

            parm_name = parm.cython_name
            handler_name = parm.ptr_complicated_type_handler(parm)
            sig_args.append(f"object {parm_name}")
            c_interface_call_args.append(
                f"\n{indent*2}<{cprefix}{parm_typename}>{handler_name}.from_pyobj({parm_name})._ptr"
            )
            parm_python_types[parm.name] = f"{handler_name}/object"

        def emit_data_handle_for_ptr_to_void_basic_enum_type_(parm: tree.Parm, cprefix: str):
            parm_typename = (
                parm.cython_global_typename
                if parm.has_typeref
                else parm.renamer(parm.cursor.type.get_canonical().spelling)  # TODO verify might be no Python/Cython keyword
            )
            emit_datahandle_(
                parm_typename,
                parm,
                cprefix=cprefix
                if not parm.is_innermost_canonical_type_layer_of_basic_type_or_void
                else "",
            )

        def handle_out_ptr_parm(parm: tree.Parm):
            nonlocal out_args
            nonlocal out_parms
            nonlocal c_interface_call_args
            nonlocal prolog
            nonlocal cprefix

            parm_name = parm.cython_name
            out_parms.append(parm) # append original name as we need to compare vs the documentation
            
            if parm.is_pointer_to_basic_type(degree=1) or parm.is_pointer_to_char(
                degree=2
            ):
                typehandler = parm._type_handler.create_from_layer(1, canonical=True)
                parm_typename = typehandler.clang_type.spelling
                prolog.append(f"cdef {parm_typename} {parm_name}")
                out_args.append(parm_name)
                c_interface_call_args.append(f"&{parm_name}")
                parm_python_types[parm.name] = CYTHON_AUTOCONV_TO_PYTHON_TYPES(parm_typename)
            elif parm.is_pointer_to_enum(degree=1):
                parm_typename = parm.lookup_innermost_type().cython_name
                prolog.append(f"cdef {cprefix}{parm_typename} {parm_name}")
                c_interface_call_args.append(f"&{parm_name}")
                out_args.append(
                    f"{parm_typename}({parm_name})"
                )  # conversion from c... type required
                parm_python_types[parm.name] = parm_typename
            elif parm.is_pointer_to_record(
                degree=2
            ) or parm.is_pointer_to_function_proto(degree=2):
                parm_typename = parm.lookup_innermost_type().cython_name
                prolog.append(f"{parm_name} = {parm_typename}.from_ptr(NULL)")
                c_interface_call_args.append(f"&{parm_name}._ptr")
                out_args.append(parm_name)
                parm_python_types[parm.name] = parm_typename
            elif parm.is_pointer_to_basic_type(degree=-2) or parm.is_pointer_to_void(
                degree=-2
            ):
                parm_typename = parm.cursor.type.get_canonical().spelling
                handler_name = parm.ptr_complicated_type_handler(parm)
                prolog.append(f"{parm_name} = {handler_name}.from_ptr(NULL)")
                c_interface_call_args.append(
                    f"\n{indent*2}<{parm_typename}>&{parm_name}._ptr"
                )
                parm_python_types[parm.name] = f"{handler_name}/object"
                out_args.append(parm_name)
            else:
                # If the argument was not removed from the parameter list,
                # we did not add an additional return value.
                # Hence, we remove the previously added original 
                # name (see top of routine) from the out_arg_names list.
                out_parms.pop(-1)

        def handle_in_inout_ptr_(parm: tree.Parm):
            global indent
            nonlocal c_interface_call_args
            nonlocal sig_args
            nonlocal cprefix

            parm_name = parm.cython_name
            if parm.is_pointer_to_record(
                degree=1, incomplete_array=True
            ) or parm.is_pointer_to_function_proto(degree=1, incomplete_array=True):
                parm_typename = parm.lookup_innermost_type().cython_name
                sig_args.append(f"object {parm_name}")
                parm_python_types[parm.name] = f"{parm_typename}/object" # use original name as key
                c_interface_call_args.append(
                    f"\n{indent*2}{parm_typename}.from_pyobj({parm_name})._ptr"
                )
            elif parm.is_pointer_to_record(
                degree=-2, incomplete_array=True
            ) or parm.is_pointer_to_function_proto(degree=-2, incomplete_array=True):
                parm_typename = parm.cython_global_typename
                emit_datahandle_(parm_typename, parm, cprefix)
            elif (
                parm.is_pointer_to_void(degree=-1, incomplete_array=True)
                or parm.is_pointer_to_basic_type(degree=-1, incomplete_array=True)
                or parm.is_pointer_to_enum(degree=-1, incomplete_array=True)
            ):
                emit_data_handle_for_ptr_to_void_basic_enum_type_(parm, cprefix)
            else:
                assert False, "should not be entered"

        def handle_value_parm_(parm: tree.Parm):
            if parm.is_autoconverted_by_cython:
                c_interface_call_args.append(f"{parm_name}")
                sig_args.append(parm.cython_repr)
                # TODO do the autoconversion
                parm_python_types[parm.name] = "/".join(CYTHON_AUTOCONV_FROM_PYTHON_TYPES(parm.cython_global_typename)) # use original name as key
            elif (
                parm.is_enum
            ):  # enums are not modelled as cdef class, so we cannot specify them as type
                parm_base_class_name = parm.lookup_innermost_type().python_base_class_name
                sig_args.append(f"object {parm_name}")
                prolog.append(
                    textwrap.dedent(
                        f"""\
                    if not isinstance({parm_name},{parm_base_class_name}):
                        raise TypeError("argument '{parm_name}' must be of type '{parm_base_class_name}'")\
                    """
                    )
                )
                c_interface_call_args.append(f"{parm_name}.value")
                parm_python_types[parm.name] = parm.cython_global_typename
            elif parm.is_record:
                parm_typename = parm.lookup_innermost_type().cython_name
                sig_args.append(f"object {parm_name}")
                c_interface_call_args.append(
                    f"\n{indent*2}{parm_typename}.from_pyobj({parm_name})._ptr[0]"
                )
                parm_python_types[parm.name] = parm_typename

        for parm in self.parms:
            parm_name = parm.cython_name
            assert isinstance(parm, ParmMixin)
            if parm.is_ptr:
                if parm.is_out_ptr:
                    assert parm.is_indirection  # make exception
                    handle_out_ptr_parm(parm)
                elif parm.is_inout_ptr:
                    handle_in_inout_ptr_(parm)
                else:  # in ptr
                    if parm.is_pointer_to_char(degree=1):  # autoconverted by Cython
                        c_interface_call_args.append(parm_name)
                        sig_args.append(parm.cython_repr)
                        parm_python_types[parm.name] = "bytes"
                    else:
                        handle_in_inout_ptr_(parm)
            else:  # no ptr
                handle_value_parm_(parm)

        fully_specified = len(list(self.parms)) == len(c_interface_call_args)
        if not fully_specified:
            warnings.warn("_codegen.cython: not all parameters could be classified for function {self.name}")
        setattr(self, "is_python_code_complete", fully_specified)
        assert len(parm_python_types) == len(c_interface_call_args), f"{self.name=} {str(parm_python_types)=}"

        return (
            fully_specified,
            sig_args,
            out_args,
            out_parms,
            c_interface_call_args,
            prolog,
            parm_python_types,
        )

    @property
    def _python_interface_retval(self):
        global python_interface_retval_template
        return python_interface_retval_template.format(name=self.cython_name)

    def _render_python_interface_c_interface_call(
        self, cprefix: str, call_args: list, out_args: list
    ):
        from . import tree

        typename = self.cython_global_typename
        retvalname = self._python_interface_retval
        comma = ","
        c_interface_call = f"{cprefix}{self.cython_name}({comma.join(call_args)})"
        assert isinstance(self, tree.Function)
        if self.is_void:
            return c_interface_call
        elif (self.is_basic_type or self.is_pointer_to_char(degree=1)):
            out_args.insert(0, retvalname)
            return f"cdef {typename} {retvalname} = {c_interface_call}"
        elif self.is_enum:
            out_args.insert(0, retvalname)
            return f"{retvalname} = {typename}({c_interface_call})"
        elif self.is_record or self.is_union:
            out_args.insert(0, retvalname)
            return f"{retvalname} = {typename}.from_value({c_interface_call})"
        else:
            warnings.warn(f"_codegen.cython: return value of function {self.name} could not be classified")
            return ""

    def render_python_interface_impl(self, cprefix: str) -> str:
        (
            fully_specified,
            sig_args,
            out_args,
            out_parms,  # required for parsing parameter documentation
            call_args,
            prolog,
            parm_python_types,
        ) = self._analyze_parms(cprefix)

        global python_interface_always_return_tuple

        result = "@cython.embedsignature(True)\n"
        result += (
            f"def {self.cython_name}({', '.join(sig_args)}):\n"
            + textwrap.indent(self._create_python_docstring([p.name for p in out_parms],parm_python_types), indent).rstrip()
            + "\n"
        )
        if self.has_python_body_prolog:
            prolog += self._python_body_prolog
        epilog = []
        if self.has_python_body_epilog:
            epilog += self._python_body_epilog
        if len(prolog):
            result += textwrap.indent("\n".join(prolog), indent).rstrip() + "\n"
        if fully_specified:
            result += f"{indent}{self._render_python_interface_c_interface_call(cprefix,call_args,out_args)}"
            result += f"{indent}# fully specified\n"
            if len(epilog):
                result += textwrap.indent("\n".join(epilog), indent).rstrip() + "\n"
            if len(out_args) > 1:
                comma = ","
                result += f"{indent}return ({comma.join(out_args)})\n"
            elif len(out_args):
                if python_interface_always_return_tuple:
                    result += f"{indent}return ({out_args[0]},)\n"
                else:
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
        ptr_parm_intent: callable = control.DEFAULT_PTR_PARM_INTENT,
        ptr_rank: callable = control.DEFAULT_PTR_RANK,
        ptr_complicated_type_handler=DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
        renamer: callable = DEFAULT_RENAMER,
        raw_comment_cleaner: callable = DEFAULT_RAW_COMMENT_CLEANER,
        docstring_cleaner: callable = DEFAULT_DOCSTRING_CLEANER,
        warn_mode: control.Warnings = control.Warnings.IGNORE,
    ):
        from . import tree

        root = tree.from_libclang_translation_unit(translation_unit, warn_mode)
        return CythonBackend(
            root,
            filename,
            node_filter,
            macro_type,
            ptr_parm_intent,
            ptr_rank,
            ptr_complicated_type_handler,
            renamer,
            raw_comment_cleaner,
            docstring_cleaner,
        )

    def __init__(
        self,
        root,
        filename: str,
        node_filter: callable = control.DEFAULT_NODE_FILTER,
        macro_type: callable = DEFAULT_MACRO_TYPE,
        ptr_parm_intent: callable = control.DEFAULT_PTR_PARM_INTENT,
        ptr_rank: callable = control.DEFAULT_PTR_RANK,
        ptr_complicated_type_handler=DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
        renamer: callable = DEFAULT_RENAMER,
        raw_comment_cleaner: callable = DEFAULT_RAW_COMMENT_CLEANER,
        docstring_cleaner: callable = DEFAULT_DOCSTRING_CLEANER,
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
        self.ptr_parm_intent = ptr_parm_intent
        self.ptr_rank = ptr_rank
        self.ptr_complicated_type_handler = ptr_complicated_type_handler
        self.renamer = renamer
        self.raw_comment_cleaner = raw_comment_cleaner
        self.docstring_cleaner = docstring_cleaner

    def walk_filtered_nodes(self):
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
                setattr(node,"raw_comment_cleaner",self.raw_comment_cleaner)
                setattr(node,"docstring_cleaner",self.docstring_cleaner)
                if isinstance(node, MacroDefinitionMixin):
                    setattr(node, "macro_type", self.macro_type)
                elif isinstance(node, (FieldMixin)):
                    setattr(node, "ptr_rank", self.ptr_rank)
                    setattr(
                        node,
                        "ptr_complicated_type_handler",
                        self.ptr_complicated_type_handler,
                    )
                elif isinstance(node, (ParmMixin)):
                    setattr(node, "ptr_rank", self.ptr_rank)
                    setattr(node, "ptr_intent", self.ptr_parm_intent)
                    setattr(
                        node,
                        "ptr_complicated_type_handler",
                        self.ptr_complicated_type_handler,
                    )
                # yield relevant nodes
                if not isinstance(node, (FieldMixin, ParmMixin)):
                    if self.node_filter(node):
                        yield node

    def create_c_interface_decl_part(self, runtime_linking: bool = False):
        """Returns the content of a Cython bindings file.

        Creates the content of a Cython bindings file.
        Contains Cython declarations per C declaration
        plus helper types that have been introduced for nested enum/struct/union types.

        Note:
            Anonymous anonymous types for which we have a tree node with
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
        for node in self.walk_filtered_nodes():
            if (
                (runtime_linking and isinstance(node, FunctionMixin))
                or isinstance(node, AnonymousFunctionPointerMixin)
                or (
                    isinstance(
                        node, (tree.AnonymousEnum, tree.AnonymousStruct, tree.AnonymousUnion)
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
        for node in self.walk_filtered_nodes():
            if isinstance(node, FunctionMixin):
                result.append(node.render_cython_lazy_loader_decl(self.renamer))
        return result

    def create_cython_lazy_loader_defs(self, dll: str):
        # TODO: Add compiler? switch to switch between MS and Linux loaders
        # TODO: Add compiler? switch to switch between HIP and CUDA backends?
        # Might be possible to implement this via the renamer and generating multiple modules
        result = []
        lib_handle = "_lib_handle"
        result.append(
            textwrap.dedent(
                f"""\
            cimport hip._util.posixloader as loader
            cdef void* {lib_handle} = NULL
            
            cdef void __init() nogil:
                global {lib_handle}
                if {lib_handle} == NULL:
                    with gil:
                        {lib_handle} = loader.open_library(\"{dll}\")

            cdef void __init_symbol(void** result, const char* name) nogil:
                global {lib_handle}
                if {lib_handle} == NULL:
                    __init()
                if result[0] == NULL:
                    with gil:
                        result[0] = loader.load_symbol({lib_handle}, name) 
            """
            )
        )
        for node in self.walk_filtered_nodes():
            if isinstance(node, FunctionMixin):
                result.append("\n" + node.render_cython_lazy_loader_def())
        return result

    def render_c_interface_decl_part(self, runtime_linking: bool = False):
        """Returns the Cython bindings file content for the given headers."""
        nl = "\n\n"
        return nl.join(self.create_c_interface_decl_part(runtime_linking))

    def render_c_interface_impl_part(
        self, runtime_linking: bool = False, dll: str = None
    ):
        """Returns the Cython bindings file content for the given headers."""
        nl = "\n"
        if runtime_linking:
            if dll is None:
                raise ValueError(
                    "argument 'dll' must not be 'None' if 'runtime_linking' is set to 'True'"
                )
            return nl.join(self.create_cython_lazy_loader_defs(dll))
        else:
            return ""

    def create_python_interface_decl_part(self, cmodule):
        """Renders Python interfaces in Cython."""
        from . import tree

        result = []
        cprefix = f"{cmodule}."
        for node in self.walk_filtered_nodes():
            contrib = node.render_python_interface_decl(cprefix=cprefix)
            if contrib != None:
                result.append(contrib)
        return result

    def create_python_interface_impl_part(self, cmodule):
        """Renders Python interfaces in Cython."""
        from . import tree

        result = []
        cprefix = f"{cmodule}."
        docstring_attributes = []
        for node in self.walk_filtered_nodes():
            setattr(node,"docstring_attributes",docstring_attributes)
            contrib = node.render_python_interface_impl(cprefix=cprefix)
            if contrib != None:
                result.append(contrib)
        return result, docstring_attributes

    def render_python_interface_decl_part(self, cython_c_bindings_module: str):
        """Returns the Python interface file content for the given headers."""
        result = self.create_python_interface_decl_part(cython_c_bindings_module)
        nl = "\n\n"
        return f"""\
{nl.join(result)}"""

    def render_python_interface_impl_part(self, cython_c_bindings_module: str):
        """Returns the Python interface file content for the given headers."""
        result, docstring_attributes = self.create_python_interface_impl_part(cython_c_bindings_module)
        nl = "\n\n"
        return ( f"""\
{nl.join(result)}""", docstring_attributes )


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
        ptr_parm_intent: callable = control.DEFAULT_PTR_PARM_INTENT,
        ptr_rank: callable = control.DEFAULT_PTR_RANK,
        ptr_complicated_type_handler=DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
        renamer: callable = DEFAULT_RENAMER,
        raw_comment_cleaner: callable = DEFAULT_RAW_COMMENT_CLEANER,
        docstring_cleaner: callable = DEFAULT_DOCSTRING_CLEANER,
        warn_mode=control.Warnings.WARN,
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
        global default_c_interface_decl_preamble
        global default_python_interface_decl_preamble
        self.pkg_name = pkg_name
        self.include_dir = include_dir
        self.header = header
        self.runtime_linking = runtime_linking
        self.dll = dll
        self.cflags = cflags
        self.c_interface_decl_preamble = default_c_interface_decl_preamble
        self.c_interface_impl_preamble = default_c_interface_impl_preamble
        self.python_interface_decl_preamble = default_python_interface_decl_preamble
        self.python_interface_impl_preamble = default_python_interface_impl_preamble

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
            ptr_complicated_type_handler,
            renamer,
            raw_comment_cleaner,
            docstring_cleaner,
            warn_mode,
        )

    def write_package_files(self, output_dir: str = None):
        """Write all files required to build this Cython/Python package.

        Args:
            pkg_name (str): Name of the package that should be generated. Influences filesnames.
        """
        python_interface_decl_preamble = (
            self.python_interface_decl_preamble + f"\nfrom . cimport c{self.pkg_name}\n"
        )

        with open(f"{output_dir}/c{self.pkg_name}.pxd", "w") as outfile:
            outfile.write(self.c_interface_decl_preamble)
            outfile.write(
                self.backend.render_c_interface_decl_part(
                    runtime_linking=self.runtime_linking
                )
            )
        with open(f"{output_dir}/c{self.pkg_name}.pyx", "w") as outfile:
            outfile.write(self.c_interface_impl_preamble)
            outfile.write(
                self.backend.render_c_interface_impl_part(
                    runtime_linking=self.runtime_linking, dll=self.dll
                )
            )
        with open(f"{output_dir}/{self.pkg_name}.pxd", "w") as outfile:
            outfile.write(python_interface_decl_preamble)
            outfile.write(
                self.backend.render_python_interface_decl_part(f"c{self.pkg_name}")
            )

        with open(f"{output_dir}/{self.pkg_name}.pyx", "w") as outfile:
            content, docstring_attributes = self.backend.render_python_interface_impl_part(f"c{self.pkg_name}")
            if len(docstring_attributes):
                DOCSTRING_ATTRIBS = "Attributes:\n" + textwrap.indent("\n".join(docstring_attributes)," "*4)
            outfile.write(self.python_interface_impl_preamble.replace("[ATTRIBUTES]",DOCSTRING_ATTRIBS))
            outfile.write(content)
