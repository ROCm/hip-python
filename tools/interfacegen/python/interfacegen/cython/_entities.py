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


"""Cython backend — entities submodule.

Carved out of the historical interfacegen.cython monolith.
Public callers should import from `interfacegen.cython`,
not from this submodule directly.
"""

__author__ = "Advanced Micro Devices, Inc."

import ctypes
import keyword
import logging
import os
import re
import sys
import textwrap
import typing

import clang.cindex
import Cython.Tempita

from .. import cparser, cythontemplates, doxyparser, tree
from ..support import cython as support
from ..support.recipes import control

_log = logging.getLogger("interfacegen")
from . import _defaults, _doxygen, _mixins
from ._defaults import *  # noqa: F401,F403
from ._doxygen import *  # noqa: F401,F403
from ._mixins import *  # noqa: F401,F403

__all__ = [
    'Root',
    'MacroDefinition',
    'Typed',
    'Field',
    'ParentIsRecordMixin',
    'Record',
    'Struct',
    'Union',
    'AnonymousStruct',
    'AnonymousUnion',
    'Enum',
    'AnonymousEnum',
    'Typedef',
    'ConstantArray',
    'FunctionPointer',
    'TypedefedFunctionPointer',
    'AnonymousFunctionPointer',
    'Parm',
]

class Root(tree.Root, CythonMixin):
    def __init__(self, *args, **kwargs):
        tree.Root.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)


class MacroDefinition(tree.MacroDefinition, CythonMixin):
    # via: https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#types
    # note: ulonglong == ulong == size_t on some operating systems
    ctypes_to_cython = {
        ctypes.c_bool: "bint",  # Python: bool
        ctypes.c_short: "short",  # Python: int
        ctypes.c_ushort: "unsigned short",  # Python: int
        ctypes.c_int: "int",  # Python: int
        ctypes.c_uint: "unsigned int",  # Python: int
        ctypes.c_long: "long",  # Python: int
        ctypes.c_ulong: "unsigned long",  # Python: int
        ctypes.c_longlong: "long long",  # Python: int
        ctypes.c_ulonglong: "unsigned long long",  # Python: int
        ctypes.c_size_t: "size_t",  # Python: int
        ctypes.c_float: "float",  # Python: float
        ctypes.c_double: "double",  # Python: float
    }

    def __init__(self, *args, **kwargs):
        tree.MacroDefinition.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)
        self.macro_type = (
            DEFAULT_MACRO_TYPE
        )  # type: typing.Callable[[MacroDefinition], None|bool|int|str]

    @property
    def no_right_hand_side(self):
        """If the `macro_type` user callback returns a Python bool 'True', this indicates a `#define <NAME>` without RHS."""
        return type(self.macro_type(self)) == bool

    @property
    def interpret_right_hand_side_as_str(self):
        """If the `macro_type` user callback returns the type ``str``, this
        means that the RHS should be interpreted as string literal."""
        return self.macro_type(self) == str

    @property
    def hardcoded_right_hand_side(self):
        """If the `macro_type` user callback returns a Python int or ctypes instance, this indicates that the value has been hardcoded."""
        return (
            type(self.macro_type(self)) == int
            or type(self.macro_type(self)) in MacroDefinition.ctypes_to_cython
        )

    def render_c_interface(self, is_decl=True):
        r"""Render declaration/definition for a macro.

        Uses the `macro_type` callback to infer a Cython type or a hardcoded
        value for the macro's RHS.

        Args:
            is_decl (bool, optional):
            If `True`, generate code for the declaration file (*.pxd),
            otherwise for the implementation file (*.pyx). This only affects
            hardcoded values, which must be emitted in the implementation.

        Behavior based on `macro_type(self)`:
        * `None`: log an error and return `None`.
        * `str`: interpret the RHS as a string literal and return `None` for the
          C interface; a Python `str` will be emitted in the Python interface.
        * `bool`: emit a `cdef bint <name>` without RHS.
        * `int` or ctypes integer value: emit a hardcoded `cdef <type> <name>`
          (and assign the value in the implementation).
        * any other `str`: emit `cdef <typename> <name>`.

        Returns:
            Optional[str]:
                The Cython declaration or definition for the macro,
                or None if there is an error or if the macro should be represented as a Python string literal.
        """

        # TODO: Simplify this by supporting only ctypes types when hardcoding

        assert isinstance(self, tree.MacroDefinition)
        type_or_typename_or_value = self.macro_type(self)

        if type_or_typename_or_value is None:
            _log.error(f"no type specified for macro definition {self.name}.")
            # FIXME: Introduce error modes: fail on error, ignore on error, ...
            return None
        elif type_or_typename_or_value == str:
            return None
        elif isinstance(type_or_typename_or_value, bool):
            if is_decl:
                return f"cdef bint {self._cython_and_c_name(self.name)}"
            return None
        elif isinstance(type_or_typename_or_value, str):
            if is_decl:
                typename = type_or_typename_or_value
                varname = self._cython_and_c_name(self.name)
                return f"cdef {typename} {varname}"
            return None

        # hardcoded values need to provide the value in the implementation file
        if isinstance(type_or_typename_or_value, int):
            typename = "int"
            varname = self.name
            value = type_or_typename_or_value
        elif (
            type(type_or_typename_or_value) in MacroDefinition.ctypes_to_cython
        ):
            typename = MacroDefinition.ctypes_to_cython[
                type_or_typename_or_value.__class__
            ]
            varname = self._cython_and_c_name(self.name)
            value = type_or_typename_or_value.value
        else:
            err_msg = f" unsupported macro type for {self.name}: {type_or_typename_or_value}"
            _log.error(err_msg)
            raise RuntimeError(err_msg)

        var_decl = f"cdef {typename} {varname}"
        # append right-hand side if is definition
        if is_decl:
            return var_decl
        else:
            return f"{var_decl} = {value}"

    def render_c_interface_decl(self):
        """Render the declaration part of the macro definition."""
        return self.render_c_interface(is_decl=True)

    def render_c_interface_impl(self):
        """Render the implementation part of the macro definition, which is only relevant for hardcoded values."""
        return self.render_c_interface(is_decl=False)

    def render_pyi_stub(
        self, cprefix: str, *, override_name: str = None,
        base: str = None,
    ):
        """Macro constants render as `<name>: Any` (no type info — the
        macro_type callback is best-effort and we don't pretend
        otherwise in the stub)."""
        name = override_name or self.cython_name
        if not name or not name.isidentifier():
            return None
        return [f"{name}: Any"]

    def render_python_interface_impl(self, cprefix: str, *, module_opts: dict):
        """Render a Python-facing macro definition assignment string.
        Args:
            cprefix: Prefix to apply to the C-level macro name when generating the right-hand side.
            module_opts: Per-module rendering options. ``module_opts["all"]``
                receives the macro's cython_global_name.
                ``module_opts["docstring_attributes"]`` receives the docs entry.
        Returns:
            A string of the form ``"{name} = {rhs}"`` where ``name`` is the
            renamed macro identifier and ``rhs`` is either the prefixed macro name or a reconstructed string literal for string-like macros.
        Notes:
            - If the macro expands to a string-like value
              (``type_or_typename_or_value == str``),
              this reconstructs the literal from cursor tokens to preserve the
              literal text.
            - If the macro is a numeric/bool or has no right-hand side, the
              Python type is inferred accordingly; otherwise the type is mapped
              from Cython/ctypes.
        """
        name = self.renamer(self.name)

        # derive docs entry type
        type_or_typename_or_value = self.macro_type(self)
        if isinstance(type_or_typename_or_value, (bool, int)):
            python_type = type(type_or_typename_or_value).__name__
        elif self.no_right_hand_side:
            python_type = "bool"
        else:
            # if we have a value of a ctypes type, map to Cython type first
            _maybe_ctypes_type = type(type_or_typename_or_value)
            if _maybe_ctypes_type in MacroDefinition.ctypes_to_cython:
                cython_typename = MacroDefinition.ctypes_to_cython[
                    _maybe_ctypes_type
                ]
            else:
                cython_typename = type_or_typename_or_value
            python_type = CYTHON_AUTOCONV_TO_PYTHON_TYPES(cython_typename)

        # side effect: add to docstring attributes
        module_opts["docstring_attributes"].append(
            textwrap.dedent(
                f"""\
                    {name} ({self.to_sphinx_pyobj(python_type)}):
                        Macro constant.
                    """
            )
        )
        # side effect: register in __all__
        module_opts["all"].append(self.cython_global_name)

        # derive right-hand side
        if type_or_typename_or_value == str:
            rhs_tokens = []
            in_arg_list = False
            for i, token in enumerate(self.cursor.get_tokens()):
                tk = token.spelling
                if i == 0:
                    continue
                if i == 1 and tk == "(":
                    in_arg_list = True
                if in_arg_list:
                    if tk == ")":
                        in_arg_list = False
                    continue
                rhs_tokens.append(tk)
            rhs = '"' + " ".join(rhs_tokens) + '"'
        else:
            rhs = f"{cprefix}{name}"
        return f"{name} = {rhs}"


class Typed:

    @property
    def cython_global_typename(self):
        """Cython type spelling with elaborated tags substituted in.

        Walks the Clang type layer hierarchy via
        ``Typed.render_type`` (no token splitting) and produces
        ``cython_decl()`` — leading ``const`` is preserved on the leaf;
        per-pointer ``const``/``restrict``/``volatile`` are kept attached
        to their ``*`` as Clang spells them.
        """
        assert isinstance(self, tree.Typed)
        return self.render_type(
            self.sep, self.renamer, prefer_canonical=True
        ).cython_decl()

    @property
    def cython_global_typename_no_const(self):
        """:pyattr:`cython_global_typename` with the leading leaf-level
        ``const`` stripped."""
        assert isinstance(self, tree.Typed)
        return self.render_type(
            self.sep, self.renamer, prefer_canonical=True
        ).cython_decl_no_const()

    @property
    def actual_rank(self):
        """The actual rank of the parameter, if this is an indirection."""
        return self.ptr_rank(self)

    @property
    def has_array_rank(self):

        assert isinstance(self, tree.Typed)
        if self.is_any_array:
            return True
        else:
            return self.actual_rank(self)

    @property
    def is_ptr(self):

        assert isinstance(self, tree.Parm)
        return self.get_pointer_degree(incomplete_array=True) > 0

    @property
    def is_indirection(self):
        """If this is not the actual value but an indirection.

        Returns:
            bool: If this is not the actual value but an indirection.
        """

        actual_rank = self.ptr_rank(self)
        assert isinstance(self, tree.Parm)
        return self.get_pointer_degree() > actual_rank

    @property
    def effective_ptr_intent(self):
        """The intent verdict from the rule chain, with the unclassified
        case (chain returns ``None``) coerced to ``INOUT``.

        INOUT is the safest fallback: it doesn't lie about read-only-ness
        (which would let the caller corrupt a ``const`` buffer) nor about
        init-state (a fully-initialized buffer is always a valid argument
        to an INOUT slot). The original C signature is emitted into every
        generated function's docstring so the user can wire the call
        manually when this fallback path triggers.
        """
        verdict = self.ptr_intent(self)
        if verdict is None:
            return control.ParmIntent.INOUT
        return verdict

    @property
    def is_ptr_intent_unclassified(self):
        """True iff the rule chain produced no verdict for this pointer
        parm and the INOUT fallback applied."""
        return self.ptr_intent(self) is None

    @property
    def is_out_ptr(self):
        """If this parameter has been specified as out parameter."""
        assert self.is_ptr
        return self.effective_ptr_intent == control.ParmIntent.OUT

    @property
    def is_inout_ptr(self):
        """If this is an inout parameter."""
        assert self.is_ptr
        return self.effective_ptr_intent == control.ParmIntent.INOUT

    @property
    def is_in_ptr(self):
        """If this is an inout parameter."""
        assert self.is_ptr
        return self.effective_ptr_intent == control.ParmIntent.IN

    @property
    def is_autoconverted_by_cython(self):

        assert isinstance(self, tree.Typed)

        return (
            self.is_basic_type
            or self.is_basic_type_constantarray(rank=1)
            or self.is_pointer_to_char(incomplete_array=True)
        )


class Field(tree.Field, CythonMixin, Typed):
    def __init__(self, *args, **kwargs):
        tree.Field.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)
        self.ptr_rank = control.DEFAULT_PTR_RANK
        self.ptr_complicated_type_handler = (
            CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER()
        )

    @property
    def cython_repr(self):

        _log.debug(
            f"<{self.render_location()}>[pre] render Cython repr. of {self.__class__.__name__},{self.cursor.kind=},{self.cursor.spelling=},type: {self.cursor.type.kind=}"
        )
        # ``field_decl`` walks the Clang type hierarchy via ``TypeHandler``
        # and positions the array suffix after the variable name (Cython
        # requires ``void *arr[8]``, not ``void *[8] arr``).
        rendered = self.render_type(
            self.sep, self.renamer, prefer_canonical=True
        )
        name = self._cython_and_c_name(self.name)
        _log.debug(
            f"<{self.render_location()}>[post] render Cython repr. of {self.__class__.__name__},{self.cursor.kind=},{self.cursor.spelling=},type: {self.cursor.type.kind=}"
        )
        return rendered.field_decl(name)

    def render_python_property(self, record_cname: str):

        attr = self.renamer(self.name)
        template = Cython.Tempita.Template(
            cythontemplates.wrapper_class_record_property_template
        )

        return template.substitute(
            record_cname=record_cname,
            handler=self.ptr_complicated_type_handler(self),
            typename=self.global_typename(
                self.sep, self.renamer, prefer_canonical=True
            ),
            attr=attr,
            is_basic_type=(
                self.is_basic_type
                or self.is_pointer_to_char()  # TODO user should be consulted if char pointer is a string
            ),
            brief_comment=_escape_for_triple_quoted_docstring(
                self.doxygen_conv.transform_text_block(
                    self.brief_comment
                    if self.brief_comment is not None
                    else "(undocumented)"
                )
            ),
            is_basic_type_constantarray=self.is_basic_type_constantarray(
                rank=1
            ),
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


class ParentIsRecordMixin:

    @property
    def parent_is_record(self):  # type: (Record|Enum) -> bool
        return isinstance(self.parent, tree.Record)


class Record(tree.Record, CythonMixin, ParentIsRecordMixin):
    def __init__(self):
        raise RuntimeError("cannot be instantiated")

    def render_pyi_stub(
        self, cprefix: str, *, override_name: str = None,
        base: str = None,
    ):
        return self._render_pyi_class_stub(cprefix, override_name, base)

    @property
    def c_record_kind(self) -> str:
        if self.cursor.kind == clang.cindex.CursorKind.STRUCT_DECL:
            return "struct"
        else:
            return "union"

    def _render_c_interface_head(self) -> str:

        name = self._cython_and_c_name(self.global_name(self.sep))
        cython_def_kind = (
            "ctypedef" if self._from_typedef_with_anon_child else "cdef"
        )
        return f"{cython_def_kind} {self.c_record_kind} {name}:\n"

    def render_c_interface_decl(self) -> str:
        """Render Cython binding for this struct/union declaration.

        Renders a Cython binding for this struct/union declaration, does
        not render declarations for nested types.

        Returns:
            str: Cython C-binding representation of this struct declaration.
        """

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

    def cname(self, cprefix: str):
        """Wrapped Cython C binding type."""
        return cprefix + self.renamer(self.global_name(self.sep))

    def render_python_interface_decl(self, cprefix: str) -> str:

        name = self.renamer(self.global_name(self.sep))
        template = Cython.Tempita.Template(
            cythontemplates.wrapper_class_decl_template
        )
        return template.substitute(
            name=name,
            cname=self.cname(cprefix),
            is_complete_type=not self.is_incomplete,
            util_types_prefix=self.util_types_prefix,
        )

    def _render_python_interface_head(
        self, cprefix: str, all_propertys_rendered: bool = False
    ) -> str:

        global python_interface_record_properties_name
        name = self.cython_global_name
        template = Cython.Tempita.Template(
            cythontemplates.wrapper_class_impl_base_template.rstrip("\n")
            + "\n\n"
            + cythontemplates.wrapper_class_record_init_template
        )

        return template.substitute(
            name=name,
            cname=self.cname(cprefix),
            is_complete_type=not self.is_incomplete,
            defaults=self._defaults if self.has_defaults else {},
            properties_name=python_interface_record_properties_name,
            all_properties_rendered=all_propertys_rendered,
            is_union=self.c_record_kind == "union",
            util_types_prefix=self.util_types_prefix,
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

    def render_python_interface_impl(self, cprefix: str, *, module_opts: dict) -> str:
        """Render the implementation part for the Python interface.

        Note:
            Python interface is defined as ``cdef class``.
        """

        global python_interface_record_properties_name
        global indent

        rendered_property_names = []
        all_properties_rendered = True
        for field in self.fields:
            prop = field.render_python_property(self.cname(cprefix))
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
        result = self._render_python_interface_head(
            cprefix, all_properties_rendered
        )
        result += textwrap.indent("\n".join(self._python_body_epilog), indent)
        module_opts["all"].append(self.cython_global_name)
        return result


class Struct(tree.Struct, Record):
    def __init__(self, *args, **kwargs):
        tree.Struct.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)


class Union(tree.Union, Record):
    def __init__(self, *args, **kwargs):
        tree.Union.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)


class AnonymousStruct(tree.AnonymousStruct, Record):
    def __init__(self, *args, **kwargs):
        tree.AnonymousStruct.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)


class AnonymousUnion(tree.AnonymousUnion, Record):
    def __init__(self, *args, **kwargs):
        tree.AnonymousUnion.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)


class Enum(tree.Enum, CythonMixin, ParentIsRecordMixin):
    def __init__(self, *args, **kwargs):
        tree.Enum.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)

    def render_pyi_stub(
        self, cprefix: str, *, override_name: str = None,
        base: str = None,
    ):
        return self._render_pyi_class_stub(cprefix, override_name, base)

    def _render_cython_enums(self):
        """Yields the enum constants' names."""

        for child_cursor in self.cursor.get_children():
            name = self._cython_and_c_name(child_cursor.spelling)
            yield name

    def _render_c_interface_head(self) -> str:

        cython_def_kind = (
            "ctypedef" if self._from_typedef_with_anon_child else "cdef"
        )
        name = self._cython_and_c_name(self.global_name(self.sep))
        return (
            f"{cython_def_kind} enum{'' if self.is_anonymous else ' '+name}:\n"
        )

    def render_c_interface_decl(self):

        #
        global indent
        return self._render_c_interface_head() + textwrap.indent(
            "\n".join(self._render_cython_enums()), indent
        )

    def _render_python_enums(self, cprefix: str):
        """Yields the enum constants' names."""

        #
        for child_cursor in self.cursor.get_children():
            name = self.renamer(child_cursor.spelling)
            yield (f"{name} = {cprefix}{name}")

    @property
    def python_base_class_name(self):
        global python_interface_int_enum_base_class_name_template
        return python_interface_int_enum_base_class_name_template.format(
            name=self.cython_global_name
        )

    def _render_python_enum_constant_docstrings(self):
        for child_cursor in self.cursor.get_children():
            name = self.renamer(child_cursor.spelling)
            docu = (
                child_cursor.brief_comment
                if child_cursor.brief_comment is not None
                else "(undocumented)"
            )
            nl = "\n"
            yield textwrap.dedent(
                f"""\
                {name}:
                    {docu.replace(nl," ").rstrip()}"""
            )

    def render_python_interface_impl(self, cprefix: str, *, module_opts: dict):
        """Renders an enum.IntEnum class.

        Note:
            Does not create an enum.IntEnum class but only exposes the enum constants
            from the Cython module corresponding to the cprefix if the
            Enum is anonymous.
        """

        global indent
        global python_interface_int_enum_base_class

        if self.is_anonymous:
            for child_cursor in self.cursor.get_children():
                name = self.renamer(child_cursor.spelling)
                module_opts["docstring_attributes"] += list(
                    self._render_python_enum_constant_docstrings()
                )
            return "\n".join(self._render_python_enums(cprefix))
        else:  # named enum
            name = self.cython_global_name
            base_class_name = self.python_base_class_name

            result = textwrap.dedent(
                f"""\
                class {base_class_name}({python_interface_int_enum_base_class}):
                    \"""Empty enum base class that allows subclassing.
                    \"""
                    pass
                class {name}({base_class_name}):
                    \"""{_escape_for_triple_quoted_docstring(self.brief_comment) if self.brief_comment is not None else name}

                    Attributes:
                """
            )
            result += textwrap.indent(
                "\n".join(self._render_python_enum_constant_docstrings()),
                indent * 2,
            )
            result += f'\n{indent}"""\n'
            # body
            result += textwrap.indent(
                "\n".join(self._render_python_enums(cprefix)), indent
            )
            # add methods
            enum_type = self.cursor.enum_type.get_canonical().spelling
            ctypes_map = {
                "signed char": "ctypes.c_byte",
                "unsigned char": "ctypes.c_ubyte",
                "short": "ctypes.c_short",
                "unsigned short": "ctypes.c_ushort",
                "int": "ctypes.c_int",
                "unsigned int": "ctypes.c_uint",
                "long": "ctypes.c_long",
                "unsigned long": "ctypes.c_ulong",
                "long long": "ctypes.c_longlong",
                "unsigned long long": "ctypes.c_ulonglong",
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
            module_opts["all"].append(base_class_name)
            module_opts["all"].append(name)
            return result


class AnonymousEnum(tree.AnonymousEnum, Enum):

    def __init__(self, *args, **kwargs):
        tree.AnonymousEnum.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)


class Typedef(tree.Typedef, CythonMixin, Typed):

    def __init__(self, *args, **kwargs):
        tree.Typedef.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)

    # override
    def actual_rank(self):

        return self.get_pointer_degree()

    def render_c_interface_decl(self):
        """Returns a Cython binding for this Typedef."""
        underlying_type_name = self.global_typename(self.sep, self.renamer)
        name = self._cython_and_c_name(self.name)

        return f"ctypedef {underlying_type_name} {name}"

    def render_python_interface_decl(self, cprefix: str) -> str:
        """cdef classes are introduced for pointers to basic types and void.

        If a type hierarchy is detected, i.e. `typedef void* A; typedef A B;`
        this will be recreated.

        Note:
            Uses the `~.cython.Typedef`'s `ptr_complicated_type_handler` callback.

        Note:
            For these types of pointers, the `~.cython.Function`
            relies on the `ptr_complicated_type_handler` callback of `~.cython.Parm` as pointers
            typically need different treatmeent.
        """

        name = self.cython_global_name
        if self.is_pointer_to_void(degree=-2) or self.is_pointer_to_basic_type(
            degree=-2
        ):
            if self.typeref is not None:
                aliased = self.renamer(self.typeref.global_name(self.sep))
            else:
                aliased = self.ptr_complicated_type_handler(self)
            return f"cdef class {name}({aliased}): pass"
        elif self.is_pointer_to_void(
            degree=-1
        ) or self.is_pointer_to_basic_type(degree=-1):
            aliased = self.ptr_complicated_type_handler(self)
            return f"cdef class {name}({aliased}): pass"
        return None

    def emits_python_alias(self):
        """If this typedef emits a Python alias type
        when the Python interface is rendered.

        This is the case if the typedef aliases

        * a record or enum, or
        * a pointer to a record or enum (any degree).

        Other typedefs are not considered.
        """
        return self.is_pointer_to_record(
            degree=(0, -1)
        ) or self.is_pointer_to_enum(degree=(0, -1))

    def render_python_interface_impl(self, cprefix: str, *, module_opts: dict) -> str:

        name = self.cython_global_name
        if self.emits_python_alias():
            aliased = self.renamer(self.typeref.global_name(self.sep))
            module_opts["docstring_attributes"].append(
                textwrap.dedent(
                    f"""\
                        {name}:
                            alias of {self.to_sphinx_pyobj(aliased)}
                        """
                )
            )
            module_opts["all"].append(name)
            return f"{name} = {aliased}"
        return None


class ConstantArray(tree.ConstantArray, CythonMixin):
    def __init__(self, *args, **kwargs):
        tree.ConstantArray.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)

    @property
    def cython_element_global_typename(self):

        if self.typehandler.match_basic_datatype(self.element_type.kind):
            return self.element_type.get_canonical().spelling
        else:
            raise NotImplementedError(
                "Record, pointer, and enum members not supported yet."
            )

    def render_c_interface_decl(self) -> str:
        """Render Cython binding for this constant array declaration."""
        name = self._cython_and_c_name(self.global_name(self.sep))
        shape = f"[{']['.join([str(i) for i in self.shape])}]"
        return f"ctypedef {self.cython_element_global_typename}{shape} {name}"

    def cname(self, cprefix: str):
        """Wrapped Cython C binding type."""
        return cprefix + self.renamer(self.global_name(self.sep))

    def render_python_interface_decl(self, cprefix: str) -> str:

        name = self.cython_global_name
        template = Cython.Tempita.Template(
            cythontemplates.wrapper_class_decl_template
        )
        return template.substitute(
            name=name,
            cname=self.cname(cprefix),
            is_complete_type=True,
            is_array=True,
            util_types_prefix=self.util_types_prefix,
        )

    def render_python_interface_impl(self, cprefix: str, *, module_opts: dict) -> str:
        global indent
        name = self.cython_global_name
        template = Cython.Tempita.Template(
            cythontemplates.wrapper_class_impl_base_template.rstrip("\n")
            + "\n"
            + cythontemplates.wrapper_class_constantarray_get_element_template
        )
        module_opts["all"].append(self.cython_global_name)
        return template.substitute(
            name=name,
            cname=self.cname(cprefix),
            is_complete_type=True,
            util_types_prefix=self.util_types_prefix,
            is_basic_type=True,
            is_array=True,
            dim=self.dim,
            shape=self.shape,
        )


class FunctionPointer(CythonMixin):

    def render_pyi_stub(
        self, cprefix: str, *, override_name: str = None,
        base: str = None,
    ):
        return self._render_pyi_class_stub(cprefix, override_name, base)

    def render_c_interface_decl(self):
        """Returns a Cython binding for this Typedef."""

        parm_types = ",".join(
            [parm.cython_global_typename for parm in self.parms]
        )
        underlying_type_name = self.renamer(self.canonical_result_typename)
        typename = self.cython_global_name  # might be AnonymousFunctionPointer
        return f"ctypedef {underlying_type_name} (*{typename}) ({parm_types})"

    def render_python_interface_decl(self, cprefix: str) -> str:

        name = self.cython_global_name
        cname = cprefix + name
        template = Cython.Tempita.Template(
            cythontemplates.wrapper_class_decl_template
        )
        return template.substitute(
            name=name,
            cname=cname,
            cptr_type=cname,  # type is already a pointer
            is_complete_type=False,
            util_types_prefix=self.util_types_prefix,
        )

    def render_python_interface_impl(self, cprefix: str, *, module_opts: dict) -> str:

        name = self.cython_global_name
        cname = cprefix + name
        template = Cython.Tempita.Template(
            cythontemplates.wrapper_class_impl_base_template
        )
        module_opts["all"].append(name)
        return template.substitute(
            name=name,
            cname=cname,
            cptr_type=cname,  # type is already a pointer
            is_funptr=True,
            is_complete_type=False,
            util_types_prefix=self.util_types_prefix,
        )


class TypedefedFunctionPointer(tree.TypedefedFunctionPointer, FunctionPointer):
    def __init__(self, *args, **kwargs):
        tree.TypedefedFunctionPointer.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)


class AnonymousFunctionPointer(tree.AnonymousFunctionPointer, FunctionPointer):
    def __init__(self, *args, **kwargs):
        tree.AnonymousFunctionPointer.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)


class Parm(tree.Parm, CythonMixin, Typed):
    def __init__(self, *args, **kwargs):
        global CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER
        tree.Parm.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)
        self.ptr_rank = control.DEFAULT_PTR_RANK
        self.ptr_intent = control.DEFAULT_PTR_PARM_INTENT
        self.ptr_complicated_type_handler = (
            CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER()
        )

    @property
    def cython_repr(self):
        """Returns the Cython (C) representation of the parameter.

        Returns ``{typename} {name}`` in most cases.
        Special care is take for pointers to constant arrays.

        Example:

            For a parameter 'arr' with name 'int (**)[][23]' the parameter representation would be
            int (**arr)[][23].
        """

        # TODO must be adjusted for

        typename: str = self.cython_global_typename
        name = self.cython_name
        if self.is_pointer_to_constantarray_of_basic_type(-1, True):
            # example typename: 'int (**)[][23]'
            parts = typename.split(")", maxsplit=1)
            return f"{parts[0]}{name}){parts[1]}"
        else:
            return f"{typename} {name}"


