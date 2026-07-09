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


"""Cython backend — defaults submodule.

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

from .. import cparser, cythontemplates, doxyparser, tree
from ..support import cython as support
from ..support.recipes import control

_log = logging.getLogger("interfacegen")

__all__ = [
    'indent',
    'restricted_names',
    'c_interface_funptr_name_template',
    'python_interface_retval_template',
    'python_interface_int_enum_base_class',
    'python_interface_int_enum_base_class_name_template',
    'python_interface_record_properties_name',
    'python_interface_pyobj_role_template',
    'default_c_interface_decl_prolog',
    'default_c_interface_impl_prolog',
    'default_python_interface_decl_prolog',
    'default_python_interface_impl_prolog',
    'LICENSE_TEXT',
    'CodegenUnsupportedPattern',
    'CYTHON_AUTOCONV_FROM_PYTHON_TYPES',
    'CYTHON_AUTOCONV_TO_PYTHON_TYPES',
    'DEFAULT_RENAMER',
    'DEFAULT_RAW_COMMENT_CLEANER',
    'DEFAULT_DOCSTRING_CLEANER',
    '_escape_for_triple_quoted_docstring',
    'DEFAULT_MACRO_TYPE',
    'CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER',
    'CallArgHoist',
]

indent = " " * 4


class CodegenUnsupportedPattern(Exception):
    """Raised when the cython backend reaches a parameter or field shape
    that ``handle_callee_allocated_ptr_parm`` /
    ``handle_caller_allocated_ptr_`` (and friends) have no branch for.

    Carries the function name, parm name, and canonical C type so callers
    (notably the gap test suite) can isolate which pattern is missing
    without grepping log lines.
    """

    def __init__(self, function_name: str, parm_name: str, canonical_type: str):
        self.function_name = function_name
        self.parm_name = parm_name
        self.canonical_type = canonical_type
        super().__init__(
            f"function {function_name}: parm {parm_name}: not handled, "
            f"canonical C type: '{canonical_type}'"
        )

restricted_names = keyword.kwlist + [
    "cdef",
    "cpdef",  # TODO extend
]

c_interface_funptr_name_template = "_{name}__funptr"

python_interface_retval_template = "_{name}__retval"

python_interface_int_enum_base_class = "enum.IntEnum"

python_interface_int_enum_base_class_name_template = "_{name}__Base"

python_interface_record_properties_name = "PROPERTIES"

python_interface_pyobj_role_template = r":py:obj:`~.{name}`"


def CYTHON_AUTOCONV_FROM_PYTHON_TYPES(canonical_ctype: str):
    """Convert a canonical C type to the Python types from which
    it is converted automatically by Cython.

    Returns:
        tuple(str): The Python types that Cython autoconverts to the C type.

    Note:
        For implementation details, see
        https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#automatic-type-conversions
    """
    tokens = [
        tk
        for tk in canonical_ctype.split(" ")
        if tk not in ("const", "unsigned")
    ]
    if tokens in [
        ["char", "*"],
        ["char", "[]"],
    ]:
        return ("bytes",)
    elif tokens in (["char"], ["short"], ["int"], ["long"], ["long", "long"]):
        return ("int",)  # no long in Python 3 anymore
    elif tokens[0] == "_Bool":  # C version of 'bool', 'bool' is a C++ type
        return ("bint",)
    elif tokens in [
        ["float"],
        ["double"],
        ["long", "double"],
    ]:
        return (
            "float",
            "int",
        )  # no long in Python 3, int can be converted to float too
    elif len(tokens) == 2 and tokens[0] in ("union", "struct", "enum"):
        raise KeyError(
            "Cython cannot autoconvert to C structs, unions, and enums from Python types."
        )
    else:
        # const unsigned char[32]
        if "char" in tokens:
            return "bytes"
        else:
            return "list"  # FIXME make configurable
        # C array and struct union are not handled yet
        # requires
        # raise NotImplementedError(f"not implemented for type '{canonical_ctype}'")


def CYTHON_AUTOCONV_TO_PYTHON_TYPES(canonical_ctype: str):
    """Convert a canonical C/Cython C type to the Python type to which
    it is converted automatically by Cython.

    Returns:
        str: The Python type that Cython autoconverts to from the C type.

    Note:
        For implementation details, see
        https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#automatic-type-conversions
    """
    tokens = [
        tk
        for tk in canonical_ctype.split(" ")
        if tk not in ("const", "unsigned")
    ]
    if tokens in [
        ["char", "*"],
        ["char", "[]"],
    ]:
        return "bytes"
    elif tokens in (
        ["char"],
        ["short"],
        ["int"],
        ["long"],
        ["long", "long"],
        ["size_t"],
    ):
        return "int"  # no long in Python 3 anymore
    elif tokens in [
        ["float"],
        ["double"],
        ["long", "double"],
    ]:
        return "float"
    elif tokens in [
        ["bint"],
        ["_Bool"],  # C's `bool`; Cython aliases it to `bint`
    ]:
        return "bool"
    elif len(tokens) == 2 and tokens[0] in ("union", "struct", "enum"):
        raise NotImplementedError("struct, union, enum types are not handled")
    else:
        return "list"


def DEFAULT_RENAMER(name):  # backend-specific
    result = name
    while result in restricted_names:
        result += "_"
    if "[]" in result:  # Cython does not like this in certain signatures
        result = result.replace("[]", "*")
    return result


def DEFAULT_RAW_COMMENT_CLEANER(raw_comment: str):
    return raw_comment


def DEFAULT_DOCSTRING_CLEANER(docstring: str):
    return docstring


def _escape_for_triple_quoted_docstring(s: str) -> str:
    """Escape a string for safe interpolation inside a `\"\"\"...\"\"\"` literal.

    Doxygen briefs that contain literal `"` (e.g. ``//!< "CPER"``) collide
    with the surrounding triple quotes when substituted directly. Replace
    every `"` with `\\"` — valid inside a triple-quoted string and visually
    identical when the docstring is rendered.
    """
    if s is None:
        return s
    return s.replace('"', '\\"')


def DEFAULT_MACRO_TYPE(node):  # backend-specific
    return "int"


# Utility types


def CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(util_types_prefix: str = ""):
    """Creates the default type handler routine.

    Args:
        util_pkg (str, optional): Prefix to the `types` from the types utility module.
        Defaults to "".
    """

    def inner(node):

        assert isinstance(node, tree.Typed)
        if node.is_pointer_to_constantarray_of_basic_type(degree=-1):
            return f"{util_types_prefix}Pointer"
        # ``char **`` OUT returns a single NUL-terminated string => ``CStr``.
        # This is decided here, independent of rank: a ``char **`` is rank-0
        # in the runtime chain (``double_indirection_out`` claims it before
        # ``string_z``), so it never reaches the rank-1 block below. An IN
        # array-of-strings (argv) or a field/return is NOT matched (no OUT
        # intent) and falls through to ``Pointer`` (recipes override to
        # ``ListOfBytes`` where needed).
        if (
            node.is_pointer_to_char(degree=2)
            and isinstance(node, tree.Parm)
            and node.is_out_ptr
        ):
            return f"{util_types_prefix}CStr"
        if node.actual_rank == 1:
            innermost_type_kind = next(
                node.clang_type_layer_kinds(postorder=-1, canonical=True)
            )
            if innermost_type_kind == clang.cindex.TypeKind.INT:
                return f"{util_types_prefix}ListOfInt"
            elif innermost_type_kind == clang.cindex.TypeKind.UINT:
                return f"{util_types_prefix}ListOfUnsigned"
            elif innermost_type_kind == clang.cindex.TypeKind.ULONG:
                return f"{util_types_prefix}ListOfUnsignedLong"
            elif (
                innermost_type_kind == clang.cindex.TypeKind.VOID
                and node.get_pointer_degree() >= 2
                and isinstance(node, tree.Parm)
                and not node.is_out_callee_allocated_ptr
            ):
                # A *caller-allocated* ``void**`` at rank 1 is an
                # array-of-pointers slot (e.g. amdsmi's ``socket_handles`` /
                # ``processor_handles``) — expose it as ``ListOfPointer`` so it
                # is list-constructible and a proper sequence.
                #
                # Deliberately NOT applied to callee-allocated ``void**``
                # out-pointers (``hipMalloc``-style single-buffer returns,
                # ``intent.allocated_by_callee``): those stay ``Pointer`` so a
                # single allocated buffer is not mistyped as a list. A plain
                # ``void *`` (degree 1) byte buffer also stays ``Pointer``
                # (handled below).
                return f"{util_types_prefix}ListOfPointer"
            elif innermost_type_kind == clang.cindex.TypeKind.CHAR_S:
                # A NUL-terminated string is rank-1 data regardless of
                # indirection depth (see ``generic.string_z``). Degree +
                # intent — not rank — pick the wrapper:
                #   * ``char *`` (degree<=1) => the buffer itself => ``CStr``
                #     (covers IN/INOUT params, return values, and fields;
                #     also ``char[]`` whose pointer degree is 0).
                #   * ``char **`` at rank 1 (e.g. numerical chain via
                #     ``DEFAULT_PTR_RANK``): an OUT slot returns one string
                #     => ``CStr``; an IN array-of-strings falls through.
                if node.get_pointer_degree() <= 1:
                    return f"{util_types_prefix}CStr"
                if isinstance(node, tree.Parm) and node.is_out_ptr:
                    return f"{util_types_prefix}CStr"
            # TODO consider other char types?
        if node.actual_rank == 2:
            return f"{util_types_prefix}ListOfPointer"
        return f"{util_types_prefix}Pointer"

    return inner


class CallArgHoist:
    """Structured description of a single pre-block hoist line emitted
    before a ``with nogil:`` cy* call.

    Two shapes are supported.

    1. **Plain hoist** — no Python wrapper temporary is involved (e.g.
       ``parm.value`` for an IntEnum arg). The expression is bound
       directly to a typed cdef local::

           cdef <c_type> _cy_<f>__arg_N = <plain_expr>

    2. **Wrapper-bound hoist** — a Python wrapper temporary owns memory
       that the resulting C pointer references (e.g.
       ``ListOfBytes.fromPyobj(options)`` allocates the ``char**``
       array). Binding *only* the raw pointer to a cdef local would let
       Python release the wrapper at end-of-statement, and the with-nogil
       cy* call would dereference dangling memory. Split the chain so
       the wrapper outlives the with-nogil block::

           cdef <wrapper_class> _cy_<f>__arg_N_obj = <wrapper_factory>
           cdef <c_type> _cy_<f>__arg_N = <cast_open><obj>.<pointer_extract><cast_close>

    The with-gil emitter has no lifetime concern (the GIL is held the
    whole time and the wrapper temporary lives as long as the C call's
    enclosing expression). For that path :pymeth:`inline_expr` returns
    the single-expression form so the call arg can be appended verbatim.
    """

    def __init__(
        self,
        *,
        c_type: str,
        plain_expr: str = None,
        wrapper_class: str = None,
        wrapper_factory: str = None,
        pointer_extract: str = None,
        cast_open: str = "",
        cast_close: str = "",
    ):
        self.c_type = c_type
        self.cast_open = cast_open
        self.cast_close = cast_close
        if plain_expr is not None:
            assert wrapper_class is None and wrapper_factory is None and \
                pointer_extract is None, (
                    "plain_expr and wrapper-bound fields are mutually exclusive"
                )
            self.plain_expr = plain_expr
            self.wrapper_class = None
            self.wrapper_factory = None
            self.pointer_extract = None
        else:
            assert wrapper_class is not None and wrapper_factory is not None \
                and pointer_extract is not None, (
                    "wrapper-bound hoist requires wrapper_class, "
                    "wrapper_factory, and pointer_extract"
                )
            self.wrapper_class = wrapper_class
            self.wrapper_factory = wrapper_factory
            self.pointer_extract = pointer_extract
            self.plain_expr = None

    def inline_expr(self) -> str:
        """Single-line form for the with-gil emitter (no hoist needed)."""
        if self.plain_expr is not None:
            return self.plain_expr
        return (
            f"{self.cast_open}{self.wrapper_factory}."
            f"{self.pointer_extract}{self.cast_close}"
        )

    # Strip a trailing ``const`` qualifier (with surrounding whitespace)
    # from the END of a c_type — e.g. ``void *const`` → ``void *``,
    # ``T * const`` → ``T *``. Cython rejects a const-qualified pointer
    # local with ``Assignment to const 'x'`` (3.1+ as a hard error;
    # 3.0.x miscompiled it silently). Stripping the outermost const
    # from the cdef *local* type is safe — local variables don't need
    # C-style const protection (the local is only assigned once at the
    # prehoist line); the cast on the right keeps the original c_type
    # so the rhs is type-correct against the C function signature.
    _STRIP_TRAILING_CONST = re.compile(r"\s*\bconst\b\s*$")

    def render_prehoist(self, arg_name: str) -> str:
        """Multi-line form for the with-nogil emitter.

        Emits the combined ``cdef T x = <T>expr`` form. This relies on
        the project-wide Cython >= 3.1.0 build floor (pinned in every
        requirements/pyproject file): Cython 3.0.x silently miscompiled
        ``cdef T x = <T>expr`` for the ``*const *`` shape (e.g.
        ``const char *const *``), dropping the initializer and leaving
        a NULL local. 3.1+ compiles that form correctly, so the
        former bare-cdef + separate-assignment split is no longer
        needed. The historical workaround (and the Cython repro) is
        preserved in
        ``share/design/UPSTREAM_BUGS/cython_const_pointer_initializer_bug.md``.

        One transformation remains:

        * **Trailing-const strip.** If the c_type ends in ``const``
          (e.g. ``void *const``, ``T * const``), strip that trailing
          const from the cdef *local* type. The cast on the rhs keeps
          the original c_type so the assignment is still type-correct
          against the C function signature. Cython 3.1+ rejects
          ``cdef T x = ...`` for const-qualified T with
          ``Assignment to const 'x'``; a local doesn't need the const —
          the const-correctness contract is between the call expression
          and the C function parameter type.

        See ``test_call_arg_hoist_double_const_pointer_uses_combined_form``,
        ``test_call_arg_hoist_trailing_const_strips_const``, and
        ``test_call_arg_hoist_const_substring_in_identifier_not_stripped``
        in test_typed_helpers.py for the regression coverage.
        """
        if self.plain_expr is not None:
            return f"cdef {self.c_type} {arg_name} = {self.plain_expr}"
        obj_name = f"{arg_name}_obj"
        prefix = f"cdef {self.wrapper_class} {obj_name} = {self.wrapper_factory}\n"
        rhs = f"{self.cast_open}{obj_name}.{self.pointer_extract}{self.cast_close}"
        cdef_type = self._STRIP_TRAILING_CONST.sub("", self.c_type)
        return f"{prefix}cdef {cdef_type} {arg_name} = {rhs}"


LICENSE_TEXT = f"""\
{support.render_license_MIT(year_start=2021)}

# This file has been autogenerated, do not modify.
"""

default_c_interface_decl_prolog = f"""\
{LICENSE_TEXT}
from libc.stdint cimport *
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
"""

default_c_interface_impl_prolog = f"""\
{LICENSE_TEXT}
"""

default_python_interface_decl_prolog = f"""\
{LICENSE_TEXT}
from libc cimport stdlib
from libc cimport string
from libc.stdint cimport *
cimport cpython.long
cimport cpython.buffer
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
"""

default_python_interface_impl_prolog = f"""\
{LICENSE_TEXT}

\"""
[MODULE_DOCSTRING]
\"""

import cython
import ctypes
import enum
# cimport (not import) is required so the with-nogil emitter can use
# the util types in `cdef` declarations — `cdef <T> name = ...` only
# works when T is visible to Cython at compile time, which means the
# defining module must be cimported. The plain Python `import` form
# would only make the names available to runtime attribute access.
cimport rocm.bindings.util.types
"""

