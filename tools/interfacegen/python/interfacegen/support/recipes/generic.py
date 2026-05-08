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
"""Library-agnostic pointer-parameter rule sets.

Each class here is shaped like a per-library class in ``rocm.py``: it exposes
``ptr_parm_intent(parm)`` and/or ``ptr_rank(node)`` callables that return a
verdict or ``None`` to defer. They are designed to be composed via
``support.recipes.control.fallback(...)`` in a per-library chain.

See ``share/design/POINTER_ARGUMENTS.md`` for the decision table the
``conservative`` ruleset is derived from and for the catalog of conventions
implemented by the other classes.
"""

__author__ = "Advanced Micro Devices, Inc."

from interfacegen.support.recipes.control import ParmIntent


# ---------------------------------------------------------------------------
# (1) baseline: only what C modifiers prove
# ---------------------------------------------------------------------------
def _is_pointer_or_array_param(parm):
    """True if the parm is one of: ``T*``, ``T[]``, ``T[N]`` (any degree)."""
    from clang.cindex import TypeKind

    if getattr(parm, "is_any_pointer", False):
        return True
    kinds = list(parm.typehandler.clang_type_layer_kinds(canonical=True))
    return bool(kinds) and kinds[0] in (
        TypeKind.INCOMPLETEARRAY,
        TypeKind.CONSTANTARRAY,
    )


def _data_is_const(parm):
    """True iff the ultimately-referred data is const-qualified.

    Strips outer POINTER layers (their own const, ``T *const``, says nothing
    about the data) and asks whether any remaining layer is const-qualified.
    Catches both ``const T *`` (const on innermost) and ``const T[]`` (const
    on the wrapping array layer — libclang's qualifier walk attaches the
    const there for array-of-const).
    """
    from clang.cindex import TypeKind

    qualifiers = list(parm.typehandler.const_qualifiers(canonical=True))
    layer_kinds = list(parm.typehandler.clang_type_layer_kinds(canonical=True))
    i = 0
    while i < len(layer_kinds) and layer_kinds[i] == TypeKind.POINTER:
        i += 1
    return any(qualifiers[i:])


class conservative:
    """Modifier-only deductions. Returns ``None`` whenever C alone doesn't
    decide; never guesses.

    Rules:
      * ``ptr_parm_intent``: pointer or array parm whose ultimately-referred
        data is ``const``-qualified (``const T *``, ``const T[]``,
        ``const T **``, …) => ``IN``. Otherwise ``None``.
      * ``ptr_rank``: presence of an INCOMPLETEARRAY or CONSTANTARRAY layer
        in the canonical type => rank 1. Otherwise ``None``.
    """

    @staticmethod
    def ptr_parm_intent(parm):
        if not _is_pointer_or_array_param(parm):
            return None
        if getattr(parm, "is_any_pointer", False) and parm.is_pointer_to_function_proto(
            degree=parm.get_pointer_degree(incomplete_array=True)
        ):
            return None
        if _data_is_const(parm):
            return ParmIntent.IN
        return None

    @staticmethod
    def ptr_rank(node):
        from interfacegen import tree
        from clang.cindex import TypeKind

        if not isinstance(node, tree.Typed):
            return None
        if node.is_any_pointer and node.is_pointer_to_function_proto(
            degree=node.get_pointer_degree(incomplete_array=True)
        ):
            return None
        for kind in node.typehandler.clang_type_layer_kinds(canonical=True):
            if kind in (TypeKind.INCOMPLETEARRAY, TypeKind.CONSTANTARRAY):
                return 1
        return None


# ---------------------------------------------------------------------------
# (4) double_indirection_out: T** non-const => OUT, scalar handle
# ---------------------------------------------------------------------------
class double_indirection_out:
    """``T**`` non-const => callee-allocates / returns a handle through the
    pointer-to-pointer slot. GIR ``(out)`` rule for double-indirection on a
    structure parameter; SAL ``_Outptr_`` equivalent.
    """

    @staticmethod
    def _matches(node):
        return (
            node.is_pointer_to_void(degree=2)
            or node.is_pointer_to_record(degree=2)
            or node.is_pointer_to_enum(degree=2)
            or node.is_pointer_to_basic_type(degree=2)
        )

    @staticmethod
    def ptr_parm_intent(parm):
        if parm.has_innermost_type_layer_const_modifier:
            return None
        if double_indirection_out._matches(parm):
            return ParmIntent.OUT
        return None

    @staticmethod
    def ptr_rank(node):
        from interfacegen import tree

        if not isinstance(node, tree.Typed):
            return None
        if double_indirection_out._matches(node):
            return 0
        return None


# ---------------------------------------------------------------------------
# (3) pointer_as_value: non-const *T => IN (Value school)
# ---------------------------------------------------------------------------
class pointer_as_value:
    """Mutually exclusive with ``pointer_as_reference``. Treats single
    ``T*`` non-const parms as IN — the function reads through the pointer
    but does not write the slot."""

    @staticmethod
    def ptr_parm_intent(parm):
        if not getattr(parm, "is_any_pointer", False):
            return None
        if parm.get_pointer_degree() != 1:
            return None
        if parm.has_innermost_type_layer_const_modifier:
            return None
        if parm.is_pointer_to_function_proto(degree=1):
            return None
        return ParmIntent.IN


# ---------------------------------------------------------------------------
# (2) pointer_as_reference: non-const *T => INOUT (Reference school)
# ---------------------------------------------------------------------------
class pointer_as_reference:
    """Mutually exclusive with ``pointer_as_value``. Treats single ``T*``
    non-const parms as INOUT — typical for in-place compute (FFT, RCCL recv
    buffers, scratch space).
    """

    @staticmethod
    def ptr_parm_intent(parm):
        if not getattr(parm, "is_any_pointer", False):
            return None
        if parm.get_pointer_degree() != 1:
            return None
        if parm.has_innermost_type_layer_const_modifier:
            return None
        if parm.is_pointer_to_function_proto(degree=1):
            return None
        return ParmIntent.INOUT


# ---------------------------------------------------------------------------
# (5) string_z: const char* => IN scalar; char** => OUT scalar
# ---------------------------------------------------------------------------
class string_z:
    """GIR / SAL ``_In_z_`` / ``_Outptr_result_z_`` convention.

    * ``const char *p``  => ``IN``,  rank 0
    * ``char **p``       => ``OUT``, rank 0  (``char *`` ambiguous => defer)
    """

    @staticmethod
    def ptr_parm_intent(parm):
        if parm.is_pointer_to_char(degree=1):
            if parm.has_innermost_type_layer_const_modifier:
                return ParmIntent.IN
            return None
        if parm.is_pointer_to_char(degree=2):
            if parm.has_innermost_type_layer_const_modifier:
                return None
            return ParmIntent.OUT
        return None

    @staticmethod
    def ptr_rank(node):
        if not hasattr(node, "is_pointer_to_char"):
            return None
        if node.is_pointer_to_char(degree=1):
            return 0
        if node.is_pointer_to_char(degree=2):
            return 0
        return None


# ---------------------------------------------------------------------------
# (9) opaque_typedef_is_handle: typedef'd T* => scalar handle
# ---------------------------------------------------------------------------
class opaque_typedef_is_handle:
    """Universal opaque-handle pattern: a typedef whose canonical form is a
    pointer (``hipStream_t = ihipStream_t*``, ``cudaStream_t``, ``FILE*``,
    ...). Without this rule the codegen treats the handle as a 1-pointer
    indirection.

    Rank-only convention: returns ``rank 0`` for typedefed pointers but
    leaves intent to other rules.
    """

    @staticmethod
    def _is_typedef_to_pointer(node):
        from clang.cindex import TypeKind

        if not hasattr(node, "typehandler"):
            return False
        kinds = list(node.typehandler.clang_type_layer_kinds())
        canonical_kinds = list(
            node.typehandler.clang_type_layer_kinds(canonical=True)
        )
        # libclang 17+ wraps parameter types in a leading ELABORATED
        # layer (e.g. a `my_handle_t` parameter reports
        # `[ELABORATED, TYPEDEF, POINTER, ELABORATED, RECORD]` rather
        # than the older `[TYPEDEF, POINTER, RECORD]`). Skip leading
        # ELABORATED entries when looking for the TYPEDEF kind so the
        # pattern still matches across both libclang versions.
        non_elab = [k for k in kinds if k != TypeKind.ELABORATED]
        return (
            len(non_elab) >= 2
            and non_elab[0] == TypeKind.TYPEDEF
            and len(canonical_kinds) >= 1
            and canonical_kinds[0] == TypeKind.POINTER
        )

    @staticmethod
    def ptr_parm_intent(parm):
        # No intent contribution; rank-only convention.
        return None

    @staticmethod
    def ptr_rank(node):
        from interfacegen import tree

        if not isinstance(node, tree.Typed):
            return None
        if opaque_typedef_is_handle._is_typedef_to_pointer(node):
            return 0
        return None


# ---------------------------------------------------------------------------
# (6) array_with_length_param  (relational; deferred — needs sibling-parm walk)
# ---------------------------------------------------------------------------
class array_with_length_param:
    """``T *buf`` adjacent to integer parm whose name names the length
    (``n``, ``len``, ``count``, ``*_size``, ``*_n``, ``n_*``) => buf is
    rank 1; intent follows pointee const.

    GIR ``(array length=N)`` / SAL ``_In_reads_(n)`` / ``_Out_writes_(n)``.

    .. note::
       Deferred. Implementing this needs a stable way to walk to the parent
       Function's parameter list, which is available via ``parm.parent`` but
       requires consensus on which sibling-name patterns count. Sketch only.
    """

    @staticmethod
    def ptr_parm_intent(parm):
        return None  # TODO

    @staticmethod
    def ptr_rank(node):
        return None  # TODO


# ---------------------------------------------------------------------------
# (7) zero_terminated_array  (deferred — typically not inferable without annot.)
# ---------------------------------------------------------------------------
class zero_terminated_array:
    """``T**`` where elements are sentinel-terminated (NULL for pointer
    arrays, 0 for numeric) => rank 1.

    GIR ``(array zero-terminated=1)``. Hard to detect from C alone; needs an
    annotation source (per-library override or external metadata).

    .. note:: Deferred. Sketch only.
    """

    @staticmethod
    def ptr_rank(node):
        return None  # TODO


# ---------------------------------------------------------------------------
# (8) status_return_out_pointer  (relational; deferred — needs return-type)
# ---------------------------------------------------------------------------
class status_return_out_pointer:
    """If the parent Function's return type is an integer/error status enum
    AND this parm is the only non-const pointer, mark it OUT.

    POSIX ``clock_gettime``, virtually every Khronos API, every ROCm runtime
    function. Not modifier-based — uses **return-type** signature inspection.

    .. note:: Deferred. Sketch only.
    """

    @staticmethod
    def ptr_parm_intent(parm):
        return None  # TODO


# ---------------------------------------------------------------------------
# (10) documented_param_intent: trust @param[in|out|in,out] doxygen tags
# ---------------------------------------------------------------------------
import re as _re


# Match a doxygen `\param[dir] <name>` or `@param[dir] <name>` directive.
# Direction bracket is optional; name is the C identifier of the parameter
# the tag describes. Body text is intentionally NOT captured — this rule
# is tag-only. Whitespace inside the bracket and between `param` and the
# name follows doxygen lenient rules (one or more spaces / single newlines
# tolerated). Both `\param` and `@param` prefixes are accepted; ROCm
# headers use both interchangeably.
_DOXY_PARAM_TAG_RE = _re.compile(
    r"[@\\]param"
    r"(?:\s*\[\s*(in|out|in\s*,\s*out|inout)\s*\])?"
    r"\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_DOXY_TAG_TO_INTENT = {
    "in": ParmIntent.IN,
    "out": ParmIntent.OUT,
    "in,out": ParmIntent.INOUT,
    "inout": ParmIntent.INOUT,
}


def _iter_doxygen_param_tags(raw_comment):
    r"""Yield ``(direction, parm_name)`` tuples from a function's
    raw doxygen comment.

    ``direction`` is one of ``'in'``, ``'out'``, ``'in,out'``,
    ``'inout'``, or ``None`` if no direction bracket was present.
    ``parm_name`` is the C identifier of the documented parameter.

    Doxygen ``\param`` body text terminates at a blank line or the
    next ``\``/``@`` directive (see ``doxyparser.py``
    SECTION_TERMINATOR + the ``param`` rule). The intent rule
    doesn't need the body text; the regex matches just the tag
    header so the body's terminator is irrelevant here. A future
    rank-keyword rule will need to transcribe the body terminator.
    """
    if not raw_comment:
        return
    for direction, name in _DOXY_PARAM_TAG_RE.findall(raw_comment):
        yield (direction or None, name)


class documented_param_intent:
    """Trust the ``@param[in|out|in,out]`` doxygen direction tag.

    For libraries with disciplined doxygen (amdsmi, hsa, hipfft,
    parts of hipblas), the `[in|out|in,out]` direction bracket is
    the most reliable signal we have for parameter intent — better
    than any structural inference because it captures author intent
    directly. Composed at the head of the per-library intent chains
    (`_RUNTIME_INTENT_CHAIN`, `_NUMERICAL_INTENT_CHAIN`,
    `_INPLACE_NUMERICAL_INTENT_CHAIN` in
    ``support/recipes/rocm.py``), so it runs after per-library
    hardcoded overrides but before all structural rules including
    `double_indirection_out` (a documented ``@param[in] T** foo``
    declares an array-of-pointers input that the structural rule
    would otherwise misclassify as OUT).
    """

    @staticmethod
    def ptr_parm_intent(parm):
        if getattr(parm, "parent", None) is None:
            return None
        raw = getattr(parm.parent, "raw_comment", None)
        if not raw:
            return None
        pname = getattr(parm, "name", "") or ""
        if not pname:
            return None
        for direction, name_in_doc in _iter_doxygen_param_tags(raw):
            if name_in_doc == pname and direction is not None:
                key = direction.replace(" ", "")
                return _DOXY_TAG_TO_INTENT.get(key)
        return None
