# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.

"""Structured Clang-type rendering driven by ``TypeHandler`` layer walks.

Produces a :class:`RenderedType` by walking the Clang type layer
hierarchy via :class:`interfacegen.cparser.TypeHandler` — no
canonical-spelling tokenization. Elaborated leaves (``struct Foo``,
``enum Bar``, ``union Baz``) and typedef chains are handled by walking
the layer hierarchy and splicing the typeref's identifier in place of
the leaf segment.

Used by ``tree.Typed.global_typename`` / ``tree.Typed.typename`` /
``Typed.cython_global_typename`` / ``Field.cython_repr`` (cython
backend) — these are now thin wrappers around this module.
"""

__author__ = "Advanced Micro Devices, Inc."

import dataclasses
import typing

import clang.cindex

from . import cparser

_TypeKind = clang.cindex.TypeKind
_TypeCategory = cparser.TypeHandler.TypeCategory


@dataclasses.dataclass(frozen=True)
class TypeLayer:
    """One outer layer above the leaf — a pointer or array indirection."""

    kind: str  # "ptr" or "array"
    is_const: bool = False  # const on this layer (e.g. ``int * const``)
    is_restrict: bool = False
    is_volatile: bool = False
    array_size: str = ""  # decimal size for ``[N]``; "" for ``[]``

    @property
    def qualifier_suffix(self) -> str:
        """Per-pointer qualifier suffix as Clang would spell it after the
        ``*``: e.g. ``const``, ``restrict``, ``const volatile``."""
        parts = []
        if self.is_const:
            parts.append("const")
        if self.is_restrict:
            parts.append("restrict")
        if self.is_volatile:
            parts.append("volatile")
        return " ".join(parts)


@dataclasses.dataclass(frozen=True)
class RenderedType:
    """Structured view of a Clang type after typeref substitution.

    ``layers`` is in outer-to-inner order — for ``void *[8]`` it is
    ``(TypeLayer(array, "8"), TypeLayer(ptr))``.
    """

    base_typename: str
    base_kind: _TypeCategory
    base_clang_kind: _TypeKind
    is_base_const: bool
    is_base_unsigned: bool
    layers: typing.Tuple[TypeLayer, ...]

    def cython_decl(self) -> str:
        """Match :pyattr:`tree.Typed.cython_global_typename`: keep leading
        ``const``; preserve per-pointer ``const``/``restrict``/``volatile``
        as Clang spells them (e.g. ``int *const``)."""
        return self._render(drop_leading_const=False)

    def cython_decl_no_const(self) -> str:
        """Match :pyattr:`tree.Typed.cython_global_typename_no_const`: also
        strip the leading ``const`` on the leaf."""
        return self._render(drop_leading_const=True)

    def global_decl(self) -> str:
        """Match :pymeth:`tree.Typed.global_typename`. Equivalent to
        :meth:`cython_decl` for the cases the legacy code covered."""
        return self._render(drop_leading_const=False)

    def field_decl(self, name: str) -> str:
        """Render as the C field declaration ``<base> <left><name><right>``.

        Cython requires trailing ``[N]`` suffixes to follow the variable
        name when a ``*`` precedes them (``void *arr[8]``, not
        ``void *[8] arr``); the declarator builder assembles the type
        from inner to outer so the suffix lands in the right place by
        construction. Pointer-to-array shapes
        (``unsigned int (**vec)[32]``) get their pointer block
        parenthesized so ``[]`` binds to the parenthesized unit (C
        declarator precedence — ``[]`` binds tighter than ``*``).
        """
        base = self._render_base(drop_leading_const=False)
        left, right = self._build_declarator()
        # Convention: a single space between the base+ptr block and the
        # variable name when the name sits OUTSIDE parens. Inside parens
        # (pointer-to-array shapes) the name is glued to the inner ``*``
        # (``(*x)`` not ``( *x)``), matching Clang/Cython style.
        if not left:
            return f"{base} {name}{right}"
        if left.startswith("("):
            return f"{base} {left}{name}{right}"
        return f"{base} {left} {name}{right}"

    def _render(self, *, drop_leading_const: bool) -> str:
        base = self._render_base(drop_leading_const=drop_leading_const)
        left, right = self._build_declarator()
        if not left and not right:
            return base
        # Clang spells ``int[8]`` with no space between base and a pure-
        # array suffix; ``int *`` with a space before pointer modifiers.
        sep = "" if not left and right.startswith("[") else " "
        return f"{base}{sep}{left}{right}"

    def _render_base(self, *, drop_leading_const: bool) -> str:
        if self.is_base_const and not drop_leading_const:
            return f"const {self.base_typename}"
        return self.base_typename

    def _build_declarator(self):
        """Return ``(left, right)`` such that ``left + <name> + right`` is
        the C declarator part that goes after the base type.

        Walker yields layers outer-to-inner; the C spelling reads
        inner-to-outer (innermost wrap of the base is leftmost). Iterate
        in reverse to construct the declarator from the inside out:

        * pointers append to ``left`` (closest to name = rightmost ``*``);
        * arrays prepend to ``right`` (outermost array = leftmost suffix);
        * when a pointer is encountered AFTER an array has been emitted
          (i.e. the pointer wraps the array — pointer-to-array shape),
          wrap the pointer block in parens so ``[]`` binds to the
          parenthesized pointer unit instead of to the variable name.
          C declarator precedence: ``[]`` binds tighter than ``*``, so
          ``int *name[8]`` parses as "array of pointers"; ``int (*name)[8]``
          forces "pointer to array".
        """
        left = ""
        right = ""
        wrapped = False  # has `left` been wrapped in `(...)`?
        have_array_in_right = False

        for layer in reversed(self.layers):
            if layer.kind == "array":
                suffix = f"[{layer.array_size}]"
                # Outer arrays appear FIRST in the spelling suffix
                # (``int x[8][16]`` — ``[8]`` is leftmost). We're iterating
                # inner-to-outer, so prepend each new array to ``right``.
                right = suffix + right
                have_array_in_right = True
            elif layer.kind == "ptr":
                piece = "*" + layer.qualifier_suffix
                if have_array_in_right and not wrapped:
                    # This ptr wraps an existing array — parens needed
                    # so ``[]`` binds to the ptr unit, not to the name.
                    left = "(" + piece
                    right = ")" + right
                    wrapped = True
                else:
                    if left and left[-1].isalpha():
                        # Previous token is a qualifier (``const``);
                        # Clang spells a space before the next ``*``.
                        left = left + " " + piece
                    else:
                        left = left + piece
        return left, right


def render(
    typed,
    sep: str,
    renamer: typing.Callable[[str], str] = lambda n: n,
    prefer_canonical: bool = False,
    local_name_only: bool = False,
    typedef_aliases: typing.Mapping[str, str] = None,
    typedef_specs: typing.Mapping[str, tuple] = None,
) -> RenderedType:
    """Walk ``typed``'s Clang type via :class:`TypeHandler` and produce a
    :class:`RenderedType` with the typeref identifier substituted in place
    of the leaf segment.

    Substitution semantics:

    * If ``prefer_canonical`` is true and the innermost canonical layer is
      a basic type or ``void``, the typeref is ignored and the canonical
      leaf spelling is used as the base.
    * Otherwise, if a typeref is present, the parent's canonical layer
      chain is split at the boundary where the typeref's canonical layer
      chain begins, the outer layers are kept, and the typeref's
      identifier (``typeref.name`` if ``local_name_only`` else
      ``typeref.global_name(sep)``, run through ``renamer``) is the base.
    * Otherwise, the canonical leaf spelling (with the leading ``const``
      stripped — re-emitted by the renderer) is the base.

    ``typedef_aliases`` and ``typedef_specs`` extend the fixed-width
    preservation below to library-specific typedefs; see
    :func:`fixed_width_typedef`.
    """
    use_canonical = (
        prefer_canonical
        and typed.is_innermost_canonical_type_layer_of_basic_type_or_void
    )
    typeref = getattr(typed, "typeref", None) if not use_canonical else None
    return render_clang_type(
        typed.typehandler.clang_type,
        typeref=typeref,
        sep=sep,
        renamer=renamer,
        local_name_only=local_name_only,
        typedef_aliases=typedef_aliases,
        typedef_specs=typedef_specs,
    )


def render_clang_type(
    clang_type: clang.cindex.Type,
    typeref=None,
    sep: str = "_",
    renamer: typing.Callable[[str], str] = lambda n: n,
    local_name_only: bool = False,
    typedef_aliases: typing.Mapping[str, str] = None,
    typedef_specs: typing.Mapping[str, tuple] = None,
) -> RenderedType:
    """Same as :func:`render` but takes a Clang type + optional typeref
    directly. Used by callers that don't have a full ``Typed`` instance
    handy (e.g. ``FunctionPointer`` rendering a result type)."""
    parent_canonical = clang_type.get_canonical()
    parent_essential = [
        l
        for l in _walk_canonical(parent_canonical)
        if not _is_passthrough_layer(l)
    ]

    base_typename: str
    leaf_layer: clang.cindex.Type
    outer_layer_clang: typing.List[clang.cindex.Type]

    # Survives canonicalization so the emitted width does not depend on the
    # host codegen ran on.
    pinned = fixed_width_typedef(clang_type, typedef_aliases, typedef_specs)
    fixed_width = pinned[0] if pinned is not None else None

    if typeref is not None and getattr(typeref, "cursor", None) is not None:
        typeref_canonical = typeref.cursor.type.get_canonical()
        typeref_essential = [
            l
            for l in _walk_canonical(typeref_canonical)
            if not _is_passthrough_layer(l)
        ]
        n_typeref = len(typeref_essential)
        if 0 < n_typeref <= len(parent_essential):
            split = len(parent_essential) - n_typeref
            outer_layer_clang = parent_essential[:split]
            leaf_layer = parent_essential[split]
            ref_name = (
                typeref.name if local_name_only else typeref.global_name(sep)
            )
            base_typename = renamer(ref_name)
        else:
            outer_layer_clang, leaf_layer, base_typename = _canonical_split(
                parent_essential, fixed_width
            )
    else:
        outer_layer_clang, leaf_layer, base_typename = _canonical_split(
            parent_essential, fixed_width
        )

    raw_layers = [_layer_from_clang_type(ct) for ct in outer_layer_clang]

    is_base_const = leaf_layer.is_const_qualified()

    # libclang reports an array's *element*-type ``const`` qualification on
    # the ARRAY layer itself rather than on the element. When an incomplete
    # array decays to a pointer (``T[]`` -> ``T *``), that ``const`` must be
    # re-attached to the element's outermost representation, otherwise it is
    # silently dropped. Concretely:
    #   ``double *const[]``  must render ``double *const *`` (const belongs to
    #                        the inner element pointer), not ``double **``;
    #   ``const double[]``   must render ``const double *`` (const belongs to
    #                        the leaf base), not ``double *``.
    # The decayed array pointer itself is never const (arrays decay to plain
    # pointers). Keeping this faithful matters because the high-level
    # ``CallArgHoist`` renders the same parm from the canonical spelling
    # (which preserves the const); a mismatch makes the C compiler warn that
    # the cy* call discards a ``const`` qualifier.
    for i, ct in enumerate(outer_layer_clang):
        if ct.kind != _TypeKind.INCOMPLETEARRAY or not ct.is_const_qualified():
            continue
        if i + 1 < len(raw_layers):
            raw_layers[i + 1] = dataclasses.replace(
                raw_layers[i + 1], is_const=True
            )
        else:
            is_base_const = True

    layers = tuple(raw_layers)
    is_base_unsigned = leaf_layer.kind in (
        _TypeKind.UCHAR,
        _TypeKind.USHORT,
        _TypeKind.UINT,
        _TypeKind.ULONG,
        _TypeKind.ULONGLONG,
        _TypeKind.UINT128,
        _TypeKind.CHAR_U,
    )
    base_kind = cparser.TypeHandler.categorize_clang_type_kind(leaf_layer.kind)

    return RenderedType(
        base_typename=base_typename,
        base_kind=base_kind,
        base_clang_kind=leaf_layer.kind,
        is_base_const=is_base_const,
        is_base_unsigned=is_base_unsigned,
        layers=layers,
    )


def _walk_canonical(clang_type: clang.cindex.Type):
    """Yield each layer of ``clang_type.get_canonical()`` in outer-to-inner
    order, using :class:`TypeHandler`'s walk semantics."""
    th = cparser.TypeHandler(clang_type)
    yield from th.walk_clang_type_layers(canonical=True)


def _canonical_split(parent_essential, fixed_width_name: str = None):
    """Use the deepest layer as the leaf and its bare canonical spelling as
    the base. Caller must have filtered out passthrough layers already.

    The canonical spelling is left verbatim — the elaborated keyword
    (``struct`` / ``union`` / ``enum``) is preserved when no typeref
    substitution applies, matching the legacy code path which only stripped
    those keywords inside the typeref-substitution branch.

    ``fixed_width_name`` overrides the canonical spelling; see
    :func:`fixed_width_typedef` for why that is necessary.
    """
    leaf_layer = parent_essential[-1]
    outer = parent_essential[:-1]
    base_typename = fixed_width_name or _canonical_leaf_verbatim(leaf_layer)
    return outer, leaf_layer, base_typename


#: What each width-pinning integer typedef promises, as ``(signed, bits)``.
#: Cython knows all of these names: ``size_t``, ``ssize_t`` and ``ptrdiff_t``
#: are builtins, and the rest arrive through the ``from libc.stdint cimport *``
#: that every generated module's prolog emits.
#:
#: The pointer-width entries claim 64 bits because every platform hip-python
#: supports is 64-bit (LP64 Linux, LLP64 Windows); that assumption lives here
#: and nowhere else.
FIXED_WIDTH_INT_SPECS = {
    "int8_t": (True, 8),
    "uint8_t": (False, 8),
    "int16_t": (True, 16),
    "uint16_t": (False, 16),
    "int32_t": (True, 32),
    "uint32_t": (False, 32),
    "int64_t": (True, 64),
    "uint64_t": (False, 64),
    "ssize_t": (True, 64),
    "size_t": (False, 64),
    "ptrdiff_t": (True, 64),
    "intptr_t": (True, 64),
    "uintptr_t": (False, 64),
}

#: Integer typedefs whose whole purpose is to pin a width (or to track the
#: platform's pointer/index width).
FIXED_WIDTH_INT_TYPEDEFS = frozenset(FIXED_WIDTH_INT_SPECS)

#: How to spell a width that a module pinned itself. Every generated prolog
#: cimports these, so Cython knows them -- a library typedef such as ``hoff_t``
#: it does not.
FIXED_WIDTH_INT_SPELLINGS = {
    (True, 8): "int8_t",
    (False, 8): "uint8_t",
    (True, 16): "int16_t",
    (False, 16): "uint16_t",
    (True, 32): "int32_t",
    (False, 32): "uint32_t",
    (True, 64): "int64_t",
    (False, 64): "uint64_t",
}


def fixed_width_typedef(
    clang_type: clang.cindex.Type,
    typedef_aliases: typing.Mapping[str, str] = None,
    typedef_specs: typing.Mapping[str, tuple] = None,
) -> tuple:
    """The width ``clang_type``'s leaf pins, as ``(spelling, (signed, bits))``.

    Returns None when the declaration pins no width, i.e. when it names a plain
    C integer (or is not an integer at all).

    Canonicalizing these typedefs away would bake the *codegen host's* data
    model into the generated bindings. Clang resolves ``uint64_t`` to
    ``unsigned long`` on LP64 Linux but to ``unsigned long long`` on LLP64
    Windows, and ``unsigned long`` is 32 bits there -- so bindings generated on
    Linux and compiled on Windows would pass a 32-bit value where the header
    says 64, truncating arguments and letting the callee write 8 bytes into the
    4-byte stack slot behind every ``size_t *`` out-parameter.

    Both halves of the answer come out of this one lookup on purpose. The
    ``spelling`` is what the renderer emits and the spec is what the wrapper
    dispatch selects on; resolving them separately is what let the ``.pxd`` and
    the chosen wrapper disagree about a width in the first place.

    Two per-module maps extend the built-in table for library typedefs that pin
    a width without being part of ``<stdint.h>`` (hipFILE's ``hoff_t``, ...),
    for which Cython has no declaration of its own:

    * ``typedef_specs`` states the width directly (``{"hoff_t": (True, 64)}``);
      the stdint spelling denoting it is emitted.
    * ``typedef_aliases`` names an existing stdint typedef to spell it like
      (``{"hoff_t": "int64_t"}``) and inherits that typedef's width.

    Only the leaf is considered. Outer pointer and array layers are rendered
    separately by the caller from the canonical walk, so they are skipped here.
    """
    for layer in _walk_declared(clang_type):
        kind = layer.kind
        if cparser.TypeHandler.match_typedef_type(kind):
            name = layer.spelling
            if name.startswith("const "):
                name = name[len("const ") :]
            if typedef_specs and name in typedef_specs:
                spec = tuple(typedef_specs[name])
                return (FIXED_WIDTH_INT_SPELLINGS[spec], spec)
            if name in FIXED_WIDTH_INT_SPECS:
                return (name, FIXED_WIDTH_INT_SPECS[name])
            if typedef_aliases and name in typedef_aliases:
                alias = typedef_aliases[name]
                return (alias, FIXED_WIDTH_INT_SPECS[alias])
            # Some other alias (``hipError_t``, ``amd_comgr_status_t``, ...);
            # keep descending in case it wraps one of ours.
            continue
        if (
            cparser.TypeHandler.match_pointer_type(kind)
            or cparser.TypeHandler.match_arraylike_type(kind)
            or cparser.TypeHandler.match_elaborated_type(kind)
        ):
            continue
        break
    return None


def _walk_declared(clang_type: clang.cindex.Type):
    """Yield each layer of ``clang_type`` as written, outer-to-inner.

    The counterpart to :func:`_walk_canonical`: typedef layers are yielded
    rather than resolved away.
    """
    th = cparser.TypeHandler(clang_type)
    yield from th.walk_clang_type_layers(canonical=False)


def _is_passthrough_layer(ct: clang.cindex.Type) -> bool:
    return cparser.TypeHandler.match_elaborated_type(ct.kind)


def _canonical_leaf_verbatim(leaf: clang.cindex.Type) -> str:
    """Return the leaf's canonical spelling stripped only of the leading
    ``const`` (re-added by the renderer via ``is_base_const``).

    Elaborated keywords (``struct``/``union``/``enum``) are preserved —
    they're part of Clang's canonical spelling and the legacy renderer
    kept them verbatim when no typeref substitution applied."""
    spelling = leaf.spelling
    if spelling.startswith("const "):
        spelling = spelling[len("const ") :]
    return spelling


def _layer_from_clang_type(ct: clang.cindex.Type) -> TypeLayer:
    if cparser.TypeHandler.match_pointer_type(ct.kind):
        return TypeLayer(
            kind="ptr",
            is_const=ct.is_const_qualified(),
            is_restrict=ct.is_restrict_qualified(),
            is_volatile=ct.is_volatile_qualified(),
        )
    if ct.kind == _TypeKind.CONSTANTARRAY:
        return TypeLayer(kind="array", array_size=str(ct.get_array_size()))
    if ct.kind == _TypeKind.INCOMPLETEARRAY:
        # Match the legacy ``DEFAULT_RENAMER`` behavior: an incomplete array
        # ``T[]`` is rewritten to a pointer ``T *`` because Cython rejects
        # ``[]`` in parameter signatures and most field positions.
        return TypeLayer(kind="ptr")
    if cparser.TypeHandler.match_arraylike_type(ct.kind):
        return TypeLayer(kind="array", array_size="")
    raise RuntimeError(f"unexpected outer layer kind in typerender: {ct.kind}")
