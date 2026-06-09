# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Layer-A unit tests for ``interfacegen.typerender``.

Each case authors a tiny synthetic header, builds the tree, locates a
function or typedef, and asserts on the structured ``RenderedType``
produced by the layer-walking renderer. These tests pin down the
substitution + decl-rendering contract independently of the rest of the
codegen pipeline and independently of the legacy token-based renderer.
"""

import pytest

from interfacegen import cython, tree, typerender

from _codegen_helpers import build_root, find_function


def _identity_renamer(name):
    return name


def _parm_render(header, fn_name, parm_index, *, prefer_canonical=False):
    root = build_root(header)
    fn = find_function(root, fn_name)
    parm = fn.get_parm(parm_index)
    return typerender.render(
        parm,
        sep="_",
        renamer=_identity_renamer,
        prefer_canonical=prefer_canonical,
    )


# ---------------------------------------------------------------------------
# Basic types and qualifiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctype,expected_decl",
    [
        ("int", "int"),
        ("unsigned int", "unsigned int"),
        ("const int", "const int"),
        ("char", "char"),
        ("_Bool", "_Bool"),
        ("long long", "long long"),
        ("unsigned char", "unsigned char"),
    ],
)
def test_basic_scalar_decl(ctype, expected_decl):
    header = f"void f({ctype} x);"
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    assert rendered.cython_decl() == expected_decl


def test_const_scalar_drops_in_no_const_form():
    header = "void f(const int x);"
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    assert rendered.cython_decl() == "const int"
    assert rendered.cython_decl_no_const() == "int"


def test_unsigned_flag_set_for_unsigned_basic():
    header = "void f(unsigned int x);"
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    assert rendered.is_base_unsigned is True


def test_unsigned_flag_unset_for_signed_basic():
    header = "void f(int x);"
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    assert rendered.is_base_unsigned is False


# ---------------------------------------------------------------------------
# Pointer chains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctype,expected_decl,expected_no_const",
    [
        ("int *", "int *", "int *"),
        # Consecutive ``*``s pack — Clang's spelling is ``int **``.
        ("int **", "int **", "int **"),
        ("int ***", "int ***", "int ***"),
        ("const int *", "const int *", "int *"),
        # Per-pointer ``const`` is preserved as ``*const`` (joined to the
        # ``*``) — matches Clang's canonical spelling.
        ("int * const", "int *const", "int *const"),
        # Leading const stripped only by ``no_const``; per-pointer const kept.
        ("const int * const", "const int *const", "int *const"),
        # Mixed pointer+qualifier: inner ``*const`` then outer ``*``.
        ("int * const *", "int *const *", "int *const *"),
        # Pointer chain with restrict on the inner ptr.
        ("int * restrict *", "int *restrict *", "int *restrict *"),
    ],
)
def test_pointer_chain(ctype, expected_decl, expected_no_const):
    header = f"void f({ctype} x);"
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    assert rendered.cython_decl() == expected_decl
    assert rendered.cython_decl_no_const() == expected_no_const


def test_global_decl_keeps_layer_const():
    header = "void f(int * const x);"
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    # Per-pointer const is part of Clang's spelling for ``int * const``.
    assert rendered.global_decl() == "int *const"


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctype,expected_decl,field_name,expected_field",
    [
        ("int[8]", "int[8]", "x", "int x[8]"),
        ("int[8][16]", "int[8][16]", "x", "int x[8][16]"),
        ("char[3][4][5]", "char[3][4][5]", "x", "char x[3][4][5]"),
        # array of pointer — pre-name keeps `*`, suffix follows the name
        ("void *[8]", "void *[8]", "arr", "void * arr[8]"),
        ("int *[8]", "int *[8]", "arr", "int * arr[8]"),
        ("const int *[16]", "const int *[16]", "p", "const int * p[16]"),
        ("char **[2]", "char **[2]", "argv", "char ** argv[2]"),
    ],
)
def test_array_shapes(ctype, expected_decl, field_name, expected_field):
    header = f"struct S {{ {ctype} _x; }};"
    root = build_root(header)
    field = next(
        n for n in root.walk(postorder=False)
        if isinstance(n, cython.Field) and n.name == "_x"
    )
    rendered = typerender.render(
        field, sep="_", renamer=_identity_renamer, prefer_canonical=True
    )
    assert rendered.cython_decl() == expected_decl
    assert rendered.field_decl(field_name) == expected_field


def test_incomplete_array_decays_to_pointer():
    """Match the legacy ``DEFAULT_RENAMER`` behavior — an incomplete array
    ``T[]`` is rewritten to a pointer because Cython rejects ``[]`` in
    most positions."""
    header = "void f(int x[]);"
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    assert rendered.cython_decl() == "int *"


@pytest.mark.parametrize(
    "ctype,expected_decl,expected_no_const",
    [
        # Array of const-pointers: the element ``T *const`` keeps its const
        # on the inner pointer once the array decays. libclang reports that
        # element const on the ARRAY layer, so the decay must re-attach it.
        # Regression: this used to drop the const and render ``double **``,
        # which made the high-level CallArgHoist (``double *const *``) clash
        # with the cy* signature (C "discards const qualifier" warning).
        ("double *const[]", "double *const *", "double *const *"),
        # hipblas ``*Batched`` shape (e.g. hipblasHaxpyBatched's ``y``).
        (
            "unsigned short *const[]",
            "unsigned short *const *",
            "unsigned short *const *",
        ),
        # Array of const elements: the const lands on the leaf base.
        ("const double[]", "const double *", "double *"),
        ("const int[]", "const int *", "int *"),
        # Array of (non-const) pointer-to-const: unchanged, const stays deep
        # (the leading leaf const is stripped only by the no_const form).
        ("const char *[]", "const char **", "char **"),
        # Array of pointer-to-const-pointer: inner const preserved, no shift.
        ("int *const *[]", "int *const **", "int *const **"),
    ],
)
def test_incomplete_array_of_const_decays_preserve_const(
    ctype, expected_decl, expected_no_const
):
    """An incomplete array whose element type is ``const``-qualified must
    propagate that ``const`` onto the element's outermost layer when it
    decays to a pointer — not onto (or off of) the decayed array pointer."""
    header = f"void f({ctype} x);"
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    assert rendered.cython_decl() == expected_decl
    assert rendered.cython_decl_no_const() == expected_no_const


# ---------------------------------------------------------------------------
# Pointer-to-array shapes (require parentheses)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header,parm_name,expected_decl,expected_field",
    [
        # Single ptr to array — Clang spells the parens.
        (
            "void f(int (*p)[8]);",
            "p",
            "int (*)[8]",
            "int (*p)[8]",
        ),
        # Double ptr to constant array — the hiprand
        # ``hiprandGetDirectionVectors{32,64}`` shape that crashed the
        # codegen with an IndexError before paren-aware rendering.
        (
            "void f(unsigned int (**vec)[32]);",
            "vec",
            "unsigned int (**)[32]",
            "unsigned int (**vec)[32]",
        ),
        (
            "void f(unsigned long long (**vec)[64]);",
            "vec",
            "unsigned long long (**)[64]",
            "unsigned long long (**vec)[64]",
        ),
    ],
)
def test_pointer_to_array_shape_emits_parens(
    header, parm_name, expected_decl, expected_field
):
    """``T (**name)[N]`` is "pointer to pointer to array of N T". C
    declarator precedence makes ``[]`` bind tighter than ``*``, so the
    pointer block needs parens to force "ptr-to-array" — otherwise the
    spelling parses as "array of pointers" (the very different shape
    ``T *name[N]``).

    Regression: the hiprand entry points
    ``hiprandGetDirectionVectors{32,64}`` declare a parameter of shape
    ``unsigned int (**vectors)[32]``. Without paren-aware rendering the
    walker emitted ``unsigned int [32]**`` and ``Parm.cython_repr``'s
    parenthesized-pointer special case (cython.py:1583-1588), which
    splits the typename on ``)`` to insert the variable name, crashed
    with an ``IndexError`` because the spelling had no ``)`` to split on.
    """
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    assert rendered.cython_decl() == expected_decl
    assert rendered.field_decl(parm_name) == expected_field
    # Parm.cython_repr's split-on-')' fallback requires a literal ')' in
    # the rendered typename — guarantee it's there.
    assert ")" in rendered.cython_decl()


def test_field_decl_positions_array_after_name_for_pointer_array():
    """Cython requires `void * arr[8]`, not `void *[8] arr` — the field
    renderer must position the suffix after the variable name."""
    header = "struct S { void *items[8]; };"
    root = build_root(header)
    field = next(
        n for n in root.walk(postorder=False)
        if isinstance(n, cython.Field) and n.name == "items"
    )
    rendered = typerender.render(
        field, sep="_", renamer=_identity_renamer, prefer_canonical=True
    )
    assert rendered.field_decl("items") == "void * items[8]"


# ---------------------------------------------------------------------------
# Elaborated types (struct / enum / union)
# ---------------------------------------------------------------------------


def test_struct_pointer_substitutes_name():
    header = """
    struct Foo { int x; };
    void f(struct Foo *p);
    """
    rendered = _parm_render(header, "f", 0)
    assert rendered.base_typename == "Foo"
    assert rendered.cython_decl() == "Foo *"


def test_const_struct_pointer_keeps_leading_const():
    header = """
    struct Foo { int x; };
    void f(const struct Foo *p);
    """
    rendered = _parm_render(header, "f", 0)
    assert rendered.cython_decl() == "const Foo *"
    assert rendered.cython_decl_no_const() == "Foo *"


def test_struct_double_pointer():
    header = """
    struct Foo { int x; };
    void f(struct Foo **pp);
    """
    rendered = _parm_render(header, "f", 0)
    assert rendered.cython_decl() == "Foo **"


def test_enum_pointer_substitutes_name():
    header = """
    enum Bar { A, B };
    void f(enum Bar *p);
    """
    rendered = _parm_render(header, "f", 0)
    assert rendered.base_typename == "Bar"
    assert rendered.cython_decl() == "Bar *"


def test_union_pointer_substitutes_name():
    header = """
    union Baz { int a; float b; };
    void f(union Baz *p);
    """
    rendered = _parm_render(header, "f", 0)
    assert rendered.base_typename == "Baz"
    assert rendered.cython_decl() == "Baz *"


# ---------------------------------------------------------------------------
# Typedefs
# ---------------------------------------------------------------------------


def test_typedef_basic_uses_canonical_when_prefer_canonical():
    header = """
    typedef int int32;
    void f(int32 x);
    """
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    # innermost canonical is basic → typeref ignored, canonical leaf used
    assert rendered.cython_decl() == "int"


def test_typedef_basic_uses_typeref_when_not_prefer_canonical():
    header = """
    typedef int int32;
    void f(int32 x);
    """
    rendered = _parm_render(header, "f", 0, prefer_canonical=False)
    assert rendered.cython_decl() == "int32"


def test_typedef_struct_uses_struct_name():
    """For tagged ``typedef struct foo_s {...} Foo;``, libclang resolves the
    TYPE_REF on a ``Foo *`` parameter to the inner struct (``foo_s``), not
    the typedef alias. The renderer mirrors this — same as the legacy
    token-based path."""
    header = """
    typedef struct foo_s { int x; } Foo;
    void f(Foo *p);
    """
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    assert rendered.cython_decl() == "foo_s *"


def test_typedef_pointer_kept_when_not_prefer_canonical():
    header = """
    typedef int * intptr;
    void f(intptr x);
    """
    rendered = _parm_render(header, "f", 0, prefer_canonical=False)
    assert rendered.cython_decl() == "intptr"


def test_typedef_pointer_decays_to_canonical_when_prefer_canonical():
    header = """
    typedef int * intptr;
    void f(intptr x);
    """
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    # innermost canonical is INT (basic) → typeref ignored
    assert rendered.cython_decl() == "int *"


def test_opaque_handle_typedef():
    header = """
    typedef struct foo_s *Foo;
    void f(Foo h);
    """
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    # Even with prefer_canonical, innermost is RECORD, so typeref kept.
    assert rendered.cython_decl() == "Foo"


def test_anonymous_typedef_struct():
    header = """
    typedef struct { int x; } Foo;
    void f(Foo *p);
    """
    rendered = _parm_render(header, "f", 0, prefer_canonical=True)
    assert rendered.cython_decl() == "Foo *"


# ---------------------------------------------------------------------------
# Typedef + array combinations (the void_typedef_array_field gap)
# ---------------------------------------------------------------------------


def test_void_typedef_array_field_decl():
    """`typedef void* opaque_handle_t; opaque_handle_t list[8];` — the
    field renderer must produce `void * list[8]`, not `void *[8] list`,
    when prefer_canonical resolves the typedef to its canonical."""
    header = """
    typedef void* opaque_handle_t;
    struct S { opaque_handle_t list[8]; };
    """
    root = build_root(header)
    field = next(
        n for n in root.walk(postorder=False)
        if isinstance(n, cython.Field) and n.name == "list"
    )
    rendered = typerender.render(
        field, sep="_", renamer=_identity_renamer, prefer_canonical=True
    )
    assert rendered.field_decl("list") == "void * list[8]"


def test_struct_pointer_typedef_array_field_decl():
    header = """
    struct foo_s;
    typedef struct foo_s* foo_ptr_t;
    struct S { foo_ptr_t arr[4]; };
    """
    root = build_root(header)
    field = next(
        n for n in root.walk(postorder=False)
        if isinstance(n, cython.Field) and n.name == "arr"
    )
    rendered = typerender.render(
        field, sep="_", renamer=_identity_renamer, prefer_canonical=True
    )
    # Innermost canonical is RECORD (not basic/void) → keep typeref name.
    assert rendered.field_decl("arr") == "foo_ptr_t arr[4]"


# ---------------------------------------------------------------------------
# Renamer round-trip
# ---------------------------------------------------------------------------


def test_renamer_applied_to_typeref_name():
    header = """
    struct Foo { int x; };
    void f(struct Foo *p);
    """
    rendered = _parm_render(header, "f", 0)
    rendered_with_renamer = typerender.render(
        rendered.layers and _parm_render.__self__ if False else
        find_function(build_root(header), "f").get_parm(0),
        sep="_",
        renamer=lambda n: n + "_t",
    )
    assert rendered_with_renamer.cython_decl() == "Foo_t *"


def test_renamer_does_not_touch_canonical_leaf_spelling():
    header = "void f(int x);"
    root = build_root(header)
    parm = find_function(root, "f").get_parm(0)
    rendered = typerender.render(
        parm, sep="_", renamer=lambda n: n + "_x", prefer_canonical=True
    )
    # No typeref → canonical leaf spelling, renamer NOT applied to "int".
    assert rendered.cython_decl() == "int"


# ---------------------------------------------------------------------------
# Function-pointer typedef (smoke test)
# ---------------------------------------------------------------------------


def test_function_pointer_typedef_parm():
    header = """
    typedef int (*cb_t)(int, void *);
    void f(cb_t cb);
    """
    rendered = _parm_render(header, "f", 0, prefer_canonical=False)
    # Function-pointer typedef carries the typedef name as its base.
    assert rendered.base_typename == "cb_t"
    assert rendered.cython_decl() == "cb_t"
