# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
"""Unit tests for the field-declaration rendering on
``interfacegen.tree.Typed``.

These pin down the field-decl contract — base type before the variable
name, array suffix(es) after — independently of the Cython codegen
pipeline so refactors to the field/parm renderers can't silently break
them. The contract is exposed through
``Typed.render_type(...).field_decl(name)`` (token-free, layer-walking
implementation in ``interfacegen.typerender``).
"""

import sys

import pytest

# pytest doesn't add the test dir to sys.path automatically — make the
# shared scaffolding importable.
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _codegen_helpers import build_root  # noqa: E402

from interfacegen import cython  # noqa: E402


def _identity_renamer(name):
    return name


def _field_decl(header, field_name, render_name=None):
    """Build the tree, find ``<field_name>``, return ``field_decl(...)``."""
    root = build_root(header)
    field = next(
        n for n in root.walk(postorder=False)
        if isinstance(n, cython.Field) and n.name == field_name
    )
    rendered = field.render_type(
        sep="_", renamer=_identity_renamer, prefer_canonical=True
    )
    return rendered.field_decl(render_name or field_name)


@pytest.mark.parametrize(
    "field_decl_in_struct,expected",
    [
        # No array suffix — base then name (one space between).
        ("int x", "int x"),
        ("void *x", "void * x"),
        ("const int *x", "const int * x"),
        ("char **x", "char ** x"),

        # Single-dim arrays — suffix follows the name (no space before ``[``).
        ("int x[8]", "int x[8]"),
        ("void *x[8]", "void * x[8]"),
        ("void *x[256]", "void * x[256]"),
        ("const int *x[16]", "const int * x[16]"),

        # Multi-dim arrays — every trailing ``[N]`` follows the name.
        ("int x[8][16]", "int x[8][16]"),
        ("void *x[8][16]", "void * x[8][16]"),
        ("char x[3][4][5]", "char x[3][4][5]"),

        # Pointer-to-pointer with array suffix.
        ("char **x[2]", "char ** x[2]"),
    ],
)
def test_field_decl_positions_array_suffix_after_name(
    field_decl_in_struct, expected
):
    """Cython requires ``void *arr[8]`` — never ``void *[8] arr``. The
    field renderer must position trailing ``[N]`` suffixes after the
    variable name. Output is always ``<base+ptrs> <name><suffix>`` with
    a single space between the type and the name and no space between
    the name and the array suffix."""
    header = f"struct S {{ {field_decl_in_struct}; }};"
    assert _field_decl(header, "x") == expected


def test_field_decl_re_renders_valid_c_declarator():
    """For each shape, the rendered declaration is a syntactically-valid
    C field declaration — ``base name<suffix>`` with the correct number of
    spaces and bracket positions."""
    cases = [
        ("struct S { void *arr[8]; };", "arr", "void * arr[8]"),
        ("struct S { int reserved[16]; };", "reserved", "int reserved[16]"),
        ("struct S { const int *p[16]; };", "p", "const int * p[16]"),
        ("struct S { char **argv[2]; };", "argv", "char ** argv[2]"),
        ("struct S { int x; };", "x", "int x"),
    ]
    for header, name, want in cases:
        got = _field_decl(header, name)
        assert got == want, (
            f"header={header!r} name={name!r} got={got!r} want={want!r}"
        )
