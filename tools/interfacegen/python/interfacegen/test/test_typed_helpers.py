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


# ---------------------------------------------------------------------------
# has_symbol codegen helper emission (Issue 4b in the typed-waddling-meteor
# plan). When `runtime_linking=True`, the generator must emit:
#   - in cy<mod>.pxd: a public `cdef bint __has_symbol(const char* name) noexcept nogil`
#                     declaration so the high-level python module can cimport it.
#   - in cy<mod>.pyx: the matching impl alongside `__init` / `__init_symbol`.
#   - in <mod>.pyx:   a python-visible `def has_symbol(name) -> bool:` wrapper
#                     that delegates to `cy<mod>.__has_symbol`.
# When `runtime_linking=False`, none of those should appear.
# ---------------------------------------------------------------------------

import os  # noqa: E402

from _codegen_helpers import make_generator  # noqa: E402


_HEADER_FOR_HAS_SYMBOL = """
int example_func(int x);
"""


def _write_module(generator, tmp_path):
    out_dir = str(tmp_path)
    os.makedirs(out_dir, exist_ok=True)
    generator.write_module_files(out_dir)
    return {
        fname: open(os.path.join(out_dir, fname)).read()
        for fname in os.listdir(out_dir)
    }


def test_has_symbol_emitted_under_runtime_linking(tmp_path):
    gen = make_generator(
        _HEADER_FOR_HAS_SYMBOL,
        module_name="hsmod",
        modifiers_lazy_loader=" noexcept nogil",
    )
    gen.runtime_linking = True
    gen.dll = "libhsmod.so"
    files = _write_module(gen, tmp_path)

    pxd = files["cyhsmod.pxd"]
    assert "cdef bint __has_symbol(const char* name) noexcept nogil" in pxd, (
        f"expected `__has_symbol` declaration in cy*.pxd; full pxd:\n{pxd}"
    )

    cy_pyx = files["cyhsmod.pyx"]
    assert "cdef bint __has_symbol(const char* name) noexcept nogil:" in cy_pyx, (
        f"expected `__has_symbol` impl in cy*.pyx; full pyx:\n{cy_pyx}"
    )
    # Sanity: the impl actually delegates to loader.has_symbol.
    assert "loader.has_symbol(" in cy_pyx, (
        f"expected delegation to loader.has_symbol; full pyx:\n{cy_pyx}"
    )

    py_pyx = files["hsmod.pyx"]
    assert "def has_symbol(name) -> bool" in py_pyx, (
        f"expected python-visible `has_symbol` def; full pyx:\n{py_pyx}"
    )
    # And the python wrapper must call into the cy* helper.
    assert "cyhsmod.__has_symbol" in py_pyx, (
        f"expected python wrapper to call cyhsmod.__has_symbol; full pyx:\n{py_pyx}"
    )
    # And `has_symbol` must show up in __all__.
    assert '"has_symbol"' in py_pyx, (
        f"expected `has_symbol` in __all__; full pyx:\n{py_pyx}"
    )


def test_has_symbol_omitted_without_runtime_linking(tmp_path):
    gen = make_generator(
        _HEADER_FOR_HAS_SYMBOL,
        module_name="hsmod_off",
        modifiers_lazy_loader=" noexcept nogil",
    )
    # runtime_linking left at the default (False).
    files = _write_module(gen, tmp_path)

    pxd = files["cyhsmod_off.pxd"]
    assert "__has_symbol" not in pxd, (
        f"`__has_symbol` should NOT be declared without runtime_linking; pxd:\n{pxd}"
    )
    py_pyx = files["hsmod_off.pyx"]
    assert "def has_symbol(" not in py_pyx, (
        f"`has_symbol` should NOT be emitted without runtime_linking; pyx:\n{py_pyx}"
    )


# ---------------------------------------------------------------------------
# Unit tests for `cython.CallArgHoist` — the structured pre-block hoist
# descriptor used by `Function._render_python_interface_c_interface_call`'s
# `_append_call_arg`. Locks down both shapes (plain / wrapper-bound) and
# the inline-form fallback used by the with-gil emitter.
# ---------------------------------------------------------------------------


def test_call_arg_hoist_plain_renders_single_cdef():
    """A plain hoist (no python wrapper temporary, e.g.
    ``IntEnumParm.value``) renders as a single ``cdef <T> arg = expr``
    line and its inline form is just the expression.
    """
    h = cython.CallArgHoist(
        c_type="cyhip.hipMemcpyKind",
        plain_expr="kind.value",
    )
    assert h.inline_expr() == "kind.value"
    assert h.render_prehoist("_cy_f__arg_3") == (
        "cdef cyhip.hipMemcpyKind _cy_f__arg_3 = kind.value"
    )


def test_call_arg_hoist_wrapper_bound_splits_into_two_cdefs():
    """A wrapper-bound hoist (Python wrapper temporary owns the C
    buffer the pointer references) renders two cdef lines: one typed
    cdef-class binding for the wrapper (so it outlives the with-nogil
    block) and one cdef for the extracted pointer.

    Both cdefs use the combined ``cdef T x = <T>expr`` form. The
    former bare-cdef + separate-assignment split for the ``*const *``
    shape (a Cython 3.0.x codegen-bug workaround) has been removed now
    that the project enforces a Cython >= 3.1.0 build floor; see
    ``test_call_arg_hoist_double_const_pointer_uses_combined_form``.
    The trailing-const strip is still exercised by
    ``test_call_arg_hoist_trailing_const_strips_const``.
    """
    h = cython.CallArgHoist(
        c_type="const char *const *",
        wrapper_class="rocm.bindings.util.types.ListOfBytes",
        wrapper_factory="rocm.bindings.util.types.ListOfBytes.fromPyobj(options)",
        pointer_extract="getPtr()",
        cast_open="<const char *const *>",
    )
    out = h.render_prehoist("_cy_f__arg_2")
    expected = (
        "cdef rocm.bindings.util.types.ListOfBytes _cy_f__arg_2_obj = "
        "rocm.bindings.util.types.ListOfBytes.fromPyobj(options)\n"
        "cdef const char *const * _cy_f__arg_2 = "
        "<const char *const *>_cy_f__arg_2_obj.getPtr()"
    )
    assert out == expected
    # Inline form (with-gil emitter) is the equivalent single
    # expression — wrapper temporary lives for the call duration.
    assert h.inline_expr() == (
        "<const char *const *>"
        "rocm.bindings.util.types.ListOfBytes.fromPyobj(options).getPtr()"
    )


def test_call_arg_hoist_wrapper_bound_records_by_value_dereferences():
    """Record-by-value hoists use ``.getElementPtr()[0]`` as the
    extract — the [0] subscript stays attached to the obj reference,
    not as a postfix on the cast.
    """
    h = cython.CallArgHoist(
        c_type="cymod.point_st",
        wrapper_class="point_st",
        wrapper_factory="point_st.fromPyobj(pt)",
        pointer_extract="getElementPtr()[0]",
    )
    out = h.render_prehoist("_cy_op_rec__arg_0")
    # The renderer emits the combined cdef-with-initializer form.
    assert out == (
        "cdef point_st _cy_op_rec__arg_0_obj = point_st.fromPyobj(pt)\n"
        "cdef cymod.point_st _cy_op_rec__arg_0 = "
        "_cy_op_rec__arg_0_obj.getElementPtr()[0]"
    )


def test_call_arg_hoist_double_const_pointer_uses_combined_form():
    """Regression: the prehoist for a wrapper-bound arg whose c_type
    contains ``*const *`` (e.g. the ``const char *const *`` shape used
    for ``hiprtcCompileProgram``'s ``options`` arg) emits the combined
    ``cdef T x = <T>expr`` form on a single line.

    Cython 3.0.x miscompiled that form for this exact shape (it parsed
    both halves but silently dropped the initializer, leaving a NULL
    local that segfaulted the vendor library on first dereference), so
    the generator used to split it into a bare ``cdef T x`` plus a
    separate ``x = <T>expr`` assignment. That workaround was removed
    once the project pinned a Cython >= 3.1.0 build floor (3.1+
    compiles the combined form correctly). The historical workaround
    and the cross-version repro live in
    ``share/design/UPSTREAM_BUGS/cython_const_pointer_initializer_bug.md``.
    """
    h = cython.CallArgHoist(
        c_type="const char *const *",
        wrapper_class="ListOfBytes",
        wrapper_factory="ListOfBytes.fromPyobj(options)",
        pointer_extract="getPtr()",
        cast_open="<const char *const *>",
    )
    rendered = h.render_prehoist("_cy_arg_2")
    # Combined cdef-with-initializer form on a single line.
    assert "cdef const char *const * _cy_arg_2 = " in rendered, (
        f"render_prehoist must emit the combined form for double-const "
        f"pointer (safe on Cython >= 3.1.0). Got:\n{rendered}"
    )
    # The former split form (bare cdef + separate assignment) must
    # NOT reappear.
    assert "cdef const char *const * _cy_arg_2\n_cy_arg_2 = " not in rendered, (
        f"render_prehoist regressed to the obsolete split form:\n{rendered}"
    )


def test_call_arg_hoist_trailing_const_strips_const():
    """Regression: the prehoist for a wrapper-bound arg whose c_type
    ends in a *trailing* ``const`` (e.g. ``void *const`` for the C
    param ``hipsparseMatDescr_t`` — a ``void *``-typedef'd opaque
    handle declared ``const`` at the parameter site) must **strip
    the trailing const from the cdef local type** while keeping
    the cast on the rhs.

    Cython 3.1+ rejects ``cdef T x = ...`` and any subsequent
    ``x = ...`` when T is a const-qualified type, with the error
    ``Assignment to const 'x'``. Cython 3.0.12 silently miscompiles
    the same form (the codegen bug also applies to trailing-const
    pointers, not just the ``*const *`` shape). Both behaviours
    are blocking. The fix: strip the outermost trailing ``const``
    qualifier from the cdef *local* type only — locals don't need
    C-style const protection (assigned exactly once at the prehoist
    line). The cast on the rhs keeps the original c_type so the
    expression is still type-correct against the C function's
    parameter type at the call site.

    In the wild this regressed when the unconditional split-form
    fix was first applied: ``hipsparseCopyMatDescr`` and
    ``hipsparseGetMatType`` (among others) failed to compile with
    ``Assignment to const ...``.
    """
    h = cython.CallArgHoist(
        c_type="void *const",
        wrapper_class="rocm.bindings.util.types.Pointer",
        wrapper_factory="rocm.bindings.util.types.Pointer.fromPyobj(src)",
        pointer_extract="getPtr()",
        cast_open="<void *const>",
    )
    rendered = h.render_prehoist("_cy_arg_1")
    # Combined cdef-with-initializer form, with trailing const
    # stripped from the cdef local type — the cast keeps it.
    expected = (
        "cdef rocm.bindings.util.types.Pointer _cy_arg_1_obj = "
        "rocm.bindings.util.types.Pointer.fromPyobj(src)\n"
        "cdef void * _cy_arg_1 = "
        "<void *const>_cy_arg_1_obj.getPtr()"
    )
    assert rendered == expected, (
        f"trailing-const must be stripped from cdef local, got:\n{rendered}"
    )
    # The const-qualified cdef must NOT be emitted (Cython rejects).
    assert "cdef void *const _cy_arg_1" not in rendered, (
        f"trailing-const cdef must be stripped (Cython rejects const "
        f"locals), got:\n{rendered}"
    )


def test_call_arg_hoist_const_substring_in_identifier_not_stripped():
    """The trailing-const strip must match the C keyword ``const`` as
    a *whole word* — substrings like ``constellation_t`` (a hypothetical
    typedef ending in 'const' as part of its name) must NOT be touched.
    """
    h = cython.CallArgHoist(
        c_type="constellation_t *",
        wrapper_class="W",
        wrapper_factory="W.fromPyobj(c)",
        pointer_extract="getPtr()",
        cast_open="<constellation_t *>",
    )
    rendered = h.render_prehoist("_cy_arg_0")
    # The cdef type must be unchanged from the c_type — no strip.
    assert "cdef constellation_t * _cy_arg_0" in rendered, (
        f"identifier ending in 'const' must not be stripped, got:\n{rendered}"
    )


def test_call_arg_hoist_rejects_mixed_or_empty_init():
    """Constructor enforces that exactly one shape is supplied."""
    # Both shapes set → reject.
    with pytest.raises(AssertionError):
        cython.CallArgHoist(
            c_type="int",
            plain_expr="x",
            wrapper_class="W",
            wrapper_factory="W.fromPyobj(x)",
            pointer_extract="getPtr()",
        )
    # Wrapper shape with one field missing → reject.
    with pytest.raises(AssertionError):
        cython.CallArgHoist(
            c_type="int *",
            wrapper_class="W",
            # wrapper_factory missing
            pointer_extract="getPtr()",
        )
    # Neither shape set → reject.
    with pytest.raises(AssertionError):
        cython.CallArgHoist(c_type="int")
