# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Regression test for the foreign-record pointer-parameter fallback.

The `is_pointer_to_record(degree=1)` branch in
`_function.py:handle_in_inout_ptr_` previously hardcoded the per-type
wrapper class as `parm_innermost_type.cython_global_name`. That assumes
the wrapper class is actually emitted in the binding — which only
happens when the recipe filter admits the innermost record. For
foreign-prefix records (libc `FILE` / `_IO_FILE`, glibc `pthread_t`,
…) the filter rejects them and the wrapper class is never defined,
so Cython compile fails with `undeclared name not builtin: …`.

The fix consults `parm.node_filter(parm_innermost_type)` and falls
back to the handler-driven generic wrapper (default
`rocm.bindings.util.types.Pointer`) when the record isn't admitted.

Motivating case: `hiptensorLoggerSetFile(FILE * file)`. With the fix,
`FILE *` parameters are bound the same way as `void *` opaque
handles (e.g. `hipblasHandle_t`), via `Pointer.fromPyobj(arg)` +
`getPtr()`. Admitted records keep their per-type wrappers verbatim
(no behavior change for `hipblasContext *` etc.).
"""

import textwrap

import pytest

from interfacegen.test._codegen_helpers import make_generator, write_module


def _admit_only_admitted_prefix(node):
    """Recipe-style filter that admits exactly one struct (`admitted_t`)
    plus the function. Used to simulate a real recipe's strict-prefix
    filter where the FILE struct (or any other foreign-prefix opaque
    type) gets rejected."""
    name = node.name or ""
    return name.startswith("admitted_") or name.startswith("f_")


def test_foreign_record_pointer_falls_back_to_pointer_wrapper(tmp_path):
    """`void f_uses_file(FILE *p);` where the recipe filter rejects
    `_IO_FILE` (the canonical struct tag of `FILE` on glibc). The
    rendered pyx must use `Pointer.fromPyobj(p)` + `getPtr()` rather
    than the hardcoded `_IO_FILE.fromPyobj(p)` + `getElementPtr()`
    that would reference an undefined wrapper class."""
    src = textwrap.dedent(
        """\
        struct _IO_FILE;
        typedef struct _IO_FILE FILE;
        void f_uses_file(FILE *p);
        """
    )
    gen = make_generator(
        src,
        node_filter=_admit_only_admitted_prefix,
        modifiers_lazy_loader=" noexcept nogil",
    )
    files = write_module(gen, tmp_path)
    pyx = files.get("mod.pyx", "")
    assert pyx, f"expected mod.pyx in {list(files)}"
    # Must NOT reference an undefined wrapper class.
    assert "_IO_FILE.fromPyobj" not in pyx, (
        "foreign-prefix record `_IO_FILE` must not be referenced as "
        "a per-type wrapper class — the class is never defined in "
        "this binding and Cython would fail with `undeclared name`. "
        f"Rendered pyx:\n{pyx}"
    )
    # Must use the generic Pointer wrapper.
    assert "Pointer.fromPyobj" in pyx, (
        "foreign-prefix record pointer must fall back to the generic "
        "Pointer wrapper. Rendered pyx:\n" + pyx
    )
    # `getPtr()` is the Pointer-class accessor; `getElementPtr()` is
    # the per-type-wrapper accessor. The fallback path uses the
    # former.
    assert "getPtr()" in pyx
    # Sanity: the cy*.pxd-level signature still types the parm as
    # `_IO_FILE *` (Cython's libc.stdio.FILE would also be valid in
    # a real binding via prolog cimport, but the in-memory test uses
    # the synthetic forward-decl).
    pxd = files.get("cymod.pxd", "")
    assert "_IO_FILE *" in pxd or "FILE *" in pxd


def test_admitted_record_pointer_keeps_per_type_wrapper(tmp_path):
    """Regression guard for the no-change-when-admitted promise. With a
    record `admitted_t` that the recipe filter accepts, the rendered
    pyx must keep using the per-type wrapper class — same shape as
    `hipblasContext *` and every other admitted opaque-handle
    parameter today."""
    src = textwrap.dedent(
        """\
        struct admitted_t;
        void f_uses_admitted(struct admitted_t *p);
        """
    )
    gen = make_generator(
        src,
        node_filter=_admit_only_admitted_prefix,
        modifiers_lazy_loader=" noexcept nogil",
    )
    files = write_module(gen, tmp_path)
    pyx = files.get("mod.pyx", "")
    assert pyx, f"expected mod.pyx in {list(files)}"
    # Must keep the per-type wrapper class reference.
    assert "admitted_t.fromPyobj" in pyx, (
        "admitted record pointers must continue using the per-type "
        "wrapper class — fallback path is for foreign-prefix records "
        "only. Rendered pyx:\n" + pyx
    )
    # `getElementPtr()` is the per-type-wrapper accessor. (`getPtr()`
    # is the Pointer-class accessor.)
    assert "getElementPtr()" in pyx
    # Negative: the generic Pointer wrapper must NOT be used here.
    assert "Pointer.fromPyobj" not in pyx, (
        "admitted record path must not route through Pointer — "
        "regression in the per-type-wrapper code path. Rendered "
        f"pyx:\n{pyx}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
