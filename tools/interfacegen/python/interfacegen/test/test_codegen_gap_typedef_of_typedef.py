# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

"""Regression test for the typedef-of-typedef chain handling gap.

`treefactory.handle_typedef_cursor_` matches typedef cursors against
the literal 2-layer shapes
``[TYPEDEF, BASIC] / [TYPEDEF, VOID] / [TYPEDEF, POINTER] /
[TYPEDEF, RECORD-or-ENUM]``. A typedef whose immediate underlying
type is itself a typedef (``typedef A B;`` where ``A`` is
``typedef X A;``) has layer shape ``[TYPEDEF, TYPEDEF, ...]`` and
falls off the if-elif chain. The cursor logs a ``Did not handle
TYPEDEF`` warning and never enters the AST — downstream the renderer
falls back to the canonical spelling for parameter types, leaking
``struct Foo *`` into Cython function signatures (which Cython
rejects with ``Expected ')', found '*'``).

HSA's `hsa_ext_finalize.h` is the motivating real-world case:

    typedef struct BrigModuleHeader* BrigModule_t;
    typedef BrigModule_t hsa_ext_module_t;
    hsa_status_t hsa_ext_program_add_module(
        hsa_ext_program_t program, hsa_ext_module_t module);

The fix: a new `Typedef.match_typedefed_typedef` matcher + a new
treefactory branch that creates a `Typedef` node and links its
`typeref` to the underlying typedef, so the renderer substitutes the
alias name (`hsa_ext_module_t`) instead of the canonical spelling.
"""

import textwrap

import pytest
from interfacegen import cython, tree
from interfacegen.test._codegen_helpers import (
    build_root,
    find_function,
    make_generator,
    write_module,
)


def test_typedef_of_typedef_basic_chain_admitted():
    """`typedef int A; typedef A B;` — B's cursor type is
    `[TYPEDEF, TYPEDEF, INT]`. Without the new matcher B wouldn't
    enter the AST at all."""
    src = textwrap.dedent(
        """\
        typedef int A;
        typedef A B;
        void f(B *p);
        """
    )
    root = build_root(src)
    typedefs = {
        n.name: n
        for n in root.walk(postorder=False)
        if isinstance(n, tree.Typedef)
    }
    assert "A" in typedefs, "single-step typedef A must still be admitted"
    assert "B" in typedefs, (
        "typedef-of-typedef B must be admitted via the new "
        "match_typedefed_typedef branch"
    )
    # B's typeref points at A so the renderer can substitute the alias.
    assert typedefs["B"].typeref is not None
    assert typedefs["B"].typeref.name == "A"


def test_typedef_of_typedef_pointer_chain_renders_alias_not_canonical(
    tmp_path,
):
    """`typedef struct foo_s* PFoo; typedef PFoo PFoo_alias;` — the
    BRIG `hsa_ext_module_t` shape. The function decl that takes
    `PFoo_alias` must render as `PFoo_alias`, not the canonical
    `struct foo_s *`. Cython rejects the latter as
    `Expected ')', found '*'`.
    """
    src = textwrap.dedent(
        """\
        struct foo_s;
        typedef struct foo_s* PFoo;
        typedef PFoo PFoo_alias;
        void g(PFoo_alias x);
        """
    )
    gen = make_generator(src)
    files = write_module(gen, tmp_path)
    pxd = files.get("cymod.pxd", "")
    assert pxd, f"expected cymod.pxd in {list(files)}"
    # Both the single-step and the chain typedef must be ctypedef'd.
    assert "ctypedef" in pxd
    assert "PFoo" in pxd
    assert "PFoo_alias" in pxd
    # Function signature must reference the alias, not the canonical
    # `struct foo_s *` spelling that breaks Cython compile.
    assert "PFoo_alias x" in pxd or "PFoo_alias) " in pxd, (
        "g's parameter must render as PFoo_alias — found:\n" + pxd
    )
    assert "struct foo_s * x" not in pxd, (
        "canonical `struct foo_s *` must NOT leak into the signature; "
        "the typedef-chain renderer fallback regression has reappeared"
    )


def test_single_step_typedef_still_uses_basic_matcher():
    """Negative regression: a single-step `typedef int A;` (layers
    `[TYPEDEF, INT]`) must continue to go through
    `match_typedefed_basic_type`, NOT the new chain matcher. The
    chain matcher only fires when the immediate underlying is itself
    a typedef (layers `[TYPEDEF, TYPEDEF, ...]`).

    Verified indirectly: a single-step int-typedef's basic-branch
    path leaves typeref=None (the basic type has no AST entity to
    point at). If the chain matcher had hijacked it, typeref would
    be set or there'd be a `Did not handle` warning at parse time.
    """
    src = "typedef int A;\nvoid h(A x);\n"
    root = build_root(src)
    typedefs = {
        n.name: n
        for n in root.walk(postorder=False)
        if isinstance(n, tree.Typedef)
    }
    assert "A" in typedefs
    assert typedefs["A"].typeref is None, (
        "single-step int-typedef must take the basic-type branch, "
        "which leaves typeref unset"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
