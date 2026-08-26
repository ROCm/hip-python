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

"""Regression test for duplicate top-level function declarations.

Some ROCm headers declare the same function more than once, each copy
in a different surrounding doxygen ``@defgroup`` block. This is legal C
(a redeclaration is the same entity), but libclang surfaces every
``FUNCTION_DECL`` cursor distinctly. Before the fix, ``treefactory``
appended one ``Function`` node per cursor, and the cython backend then
emitted one lazy-loader block per node — producing duplicate
``cdef void* _<name>__funptr = NULL`` declarations that Cython rejects
with ``'..._funptr' redeclared`` / ``cdef variable declared after it is
used``.

The fix dedups ``FUNCTION_DECL`` cursors by spelling at the single
insertion site in ``treefactory.handle_top_level_cursor_``. This test
pins both the tree-level invariant (one ``Function`` node per spelling)
and the emitted-text invariant (one funptr declaration).

The matching upstream header bug is tracked in
``share/design/UPSTREAM_BUGS/hip_runtime_api_duplicate_function_declarations.md``.
"""

import textwrap

import pytest
from interfacegen import cython
from interfacegen.test._codegen_helpers import (
    build_root,
    make_generator,
    write_module,
)


def test_duplicate_function_decl_collapsed_to_one_node(tmp_path):
    """A function declared twice (identical signatures) must yield a
    single ``Function`` node, not one per ``FUNCTION_DECL`` cursor."""
    src = textwrap.dedent(
        """\
        typedef int my_status_t;
        my_status_t my_fn(int a, int b);
        my_status_t my_fn(int a, int b);
        """
    )
    root = build_root(src)
    fns = [
        n
        for n in root.walk()
        if isinstance(n, cython.Function) and n.name == "my_fn"
    ]
    assert len(fns) == 1, f"expected exactly one Function node, got {len(fns)}"


def test_triple_function_decl_with_doxygen_collapsed_to_one_node(tmp_path):
    """Three declarations, one preceded by a doxygen-style ``/**…*/``
    block — mirrors the real ROCm shape where each copy sits in a
    different ``@defgroup``. Still must collapse to a single node."""
    src = textwrap.dedent(
        """\
        typedef int my_status_t;
        my_status_t my_fn(int a, int b);
        /**
         * @brief Same function, different doxygen group.
         * @defgroup Other
         */
        my_status_t my_fn(int a, int b);
        my_status_t my_fn(int a, int b);
        """
    )
    root = build_root(src)
    fns = [
        n
        for n in root.walk()
        if isinstance(n, cython.Function) and n.name == "my_fn"
    ]
    assert len(fns) == 1, f"expected exactly one Function node, got {len(fns)}"


def test_duplicate_function_decl_emits_one_funptr(tmp_path):
    """End-to-end: the runtime-linking lazy loader must declare the
    function pointer exactly once. Before the fix this emitted one
    ``cdef void* _my_fn__funptr = NULL`` per duplicate cursor, which
    Cython rejects as a redeclaration."""
    src = textwrap.dedent(
        """\
        typedef int my_status_t;
        my_status_t my_fn(int a, int b);
        /**
         * @brief Same function, different doxygen group.
         */
        my_status_t my_fn(int a, int b);
        my_status_t my_fn(int a, int b);
        """
    )
    gen = make_generator(
        src,
        module_name="mod",
        runtime_linking=True,
        dll="libmy.so",
        modifiers_lazy_loader=" nogil",
    )
    files = write_module(gen, tmp_path)
    pyx = files.get("cymod.pyx", "")
    assert pyx, f"expected cymod.pyx in {list(files)}"
    assert pyx.count("cdef void* _my_fn__funptr = NULL") == 1, (
        "lazy-loader funptr must be declared exactly once; got "
        f"{pyx.count('cdef void* _my_fn__funptr = NULL')}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
