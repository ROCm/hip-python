# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Regression test for duplicate top-level record declarations.

Some ROCm headers define the same record type at two sites in the include
graph. The motivating case is ROCm 7.14 ``rocrand.h``, which provides a
C-mode fallback definition::

    #if defined(__cplusplus)
        #include <hip/hip_runtime.h>
        #include <hip/hip_vector_types.h>
    #else
        ...
        typedef struct { uint32_t x, y, z, w; } uint4;   // fallback
    #endif

The codegen parses with ``-x c`` (so ``__cplusplus`` is undefined), while
``hiprand.h`` independently ``#include``s ``<hip/hip_runtime.h>`` — which
pulls in HIP's own ``uint4`` from ``amd_hip_vector_types.h``. libclang
therefore surfaces two top-level ``uint4`` definitions. Before the fix,
``treefactory`` appended one ``Record`` node per definition, and the cython
backend emitted a duplicate ``cdef class uint4`` / ``ctypedef struct
uint4:`` that Cython rejects with ``C class 'uint4' already defined``.

The fix dedups top-level ``STRUCT_DECL``/``UNION_DECL``/``ENUM_DECL``
cursors by canonical typename in ``treefactory`` (analogous to the
``FUNCTION_DECL`` dedup), preferring a definition over a forward
declaration. This test pins both the tree-level invariant (one ``Record``
node per canonical typename) and the emitted-text invariant (one class /
one struct block).

The upstream header bug is tracked in
``share/design/UPSTREAM_BUGS/rocrand_duplicate_uint4_definition.md``.
"""

import re
import textwrap

from interfacegen import cython
from interfacegen.test._codegen_helpers import build_root, make_generator, write_module


# Two identical ``typedef struct {...} uint4;`` definitions mimic the HIP +
# rocrand C-mode fallback collision (same canonical typename ``uint4``).
DUPLICATE_RECORD_HEADER = textwrap.dedent(
    """\
    typedef struct { unsigned int x, y, z, w; } uint4;
    typedef struct { unsigned int x, y, z, w; } uint4;
    int hiprandFoo(uint4 seed);
    """
)


def _record_nodes(root, name):
    out = []
    for n in root.walk():
        if isinstance(n, (cython.Struct, cython.Union, cython.Enum)) and n.name == name:
            out.append(n)
    return out


def test_duplicate_record_decl_collapsed_to_one_node():
    """A record defined twice (identical, same canonical typename) must
    yield a single record node, not one per ``STRUCT_DECL`` cursor."""
    root = build_root(DUPLICATE_RECORD_HEADER)
    recs = _record_nodes(root, "uint4")
    assert len(recs) == 1, f"expected exactly one uint4 record node, got {len(recs)}"


def test_forward_decl_then_definition_prefers_definition():
    """A forward declaration followed by the definition must collapse to a
    single node that is the *definition* (not the opaque forward decl)."""
    src = textwrap.dedent(
        """\
        struct Foo;
        struct Foo { int a; };
        """
    )
    root = build_root(src)
    recs = _record_nodes(root, "Foo")
    assert len(recs) == 1, f"expected exactly one Foo node, got {len(recs)}"
    assert recs[0].cursor.is_definition(), (
        "the surviving Foo node must be the definition, not the forward decl"
    )


def test_duplicate_record_decl_emits_one_class(tmp_path):
    """End-to-end: the duplicated record must be emitted exactly once as a
    ``ctypedef struct uint4:`` (cy* module) and one ``cdef class uint4``
    (wrapper module). Before the fix each appeared twice, which Cython
    rejects with ``C class 'uint4' already defined``."""
    gen = make_generator(DUPLICATE_RECORD_HEADER, module_name="mod_dup")
    files = write_module(gen, tmp_path)

    cy_pxd = files["cymod_dup.pxd"]
    assert len(re.findall(r"ctypedef struct uint4:", cy_pxd)) == 1, (
        f"expected exactly one `ctypedef struct uint4:`; full pxd:\n{cy_pxd}"
    )

    wrapper_pxd = files["mod_dup.pxd"]
    assert len(re.findall(r"cdef class uint4\(", wrapper_pxd)) == 1, (
        f"expected exactly one `cdef class uint4(`; full pxd:\n{wrapper_pxd}"
    )
