# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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

"""Regression test: array field whose element type is a typedef of ``void *``.

Real-world hit:

    typedef void* amdsmi_processor_handle;
    typedef struct {
      uint32_t count;
      amdsmi_processor_handle processor_list[256];
      uint64_t reserved[15];
    } amdsmi_topology_nearest_t;

Before the fix, ``Field.cython_repr`` joined typename + name without
recognising that the canonical typename ends in ``[N]`` and emitted
``void *[256] processor_list`` — Cython rejects with "Empty declarator".

The fix lives in ``interfacegen.typerender.RenderedType.field_decl``,
which positions trailing ``[N]`` suffixes after the variable name
regardless of pointer indirection in the base type. Covers the same
shape with struct-pointer typedefs
(``typedef struct foo* foo_t; foo_t arr[8];``).
"""

import re

from _codegen_helpers import make_generator, write_module

HEADER_VOID_TYPEDEF = """
typedef void* handle_t;
typedef struct container_s {
    unsigned int n;
    handle_t arr[8];
} container_t;
"""


HEADER_STRUCT_TYPEDEF = """
struct foo;
typedef struct foo* foo_t;
typedef struct container2_s {
    unsigned int n;
    foo_t arr[8];
} container2_t;
"""


def _emitted_field_line(text: str, struct_name: str, field_name: str) -> str:
    """Pull the line that declares `field_name` inside `struct_name`."""
    in_struct = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(
            f"cdef struct {struct_name}"
        ) or stripped.startswith(f"ctypedef struct {struct_name}"):
            in_struct = True
            continue
        if in_struct:
            if not line.startswith(" "):
                in_struct = False
                continue
            if field_name in stripped:
                return stripped
    raise AssertionError(
        f"field {field_name} not found in struct {struct_name}"
    )


def test_void_typedef_array_field_renders_suffix_after_name(tmp_path):
    gen = make_generator(HEADER_VOID_TYPEDEF, module_name="mod_v")
    files = write_module(gen, tmp_path)
    pxd = files["cymod_v.pxd"]
    line = _emitted_field_line(pxd, "container_s", "arr")
    # Want: void *arr[8]   (suffix after name)
    # Got:  void *[8] arr  (suffix before name — Cython rejects)
    assert re.search(
        r"void\s*\*\s*arr\s*\[\s*8\s*\]", line
    ), f"expected `void *arr[8]`-shape, got: {line!r}"
    assert not re.search(
        r"void\s*\*\s*\[\s*8\s*\]\s*arr", line
    ), f"emitted broken `void *[8] arr` shape: {line!r}"


def test_struct_pointer_typedef_array_field_renders_suffix_after_name(
    tmp_path,
):
    """For a `typedef struct foo* foo_t;` typedef, the renderer keeps the
    typedef name (no canonical expansion to ``foo *``) because the
    innermost canonical layer is a record, not a basic type — so
    ``prefer_canonical=True`` falls back to the typeref. The emitted line
    is ``foo_t arr[8]``. ``field_decl`` still positions ``[8]`` after the
    name; the broken ``T *[N] name`` shape never appears here because the
    type stays a single token.
    """
    gen = make_generator(HEADER_STRUCT_TYPEDEF, module_name="mod_s")
    files = write_module(gen, tmp_path)
    pxd = files["cymod_s.pxd"]
    line = _emitted_field_line(pxd, "container2_s", "arr")
    assert re.search(
        r"foo_t\s+arr\s*\[\s*8\s*\]", line
    ), f"expected `foo_t arr[8]`-shape, got: {line!r}"
    assert not re.search(
        r"\[\s*8\s*\]\s*arr", line
    ), f"emitted broken `[8] arr` shape: {line!r}"
