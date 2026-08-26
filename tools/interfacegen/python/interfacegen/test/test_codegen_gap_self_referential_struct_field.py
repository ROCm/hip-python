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

"""Regression test: self-referential struct pointer field.

Real-world hit (ROCm 7.14 ``hip/hip_runtime.h``):

    typedef struct hipDevResource_st {
      hipDevResourceType type;
      /* ...union/padding... */
      struct hipDevResource_st * nextResource;
    } hipDevResource;

The ``nextResource`` field references the record by its own tag
(``struct hipDevResource_st``). Because the tree factory registered the
record in the type registry only AFTER descending into its fields, the
field's typeref lookup missed, ``Field.typeref`` stayed ``None``, and the
renderer fell back to the verbatim canonical spelling — leaking the
elaborated ``struct`` keyword:

    struct hipDevResource_st * nextResource   # Cython: "Syntax error in
                                              # C variable declaration"

The fix (in ``interfacegen.treefactory``) appends the record to the type
registry BEFORE descending into its children, so the self-referential
field resolves its typeref the same way a function parameter of the same
type already does, and the renderer substitutes the bare tag name.
"""

import re

from _codegen_helpers import make_generator, write_module

HEADER_SELF_REF = """
typedef struct hipDevResource_st {
    int type;
    struct hipDevResource_st * nextResource;
} hipDevResource;
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


def test_self_referential_pointer_field_drops_struct_keyword(tmp_path):
    gen = make_generator(HEADER_SELF_REF, module_name="mod_selfref")
    files = write_module(gen, tmp_path)
    pxd = files["cymod_selfref.pxd"]
    line = _emitted_field_line(pxd, "hipDevResource_st", "nextResource")
    # Want: hipDevResource_st * nextResource   (bare tag name)
    # Got:  struct hipDevResource_st * nextResource   (Cython rejects)
    assert re.search(
        r"hipDevResource_st\s*\*\s*nextResource", line
    ), f"expected `hipDevResource_st * nextResource`-shape, got: {line!r}"
    assert (
        "struct hipDevResource_st" not in line
    ), f"emitted illegal elaborated `struct` keyword in field: {line!r}"
