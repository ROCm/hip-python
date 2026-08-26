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

"""Tests that a fixed-size char field is read at its declared extent.

Cython treats ``char[N]`` exactly like ``char*``: it decays the array and
converts with ``__Pyx_PyBytes_FromString``, which is ``strlen`` followed by a
copy of that many bytes. When the field holds no NUL -- ``hipUUID_t``'s ``char
bytes[16]`` is filled with 16 ASCII characters and nothing else -- ``strlen``
runs off the end of the field and the getter returns whatever followed the
struct in memory. Cython declines to fix this upstream (cython#3234), so the
extent has to reach the emitted expression: a counted slice compiles to
``PyBytes_FromStringAndSize`` and cannot leave the field.

A field the generator classifies as text takes the same bound one step
further and decodes up to the first NUL within it, so the flag that selects
that form is pinned here too.

The last two tests are the ones that pin the fix rather than its spelling.
They compile the expression the emitter produced and assert on the C function
Cython chose for it, which is where the bug lived; every text assertion above
them would pass just as happily for the decayed form.
"""

import os
import re
import subprocess
import sys

import interfacegen.tree
import pytest
from _codegen_helpers import cython_check, make_generator, write_module

HEADER = """
/* hipUUID_t-shaped: an all-ASCII payload with no room for a terminator. */
typedef struct {
    char bytes[16];
} probe_uuid_t;

/* hipDeviceProp_tR0600-shaped: NUL-terminated text next to a binary blob. */
typedef struct {
    char name[256];
    char gcn_arch_name[256];
    signed char luid[8];
} probe_props_t;

/* hipIpcMemHandle_st-shaped: an opaque unsigned payload. */
typedef struct {
    unsigned char reserved[64];
} probe_handle_t;

/* Regression guard: non-char basic arrays keep their list conversion, which
   is already length-aware. */
typedef struct {
    int vals[4];
    unsigned int flags[2];
} probe_ints_t;

/* Regression guard: rank > 1 is out of scope. */
typedef struct {
    char grid[4][8];
} probe_grid_t;
"""

# (field, declared extent) for every rank-1 char array in HEADER, signed and
# unsigned alike. The C type says nothing about whether the payload is text or
# opaque bytes, so all of them get the same bound.
CHAR_FIELDS = [
    ("bytes", 16),
    ("name", 256),
    ("gcn_arch_name", 256),
    ("luid", 8),
    ("reserved", 64),
]


def _emit(tmp_path, module_name: str = "mod", node_init=None) -> dict:
    gen = make_generator(HEADER, module_name=module_name, node_init=node_init)
    return write_module(gen, tmp_path)


def _flag_as_text(*fields):
    """A ``node_init`` in the shape the generators use to classify a field.

    The C type cannot say whether a payload is text, so the generators carry
    a table of the fields that are and set this flag on them.
    """

    def _init(node):
        if isinstance(node, interfacegen.tree.Field) and node.name in fields:
            node.is_text_char_array = True

    return _init


def _getter(pyx: str, field: str) -> str:
    """Return the ``def get_<field>(self, i):`` block, docstring included."""
    lines = pyx.splitlines()
    opening = f"    def get_{field}(self, i):"
    start = next((i for i, ln in enumerate(lines) if ln == opening), None)
    assert start is not None, f"no `get_{field}` getter emitted; pyx:\n{pyx}"
    end = start + 1
    while end < len(lines) and (
        not lines[end].strip() or lines[end].startswith("        ")
    ):
        end += 1
    return "\n".join(lines[start:end])


def _return_expression(body: str) -> str:
    """The single expression the getter returns."""
    returns = re.findall(r"^\s*return (.+)$", body, flags=re.MULTILINE)
    assert len(returns) == 1, f"expected exactly one return; getter:\n{body}"
    return returns[0]


@pytest.mark.parametrize("field,extent", CHAR_FIELDS)
def test_char_array_getter_reads_exactly_the_extent(tmp_path, field, extent):
    """The getter slices the C expression at the field's declared extent."""
    body = _getter(_emit(tmp_path)["mod.pyx"], field)
    expr = _return_expression(body)

    # The slice has to be applied to the C expression, and it has to stop at
    # the declared extent. Materializing the field first and slicing the
    # result runs the unbounded conversion, then trims the leaked bytes --
    # which reads correctly and fixes nothing.
    assert expr.endswith(f"[i].{field}[:{extent}])"), (
        f"`{field}` must be sliced at its extent {extent} on the C "
        f"expression, got: {expr}"
    )
    assert expr.startswith("<bytes>("), (
        f"`{field}` must keep the bytes return type the stub promises, "
        f"got: {expr}"
    )
    assert (
        "cdef bytes" not in body
    ), f"`{field}` must not materialize the field before slicing; getter:\n{body}"


def test_char_array_extent_is_documented(tmp_path):
    """The docstring states how many bytes come back, since that is no longer
    the length of a string but the size of the field."""
    body = _getter(_emit(tmp_path)["mod.pyx"], "bytes")
    assert "16 bytes" in body, f"getter must document its extent:\n{body}"


def test_char_array_fields_stay_read_only(tmp_path):
    """Only the read changes. The setter stays commented out, as it is for
    every other fixed-size array field."""
    pyx = _emit(tmp_path)["mod.pyx"]
    assert not re.search(
        r"^\s*def set_bytes\(", pyx, flags=re.MULTILINE
    ), f"char array fields must stay read-only; pyx:\n{pyx}"
    assert re.search(
        r"^\s*#def set_bytes\(", pyx, flags=re.MULTILINE
    ), f"the commented-out setter should be kept; pyx:\n{pyx}"
    assert re.search(
        r"@property\s*\n\s*def bytes\(self\):", pyx
    ), f"the property itself must survive; pyx:\n{pyx}"


def test_a_field_flagged_as_text_decodes_within_the_extent(tmp_path):
    """A text field returns ``str``, trimmed at the first NUL *inside* the
    field: same bound, one more step. The extent still reaches the call, since
    that is what keeps the NUL search from running off the end."""
    pyx = _emit(tmp_path, node_init=_flag_as_text("name"))["mod.pyx"]
    expr = _return_expression(_getter(pyx, "name"))
    assert expr == (
        "rocm.bindings.util.types.to_str_n("
        "&(<cymod.probe_props_t*>self._ptr)[i].name[0], 256)"
    ), f"`name` must decode its 256 bytes through to_str_n, got: {expr}"


def test_the_flag_is_per_field(tmp_path):
    """Its neighbour in the same record is untouched. A field is text because
    a table says so, not because the record it sits in has one."""
    pyx = _emit(tmp_path, node_init=_flag_as_text("name"))["mod.pyx"]
    expr = _return_expression(_getter(pyx, "gcn_arch_name"))
    assert expr.startswith(
        "<bytes>("
    ), f"an unflagged field must stay counted bytes, got: {expr}"
    assert expr.endswith(
        "[i].gcn_arch_name[:256])"
    ), f"an unflagged field must keep its bound, got: {expr}"


@pytest.mark.parametrize("field", ["vals", "flags"])
def test_non_char_basic_arrays_are_untouched(tmp_path, field):
    """``int[4]`` keeps Cython's ``__Pyx_carray_to_py_*`` list conversion,
    which already carries the length."""
    body = _getter(_emit(tmp_path)["mod.pyx"], field)
    expr = _return_expression(body)
    assert expr.endswith(
        f"[i].{field}"
    ), f"`{field}` should keep its plain array conversion, got: {expr}"
    assert "[:" not in expr, f"`{field}` must not be sliced, got: {expr}"


def test_multidimensional_char_array_is_untouched(tmp_path):
    """A ``char[4][8]`` is not a string in any language and has no single
    extent to bound; whatever the emitter does with it today, it is not
    this branch."""
    pyx = _emit(tmp_path)["mod.pyx"]
    assert not re.search(
        r"\.grid\[:\d+\]", pyx
    ), f"rank-2 arrays must not be sliced; pyx:\n{pyx}"


def test_declarations_still_compile(tmp_path):
    """The change is confined to the ``.pyx`` body, so the emitted
    declarations must be exactly as compilable as before."""
    _emit(tmp_path)
    res = cython_check(tmp_path)
    assert res.returncode == 0, (
        f"emitted declarations do not compile:\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )


# ---------------------------------------------------------------------------
# what the emitted expression compiles to
# ---------------------------------------------------------------------------

# Cython spells the counted conversion with the unbounded one as a prefix, so
# telling them apart needs the opening parenthesis.
_COUNTED_CALL = "__Pyx_PyBytes_FromStringAndSize("
_STRLEN_CALL = re.compile(r"__Pyx_PyBytes_FromString\(")


def _compile_probe(tmp_path, name: str, body: str) -> str:
    """Cythonize ``body`` against the emitted ``cymod.pxd``, return the C.

    Cython needs neither the C header nor a C compiler to get here, and the
    conversion function is already chosen by the time it writes the ``.c``.
    ``self._ptr`` and the ``[i]`` subscript are all that tie a getter's
    expression to its wrapper class, so a module-level pointer and index
    stand in for them; the cast, the field and the slice are used exactly as
    the emitter produced them.
    """
    out_dir = str(tmp_path)
    source = (
        "cimport cymod\n" "cdef void* _ptr = NULL\n" "cdef Py_ssize_t i = 0\n"
    ) + body
    with open(os.path.join(out_dir, f"{name}.pyx"), "w") as fh:
        fh.write(source)
    res = subprocess.run(
        [sys.executable, "-m", "cython", "--3str", f"{name}.pyx"],
        cwd=out_dir,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, (
        f"probe does not compile:\n{source}\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )
    with open(os.path.join(out_dir, f"{name}.c"), encoding="utf-8") as fh:
        return fh.read()


def _conversion_calls(c_source: str) -> list:
    """Lines calling the ``strlen``-based conversion, ``#define``s aside."""
    return [
        ln.strip()
        for ln in c_source.splitlines()
        if _STRLEN_CALL.search(ln) and not ln.lstrip().startswith("#define")
    ]


def test_emitted_slice_compiles_to_a_counted_conversion(tmp_path):
    """Every char field's emitted expression reaches
    ``PyBytes_FromStringAndSize``, and none of them reaches the ``strlen``
    conversion that reads past the field.
    """
    pyx = _emit(tmp_path)["mod.pyx"]
    exprs = [
        _return_expression(_getter(pyx, field)) for field, _ in CHAR_FIELDS
    ]
    c_source = _compile_probe(
        tmp_path,
        "_bounds_probe",
        "".join(
            f"def probe_{n}():\n    return {e.replace('self._ptr', '_ptr')}\n"
            for n, e in enumerate(exprs)
        ),
    )

    assert c_source.count(_COUNTED_CALL) >= len(exprs), (
        f"expected a counted conversion for each of the {len(exprs)} fields, "
        f"found {c_source.count(_COUNTED_CALL)}"
    )
    calls = _conversion_calls(c_source)
    assert not calls, (
        "a field is still converted with strlen, which reads past its "
        "declared extent:\n" + "\n".join(calls)
    )


def test_the_decayed_form_really_does_use_strlen(tmp_path):
    """Control for the test above: without the slice, Cython picks the
    ``strlen``-based conversion. If this ever stops holding, the bound is no
    longer what keeps the read inside the field.
    """
    _emit(tmp_path)
    c_source = _compile_probe(
        tmp_path,
        "_decay_probe",
        "def probe():\n    return (<cymod.probe_uuid_t*>_ptr)[i].bytes\n",
    )
    assert _conversion_calls(c_source), (
        "expected the decayed form to convert with strlen; Cython may have "
        "changed behaviour, in which case this whole fix wants revisiting"
    )


if __name__ == "__main__":
    pytest.main([__file__])
