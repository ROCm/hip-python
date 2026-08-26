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

"""Tests for the ``ListOf*`` allocator + sequence protocol on
``rocm.bindings.util.types``.

The ``ListOfPointer`` / ``ListOfInt`` / ``ListOfLong`` / ``ListOfUnsigned`` /
``ListOfUnsignedLong`` / ``ListOfInt64`` / ``ListOfUInt64`` / ``ListOfBytes``
wrapper classes gained:

  * ``allocate(count)`` — an owned, zero-initialized buffer of ``count``
    elements (freed on garbage collection), with a known length.
  * ``__len__`` / ``__getitem__`` (int + slice, negative indices) /
    ``__iter__`` — element access at the natural element granularity
    (not the byte-offset semantics inherited from ``Pointer``).
  * ``to_list`` / ``to_tuple`` — Python conversions.

Length is tracked when the instance is created via ``allocate`` or from
a ``list`` / ``tuple``. Instances wrapping a raw pointer (``fromObj`` of
an ``int`` / buffer, the codegen ``fromPtr`` return path) have an unknown
length and must raise ``TypeError`` on any length-dependent operation.

These run against the *built* extension, so they exercise the compiled
Cython directly.
"""

import ctypes

import pytest
from rocm.bindings.util import types as _t

_INT_CLASSES = [
    (_t.ListOfInt, ctypes.c_int),
    (_t.ListOfLong, ctypes.c_long),
    (_t.ListOfUnsigned, ctypes.c_uint),
    (_t.ListOfUnsignedLong, ctypes.c_ulong),
    (_t.ListOfInt64, ctypes.c_int64),
    (_t.ListOfUInt64, ctypes.c_uint64),
]


def _write_scalars(wrapper, ctype, values):
    """Write ``values`` into the wrapper's owned buffer via ctypes."""
    arr = (ctype * len(values)).from_address(int(wrapper))
    for i, v in enumerate(values):
        arr[i] = v


# ---------------------------------------------------------------------------
# allocate + zero-init + length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,ctype", _INT_CLASSES)
def test_allocate_is_zero_initialized_with_known_length(cls, ctype):
    w = cls.allocate(5)
    assert len(w) == 5
    assert w.to_list() == [0, 0, 0, 0, 0]


@pytest.mark.parametrize("cls,ctype", _INT_CLASSES)
def test_allocate_zero_count_is_empty(cls, ctype):
    w = cls.allocate(0)
    assert len(w) == 0
    assert w.to_list() == []
    assert list(w) == []


@pytest.mark.parametrize("cls,ctype", _INT_CLASSES)
def test_allocate_negative_count_raises(cls, ctype):
    with pytest.raises(ValueError):
        cls.allocate(-1)


# ---------------------------------------------------------------------------
# element access / iteration / conversion (integer ListOf*)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,ctype", _INT_CLASSES)
def test_getitem_iter_and_conversions(cls, ctype):
    w = cls.allocate(4)
    _write_scalars(w, ctype, [0, 10, 20, 30])
    assert w[0] == 0
    assert w[2] == 20
    assert w[-1] == 30  # negative index
    assert w[1:3] == [10, 20]  # slice -> list
    assert list(w) == [0, 10, 20, 30]
    assert w.to_list() == [0, 10, 20, 30]
    assert w.to_tuple() == (0, 10, 20, 30)


@pytest.mark.parametrize("cls,ctype", _INT_CLASSES)
def test_getitem_out_of_range_raises_indexerror(cls, ctype):
    w = cls.allocate(3)
    with pytest.raises(IndexError):
        w[3]
    with pytest.raises(IndexError):
        w[-4]


@pytest.mark.parametrize("cls,ctype", _INT_CLASSES)
def test_constructed_from_list_tracks_length(cls, ctype):
    w = cls([1, 2, 3])
    assert len(w) == 3
    assert w.to_list() == [1, 2, 3]


# ---------------------------------------------------------------------------
# fixed-width ListOf* — 8-byte elements on every data model
# ---------------------------------------------------------------------------


def _fill(wrapper, offset, count, value):
    """Write ``value`` into ``count`` raw bytes starting at ``offset``."""
    raw = (ctypes.c_ubyte * (offset + count)).from_address(int(wrapper))
    for i in range(offset, offset + count):
        raw[i] = value


@pytest.mark.parametrize(
    "cls,expected",
    [(_t.ListOfInt64, -1), (_t.ListOfUInt64, 0xFFFFFFFFFFFFFFFF)],
)
def test_fixed_width_elements_are_eight_bytes(cls, expected):
    # The point of these classes: the element is 64 bits wherever the
    # extension was compiled, unlike ``long`` / ``unsigned long`` which are
    # 32 bits on Windows. Setting all bits of the first slot must read back
    # as a full 64-bit value and must not spill into the second slot.
    w = cls.allocate(2)
    _fill(w, 0, 8, 0xFF)
    assert w[0] == expected
    assert w[1] == 0


@pytest.mark.parametrize(
    "cls,value",
    [(_t.PointerToInt64, -(2**40)), (_t.PointerToUInt64, 2**40)],
)
def test_fixed_width_value_survives_beyond_32_bits(cls, value):
    w = cls.allocate()
    w.value = value
    assert w.value == value


# ---------------------------------------------------------------------------
# ListOfPointer — elements come back as Pointer
# ---------------------------------------------------------------------------


def test_listofpointer_allocate_elements_are_pointers():
    w = _t.ListOfPointer.allocate(3)
    assert len(w) == 3
    arr = (ctypes.c_void_p * 3).from_address(int(w))
    arr[1] = 0xDEADBEEF
    el = w[1]
    assert isinstance(el, _t.Pointer)
    assert int(el) == 0xDEADBEEF
    assert [int(x) for x in w] == [0, 0xDEADBEEF, 0]


def test_listofpointer_from_list_roundtrips_addresses():
    w = _t.ListOfPointer([0x10, 0x20])
    assert [int(x) for x in w.to_list()] == [0x10, 0x20]


# ---------------------------------------------------------------------------
# ListOfBytes — elements come back as bytes (None for NULL slots)
# ---------------------------------------------------------------------------


def test_listofbytes_from_list_reads_back_bytes():
    w = _t.ListOfBytes(["aa", "bb"])
    assert len(w) == 2
    assert w.to_list() == [b"aa", b"bb"]


def test_listofbytes_allocate_slots_read_back_none():
    w = _t.ListOfBytes.allocate(2)
    assert len(w) == 2
    assert w[0] is None
    assert w.to_list() == [None, None]


# ---------------------------------------------------------------------------
# unknown-length instances raise on length-dependent operations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        _t.ListOfInt,
        _t.ListOfUnsigned,
        _t.ListOfUnsignedLong,
        _t.ListOfPointer,
        _t.ListOfBytes,
    ],
)
def test_unknown_length_raises_typeerror(cls):
    # fromObj(int) wraps a raw address -> length is unknown (_len == -1).
    w = cls.fromObj(0)
    with pytest.raises(TypeError):
        len(w)
    with pytest.raises(TypeError):
        w[0]
    with pytest.raises(TypeError):
        list(w)
    with pytest.raises(TypeError):
        w.to_list()


# ---------------------------------------------------------------------------
# PointerTo* — rank-0 scalar-pointer subclasses of the matching ListOf*
# ---------------------------------------------------------------------------


_POINTER_CLASSES = [
    (_t.PointerToInt, _t.ListOfInt, ctypes.c_int),
    (_t.PointerToLong, _t.ListOfLong, ctypes.c_long),
    (_t.PointerToUnsigned, _t.ListOfUnsigned, ctypes.c_uint),
    (_t.PointerToUnsignedLong, _t.ListOfUnsignedLong, ctypes.c_ulong),
    (_t.PointerToInt64, _t.ListOfInt64, ctypes.c_int64),
    (_t.PointerToUInt64, _t.ListOfUInt64, ctypes.c_uint64),
]


@pytest.mark.parametrize("cls,base,ctype", _POINTER_CLASSES)
def test_pointerto_is_listof_subclass(cls, base, ctype):
    assert issubclass(cls, base)
    assert isinstance(cls.allocate(), base)


@pytest.mark.parametrize("cls,base,ctype", _POINTER_CLASSES)
def test_pointerto_allocate_defaults_to_single_slot(cls, base, ctype):
    w = cls.allocate()
    assert len(w) == 1
    assert w[0] == 0
    assert w.value == 0
    assert w.to_list() == [0]


@pytest.mark.parametrize("cls,base,ctype", _POINTER_CLASSES)
def test_pointerto_value_get_set_roundtrip(cls, base, ctype):
    w = cls.allocate()
    w.value = 123
    assert w.value == 123
    assert w[0] == 123
    # value tracks a mutation performed through the raw buffer too.
    _write_scalars(w, ctype, [456])
    assert w.value == 456


@pytest.mark.parametrize("cls,base,ctype", _POINTER_CLASSES)
def test_pointerto_allocate_count_still_supported(cls, base, ctype):
    w = cls.allocate(3)
    assert len(w) == 3
    assert w.to_list() == [0, 0, 0]


@pytest.mark.parametrize("cls,base,ctype", _POINTER_CLASSES)
def test_pointerto_from_single_element_list(cls, base, ctype):
    w = cls([7])
    assert isinstance(w, cls)
    assert len(w) == 1
    assert w.to_list() == [7]
    assert w.value == 7
    # a single-element tuple works too, via fromObj / fromPyobj.
    w2 = cls.fromObj((9,))
    assert w2.value == 9


@pytest.mark.parametrize("cls,base,ctype", _POINTER_CLASSES)
@pytest.mark.parametrize("seq", [[], [1, 2], (1, 2, 3)])
def test_pointerto_rejects_non_scalar_sequence(cls, base, ctype, seq):
    # a PointerTo* wraps a single scalar: list/tuple init must have len 1.
    with pytest.raises(ValueError):
        cls(seq)
    with pytest.raises(ValueError):
        cls.fromObj(seq)


@pytest.mark.parametrize("cls,base,ctype", _POINTER_CLASSES)
def test_pointerto_fromobj_passthrough_identity(cls, base, ctype):
    w = cls.allocate()
    assert cls.fromObj(w) is w


@pytest.mark.parametrize("cls,base,ctype", _POINTER_CLASSES)
def test_pointerto_null_value_raises(cls, base, ctype):
    w = cls.fromObj(0)  # NULL address
    with pytest.raises(ValueError):
        w.value
    with pytest.raises(ValueError):
        w.value = 1
