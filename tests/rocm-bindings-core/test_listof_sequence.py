# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Tests for the ``ListOf*`` allocator + sequence protocol on
``rocm.bindings.util.types``.

The ``ListOfPointer`` / ``ListOfInt`` / ``ListOfLong`` / ``ListOfUnsigned`` /
``ListOfUnsignedLong`` / ``ListOfBytes`` wrapper classes gained:

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
    [_t.ListOfInt, _t.ListOfUnsigned, _t.ListOfUnsignedLong,
     _t.ListOfPointer, _t.ListOfBytes],
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
