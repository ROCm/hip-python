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

"""Contract tests for ``rocm.bindings.util.types.NDBuffer``'s CUDA array
interface, on both the producing and the consuming side.

`NDBuffer` used to keep its interface dict, a keep-alive reference and a
format string in a ``cdef dict __dict__`` attribute bag. Cython 3.3.0
made that illegal (cython#7823), so the three are separate typed members
now and the interface is a property. These tests pin the observable
contract that survived the move:

  * The protocol resolves by plain attribute access, by `hasattr` and by
    `getattr` -- the only ways `numba` and CuPy look for it.
  * The dict handed out is a *copy*, so a consumer that adds a key to it
    (as `numba` does with ``'strides'``) cannot disturb the buffer.
  * Instances no longer carry a Python instance dict at all.
  * The ``'data'`` entry always agrees with the buffer's own pointer.

On the consuming side, building an `NDBuffer` from a foreign producer
used to raise `TypeError` unconditionally. Now that it works, the layouts
an `NDBuffer` cannot represent have to be refused rather than silently
misread: its addressing derives every offset from the shape alone, so a
strided, masked or offset input would be read at the wrong addresses.
Strides that merely spell out the contiguous layout the shape already
implies are accepted, which is what `numba` emits.
"""

import ctypes
import gc
import math

import pytest
from rocm.bindings.util import types as _t


def _contiguous_strides(shape, itemsize):
    """The C-contiguous strides `shape` and `itemsize` imply, in bytes."""
    strides = []
    stride = itemsize
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


class _Omit:
    """Sentinel asking `_ForeignArray` to leave a key out entirely."""


_OMIT = _Omit()


class _ForeignArray:
    """A minimal CUDA array interface producer.

    Mirrors `numba.hip.testing.ForeignArray` and CuPy's
    ``DummyObjectWithCudaArrayInterface``: the protocol comes from a
    property, and the instance owns the allocation its pointer refers to.
    Deliberately does *not* implement the Python buffer protocol, which
    `NDBuffer` would otherwise prefer over the interface.
    """

    def __init__(self, shape=(4,), typestr="<f8", itemsize=8, **overrides):
        self._allocation = ctypes.create_string_buffer(
            max(1, math.prod(shape) * itemsize)
        )
        self._desc = dict(
            shape=shape,
            typestr=typestr,
            data=(ctypes.addressof(self._allocation), False),
            strides=None,
            offset=0,
            mask=None,
            version=3,
            stream=None,
        )
        for key, value in overrides.items():
            if value is _OMIT:
                del self._desc[key]
            else:
                self._desc[key] = value

    @property
    def address(self):
        return ctypes.addressof(self._allocation)

    @property
    def __cuda_array_interface__(self):
        return dict(self._desc)


# ---------------------------------------------------------------------------
# Producing: how the interface is exposed
# ---------------------------------------------------------------------------


def test_interface_resolves_by_every_access_pattern():
    """`numba` and CuPy detect the protocol with `hasattr` and read it
    with `getattr`; a property has to satisfy both."""
    buf = _t.NDBuffer(bytearray(16))

    assert hasattr(buf, "__cuda_array_interface__")
    assert isinstance(getattr(buf, "__cuda_array_interface__", None), dict)
    assert buf.__cuda_array_interface__ is not None


def test_interface_carries_the_documented_keys():
    buf = _t.NDBuffer(bytearray(16))
    interface = buf.__cuda_array_interface__

    for key in ("shape", "typestr", "data", "version", "stream"):
        assert key in interface, f"'{key}' missing from the interface"
    assert interface["shape"] == (16,)
    assert interface["typestr"] == "B"
    assert interface["version"] == 3
    assert interface["strides"] is None


def test_returned_dict_is_a_copy():
    """A consumer may add or overwrite entries in what it was handed.

    `numba`'s ``test_consuming_strides`` does exactly this: it assigns
    ``face['strides'] = ...`` and consumes the dict again. Handing out
    the live dict would let that write a non-contiguous layout into the
    buffer permanently -- which the buffer itself then refuses.
    """
    buf = _t.NDBuffer(bytearray(16))

    first = buf.__cuda_array_interface__
    first["strides"] = (1,)
    first["mask"] = object()
    first["shape"] = (99,)

    second = buf.__cuda_array_interface__
    assert second["strides"] is None
    assert second["mask"] is None
    assert second["shape"] == (16,)
    assert first is not second
    assert buf.shape == (16,)


def test_data_entry_agrees_with_the_pointer():
    """The 'data' address is refreshed on read, so it tracks the buffer
    through slicing and reconfiguration."""
    buf = _t.NDBuffer(bytearray(16))
    assert buf.__cuda_array_interface__["data"][0] == int(buf)

    sub = buf[8:]
    assert sub.__cuda_array_interface__["data"][0] == int(sub)
    assert int(sub) == int(buf) + 8

    buf.configure(shape=(4,), typestr="<f8", _force=True)
    assert buf.__cuda_array_interface__["data"][0] == int(buf)


def test_configure_changes_are_visible_on_the_next_read():
    buf = _t.NDBuffer(bytearray(16))
    assert buf.__cuda_array_interface__["typestr"] == "B"

    buf.configure(shape=(2,), typestr="<f8", _force=True)
    interface = buf.__cuda_array_interface__
    assert interface["typestr"] == "<f8"
    assert interface["shape"] == (2,)
    assert buf.itemsize == 8

    buf.configure(read_only=True)
    assert buf.__cuda_array_interface__["data"][1] is True

    buf.configure(stream=7)
    assert buf.__cuda_array_interface__["stream"] == 7


def test_slicing_keeps_an_absent_stream_absent():
    """`None` must not be mapped onto the address 0, which the interface
    specification disallows as ambiguous."""
    buf = _t.NDBuffer(bytearray(16))
    assert buf.__cuda_array_interface__["stream"] is None
    assert buf[8:].__cuda_array_interface__["stream"] is None


def test_instances_carry_no_python_dict():
    """The attribute bag is gone, which is the point of the change."""
    buf = _t.NDBuffer(bytearray(16))

    assert not hasattr(buf, "__dict__")
    with pytest.raises(TypeError):
        vars(buf)
    with pytest.raises(AttributeError):
        buf.some_new_attribute = 1


def test_wrapped_exporter_stays_alive():
    """The keep-alive reference moved to a typed member; the wrapped
    object must still outlive the caller's own reference to it."""
    payload = bytearray(b"keep-me-alive!!!")
    buf = _t.NDBuffer(payload)
    address = int(buf)

    del payload
    gc.collect()

    assert ctypes.string_at(address, 16) == b"keep-me-alive!!!"
    assert buf.__cuda_array_interface__["data"][0] == address


def test_buffer_protocol_still_reports_the_format():
    """The NUL-terminated format string moved to a typed member too, and
    it has to outlive the view that points at it."""
    buf = _t.NDBuffer(bytearray(b"0123456789abcdef"))
    assert bytes(memoryview(buf)) == b"0123456789abcdef"


# ---------------------------------------------------------------------------
# Consuming: building an NDBuffer from a foreign producer
# ---------------------------------------------------------------------------


def test_consumes_a_foreign_producer():
    """Used to raise `TypeError` for every input: `configure` takes
    keyword arguments and was handed the interface dict positionally."""
    foreign = _ForeignArray(shape=(4,), typestr="<f8")
    buf = _t.NDBuffer(foreign)

    assert int(buf) == foreign.address
    assert buf.shape == (4,)
    assert buf.typestr == "<f8"
    assert buf.itemsize == 8
    assert buf.size == 4
    assert buf.is_read_only is False


def test_consumes_a_read_only_producer_and_its_stream():
    foreign = _ForeignArray(shape=(2, 3), data=(1024, True), stream=7)
    buf = _t.NDBuffer(foreign)

    assert int(buf) == 1024
    assert buf.is_read_only is True
    assert buf.stream_as_int == 7
    assert buf.shape == (2, 3)


def test_absent_stream_stays_absent():
    buf = _t.NDBuffer(_ForeignArray(shape=(4,)))
    assert buf.__cuda_array_interface__["stream"] is None


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 4)])
def test_explicit_contiguous_strides_are_accepted(shape):
    """The case `numba`'s ``test_consuming_strides`` exercises: a
    producer may spell out the strides the shape already implies.
    Previously any non-`None` 'strides' was refused outright."""
    strides = _contiguous_strides(shape, 8)
    explicit = _t.NDBuffer(_ForeignArray(shape=shape, strides=strides))
    implicit = _t.NDBuffer(_ForeignArray(shape=shape, strides=None))

    assert explicit.shape == implicit.shape == shape
    assert explicit.itemsize == implicit.itemsize == 8
    # The interface an NDBuffer produces is always contiguous.
    assert explicit.__cuda_array_interface__["strides"] is None


@pytest.mark.parametrize(
    "shape,strides",
    [
        ((4,), (16,)),  # every other element
        ((4,), (-8,)),  # reversed, as numpy's [::-1] yields
        ((2, 3), (8, 32)),  # column major
        ((2, 3), (48, 8)),  # a row-major view into a wider array
    ],
)
def test_non_contiguous_strides_are_refused(shape, strides):
    with pytest.raises(RuntimeError, match="not contiguous"):
        _t.NDBuffer(_ForeignArray(shape=shape, strides=strides))


@pytest.mark.parametrize(
    "shape,strides",
    [
        ((1, 4), (12345, 8)),
        ((4, 1), (8, 12345)),
        ((1, 1), (99, 99)),
    ],
)
def test_the_stride_of_a_single_element_axis_is_ignored(shape, strides):
    """Such a stride is arbitrary -- numpy ignores it in its own
    contiguity test, so a producer's choice must not be second-guessed."""
    buf = _t.NDBuffer(_ForeignArray(shape=shape, strides=strides))
    assert buf.shape == shape


def test_an_empty_array_is_contiguous_either_way():
    """`numba`'s ``test_zero_size_array`` produces exactly this: no
    elements, and a legitimate data pointer of 0."""
    buf = _t.NDBuffer(_ForeignArray(shape=(0,), data=(0, False), strides=(8,)))
    assert buf.shape == (0,)
    assert buf.size == 0
    assert int(buf) == 0


def test_a_mask_is_refused():
    """An NDBuffer cannot represent one, and reading masked data as
    dense would be silently wrong rather than merely unsupported."""
    with pytest.raises(NotImplementedError, match="[Mm]asked"):
        _t.NDBuffer(_ForeignArray(shape=(4,), mask=object()))


def test_a_non_zero_offset_is_refused():
    """Ignoring it would leave the pointer at the wrong element."""
    with pytest.raises(NotImplementedError, match="offset"):
        _t.NDBuffer(_ForeignArray(shape=(4,), offset=8))


def test_optional_keys_may_be_absent():
    """'strides', 'mask' and 'offset' are all read defensively, so a
    sparser producer is still consumable."""
    foreign = _ForeignArray(
        shape=(4,), strides=_OMIT, mask=_OMIT, offset=_OMIT, stream=_OMIT
    )
    buf = _t.NDBuffer(foreign)

    assert buf.shape == (4,)
    assert int(buf) == foreign.address


def test_a_missing_data_key_is_reported():
    with pytest.raises(ValueError, match="'data'"):
        _t.NDBuffer(_ForeignArray(shape=(4,), data=_OMIT))


def test_strides_of_the_wrong_length_are_reported():
    with pytest.raises(ValueError, match="'strides'"):
        _t.NDBuffer(_ForeignArray(shape=(2, 3), strides=(8,)))


# ---------------------------------------------------------------------------
# configure() reports invalid input instead of returning it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [0, -1])
def test_configure_raises_on_an_invalid_stream(stream):
    """Both branches used to ``return`` the `ValueError`, so the caller
    got an exception object where it expected the buffer."""
    buf = _t.NDBuffer(bytearray(16))
    with pytest.raises(ValueError, match="'stream'"):
        buf.configure(stream=stream)


def test_configure_returns_the_buffer_on_success():
    buf = _t.NDBuffer(bytearray(16))
    assert buf.configure(stream=7) is buf
