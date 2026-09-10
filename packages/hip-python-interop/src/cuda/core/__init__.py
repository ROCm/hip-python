# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

"""Minimal ``cuda.core`` compatibility shim, backed by HIP.

This is NOT a full port of NVIDIA's ``cuda.core`` / ``cuda.core.experimental``
package. It implements just enough of the high-level surface for HIP ports of
CUDA-Python consumers: a `~.Device` exposing ``uuid``, the `~.Stream`
vocabulary type behind the CUDA stream protocol, and stream-ordered device
allocation through `~.DeviceMemoryResource` and `~.Buffer`.

Everything is implemented on top of the high-level ``rocm.bindings.hip`` HIP
runtime API. Additional high-level abstractions (Event, Program, Linker, ...)
are intentionally out of scope and can be added on demand.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

from rocm.bindings import hip

__all__ = ["Buffer", "Device", "DeviceMemoryResource", "Stream"]

#: The ``cuda.core`` API level this shim emulates. Consumers gate features on
#: it --- ``pytest.importorskip("cuda.core", minversion=...)`` and the like ---
#: so it names the surface implemented below rather than the HIP Python
#: release the shim ships in.
__version__ = "0.5.0"


def _check(err):
    """Raise on a non-success HIP status returned by a ``rocm.bindings.hip`` call."""
    if int(err) != int(hip.hipError_t.hipSuccess):
        raise RuntimeError(f"HIP error in cuda.core: {err!r}")


def _stream_address(obj, argname="stream"):
    """Return the ``hipStream_t`` address that ``obj`` stands for, as an int.

    :py:obj:`None` selects the NULL stream, which is what the ``cuda.core``
    entry points taking an optional stream fall back to. Any other object must
    implement the `CUDA stream protocol
    <https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol>`__,
    which is how streams from RMM, CuPy, Numba and ``cuda.core`` itself are all
    accepted interchangeably.
    """
    if obj is None:
        return 0
    protocol = getattr(obj, "__cuda_stream__", None)
    if protocol is None:
        raise TypeError(
            f"{argname} must be None or implement the CUDA stream protocol "
            f"(__cuda_stream__), got {type(obj).__name__}"
        )
    version, address = protocol()
    if version != 0:
        raise NotImplementedError(
            f"cuda.core does not currently support the CUDA stream protocol "
            f"version: '{version}'."
        )
    return int(address)


class Device:
    """HIP-backed stand-in for ``cuda.core.Device``.

    Only the members required by current consumers are implemented. Passing no
    ``device_id`` selects the current device (via ``hipGetDevice``).
    """

    def __init__(self, device_id=None):
        if device_id is None:
            err, device_id = hip.hipGetDevice()
            _check(err)
        self._id = int(device_id)
        self._default_stream = None

    @property
    def device_id(self) -> int:
        """Ordinal of the underlying HIP device."""
        return self._id

    @property
    def uuid(self) -> str:
        """Device UUID as a ``GPU-<hex>`` string.

        Backed by ``hipDeviceGetUuid``. ROCm stores the UUID as ASCII text in
        the 16-byte field (e.g. ``b"50a437de77a657b8"``), so the result matches
        the ``GPU-<hex>`` string reported by ``rocminfo`` / ``amd-smi``. Falls
        back to a raw hex rendering if a runtime ever returns non-text bytes.
        """
        err, handle = hip.hipDeviceGetUuid(self._id)
        _check(err)
        raw = handle.get_bytes(0)
        stripped = raw.rstrip(b"\x00")
        try:
            text = stripped.decode("ascii")
            is_text = bool(text) and text.isprintable()
        except UnicodeDecodeError:
            is_text = False
        if not is_text:
            text = raw.hex()
        return text if text.startswith("GPU-") else f"GPU-{text}"

    def __repr__(self):
        return f"<cuda.core.Device id={self._id} (HIP)>"

    def set_current(self):
        err = hip.hipSetDevice(self._id)[0]
        _check(err)

    def sync(self):
        err = hip.hipDeviceSynchronize()[0]
        _check(err)

    @property
    def default_stream(self):
        """The NULL stream, as a non-owning `~.Stream` token.

        HIP has no per-thread default stream selected by an environment
        variable the way CUDA Python's
        ``CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`` does, so this is always
        the legacy NULL stream. The token is created once per `~.Device` so
        that repeated reads compare equal.
        """
        if self._default_stream is None:
            self._default_stream = Stream._wrap(0)
        return self._default_stream

    def create_stream(self, obj=None):
        """Create a HIP stream, or wrap a foreign one.

        With no argument a new non-blocking stream is created --- non-blocking
        being the ``cuda.core`` default --- and destroyed again when the
        returned `~.Stream` is closed or collected. HIP associates a new stream
        with the *current* device rather than with the device this method was
        called on, so the two must agree.

        Passing ``obj`` instead wraps an existing stream from any library
        implementing the CUDA stream protocol. The wrapper keeps ``obj`` alive
        and never destroys the borrowed stream.
        """
        if obj is not None:
            return Stream._wrap(_stream_address(obj, "obj"), owner=obj)
        err, current = hip.hipGetDevice()
        _check(err)
        if current != self._id:
            raise RuntimeError(
                f"cannot create a stream on device {self._id} while device "
                f"{current} is current; call Device.set_current() first"
            )
        err, handle = hip.hipStreamCreateWithFlags(hip.hipStreamNonBlocking)
        _check(err)
        return Stream._own(int(handle))


class Stream:
    """HIP-backed stand-in for ``cuda.core.Stream``.

    Instances come from :py:meth:`Device.create_stream` or
    :py:attr:`Device.default_stream`. As in ``cuda.core``, constructing one
    directly is refused: whether the stream would be owned or borrowed is
    ambiguous, and the two differ in what closing them does.
    """

    __slots__ = ("_address", "_owned", "_owner")

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Stream objects cannot be instantiated directly. Please use "
            "Device.create_stream() or Device.default_stream."
        )

    @classmethod
    def _own(cls, address):
        """Take ownership of ``address``; closing destroys the HIP stream."""
        return cls._make(address, owned=True, owner=None)

    @classmethod
    def _wrap(cls, address, owner=None):
        """Borrow ``address``, holding ``owner`` alive for as long as we are."""
        return cls._make(address, owned=False, owner=owner)

    @classmethod
    def _make(cls, address, owned, owner):
        self = cls.__new__(cls)
        self._address = address
        self._owned = owned
        self._owner = owner
        return self

    @property
    def handle(self):
        """The underlying ``hipStream_t``.

        As in ``cuda.core``, this is a Python object; use ``int()`` on it to
        get the address of the C handle.
        """
        return hip.ihipStream_t(self._address)

    def __cuda_stream__(self):
        """CUDA stream protocol, version 0: ``(0, <hipStream_t address>)``."""
        return (0, self._address)

    def sync(self):
        """Block until all work queued on this stream has completed."""
        err = hip.hipStreamSynchronize(self._address)[0]
        _check(err)

    def close(self):
        """Destroy an owned stream, or drop the reference to a borrowed one."""
        if self._owned and self._address:
            self._owned = False
            err = hip.hipStreamDestroy(self._address)[0]
            _check(err)
        self._owner = None
        self._address = 0

    def __del__(self):
        # A finalizer must not raise, and by the time one runs the HIP runtime
        # may already have been torn down.
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown only
            pass

    def __repr__(self):
        kind = "owned" if self._owned else "borrowed"
        return f"<cuda.core.Stream {self._address:#x} ({kind}, HIP)>"


class Buffer:
    """HIP-backed stand-in for ``cuda.core.Buffer``.

    Instances come from :py:meth:`DeviceMemoryResource.allocate`, and hold on
    to the stream that allocated them: closing without naming a stream frees
    the allocation in the order it was created, and a stream-ordered free onto
    an already destroyed stream would not be valid.
    """

    __slots__ = ("_mr", "_ptr", "_size", "_stream")

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Buffer objects cannot be instantiated directly. Please use "
            "DeviceMemoryResource.allocate()."
        )

    @classmethod
    def _make(cls, ptr, size, mr, stream):
        self = cls.__new__(cls)
        self._ptr = ptr
        self._size = size
        self._mr = mr
        self._stream = stream
        return self

    @property
    def handle(self):
        """The allocation, as the ``DeviceArray`` HIP returned for it.

        As in ``cuda.core``, this is a Python object; use ``int()`` on it to
        get the device address.
        """
        return self._ptr

    @property
    def size(self):
        """Size of the allocation in bytes."""
        return self._size

    def close(self, stream=None):
        """Free the allocation, ordered on ``stream``.

        ``stream`` defaults to the one that allocated the buffer. The free is
        stream-ordered, so it takes effect once preceding work on that stream
        has run, not when this call returns.
        """
        if self._ptr is None:
            return
        ptr, self._ptr = self._ptr, None
        self._mr.deallocate(
            ptr, self._size, self._stream if stream is None else stream
        )
        self._stream = None

    def __del__(self):
        # A finalizer must not raise, and by the time one runs the HIP runtime
        # may already have been torn down.
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown only
            pass

    def __repr__(self):
        if self._ptr is None:
            return "<cuda.core.Buffer closed (HIP)>"
        return (
            f"<cuda.core.Buffer {int(self._ptr):#x} size={self._size} (HIP)>"
        )


class DeviceMemoryResource:
    """HIP-backed stand-in for ``cuda.core.DeviceMemoryResource``.

    Allocations are stream-ordered and come from the device's current HIP
    memory pool, which is the device's default pool unless one has been set
    with ``hipDeviceSetMemPool``. The pool therefore outlives this object and
    is never destroyed by it, matching how ``cuda.core`` treats a memory
    resource built over a pool it does not own.
    """

    __slots__ = ("_device_id", "_pool")

    def __init__(self, device_id):
        self._device_id = int(device_id)
        err, pool = hip.hipDeviceGetMemPool(self._device_id)
        _check(err)
        self._pool = pool

    @property
    def device_id(self):
        """Ordinal of the device whose pool backs this resource."""
        return self._device_id

    @property
    def handle(self):
        """The underlying ``hipMemPool_t``."""
        return self._pool

    def allocate(self, size, stream=None):
        """Allocate ``size`` bytes from the pool, ordered on ``stream``.

        The allocation is only safe to access once preceding work on
        ``stream`` has run. ``stream`` defaults to the NULL stream.
        """
        err, ptr = hip.hipMallocFromPoolAsync(
            size, self._pool, _stream_address(stream)
        )
        _check(err)
        return Buffer._make(ptr, int(size), self, stream)

    def deallocate(self, ptr, size, stream=None):
        """Return ``ptr`` to the pool, ordered on ``stream``."""
        err = hip.hipFreeAsync(ptr, _stream_address(stream))[0]
        _check(err)

    def __repr__(self):
        return f"<cuda.core.DeviceMemoryResource device_id={self._device_id} (HIP)>"
