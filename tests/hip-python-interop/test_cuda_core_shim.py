#!/usr/bin/env -S python3 -m pytest -v -s
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

"""Unit tests for the hip-python-interop ``cuda.core`` compatibility shim.

The shim is pure Python over ``rocm.bindings.hip``. Its declared surface, the
refusal to construct the vocabulary types directly, and the CUDA stream
protocol guards all raise before reaching HIP, so those run anywhere; the
stream and memory-pool round trips need a GPU and are gated behind a probe.

The shim lives inside the installed ``hip_python_interop`` wheel, not in this
tree, so the test imports the installed ``cuda.core`` package.
"""

__author__ = "Advanced Micro Devices, Inc."

import pytest

cuda_core = pytest.importorskip("cuda.core")


def _gpu_count():
    """Number of HIP devices, or 0 where the runtime cannot be reached."""
    try:
        from rocm.bindings import hip

        err, count = hip.hipGetDeviceCount()
        return count if int(err) == int(hip.hipError_t.hipSuccess) else 0
    except Exception:
        return 0


_gpus = _gpu_count()

_needs_gpu = pytest.mark.skipif(not _gpus, reason="requires a HIP device")
_needs_two_gpus = pytest.mark.skipif(
    _gpus < 2, reason="requires two HIP devices"
)


@pytest.fixture
def device():
    """The current device, made current, as ``cuda.core`` consumers use it."""
    dev = cuda_core.Device()
    dev.set_current()
    return dev


# ---------------------------------------------------------------------------
# Declared surface (no GPU needed)
# ---------------------------------------------------------------------------


def test_exports_the_documented_names():
    assert sorted(cuda_core.__all__) == [
        "Buffer",
        "Device",
        "DeviceMemoryResource",
        "Stream",
    ]
    for name in cuda_core.__all__:
        assert hasattr(cuda_core, name)


def test_version_is_the_emulated_cuda_core_api_level():
    # Consumers gate on this with pytest.importorskip(minversion=...), so it
    # must parse as a version and name the cuda.core surface rather than the
    # HIP Python release.
    from packaging.version import Version

    assert Version(cuda_core.__version__) >= Version("0.5.0")


@pytest.mark.parametrize("cls", ["Stream", "Buffer"])
def test_vocabulary_types_refuse_direct_construction(cls):
    with pytest.raises(RuntimeError, match="cannot be instantiated directly"):
        getattr(cuda_core, cls)()


# ---------------------------------------------------------------------------
# CUDA stream protocol handling (no GPU needed)
# ---------------------------------------------------------------------------


def test_stream_address_reads_the_protocol():
    class Stream0:
        def __cuda_stream__(self):
            return (0, 0xF00D)

    assert cuda_core._stream_address(Stream0()) == 0xF00D
    assert cuda_core._stream_address(None) == 0


def test_stream_address_rejects_a_future_protocol_version():
    class Stream1:
        def __cuda_stream__(self):
            return (1, 2)

    with pytest.raises(NotImplementedError, match="version: '1'"):
        cuda_core._stream_address(Stream1())


def test_stream_address_rejects_a_non_stream():
    with pytest.raises(TypeError, match="__cuda_stream__"):
        cuda_core._stream_address(object())


# ---------------------------------------------------------------------------
# Streams (GPU)
# ---------------------------------------------------------------------------


@_needs_gpu
def test_default_stream_is_a_stable_null_stream_token(device):
    assert device.default_stream.__cuda_stream__() == (0, 0)
    assert device.default_stream is device.default_stream


@_needs_gpu
def test_created_stream_reports_its_own_handle(device):
    stream = device.create_stream()
    try:
        version, address = stream.__cuda_stream__()
        assert version == 0
        assert address == int(stream.handle) != 0
        stream.sync()
    finally:
        stream.close()


@_needs_gpu
def test_created_stream_close_is_idempotent(device):
    stream = device.create_stream()
    stream.close()
    stream.close()


@_needs_gpu
def test_wrapping_a_foreign_stream_adopts_its_address(device):
    backing = device.create_stream()
    try:

        class Foreign:
            def __cuda_stream__(self):
                return (0, int(backing.handle))

        foreign = Foreign()
        wrapper = device.create_stream(foreign)
        assert wrapper.__cuda_stream__() == backing.__cuda_stream__()

        # Closing a wrapper releases the reference without destroying the
        # stream it borrowed, which stays usable afterwards.
        wrapper.close()
        backing.sync()
    finally:
        backing.close()


@_needs_two_gpus
def test_create_stream_requires_its_device_to_be_current(device):
    other = cuda_core.Device(1 if device.device_id == 0 else 0)
    with pytest.raises(RuntimeError, match="call Device.set_current"):
        other.create_stream()


# ---------------------------------------------------------------------------
# Stream-ordered device memory (GPU)
# ---------------------------------------------------------------------------


@_needs_gpu
def test_memory_resource_wraps_the_device_pool(device):
    mr = cuda_core.DeviceMemoryResource(device.device_id)
    assert mr.device_id == device.device_id
    assert int(mr.handle) != 0


@_needs_gpu
def test_allocate_and_close_round_trip_on_a_stream(device):
    mr = cuda_core.DeviceMemoryResource(device.device_id)
    stream = device.create_stream()
    try:
        buffer = mr.allocate(1024, stream=stream)
        assert buffer.size == 1024
        assert int(buffer.handle) != 0
        buffer.close(stream=stream)
        stream.sync()
    finally:
        stream.close()


@_needs_gpu
def test_close_without_a_stream_uses_the_allocating_one(device):
    mr = cuda_core.DeviceMemoryResource(device.device_id)
    stream = device.create_stream()
    try:
        buffer = mr.allocate(1024, stream=stream)
        buffer.close()
        buffer.close()  # idempotent
        stream.sync()
    finally:
        stream.close()


@_needs_gpu
def test_allocate_defaults_to_the_null_stream(device):
    mr = cuda_core.DeviceMemoryResource(device.device_id)
    buffer = mr.allocate(1024)
    assert int(buffer.handle) != 0
    buffer.close()
    device.sync()
