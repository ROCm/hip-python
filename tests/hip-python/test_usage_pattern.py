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

"""The way applications actually drive the ``hip`` compat package.

Modelled on a real benchmarking script: reach the bindings through
``from hip import hip, hiprtc``, wrap every call in the usual ``hip_check``
over ``(err, *rest)`` tuples, read device properties and attributes, then
time work with events. These are the pieces that break when a signature
changes underneath a consumer, as ``hipGetDeviceProperties`` did when it
stopped taking a ``props`` argument and started returning the struct.

Timing is asserted for shape only, never for speed, so this stays a
correctness test even on a busy machine.
"""

__author__ = "Advanced Micro Devices, Inc."

import math

import pytest

hip_pkg = pytest.importorskip("hip")

hip = hip_pkg.hip
hiprtc = hip_pkg.hiprtc


def hip_check(call_result):
    """Unwrap a binding's ``(err, *rest)`` return, raising on failure."""
    if isinstance(call_result, tuple):
        err = call_result[0]
        result = call_result[1:]
        if len(result) == 1:
            result = result[0]
    else:
        # Single-output functions return the bare error enum.
        err = call_result
        result = ()
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    if (
        isinstance(err, hiprtc.hiprtcResult)
        and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result


def _runtime_available():
    """True when libamdhip64 is loadable; no device required."""
    try:
        hip.hipGetErrorString(hip.hipError_t.hipSuccess)
    except Exception:
        return False
    return True


def _gpu_count():
    """Number of HIP devices, or 0 where the runtime cannot be reached."""
    try:
        err, count = hip.hipGetDeviceCount()
        return count if int(err) == int(hip.hipError_t.hipSuccess) else 0
    except Exception:
        return 0


_needs_runtime = pytest.mark.skipif(
    not _runtime_available(), reason="requires a loadable HIP runtime"
)
_needs_gpu = pytest.mark.skipif(
    not _gpu_count(), reason="requires a HIP device"
)


# ---------------------------------------------------------------------------
# Types and enums (no runtime needed)
# ---------------------------------------------------------------------------


def test_hip_check_passes_success_and_raises_on_failure():
    assert hip_check((hip.hipError_t.hipSuccess, 42)) == 42
    assert hip_check(hip.hipError_t.hipSuccess) == ()
    invalid = hip.hipError_t.hipErrorInvalidValue
    with pytest.raises(RuntimeError) as failure:
        hip_check((invalid, None))
    assert str(invalid) in str(failure.value)


def test_device_properties_struct_exposes_the_documented_fields():
    props = hip.hipDeviceProp_t()
    exposed = props.PROPERTIES()
    assert "gcnArchName" in exposed
    assert "multiProcessorCount" in exposed


def test_device_attributes_enumerate_and_sort():
    attributes = sorted(hip.hipDeviceAttribute_t)
    assert attributes
    assert hip.hipDeviceAttribute_t.hipDeviceAttributeWarpSize in attributes


def test_stream_and_event_handle_types_are_reachable():
    assert hip.hipStream_t is not None
    assert hip.hipEvent_t is not None


# ---------------------------------------------------------------------------
# Runtime calls that need no device
# ---------------------------------------------------------------------------


@_needs_runtime
def test_error_strings_are_readable():
    # The message comes back as a `CStr` view of the runtime's own buffer.
    message = hip_check(
        hip.hipGetErrorString(hip.hipError_t.hipErrorInvalidValue)
    )
    assert "invalid" in str(message)
    assert str(
        hip_check(
            hiprtc.hiprtcGetErrorString(hiprtc.hiprtcResult.HIPRTC_SUCCESS)
        )
    )


# ---------------------------------------------------------------------------
# Device work
# ---------------------------------------------------------------------------


@_needs_gpu
def test_device_properties_are_returned_not_filled_in():
    # The pre-4.0 spelling passed a `props` struct in. Today the struct
    # comes back as the second half of the `(err, props)` tuple.
    props = hip_check(hip.hipGetDeviceProperties(0))
    assert props.gcnArchName.startswith("gfx")
    assert props.multiProcessorCount > 0


@_needs_gpu
def test_attribute_query_agrees_with_the_properties_struct():
    props = hip_check(hip.hipGetDeviceProperties(0))
    count = hip_check(
        hip.hipDeviceGetAttribute(
            hip.hipDeviceAttribute_t.hipDeviceAttributeMultiprocessorCount, 0
        )
    )
    assert count == props.multiProcessorCount


@_needs_gpu
def test_stream_round_trip():
    stream = hip_check(hip.hipStreamCreate())
    try:
        hip_check(hip.hipStreamSynchronize(stream))
    finally:
        hip_check(hip.hipStreamDestroy(stream))


@_needs_gpu
def test_event_timing_round_trip():
    start = hip_check(hip.hipEventCreate())
    stop = hip_check(hip.hipEventCreate())
    try:
        hip_check(hip.hipEventRecord(start, None))
        hip_check(hip.hipDeviceSynchronize())
        hip_check(hip.hipEventRecord(stop, None))
        hip_check(hip.hipEventSynchronize(stop))
        elapsed_ms = hip_check(hip.hipEventElapsedTime(start, stop))
    finally:
        hip_check(hip.hipEventDestroy(start))
        hip_check(hip.hipEventDestroy(stop))
    assert math.isfinite(elapsed_ms)
    assert elapsed_ms >= 0.0
