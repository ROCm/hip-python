#!/usr/bin/env -S python3 -m pytest -v -s
# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
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

"""Unit tests for the hip-python-interop ``cuda.bindings.cufile`` shim.

The shim is a compiled Cython module layered on hipFILE (``rocm.bindings``'
``cyhipfile``). These tests exercise the GPU/driver-independent surface: symbol
presence, enum names/values (cross-checked against ``rocm.bindings.hipfile``),
and the ``Descr``/``IOParams``/``IOEvents`` array helpers -- none of which touch
``libhipfile.so``.

Paths that dispatch into ``libhipfile.so`` (``get_version``, ``op_status_error``,
and hence ``cuFileError``'s message lookup) are guarded behind a runtime probe so
the module still runs where the hipFILE bindings were built without a loadable
``libhipfile.so``.

The shim lives inside the installed ``hip_python_interop`` wheel, not in this
tree, so the test imports the installed ``cuda.bindings.cufile`` module.
"""

__author__ = "Advanced Micro Devices, Inc."

import pytest

# Skip the whole module when the interop wheel (built with the optional hipFILE
# bindings) is not importable, rather than erroring at collection time.
cufile = pytest.importorskip("cuda.bindings.cufile")


def _hipfile_runtime_available():
    """True when libhipfile.so is loadable (needed for the C-dispatch paths)."""
    try:
        cufile.get_version()
    except Exception:
        return False
    return True


_have_hipfile = _hipfile_runtime_available()

_needs_runtime = pytest.mark.skipif(
    not _have_hipfile,
    reason="requires the hipFILE bindings with a loadable libhipfile.so",
)


# ---------------------------------------------------------------------------
# Surface presence (no libhipfile.so needed)
# ---------------------------------------------------------------------------

_EXPECTED_FUNCTIONS = [
    "driver_open",
    "driver_close",
    "use_count",
    "driver_get_properties",
    "driver_set_poll_mode",
    "driver_set_max_direct_io_size",
    "driver_set_max_cache_size",
    "driver_set_max_pinned_mem_size",
    "handle_register",
    "handle_deregister",
    "buf_register",
    "buf_deregister",
    "read",
    "write",
    "get_parameter_size_t",
    "get_parameter_bool",
    "get_parameter_string",
    "set_parameter_size_t",
    "set_parameter_bool",
    "set_parameter_string",
    "get_version",
    "op_status_error",
    "batch_io_set_up",
    "batch_io_submit",
    "batch_io_get_status",
    "batch_io_cancel",
    "batch_io_destroy",
    "read_async",
    "write_async",
    "stream_register",
    "stream_deregister",
]

_EXPECTED_ENUMS = [
    "OpError",
    "Opcode",
    "Status",
    "BatchMode",
    "FileHandleType",
    "FeatureFlags",
    "DriverStatusFlags",
    "DriverControlFlags",
    "SizeTConfigParameter",
    "BoolConfigParameter",
    "StringConfigParameter",
]


@pytest.mark.parametrize("name", _EXPECTED_FUNCTIONS)
def test_function_is_exposed(name):
    assert callable(getattr(cufile, name)), f"missing cufile.{name}"


@pytest.mark.parametrize("name", _EXPECTED_ENUMS)
def test_enum_is_exposed(name):
    import enum

    obj = getattr(cufile, name)
    assert isinstance(obj, type) and issubclass(obj, enum.IntEnum)


@pytest.mark.parametrize("name", ["Descr", "IOParams", "IOEvents"])
def test_array_helper_type_is_exposed(name):
    assert isinstance(getattr(cufile, name), type)


def test_cufile_error_is_exception_subclass():
    assert issubclass(cufile.cuFileError, Exception)


def test_op_error_has_cuda_python_member_names():
    # A representative sample of the CUfileOpError names cuda-python exposes.
    for member in (
        "SUCCESS",
        "DRIVER_NOT_INITIALIZED",
        "INVALID_VALUE",
        "CUDA_DRIVER_ERROR",
        "IO_MAX_ERROR",
    ):
        assert hasattr(cufile.OpError, member)
    assert cufile.OpError.SUCCESS == 0


def test_file_handle_type_member_names():
    for member in ("OPAQUE_FD", "OPAQUE_WIN32", "USERSPACE_FS"):
        assert hasattr(cufile.FileHandleType, member)


# ---------------------------------------------------------------------------
# Enum values match the generated rocm.bindings.hipfile constants
# ---------------------------------------------------------------------------

def test_op_error_values_match_hipfile():
    hipfile = pytest.importorskip("rocm.bindings.hipfile")
    assert int(cufile.OpError.SUCCESS) == int(
        hipfile.hipFileOpError.hipFileSuccess
    )
    assert int(cufile.OpError.INVALID_VALUE) == int(
        hipfile.hipFileOpError.hipFileInvalidValue
    )
    assert int(cufile.OpError.CUDA_DRIVER_ERROR) == int(
        hipfile.hipFileOpError.hipFileHipDriverError
    )


def test_file_handle_type_values_match_hipfile():
    hipfile = pytest.importorskip("rocm.bindings.hipfile")
    assert int(cufile.FileHandleType.OPAQUE_FD) == int(
        hipfile.hipFileFileHandleType.hipFileHandleTypeOpaqueFD
    )


# ---------------------------------------------------------------------------
# Descr / IOParams / IOEvents array helpers (no libhipfile.so needed)
# ---------------------------------------------------------------------------

def test_descr_default_is_single_element_with_address():
    descr = cufile.Descr()
    assert len(descr) == 1
    assert descr.ptr != 0


def test_descr_type_and_fd_round_trip():
    descr = cufile.Descr()
    descr.type = cufile.FileHandleType.OPAQUE_FD
    assert descr.type == int(cufile.FileHandleType.OPAQUE_FD)
    descr.handle.fd = 7
    assert descr.handle.fd == 7


def test_descr_array_indexing():
    descr = cufile.Descr(3)
    assert len(descr) == 3
    # Element views share the parent's contiguous storage.
    assert descr[0].ptr == descr.ptr
    stride = descr[1].ptr - descr[0].ptr
    assert stride > 0
    assert descr[2].ptr == descr.ptr + 2 * stride
    with pytest.raises(IndexError):
        descr[3]


def test_io_params_and_events_expose_address():
    params = cufile.IOParams(2)
    assert len(params) == 2
    assert params.ptr != 0
    events = cufile.IOEvents(2)
    assert len(events) == 2
    assert events.ptr != 0


def test_io_events_ret_and_status_round_trip():
    events = cufile.IOEvents()
    events.status = cufile.Status.COMPLETE
    assert events.status == int(cufile.Status.COMPLETE)
    events.ret = 4096
    assert events.ret == 4096


# ---------------------------------------------------------------------------
# C-dispatch paths (require a loadable libhipfile.so)
# ---------------------------------------------------------------------------

@_needs_runtime
def test_get_version_returns_packed_int():
    version = cufile.get_version()
    assert isinstance(version, int)
    assert version >= 0


@_needs_runtime
def test_op_status_error_returns_str():
    message = cufile.op_status_error(int(cufile.OpError.SUCCESS))
    assert isinstance(message, str)


@_needs_runtime
def test_cufile_error_carries_status_and_message():
    err = cufile.cuFileError(int(cufile.OpError.INVALID_VALUE))
    assert err.status == int(cufile.OpError.INVALID_VALUE)
    assert err.cu_err is None
    assert "INVALID_VALUE" in str(err)
