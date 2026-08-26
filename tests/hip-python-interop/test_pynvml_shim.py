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

"""Unit tests for the hip-python-interop ``pynvml`` (NVML) compatibility shim.

These exercise the GPU-independent code paths of the shim by stubbing the
``rocm.bindings.amdsmi`` calls it dispatches to, so they run without a GPU
present (the genuinely GPU-dependent paths are covered by the examples suite).

The shim lives inside the installed ``hip_python_interop`` wheel, not in this
tree, so the test imports the installed top-level ``pynvml`` package.
"""

__author__ = "Advanced Micro Devices, Inc."

import sys

import pytest

# ROCm ships no AMD SMI library on Windows, so the rocm-bindings-systems wheel
# that backs the shim is not built there. State that verdict up front rather
# than letting it arrive as an ImportError about a module the reader of this
# file never mentioned: `import pynvml` reaches the shim's own
# `from rocm.bindings import amdsmi`, and importorskip("pynvml") would not skip
# on that, it would fail collection.
if sys.platform == "win32":
    pytest.skip(
        "the pynvml shim is backed by AMD SMI, which ROCm does not ship on Windows",
        allow_module_level=True,
    )

# Elsewhere the backing module is assumed present, so import it plainly. Doing
# so before pynvml keeps a genuinely broken install distinguishable from the
# platform case above.
import rocm.bindings.amdsmi  # noqa: E402,F401

pynvml = pytest.importorskip("pynvml")


@pytest.fixture
def initialized_single_device(monkeypatch):
    """Pretend NVML is initialized with one device, without touching a GPU."""
    device = pynvml._NvmlDevice(0, 0)
    monkeypatch.setattr(pynvml, "_init_count", 1, raising=False)
    monkeypatch.setattr(pynvml, "_devices", [device], raising=False)
    return device


def test_get_index_returns_ordinal(initialized_single_device):
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    assert pynvml.nvmlDeviceGetIndex(handle) == 0


def test_mig_mode_reports_disabled(initialized_single_device):
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mode = pynvml.nvmlDeviceGetMigMode(handle)
    assert mode == (
        pynvml.NVML_DEVICE_MIG_DISABLE,
        pynvml.NVML_DEVICE_MIG_DISABLE,
    )
    # dask-cuda reads the current mode via ``[0]``.
    assert mode[0] == pynvml.NVML_DEVICE_MIG_DISABLE
    assert pynvml.nvmlDeviceGetMaxMigDeviceCount(handle) == 0


def test_mig_device_handle_is_not_supported(initialized_single_device):
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    with pytest.raises(pynvml.NVMLError_NotSupported):
        pynvml.nvmlDeviceGetMigDeviceHandleByIndex(device=handle, index=0)


def test_error_value_dispatches_to_subclass():
    not_supported = pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)
    assert isinstance(not_supported, pynvml.NVMLError_NotSupported)
    assert not_supported.value == pynvml.NVML_ERROR_NOT_SUPPORTED

    lib_not_found = pynvml.NVMLError(pynvml.NVML_ERROR_LIBRARY_NOT_FOUND)
    assert isinstance(lib_not_found, pynvml.NVMLError_LibraryNotFound)


def test_cpu_affinity_returns_bitmask_words(
    initialized_single_device, monkeypatch
):
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    expected = [0b1011, 0b0]

    def fake_affinity(processor_handle, cpu_set_size, cpu_set, scope):
        assert cpu_set_size == len(expected)
        assert (
            scope
            == pynvml.amdsmi.amdsmi_affinity_scope_t.AMDSMI_AFFINITY_SCOPE_NODE
        )
        for i, word in enumerate(expected):
            cpu_set[i] = word
        return pynvml._OK

    monkeypatch.setattr(
        pynvml.amdsmi,
        "amdsmi_get_cpu_affinity_with_scope",
        fake_affinity,
        raising=False,
    )
    assert pynvml.nvmlDeviceGetCpuAffinity(handle, len(expected)) == expected


def test_cpu_affinity_rejects_nonpositive_size(initialized_single_device):
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    with pytest.raises(pynvml.NVMLError_InvalidArgument):
        pynvml.nvmlDeviceGetCpuAffinity(handle, 0)


def test_cpu_affinity_not_supported_propagates(
    initialized_single_device, monkeypatch
):
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def fake_affinity(processor_handle, cpu_set_size, cpu_set, scope):
        return int(pynvml._S.AMDSMI_STATUS_NOT_SUPPORTED)

    monkeypatch.setattr(
        pynvml.amdsmi,
        "amdsmi_get_cpu_affinity_with_scope",
        fake_affinity,
        raising=False,
    )
    with pytest.raises(pynvml.NVMLError_NotSupported):
        pynvml.nvmlDeviceGetCpuAffinity(handle, 2)


def test_cpu_affinity_missing_symbol_is_not_supported(
    initialized_single_device, monkeypatch
):
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    monkeypatch.delattr(
        pynvml.amdsmi, "amdsmi_get_cpu_affinity_with_scope", raising=False
    )
    with pytest.raises(pynvml.NVMLError_NotSupported):
        pynvml.nvmlDeviceGetCpuAffinity(handle, 2)
