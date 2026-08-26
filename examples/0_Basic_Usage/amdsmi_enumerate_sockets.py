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

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

"""Enumerate AMD GPUs with the AMD SMI binding.

Mirrors the lifecycle and enumeration portions of the nodrm C example
(`rocm-systems/projects/amdsmi/example/amd_smi_nodrm_example.cc`):
init, query library version, walk sockets -> processors, keep the AMD
GPUs, then shut down.

Socket / processor enumeration uses AMD SMI's C two-call (count ->
allocate -> fill) pattern. The handle-array out-parameters bind to
:py:obj:`.rocm.bindings.util.types.ListOfPointer`, so this example uses
``ListOfPointer.allocate(n)``: a NULL (``None``) buffer first to read the
count, then a wrapper sized to that count to fill. The filled wrapper is
iterable and indexable, so the results read back as
:py:obj:`.rocm.bindings.util.types.Pointer` elements. The caller-driven
``count`` out-parameter is a plain ``ctypes`` scalar whose address is
passed via :py:func:`ctypes.addressof`, while the pure-out
``processor_type`` is returned directly by the binding.
"""

import sys

if sys.platform == "win32":
    raise NotImplementedError(
        "This example needs AMD SMI. ROCm ships no AMD SMI library on Windows, "
        "so the rocm-bindings-systems wheel that provides rocm.bindings.amdsmi "
        "is not built there."
    )

# [literalinclude-begin]
import ctypes

from rocm.bindings import amdsmi
from rocm.bindings.util import types


def amdsmi_check(call_result):
    if isinstance(call_result, amdsmi.amdsmi_status_t):
        err, result = call_result, ()
    else:
        err, result = call_result[0], call_result[1:]
    if err != amdsmi.amdsmi_status_t.AMDSMI_STATUS_SUCCESS:
        raise RuntimeError(str(err))
    if len(result) == 1:
        return result[0]
    return result


def get_socket_handles():
    """Return the system socket handles as a list of ``Pointer``."""
    count = ctypes.c_uint()
    amdsmi_check(
        amdsmi.amdsmi_get_socket_handles(ctypes.addressof(count), None)
    )
    handles = types.ListOfPointer.allocate(count.value)
    amdsmi_check(
        amdsmi.amdsmi_get_socket_handles(ctypes.addressof(count), handles)
    )
    return handles.to_list()


def get_processor_handles(socket_handle):
    """Return the processor handles on a socket as a list of ``Pointer``."""
    count = ctypes.c_uint()
    amdsmi_check(
        amdsmi.amdsmi_get_processor_handles(
            socket_handle, ctypes.addressof(count), None
        )
    )
    handles = types.ListOfPointer.allocate(count.value)
    amdsmi_check(
        amdsmi.amdsmi_get_processor_handles(
            socket_handle, ctypes.addressof(count), handles
        )
    )
    return handles.to_list()


def is_amd_gpu(processor_handle):
    processor_type = amdsmi_check(
        amdsmi.amdsmi_get_processor_type(processor_handle)
    )
    return (
        processor_type == amdsmi.processor_type_t.AMDSMI_PROCESSOR_TYPE_AMD_GPU
    )


amdsmi_check(
    amdsmi.amdsmi_init(amdsmi.amdsmi_init_flags_t.AMDSMI_INIT_AMD_GPUS)
)
try:
    version = amdsmi_check(amdsmi.amdsmi_get_lib_version())
    print(
        f"AMD SMI library version: "
        f"{version.major}.{version.minor}.{version.release}"
    )

    sockets = get_socket_handles()
    print(f"sockets: {len(sockets)}")

    gpus = []
    for socket_handle in sockets:
        for processor_handle in get_processor_handles(socket_handle):
            if is_amd_gpu(processor_handle):
                gpus.append(processor_handle)
    print(f"AMD GPUs: {len(gpus)}")

    for index, processor_handle in enumerate(gpus):
        asic = amdsmi_check(amdsmi.amdsmi_get_gpu_asic_info(processor_handle))
        name = asic.market_name
        if isinstance(name, bytes):
            name = name.split(b"\x00", 1)[0].decode("utf-8", "replace")
        print(f"  GPU {index}: {name}")
finally:
    amdsmi_check(amdsmi.amdsmi_shut_down())
print("ok")
