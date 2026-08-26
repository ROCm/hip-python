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

"""Query GPUs through the NVML (``pynvml``) Python API.

On AMD GPUs the ``hip-python-interop`` package supplies a ``pynvml``
compatibility shim backed by AMD SMI (``rocm.bindings.amdsmi``); elsewhere
the upstream ``pynvml`` package does. The Python source below is identical
in both environments, which is the whole point of the shim: code that
already speaks NVML keeps working on AMD hardware unchanged.
"""

import sys

if sys.platform == "win32":
    raise NotImplementedError(
        "This example needs the pynvml shim, which is backed by AMD SMI. ROCm "
        "ships no AMD SMI library on Windows, so the rocm-bindings-systems "
        "wheel that provides rocm.bindings.amdsmi is not built there."
    )

# [literalinclude-begin]
import pynvml


def _gib(num_bytes):
    return num_bytes / (1024**3)


pynvml.nvmlInit()
try:
    count = pynvml.nvmlDeviceGetCount()
    print(f"NVML reports {count} device(s)")

    for index in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)

        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(
            name, bytes
        ):  # upstream pynvml returns bytes on older versions
            name = name.decode("utf-8", "replace")
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        print(f"device {index}: {name}")
        print(f"  uuid:   {uuid}")
        print(
            f"  memory: {_gib(mem.used):.2f} / {_gib(mem.total):.2f} GiB used"
        )

        # Telemetry that may be unsupported on some platforms/drivers; NVML
        # signals that with NVMLError_NotSupported, which we treat as "n/a".
        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            print(f"  temp:   {temp} C")
        except pynvml.NVMLError_NotSupported:
            print("  temp:   n/a")

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"  util:   gpu {util.gpu}% / mem {util.memory}%")
        except pynvml.NVMLError_NotSupported:
            print("  util:   n/a")

        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            print(f"  power:  {power_w:.1f} W")
        except pynvml.NVMLError_NotSupported:
            print("  power:  n/a")
finally:
    pynvml.nvmlShutdown()

print("ok")
