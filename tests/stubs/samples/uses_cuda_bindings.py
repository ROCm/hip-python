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

"""Annotated CUDA code running on the interop shim.

This is the port case: the file would type-check the same against
`cuda-bindings` upstream, which is the point of shipping types here.
Type-checked, never run. See `../test_pyright_samples.py`.
"""

from cuda.bindings import driver, runtime


def device_count() -> int:
    """Number of devices, through the runtime API."""
    err, count = runtime.cudaGetDeviceCount()
    if int(err) != 0:
        raise RuntimeError(f"cudaGetDeviceCount: {err}")
    return int(count)


def driver_device_count() -> int:
    """The same, through the driver API."""
    driver.cuInit(0)
    _, count = driver.cuDeviceGetCount()
    return int(count)


def allocate(nbytes: int) -> object:
    err, pointer = runtime.cudaMalloc(nbytes)
    if int(err) != 0:
        raise MemoryError(f"cudaMalloc({nbytes}): {err}")
    return pointer


def release(pointer: object) -> None:
    runtime.cudaFree(pointer)
