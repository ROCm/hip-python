# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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

# [literalinclude-begin]
import ctypes

import numpy as np
from hip import hip, hipblas


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hipblas.hipblasStatus_t)
        and err != hipblas.hipblasStatus_t.HIPBLAS_STATUS_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result


num_elements = 100

# input data on host
alpha = ctypes.c_float(2)
x_h = np.random.rand(num_elements).astype(dtype=np.float32)
y_h = np.random.rand(num_elements).astype(dtype=np.float32)

# expected result
y_expected = alpha * x_h + y_h

# stream with 2 enabled CUs (we pass a 2x 32-bit mask with two set bits)
stream = hip_check(hip.hipExtStreamCreateWithCUMask(2, [0b1, 0b10]))

# device vectors
num_bytes = num_elements * np.dtype(np.float32).itemsize
x_d = hip_check(hip.hipMallocAsync(num_bytes, stream))
y_d = hip_check(hip.hipMallocAsync(num_bytes, stream))

# copy input data to device
hip_check(
    hip.hipMemcpyAsync(
        x_d, x_h, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice, stream
    )
)
hip_check(
    hip.hipMemcpyAsync(
        y_d, y_h, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice, stream
    )
)

# call hipblasSaxpy + initialization & destruction of handle
handle = hip_check(hipblas.hipblasCreate())
hip_check(hipblas.hipblasSetStream(handle, stream))
hip_check(
    hipblas.hipblasSaxpy(
        handle, num_elements, ctypes.addressof(alpha), x_d, 1, y_d, 1
    )
)
hip_check(hipblas.hipblasDestroy(handle))

# copy result (stored in y_d) back to host (store in y_h)
hip_check(
    hip.hipMemcpyAsync(
        y_h, y_d, num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost, stream
    )
)

hip_check(hip.hipStreamSynchronize(stream))
hip_check(hip.hipStreamDestroy(stream))

# compare to expected result
if np.allclose(y_expected, y_h):
    print("ok")
else:
    print("FAILED")
# print(f"{y_h=}")
# print(f"{y_expected=}")

# clean up
hip_check(hip.hipFree(x_d))
hip_check(hip.hipFree(y_d))
