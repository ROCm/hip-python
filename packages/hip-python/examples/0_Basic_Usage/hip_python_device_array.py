# MIT License
# 
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

"""This example demonstrates how to configure the shape and data type
of a device array returned by hipMalloc.

This example demonstrates how to configure the shape and data type
of a device array returned by hipMalloc (and related routines).
Further showns how to retrieve single elements / contiguous subarrays
with respect to specified type and shape information.
"""

# [literalinclude-begin]
verbose = False

import ctypes

from hip import hip, hipblas
import numpy as np

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err,hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif isinstance(err,hipblas.hipblasStatus_t) and err != hipblas.hipblasStatus_t.HIPBLAS_STATUS_SUCCESS:
        raise RuntimeError(str(err))
    return result

# init host array and fill with ones
shape = (3,20) # shape[1]: inner dim
x_h = np.ones(shape,dtype="float32")
num_bytes = x_h.size * x_h.itemsize

# init device array and upload host data
x_d = hip_check(hip.hipMalloc(num_bytes)).configure(
    typestr="float32",shape=shape
)
hip_check(hip.hipMemcpy(x_d,x_h,num_bytes,hip.hipMemcpyKind.hipMemcpyHostToDevice))

# scale device array entries by row index using hipblasSscal
handle = hip_check(hipblas.hipblasCreate())
for r in range(0,shape[0]):
    row = x_d[r,:] # extract subarray
    row_len = row.size
    alpha = ctypes.c_float(r)
    hip_check(hipblas.hipblasSscal(handle, row_len, ctypes.addressof(alpha), row, 1))
    hip_check(hip.hipDeviceSynchronize())
hip_check(hipblas.hipblasDestroy(handle))

# copy device data back to host
hip_check(hip.hipMemcpy(x_h,x_d,num_bytes,hip.hipMemcpyKind.hipMemcpyDeviceToHost))

# deallocate device data
hip_check(hip.hipFree(x_d))

for r in range(0,shape[0]):
    row_rounded = [round(el) for el in x_h[r,:]]
    for c,e in enumerate(row_rounded):
        if e != r:
            raise ValueError(f"expected '{r}' for element ({r},{c}), is '{e}")
    if verbose:
        print("\t".join((str(i) for i in row_rounded))+"\n")
print("ok")
