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

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

"""hipSPARSE sparse Saxpyi on float32 numpy arrays.

Sparse-vector counterpart of `hipblas_with_numpy.py`. Computes
`y[xInd[i]] += alpha * xVal[i]` for a sparse `x` (held as
value/index pairs) and a dense `y`. Single GPU.
"""

# [literalinclude-begin]
import ctypes

import numpy as np
from rocm.bindings import hip, hipsparse


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hipsparse.hipsparseStatus_t)
        and err != hipsparse.hipsparseStatus_t.HIPSPARSE_STATUS_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result


# Sparse x of length nnz scattered into a dense y of length n.
n = 16
nnz = 4
alpha = ctypes.c_float(2.0)

# Host data.
x_val_h = np.array([1.0, -2.0, 3.5, 4.25], dtype=np.float32)
x_ind_h = np.array([1, 4, 9, 12], dtype=np.int32)
y_h = np.arange(n, dtype=np.float32)

# Reference: y[xInd[i]] += alpha * xVal[i]
y_expected = y_h.copy()
for i in range(nnz):
    y_expected[x_ind_h[i]] += alpha.value * x_val_h[i]

# Device buffers.
x_val_d = hip_check(hip.hipMalloc(x_val_h.nbytes))
x_ind_d = hip_check(hip.hipMalloc(x_ind_h.nbytes))
y_d = hip_check(hip.hipMalloc(y_h.nbytes))

hip_check(hip.hipMemcpy(x_val_d, x_val_h, x_val_h.nbytes,
                         hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(x_ind_d, x_ind_h, x_ind_h.nbytes,
                         hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(y_d, y_h, y_h.nbytes,
                         hip.hipMemcpyKind.hipMemcpyHostToDevice))

# Run hipsparseSaxpyi.
handle = hip_check(hipsparse.hipsparseCreate())
hip_check(
    hipsparse.hipsparseSaxpyi(
        handle, nnz,
        ctypes.addressof(alpha),
        x_val_d, x_ind_d, y_d,
        hipsparse.hipsparseIndexBase_t.HIPSPARSE_INDEX_BASE_ZERO,
    )
)
hip_check(hipsparse.hipsparseDestroy(handle))

# Copy result back and verify.
hip_check(hip.hipMemcpy(y_h, y_d, y_h.nbytes,
                         hip.hipMemcpyKind.hipMemcpyDeviceToHost))

if np.allclose(y_h, y_expected):
    print("ok")
else:
    print("FAILED")

# Clean up.
hip_check(hip.hipFree(x_val_d))
hip_check(hip.hipFree(x_ind_d))
hip_check(hip.hipFree(y_d))
