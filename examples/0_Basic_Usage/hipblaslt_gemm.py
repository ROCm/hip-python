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

"""hipBLASLt FP16 GEMM with NumPy.

Mirrors the upstream `01_hipblaslt_gemm` C++ sample
(`rocm-libraries/projects/hipblaslt/clients/samples/`) using the
hip-python bindings: compute D = alpha * A @ B + beta * C with
half-precision inputs, FP32 accumulation, on a single GPU. Verifies
the result against a NumPy reference.

Single-GPU; no batching.
"""

import ctypes
import numpy as np
from rocm.bindings import hip, hipblas, hipblaslt


def hip_check(call_result):
    if isinstance(call_result, tuple):
        err = call_result[0]
        result = call_result[1:]
        if len(result) == 1:
            result = result[0]
    else:
        # Single-output funcs (e.g. hipMemcpy after the with-nogil
        # codegen refactor) return the bare hipError_t enum, not a
        # 1-tuple. Treat that as an empty-result call.
        err = call_result
        result = ()
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hipblas.hipblasStatus_t)
        and err != hipblas.hipblasStatus_t.HIPBLAS_STATUS_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result


# Problem: D = alpha * A @ B + beta * C, all NN (no transpose).
m, n, k = 64, 64, 64
alpha = ctypes.c_float(1.0)
beta = ctypes.c_float(1.0)
max_workspace_size = 32 * 1024 * 1024  # 32 MiB

# Host data. hipBLASLt is column-major, so the host buffers are stored in
# Fortran order — their raw bytes then match the column-major layout the
# device reads with leading dimensions lda=m, ldb=k, ldc=ldd=m.
rng = np.random.default_rng(seed=0)
a_h = np.asfortranarray(
    rng.standard_normal((m, k), dtype=np.float32).astype(np.float16)
)
b_h = np.asfortranarray(
    rng.standard_normal((k, n), dtype=np.float32).astype(np.float16)
)
c_h = np.asfortranarray(
    rng.standard_normal((m, n), dtype=np.float32).astype(np.float16)
)
# Reference computed in fp32 for accuracy comparison.
d_expected = (
    alpha.value * a_h.astype(np.float32) @ b_h.astype(np.float32)
    + beta.value * c_h.astype(np.float32)
).astype(np.float16)

# Device buffers.
a_bytes = a_h.nbytes
b_bytes = b_h.nbytes
c_bytes = c_h.nbytes
d_a = hip_check(hip.hipMalloc(a_bytes))
d_b = hip_check(hip.hipMalloc(b_bytes))
d_c = hip_check(hip.hipMalloc(c_bytes))
d_d = hip_check(hip.hipMalloc(c_bytes))
d_workspace = hip_check(hip.hipMalloc(max_workspace_size))

hip_check(
    hip.hipMemcpy(d_a, a_h, a_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
)
hip_check(
    hip.hipMemcpy(d_b, b_h, b_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
)
hip_check(
    hip.hipMemcpy(d_c, c_h, c_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
)

# Build matrix layout descriptors (col-major: lda=m, ldb=k, ldc=ldd=m).
matA = hip_check(
    hipblaslt.hipblasLtMatrixLayoutCreate(hip.hipDataType.HIP_R_16F, m, k, m)
)
matB = hip_check(
    hipblaslt.hipblasLtMatrixLayoutCreate(hip.hipDataType.HIP_R_16F, k, n, k)
)
matC = hip_check(
    hipblaslt.hipblasLtMatrixLayoutCreate(hip.hipDataType.HIP_R_16F, m, n, m)
)
matD = hip_check(
    hipblaslt.hipblasLtMatrixLayoutCreate(hip.hipDataType.HIP_R_16F, m, n, m)
)

# Matmul descriptor: FP32 accumulation, FP32 scale.
matmul = hip_check(
    hipblaslt.hipblasLtMatmulDescCreate(
        hipblas.hipblasComputeType_t.HIPBLAS_COMPUTE_32F,
        hip.hipDataType.HIP_R_32F,
    )
)
trans_n = ctypes.c_int32(int(hipblas.hipblasOperation_t.HIPBLAS_OP_N))
hip_check(
    hipblaslt.hipblasLtMatmulDescSetAttribute(
        matmul,
        hipblaslt.hipblasLtMatmulDescAttributes_t.HIPBLASLT_MATMUL_DESC_TRANSA,
        ctypes.addressof(trans_n),
        ctypes.sizeof(trans_n),
    )
)
hip_check(
    hipblaslt.hipblasLtMatmulDescSetAttribute(
        matmul,
        hipblaslt.hipblasLtMatmulDescAttributes_t.HIPBLASLT_MATMUL_DESC_TRANSB,
        ctypes.addressof(trans_n),
        ctypes.sizeof(trans_n),
    )
)

# Heuristics: ask for 1 algorithm.
handle = hip_check(hipblaslt.hipblasLtCreate())
pref = hip_check(hipblaslt.hipblasLtMatmulPreferenceCreate())
ws = ctypes.c_uint64(max_workspace_size)
hip_check(
    hipblaslt.hipblasLtMatmulPreferenceSetAttribute(
        pref,
        hipblaslt.hipblasLtMatmulPreferenceAttributes_t.HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        ctypes.addressof(ws),
        ctypes.sizeof(ws),
    )
)

heuristic_results = hipblaslt.hipblasLtMatmulHeuristicResult_t.allocate(1)
returned = hip_check(
    hipblaslt.hipblasLtMatmulAlgoGetHeuristic(
        handle,
        matmul,
        matA,
        matB,
        matC,
        matD,
        pref,
        1,
        heuristic_results,
    )
)
assert returned > 0, "hipBLASLt found no algorithm for this problem"

# Run the matmul on the default stream.
hip_check(
    hipblaslt.hipblasLtMatmul(
        handle,
        matmul,
        ctypes.addressof(alpha),
        d_a,
        matA,
        d_b,
        matB,
        ctypes.addressof(beta),
        d_c,
        matC,
        d_d,
        matD,
        heuristic_results.algo,
        d_workspace,
        heuristic_results.workspaceSize,
        None,  # default stream
    )
)
hip_check(hip.hipDeviceSynchronize())

# Copy result back and verify.
d_h = np.empty_like(c_h)
hip_check(
    hip.hipMemcpy(d_h, d_d, c_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost)
)

if np.allclose(
    d_h.astype(np.float32), d_expected.astype(np.float32), atol=1e-1, rtol=1e-2
):
    print("ok")
else:
    diff = np.abs(d_h.astype(np.float32) - d_expected.astype(np.float32))
    print(
        f"FAILED: max abs diff = {diff.max():.4f}, mean abs diff = {diff.mean():.4f}"
    )

# Clean up.
hip_check(hipblaslt.hipblasLtMatmulPreferenceDestroy(pref))
hip_check(hipblaslt.hipblasLtMatmulDescDestroy(matmul))
hip_check(hipblaslt.hipblasLtMatrixLayoutDestroy(matA))
hip_check(hipblaslt.hipblasLtMatrixLayoutDestroy(matB))
hip_check(hipblaslt.hipblasLtMatrixLayoutDestroy(matC))
hip_check(hipblaslt.hipblasLtMatrixLayoutDestroy(matD))
hip_check(hipblaslt.hipblasLtDestroy(handle))
hip_check(hip.hipFree(d_a))
hip_check(hip.hipFree(d_b))
hip_check(hip.hipFree(d_c))
hip_check(hip.hipFree(d_d))
hip_check(hip.hipFree(d_workspace))
