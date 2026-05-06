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

"""hipSPARSELt 2:4 structured-sparsity matmul.

Mirrors the upstream `example_spmm_strided_batched.cpp` sample
(`/src/rocm-libraries/projects/hipsparselt/clients/samples/`) using
the hip-python bindings: 2:4 structured-sparsity matmul on a single
GPU. A is structured-sparse (50% zeros, 2-of-4 pattern), B is dense.

Single-GPU; no batching.
"""

import ctypes
import numpy as np
from rocm.bindings import hip, hipsparse, hipsparselt


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


# Problem: D = alpha * A @ B + beta * C, A is 2:4 structured sparse.
m, n, k = 128, 128, 128
alpha = ctypes.c_float(1.0)
beta = ctypes.c_float(0.0)

# Column-major host data; A will be pruned in-place by hipSPARSELt to satisfy
# the 2:4 sparsity pattern (every group of 4 contiguous values keeps ≤ 2).
rng = np.random.default_rng(seed=0)
a_h = rng.standard_normal((m, k), dtype=np.float32).astype(np.float16)
b_h = rng.standard_normal((k, n), dtype=np.float32).astype(np.float16)
c_h = np.zeros((m, n), dtype=np.float16)

a_bytes = a_h.nbytes
b_bytes = b_h.nbytes
c_bytes = c_h.nbytes

# Device buffers.
d_a = hip_check(hip.hipMalloc(a_bytes))
d_b = hip_check(hip.hipMalloc(b_bytes))
d_c = hip_check(hip.hipMalloc(c_bytes))
d_d = hip_check(hip.hipMalloc(c_bytes))
d_pruned = hip_check(hip.hipMalloc(a_bytes))  # pruned A goes here

hip_check(hip.hipMemcpy(d_a, a_h, a_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(d_b, b_h, b_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(d_c, c_h, c_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

# hipSPARSELt handle + descriptors. Note: every hipSPARSELt API takes
# `&handle` / `&desc` (pointer-style). The Python bindings accept the
# Python wrapper objects directly.
handle = hipsparselt.hipsparseLtHandle_t()
hip_check(hipsparselt.hipsparseLtInit(handle))

# A: structured-sparse (2:4), B/C/D: dense. All col-major.
matA = hipsparselt.hipsparseLtMatDescriptor_t()
matB = hipsparselt.hipsparseLtMatDescriptor_t()
matC = hipsparselt.hipsparseLtMatDescriptor_t()
matD = hipsparselt.hipsparseLtMatDescriptor_t()
hip_check(
    hipsparselt.hipsparseLtStructuredDescriptorInit(
        handle, matA, m, k, m, 16, hip.hipDataType.HIP_R_16F,
        hipsparse.hipsparseOrder_t.HIPSPARSE_ORDER_COL,
        hipsparselt.hipsparseLtSparsity_t.HIPSPARSELT_SPARSITY_50_PERCENT,
    )
)
hip_check(
    hipsparselt.hipsparseLtDenseDescriptorInit(
        handle, matB, k, n, k, 16, hip.hipDataType.HIP_R_16F,
        hipsparse.hipsparseOrder_t.HIPSPARSE_ORDER_COL,
    )
)
hip_check(
    hipsparselt.hipsparseLtDenseDescriptorInit(
        handle, matC, m, n, m, 16, hip.hipDataType.HIP_R_16F,
        hipsparse.hipsparseOrder_t.HIPSPARSE_ORDER_COL,
    )
)
hip_check(
    hipsparselt.hipsparseLtDenseDescriptorInit(
        handle, matD, m, n, m, 16, hip.hipDataType.HIP_R_16F,
        hipsparse.hipsparseOrder_t.HIPSPARSE_ORDER_COL,
    )
)

matmul = hipsparselt.hipsparseLtMatmulDescriptor_t()
alg_sel = hipsparselt.hipsparseLtMatmulAlgSelection_t()
plan = hipsparselt.hipsparseLtMatmulPlan_t()
hip_check(
    hipsparselt.hipsparseLtMatmulDescriptorInit(
        handle, matmul,
        hipsparse.hipsparseOperation_t.HIPSPARSE_OPERATION_NON_TRANSPOSE,
        hipsparse.hipsparseOperation_t.HIPSPARSE_OPERATION_NON_TRANSPOSE,
        matA, matB, matC, matD,
        hipsparselt.hipsparseLtComputetype_t.HIPSPARSELT_COMPUTE_32F,
    )
)
hip_check(
    hipsparselt.hipsparseLtMatmulAlgSelectionInit(
        handle, alg_sel, matmul,
        hipsparselt.hipsparseLtMatmulAlg_t.HIPSPARSELT_MATMUL_ALG_DEFAULT,
    )
)

# Prune A to 2:4 pattern (writes pruned data to d_pruned).
hip_check(
    hipsparselt.hipsparseLtSpMMAPrune(
        handle, matmul, d_a, d_pruned,
        hipsparselt.hipsparseLtPruneAlg_t.HIPSPARSELT_PRUNE_SPMMA_STRIP,
        None,  # default stream
    )
)

# Build the matmul plan and discover required scratch sizes.
hip_check(hipsparselt.hipsparseLtMatmulPlanInit(handle, plan, matmul, alg_sel))
workspace_size = ctypes.c_size_t(0)
hip_check(hipsparselt.hipsparseLtMatmulGetWorkspace(handle, plan, ctypes.addressof(workspace_size)))
compressed_size = ctypes.c_size_t(0)
compress_buffer_size = ctypes.c_size_t(0)
hip_check(
    hipsparselt.hipsparseLtSpMMACompressedSize(
        handle, plan,
        ctypes.addressof(compressed_size),
        ctypes.addressof(compress_buffer_size),
    )
)

d_compressed = hip_check(hip.hipMalloc(compressed_size.value))
d_compress_buffer = hip_check(hip.hipMalloc(compress_buffer_size.value))
d_workspace = hip_check(hip.hipMalloc(max(workspace_size.value, 1)))

# Compress the pruned A into the format the kernel consumes.
hip_check(
    hipsparselt.hipsparseLtSpMMACompress(
        handle, plan, d_pruned, d_compressed, d_compress_buffer, None
    )
)

# Run the matmul on a 1-stream array.
streams = (ctypes.c_void_p * 1)(0)  # default stream
hip_check(
    hipsparselt.hipsparseLtMatmul(
        handle, plan,
        ctypes.addressof(alpha),
        d_compressed, d_b,
        ctypes.addressof(beta),
        d_c, d_d,
        d_workspace, streams, 1,
    )
)
hip_check(hip.hipDeviceSynchronize())

# Verify against a NumPy reference using the pruned A (read back from d_pruned).
a_pruned_h = np.empty_like(a_h)
hip_check(hip.hipMemcpy(a_pruned_h, d_pruned, a_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
d_h = np.empty_like(c_h)
hip_check(hip.hipMemcpy(d_h, d_d, c_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
d_expected = (
    alpha.value * a_pruned_h.astype(np.float32) @ b_h.astype(np.float32)
    + beta.value * c_h.astype(np.float32)
).astype(np.float16)

if np.allclose(d_h.astype(np.float32), d_expected.astype(np.float32), atol=1e-1, rtol=1e-2):
    print("ok")
else:
    diff = np.abs(d_h.astype(np.float32) - d_expected.astype(np.float32))
    print(f"FAILED: max abs diff = {diff.max():.4f}, mean abs diff = {diff.mean():.4f}")

# Clean up.
hip_check(hipsparselt.hipsparseLtMatmulPlanDestroy(plan))
hip_check(hipsparselt.hipsparseLtMatDescriptorDestroy(matA))
hip_check(hipsparselt.hipsparseLtMatDescriptorDestroy(matB))
hip_check(hipsparselt.hipsparseLtMatDescriptorDestroy(matC))
hip_check(hipsparselt.hipsparseLtMatDescriptorDestroy(matD))
hip_check(hipsparselt.hipsparseLtDestroy(handle))
hip_check(hip.hipFree(d_a))
hip_check(hip.hipFree(d_b))
hip_check(hip.hipFree(d_c))
hip_check(hip.hipFree(d_d))
hip_check(hip.hipFree(d_pruned))
hip_check(hip.hipFree(d_compressed))
hip_check(hip.hipFree(d_compress_buffer))
hip_check(hip.hipFree(d_workspace))
