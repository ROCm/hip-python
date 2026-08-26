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

"""hipTensor reduction with NumPy.

Mirrors the upstream `samples/03_reduction/reduction_c.c` C-API
sample (`rocm-libraries/projects/hiptensor/`) using the
hip-python bindings: FP32 4-D tensor reduction
`C_{k,v} = alpha * sum_{m,h} A_{m,h,k,v} + beta * C_{k,v}` on a
single GPU. Verified against `numpy.sum(A, axis=(0,1))`.
"""

import ctypes

# The hipTensor bindings are generated but built on no platform: `hiptensor`
# is absent from HIP_PYTHON_ALL_LIBRARIES in the rocm-bindings-libraries
# CMakeLists, because ROCm installs no hipTensor headers to compile them
# against. The example is kept as a reference for the hipTensor C API, and
# runs against a wheel that was built with the bindings enabled.
try:
    from rocm.bindings import hiptensor
except ImportError as e:
    raise NotImplementedError(
        "This example needs the rocm.bindings.hiptensor bindings, which the "
        "rocm-bindings-libraries wheel does not build on any platform: ROCm "
        f"installs no hipTensor headers to compile them against ({e})."
    )

import numpy as np
from rocm.bindings import hip


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
        isinstance(err, hiptensor.hiptensorStatus_t)
        and err != hiptensor.hiptensorStatus_t.HIPTENSOR_STATUS_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result


# Problem: C_{k,v} = alpha * sum_{m,h} A_{m,h,k,v} + beta * C_{k,v}.
# Mode labels are arbitrary int32 ASCII codes by hipTensor convention.
mode_a = (ctypes.c_int32 * 4)(ord("m"), ord("h"), ord("k"), ord("v"))
mode_c = (ctypes.c_int32 * 2)(ord("k"), ord("v"))
extent_a = (ctypes.c_int64 * 4)(8, 8, 16, 16)  # m, h, k, v — kept small
extent_c = (ctypes.c_int64 * 2)(16, 16)  # k, v

alpha = ctypes.c_float(1.1)
beta = ctypes.c_float(0.0)

# Host data.
rng = np.random.default_rng(seed=0)
a_h = rng.standard_normal((8, 8, 16, 16), dtype=np.float32)
c_h = np.zeros((16, 16), dtype=np.float32)
c_expected = (alpha.value * a_h.sum(axis=(0, 1)) + beta.value * c_h).astype(
    np.float32
)

# Device buffers.
d_a = hip_check(hip.hipMalloc(a_h.nbytes))
d_c = hip_check(hip.hipMalloc(c_h.nbytes))
hip_check(
    hip.hipMemcpy(
        d_a, a_h, a_h.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice
    )
)
hip_check(
    hip.hipMemcpy(
        d_c, c_h, c_h.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice
    )
)

# Library handle and tensor descriptors (FP32 throughout).
handle = hip_check(hiptensor.hiptensorCreate())
desc_a = hip_check(
    hiptensor.hiptensorCreateTensorDescriptor(
        handle,
        4,
        mode_a,
        None,
        hiptensor.hiptensorDataType_t.HIPTENSOR_R_32F,
        0,
    )
)
desc_c = hip_check(
    hiptensor.hiptensorCreateTensorDescriptor(
        handle,
        2,
        mode_c,
        None,
        hiptensor.hiptensorDataType_t.HIPTENSOR_R_32F,
        0,
    )
)

# Reduction operator descriptor: D_{k,v} = sum_{m,h} A_{m,h,k,v}; identity unary
# pre-/post-ops on A and C/D; FP32 compute.
op = hip_check(
    hiptensor.hiptensorCreateReduction(
        handle,
        desc_a,
        mode_a,
        hiptensor.hiptensorOperator_t.HIPTENSOR_OP_IDENTITY,
        desc_c,
        mode_c,
        hiptensor.hiptensorOperator_t.HIPTENSOR_OP_IDENTITY,
        desc_c,
        mode_c,
        hiptensor.hiptensorOperator_t.HIPTENSOR_OP_ADD,
        hiptensor.hiptensorComputeDescriptor_t.HIPTENSOR_COMPUTE_DESC_32F,
    )
)

# Plan + workspace.
pref = hip_check(
    hiptensor.hiptensorCreatePlanPreference(
        handle,
        hiptensor.hiptensorAlgo_t.HIPTENSOR_ALGO_DEFAULT,
        hiptensor.hiptensorJitMode_t.HIPTENSOR_JIT_MODE_NONE,
    )
)
worksize = hip_check(
    hiptensor.hiptensorEstimateWorkspaceSize(
        handle,
        op,
        pref,
        hiptensor.hiptensorWorksizePreference_t.HIPTENSOR_WORKSPACE_DEFAULT,
    )
)
d_work = hip_check(hip.hipMalloc(max(worksize, 1)))
plan = hip_check(hiptensor.hiptensorCreatePlan(handle, op, pref, worksize))

# Run the reduction (default stream = 0).
hip_check(
    hiptensor.hiptensorReduce(
        handle,
        plan,
        ctypes.addressof(alpha),
        d_a,
        ctypes.addressof(beta),
        d_c,
        d_c,
        d_work,
        worksize,
        0,
    )
)
hip_check(hip.hipDeviceSynchronize())

# Copy result back and verify.
hip_check(
    hip.hipMemcpy(
        c_h, d_c, c_h.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost
    )
)
if np.allclose(c_h, c_expected, atol=1e-3, rtol=1e-3):
    print("ok")
else:
    diff = np.abs(c_h - c_expected)
    print(
        f"FAILED: max abs diff = {diff.max():.4e}, mean abs diff = {diff.mean():.4e}"
    )

# Clean up.
hip_check(hiptensor.hiptensorDestroyPlan(plan))
hip_check(hiptensor.hiptensorDestroyPlanPreference(pref))
hip_check(hiptensor.hiptensorDestroyOperationDescriptor(op))
hip_check(hiptensor.hiptensorDestroyTensorDescriptor(desc_a))
hip_check(hiptensor.hiptensorDestroyTensorDescriptor(desc_c))
hip_check(hiptensor.hiptensorDestroy(handle))
hip_check(hip.hipFree(d_a))
hip_check(hip.hipFree(d_c))
hip_check(hip.hipFree(d_work))
