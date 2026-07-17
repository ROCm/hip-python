# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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

"""hipSOLVER LU factorization (``getrf``) of a small dense matrix.

Ported from the upstream hipSOLVER C sample
``clients/samples/example_basic.c``: it factors a real ``M``-by-``N``
matrix ``A`` on the GPU via :py:obj:`~.hipsolverDgetrf`, so that
:math:`PA = LU`. Unlike the C sample (which only prints ``A`` and ``U``),
this port reconstructs :math:`LU` with :py:obj:`numpy` and checks it
against the row-pivoted input, printing ``ok`` on success.

hipSOLVER, like LAPACK, expects **column-major** matrices, so ``A`` is a
Fortran-ordered :py:obj:`numpy` array. The workspace size is queried with
:py:obj:`~.hipsolverDgetrf_bufferSize`, whose ``lwork`` output is returned
directly (a callee-allocated scalar), not written through a caller buffer.
"""

# [literalinclude-begin]
import numpy as np
from rocm.bindings import hip, hipsolver


def hip_check(call_result):
    if isinstance(call_result, tuple):
        err = call_result[0]
        result = call_result[1:]
        if len(result) == 1:
            result = result[0]
    else:
        err = call_result
        result = ()
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hipsolver.hipsolverStatus_t)
        and err != hipsolver.hipsolverStatus_t.HIPSOLVER_STATUS_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result


# input matrix on the host, column-major (Fortran order); same values as
# the upstream C sample.
m = n = lda = 3
A = np.asfortranarray(
    np.array(
        [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]],
        dtype=np.float64,
    )
)
A_orig = A.copy()
size_piv = min(m, n)

# allocate device memory: factored matrix, pivot indices and the info flag.
dA = hip_check(hip.hipMalloc(A.nbytes))
dIpiv = hip_check(hip.hipMalloc(size_piv * np.dtype(np.int32).itemsize))
dInfo = hip_check(hip.hipMalloc(np.dtype(np.int32).itemsize))

# copy the input matrix to the device.
hip_check(
    hip.hipMemcpy(dA, A, A.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
)

# create the solver handle and query + allocate the workspace. `lwork` (the
# workspace size in bytes) is returned by the buffer-size query.
handle = hip_check(hipsolver.hipsolverCreate())
lwork = hip_check(hipsolver.hipsolverDgetrf_bufferSize(handle, m, n, dA, lda))
dWork = hip_check(hip.hipMalloc(lwork))

# compute the LU factorization in place on the GPU.
hip_check(
    hipsolver.hipsolverDgetrf(
        handle, m, n, dA, lda, dWork, lwork, dIpiv, dInfo
    )
)

# copy the factored matrix, pivots and info flag back to the host.
LU = np.zeros((m, n), dtype=np.float64, order="F")
ipiv = np.zeros(size_piv, dtype=np.int32)
info = np.zeros(1, dtype=np.int32)
hip_check(
    hip.hipMemcpy(LU, dA, LU.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost)
)
hip_check(
    hip.hipMemcpy(
        ipiv, dIpiv, ipiv.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost
    )
)
hip_check(
    hip.hipMemcpy(
        info, dInfo, info.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost
    )
)

# verify: getrf stores unit-lower L (below the diagonal) and upper U (on and
# above it) in the factored matrix, with PA == L @ U. Rebuild L and U and
# replay the LAPACK 1-based row pivots on a copy of the original A to form PA.
L = np.tril(LU, -1) + np.eye(m, n)
U = np.triu(LU)
PA = A_orig.copy()
for i in range(size_piv):
    p = int(ipiv[i]) - 1  # ipiv is 1-based (LAPACK convention)
    if p != i:
        PA[[i, p], :] = PA[[p, i], :]

if info[0] == 0 and np.allclose(PA, L @ U):
    print("ok")
else:
    print("FAILED")
# print(f"{LU=}")
# print(f"{ipiv=}")

# clean up
hip_check(hip.hipFree(dA))
hip_check(hip.hipFree(dIpiv))
hip_check(hip.hipFree(dInfo))
hip_check(hip.hipFree(dWork))
hip_check(hipsolver.hipsolverDestroy(handle))
