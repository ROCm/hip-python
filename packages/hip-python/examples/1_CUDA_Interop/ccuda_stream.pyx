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

# [literalinclude-begin]
cimport cuda.ccudart as ccudart

cdef ccudart.cudaError_t err
cdef ccudart.cudaStream_t stream
DEF num_bytes = 4*100
cdef char[num_bytes] x_h
cdef void* x_d
cdef int x

def cuda_check(ccudart.cudaError_t err):
    IF HIP_PYTHON: # HIP Python CUDA interop layer Cython interfaces are used like C API
        success_status = ccudart.cudaSuccess
    ELSE:
        success_status = ccudart.cudaError_t.cudaSuccess
    if err != success_status:
        raise RuntimeError(f"reason: {err}")

IF HIP_PYTHON:
    print("using HIP Python wrapper for CUDA Python")

cuda_check(ccudart.cudaStreamCreate(&stream))
cuda_check(ccudart.cudaMalloc(&x_d, num_bytes))
cuda_check(ccudart.cudaMemcpyAsync(x_d,x_h, num_bytes, ccudart.cudaMemcpyHostToDevice, stream))
cuda_check(ccudart.cudaMemsetAsync(x_d, 0, num_bytes, stream))
cuda_check(ccudart.cudaMemcpyAsync(x_h, x_d, num_bytes, ccudart.cudaMemcpyDeviceToHost, stream))
cuda_check(ccudart.cudaStreamSynchronize(stream))
cuda_check(ccudart.cudaStreamDestroy(stream))

# deallocate device data
cuda_check(ccudart.cudaFree(x_d))

for i in range(0,round(num_bytes/4)):
    x = (<int*>&x_h[4*i])[0]
    if x != 0:
        raise ValueError(f"expected '0' for element {i}, is: '{x}'")
print("ok")
