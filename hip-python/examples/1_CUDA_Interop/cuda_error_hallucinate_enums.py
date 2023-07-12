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

"""Enum constant hallucination example.

Demonstrates the enum constant hallucination feature of HIP Python's enum types
that can be used for porting CUDA applications with enum constants
that have no HIP equivalent.

The hallucinated enum types act as instances of the original type,
they all have the same value, which does not conflict with the values
of the original enum constants.

This script:

* Will fail while initializing error_kinds if you just run it as it is.
    
* Will pass if you specify environment variable HIP_PYTHON_ENUM_HALLUCINATE_MEMBERS=1 before running
"""

# [literalinclude-begin]
from cuda.cudart import cudaError_t

error_kinds = ( # some of those do not exist in HIP
    cudaError_t.cudaErrorInitializationError,
    cudaError_t.cudaErrorInsufficientDriver,
    cudaError_t.cudaErrorInvalidDeviceFunction,
    cudaError_t.cudaErrorInvalidDevice,
    cudaError_t.cudaErrorStartupFailure, # no HIP equivalent
    cudaError_t.cudaErrorInvalidKernelImage,
    cudaError_t.cudaErrorAlreadyAcquired,
    cudaError_t.cudaErrorOperatingSystem,
    cudaError_t.cudaErrorNotPermitted, # no HIP equivalent
    cudaError_t.cudaErrorNotSupported,
    cudaError_t.cudaErrorSystemNotReady, # no HIP equivalent
    cudaError_t.cudaErrorSystemDriverMismatch, # no HIP equivalent
    cudaError_t.cudaErrorCompatNotSupportedOnDevice, # no HIP equivalent
    cudaError_t.cudaErrorDeviceUninitialized,
    cudaError_t.cudaErrorTimeout, # no HIP equivalent
    cudaError_t.cudaErrorUnknown,
    cudaError_t.cudaErrorApiFailureBase, # no HIP equivalent
)

for err in error_kinds:
    assert isinstance(err,cudaError_t)
    assert (err != cudaError_t.cudaSuccess)
print("ok")
