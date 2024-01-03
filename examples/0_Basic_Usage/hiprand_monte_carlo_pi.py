# MIT License
# 
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

"""
This example uses hiprand to estimate Pi by means of the Monte-Carlo method.
    
The unit square has the area 1^2, while the unit circle has the area
pi*(1/2)^2. Therefore, the ratio of the areas is pi/4.

Using the Monte-Carlo method, we randomly choose N (x,y)-coordinates in the unit square.
We then estimate the ratio of areas as the ratio between those samples
within the unit circle and N.
The accuracy of the approach increases with N.

Note:
    This example was originally taken from the ROCRAND repository on Github.
    See this repository for a "more pythonic" object-oriented interface to hiprand/rocrand (ctypes-based, Python-only).
"""

# [literalinclude-begin]
from hip import hip, hiprand
import numpy as np
import math

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hiprand.hiprandStatus) and err != hiprand.hiprandStatus.HIPRAND_STATUS_SUCCESS:
        raise RuntimeError(str(err))
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

print("Estimating Pi via the Monte Carlo method:\n")

def calculate_pi(n):
    """Calculate Pi for the given number of samples.
    """
    xy = np.empty(shape=(2, n)) # host array, default type is float64
    gen = hip_check(hiprand.hiprandCreateGenerator(hiprand.hiprandRngType.HIPRAND_RNG_PSEUDO_DEFAULT))
    xy_d = hip_check(hip.hipMalloc(xy.size*xy.itemsize)) # create same size device array
    hip_check(hiprand.hiprandGenerateUniformDouble(gen,xy_d,xy.size)) # generate device random numbers
    hip_check(hip.hipMemcpy(xy,xy_d,xy.size*xy.itemsize,hip.hipMemcpyKind.hipMemcpyDeviceToHost)) # copy to host
    hip_check(hip.hipFree(xy_d)) # free device array
    hip_check(hiprand.hiprandDestroyGenerator(gen))

    inside = xy[0]**2 + xy[1]**2 <= 1.0
    in_xy  = xy[:,  inside]
    estimate = 4*in_xy[0,:].size/n
    return estimate

print(f"#samples\testimate\trelative error")
n = 100
imax = 5
for i in range(1,imax):
    n *= 10
    estimate = calculate_pi(n)
    print(f"{n:12}\t{estimate:1.9f}\t{abs(estimate-math.pi)/math.pi:1.9f}")
print("ok")