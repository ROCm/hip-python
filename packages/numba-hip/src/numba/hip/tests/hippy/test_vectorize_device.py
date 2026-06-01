# MIT License
#
# Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

from numba import hip

hip.pose_as_cuda()

# unchanged original unit Numba CUDA test code below:

from numba import vectorize
from numba import cuda, float32
import numpy as np
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest


@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestCudaVectorizeDeviceCall(CUDATestCase):
    def test_cuda_vectorize_device_call(self):

        @cuda.jit(float32(float32, float32, float32), device=True)
        def cu_device_fn(x, y, z):
            return x ** y / z

        def cu_ufunc(x, y, z):
            return cu_device_fn(x, y, z)

        ufunc = vectorize([float32(float32, float32, float32)], target='cuda')(
            cu_ufunc)

        N = 100

        X = np.array(np.random.sample(N), dtype=np.float32)
        Y = np.array(np.random.sample(N), dtype=np.float32)
        Z = np.array(np.random.sample(N), dtype=np.float32) + 0.1

        out = ufunc(X, Y, Z)

        gold = (X ** Y) / Z

        self.assertTrue(np.allclose(out, gold))


if __name__ == '__main__':
    unittest.main()
