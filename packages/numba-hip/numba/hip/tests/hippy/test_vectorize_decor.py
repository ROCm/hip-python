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

import numpy as np

from numba import vectorize, cuda
from numba.tests.npyufunc.test_vectorize_decor import BaseVectorizeDecor, \
    BaseVectorizeNopythonArg, BaseVectorizeUnrecognizedArg
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest


@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestVectorizeDecor(CUDATestCase, BaseVectorizeDecor):
    """
    Runs the tests from BaseVectorizeDecor with the CUDA target.
    """
    target = 'cuda'


@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestGPUVectorizeBroadcast(CUDATestCase):
    def test_broadcast(self):
        a = np.random.randn(100, 3, 1)
        b = a.transpose(2, 1, 0)

        def fn(a, b):
            return a - b

        @vectorize(['float64(float64,float64)'], target='cuda')
        def fngpu(a, b):
            return a - b

        expect = fn(a, b)
        got = fngpu(a, b)
        np.testing.assert_almost_equal(expect, got)

    def test_device_broadcast(self):
        """
        Same test as .test_broadcast() but with device array as inputs
        """

        a = np.random.randn(100, 3, 1)
        b = a.transpose(2, 1, 0)

        def fn(a, b):
            return a - b

        @vectorize(['float64(float64,float64)'], target='cuda')
        def fngpu(a, b):
            return a - b

        expect = fn(a, b)
        got = fngpu(cuda.to_device(a), cuda.to_device(b))
        np.testing.assert_almost_equal(expect, got.copy_to_host())


@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestVectorizeNopythonArg(BaseVectorizeNopythonArg, CUDATestCase):
    def test_target_cuda_nopython(self):
        warnings = ["nopython kwarg for cuda target is redundant"]
        self._test_target_nopython('cuda', warnings)


@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestVectorizeUnrecognizedArg(BaseVectorizeUnrecognizedArg, CUDATestCase):
    def test_target_cuda_unrecognized_arg(self):
        self._test_target_unrecognized_arg('cuda')


if __name__ == '__main__':
    unittest.main()
