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
from numba import vectorize
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest


@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestVectorizeComplex(CUDATestCase):
    def test_vectorize_complex(self):
        @vectorize(['complex128(complex128)'], target='cuda')
        def vcomp(a):
            return a * a + 1.

        A = np.arange(5, dtype=np.complex128)
        B = vcomp(A)
        self.assertTrue(np.allclose(A * A + 1., B))


if __name__ == '__main__':
    unittest.main()
