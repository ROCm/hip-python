# Copyright (c) 2012, Anaconda, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# MIT License
#
# Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import unittest

from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout

@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestVecAdd(CUDATestCase):
    """
    Test simple vector addition
    """

    def setUp(self):
        # Prevent output from this test showing
        # up when running the test suite
        self._captured_stdout = captured_stdout()
        self._captured_stdout.__enter__()
        super().setUp()

    def tearDown(self):
        # No exception type, value, or traceback
        self._captured_stdout.__exit__(None, None, None)
        super().tearDown()

    def test_ex_vecadd(self):
        # ex_vecadd.import.begin
        import numpy as np
        from numba import cuda
        # ex_vecadd.import.end

        # ex_vecadd.kernel.begin
        @cuda.jit
        def f(a, b, c):
            # like threadIdx.x + (blockIdx.x * blockDim.x)
            tid = cuda.grid(1)
            size = len(c)

            if tid < size:
                c[tid] = a[tid] + b[tid]
        # ex_vecadd.kernel.end

        # Seed RNG for test repeatability
        np.random.seed(1)

        # ex_vecadd.allocate.begin
        N = 100000
        a = cuda.to_device(np.random.random(N))
        b = cuda.to_device(np.random.random(N))
        c = cuda.device_array_like(a)
        # ex_vecadd.allocate.end

        # ex_vecadd.forall.begin
        f.forall(len(a))(a, b, c)
        print(c.copy_to_host())
        # ex_vecadd.forall.end

        # ex_vecadd.launch.begin
        # Enough threads per block for several warps per block
        nthreads = 256
        # Enough blocks to cover the entire vector depending on its length
        nblocks = (len(a) // nthreads) + 1
        f[nblocks, nthreads](a, b, c)
        print(c.copy_to_host())
        # ex_vecadd.launch.end

        np.testing.assert_equal(
            c.copy_to_host(),
            a.copy_to_host() + b.copy_to_host()
        )


if __name__ == "__main__":
    unittest.main()
