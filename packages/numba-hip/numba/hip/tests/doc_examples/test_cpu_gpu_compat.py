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

# unchanged original unit Numba CUDA test code below:

import unittest

from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout
import numpy as np


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestCpuGpuCompat(CUDATestCase):
    """
    Test compatibility of CPU and GPU functions
    """

    def setUp(self):
        # Prevent output from this test showing up when running the test suite
        self._captured_stdout = captured_stdout()
        self._captured_stdout.__enter__()
        super().setUp()

    def tearDown(self):
        # No exception type, value, or traceback
        self._captured_stdout.__exit__(None, None, None)
        super().tearDown()

    def test_ex_cpu_gpu_compat(self):
        # ex_cpu_gpu_compat.import.begin
        from math import pi

        import numba
        from numba import cuda
        # ex_cpu_gpu_compat.import.end

        # ex_cpu_gpu_compat.allocate.begin
        X = cuda.to_device([1, 10, 234])
        Y = cuda.to_device([2, 2, 4014])
        Z = cuda.to_device([3, 14, 2211])
        results = cuda.to_device([0.0, 0.0, 0.0])
        # ex_cpu_gpu_compat.allocate.end

        # ex_cpu_gpu_compat.define.begin
        @numba.jit
        def business_logic(x, y, z):
            return 4 * z * (2 * x - (4 * y) / 2 * pi)
        # ex_cpu_gpu_compat.define.end

        # ex_cpu_gpu_compat.cpurun.begin
        print(business_logic(1, 2, 3))  # -126.79644737231007
        # ex_cpu_gpu_compat.cpurun.end

        # ex_cpu_gpu_compat.usegpu.begin
        @cuda.jit
        def f(res, xarr, yarr, zarr):
            tid = cuda.grid(1)
            if tid < len(xarr):
                # The function decorated with numba.jit may be directly reused
                res[tid] = business_logic(xarr[tid], yarr[tid], zarr[tid])
        # ex_cpu_gpu_compat.usegpu.end

        # ex_cpu_gpu_compat.launch.begin
        f.forall(len(X))(results, X, Y, Z)
        print(results)
        # [-126.79644737231007, 416.28324559588634, -218912930.2987788]
        # ex_cpu_gpu_compat.launch.end

        expect = [
            business_logic(x, y, z) for x, y, z in zip(X, Y, Z)
        ]

        np.testing.assert_equal(
            expect,
            results.copy_to_host()
        )


if __name__ == "__main__":
    unittest.main()
