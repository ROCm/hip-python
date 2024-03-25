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

# Contents in this file are referenced from the sphinx-generated docs.
# "magictoken" is used for markers as beginning and ending of example text.

import unittest
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim)
from numba.tests.support import skip_unless_cffi


@skip_unless_cffi
@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestFFI(CUDATestCase):
    def test_ex_linking_cu(self):
        # magictoken.ex_linking_cu.begin
        from numba import cuda
        import numpy as np
        import os

        # Declaration of the foreign function
        mul = cuda.declare_device('mul_f32_f32', 'float32(float32, float32)')

        # Path to the source containing the foreign function
        # (here assumed to be in a subdirectory called "ffi")
        basedir = os.path.dirname(os.path.abspath(__file__))
        functions_cu = os.path.join(basedir, 'ffi', 'functions.cu')

        # Kernel that links in functions.cu and calls mul
        @cuda.jit(link=[functions_cu])
        def multiply_vectors(r, x, y):
            i = cuda.grid(1)

            if i < len(r):
                r[i] = mul(x[i], y[i])

        # Generate random data
        N = 32
        np.random.seed(1)
        x = np.random.rand(N).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        r = np.zeros_like(x)

        # Run the kernel
        multiply_vectors[1, 32](r, x, y)

        # Sanity check - ensure the results match those expected
        np.testing.assert_array_equal(r, x * y)
        # magictoken.ex_linking_cu.end

    def test_ex_from_buffer(self):
        from numba import cuda
        import os

        basedir = os.path.dirname(os.path.abspath(__file__))
        functions_cu = os.path.join(basedir, 'ffi', 'functions.cu')

        # magictoken.ex_from_buffer_decl.begin
        signature = 'float32(CPointer(float32), int32)'
        sum_reduce = cuda.declare_device('sum_reduce', signature)
        # magictoken.ex_from_buffer_decl.end

        # magictoken.ex_from_buffer_kernel.begin
        import cffi
        ffi = cffi.FFI()

        @cuda.jit(link=[functions_cu])
        def reduction_caller(result, array):
            array_ptr = ffi.from_buffer(array)
            result[()] = sum_reduce(array_ptr, len(array))
        # magictoken.ex_from_buffer_kernel.end

        import numpy as np
        x = np.arange(10).astype(np.float32)
        r = np.ndarray((), dtype=np.float32)

        reduction_caller[1, 1](r, x)

        expected = np.sum(x)
        actual = r[()]
        np.testing.assert_allclose(expected, actual)


if __name__ == '__main__':
    unittest.main()
