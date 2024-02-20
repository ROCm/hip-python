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

import numpy as np
from numba import hip as cuda
from numba.hip.testing import unittest, HIPTestCase as CUDATestCase, skip_on_hipsim as skip_on_cudasim


class TestArrayAttr(CUDATestCase):

    def test_contigous_2d(self):
        ary = np.arange(10)
        cary = ary.reshape(2, 5)
        fary = np.asfortranarray(cary)

        dcary = cuda.to_device(cary)
        dfary = cuda.to_device(fary)
        self.assertTrue(dcary.is_c_contiguous())
        self.assertTrue(not dfary.is_c_contiguous())
        self.assertTrue(not dcary.is_f_contiguous())
        self.assertTrue(dfary.is_f_contiguous())

    def test_contigous_3d(self):
        ary = np.arange(20)
        cary = ary.reshape(2, 5, 2)
        fary = np.asfortranarray(cary)

        dcary = cuda.to_device(cary)
        dfary = cuda.to_device(fary)
        self.assertTrue(dcary.is_c_contiguous())
        self.assertTrue(not dfary.is_c_contiguous())
        self.assertTrue(not dcary.is_f_contiguous())
        self.assertTrue(dfary.is_f_contiguous())

    def test_contigous_4d(self):
        ary = np.arange(60)
        cary = ary.reshape(2, 5, 2, 3)
        fary = np.asfortranarray(cary)

        dcary = cuda.to_device(cary)
        dfary = cuda.to_device(fary)
        self.assertTrue(dcary.is_c_contiguous())
        self.assertTrue(not dfary.is_c_contiguous())
        self.assertTrue(not dcary.is_f_contiguous())
        self.assertTrue(dfary.is_f_contiguous())

    def test_ravel_1d(self):
        ary = np.arange(60)
        dary = cuda.to_device(ary)
        for order in 'CFA':
            expect = ary.ravel(order=order)
            dflat = dary.ravel(order=order)
            flat = dflat.copy_to_host()
            self.assertTrue(dary is not dflat)  # ravel returns new array
            self.assertEqual(flat.ndim, 1)
            self.assertPreciseEqual(expect, flat)

    @skip_on_cudasim('CUDA Array Interface is not supported in the simulator')
    def test_ravel_stride_1d(self):
        ary = np.arange(60)
        dary = cuda.to_device(ary)
        # No-copy stride device array
        darystride = dary[::2]
        dary_data = dary.__cuda_array_interface__['data'][0]
        ddarystride_data = darystride.__cuda_array_interface__['data'][0]
        self.assertEqual(dary_data, ddarystride_data)
        # Fail on ravel on non-contiguous array
        with self.assertRaises(NotImplementedError):
            darystride.ravel()

    def test_ravel_c(self):
        ary = np.arange(60)
        reshaped = ary.reshape(2, 5, 2, 3)

        expect = reshaped.ravel(order='C')
        dary = cuda.to_device(reshaped)
        dflat = dary.ravel()
        flat = dflat.copy_to_host()
        self.assertTrue(dary is not dflat)
        self.assertEqual(flat.ndim, 1)
        self.assertPreciseEqual(expect, flat)

        # explicit order kwarg
        for order in 'CA':
            expect = reshaped.ravel(order=order)
            dary = cuda.to_device(reshaped)
            dflat = dary.ravel(order=order)
            flat = dflat.copy_to_host()
            self.assertTrue(dary is not dflat)
            self.assertEqual(flat.ndim, 1)
            self.assertPreciseEqual(expect, flat)

    @skip_on_cudasim('CUDA Array Interface is not supported in the simulator')
    def test_ravel_stride_c(self):
        ary = np.arange(60)
        reshaped = ary.reshape(2, 5, 2, 3)

        dary = cuda.to_device(reshaped)
        darystride = dary[::2, ::2, ::2, ::2]
        dary_data = dary.__cuda_array_interface__['data'][0]
        ddarystride_data = darystride.__cuda_array_interface__['data'][0]
        self.assertEqual(dary_data, ddarystride_data)
        with self.assertRaises(NotImplementedError):
            darystride.ravel()

    def test_ravel_f(self):
        ary = np.arange(60)
        reshaped = np.asfortranarray(ary.reshape(2, 5, 2, 3))
        for order in 'FA':
            expect = reshaped.ravel(order=order)
            dary = cuda.to_device(reshaped)
            dflat = dary.ravel(order=order)
            flat = dflat.copy_to_host()
            self.assertTrue(dary is not dflat)
            self.assertEqual(flat.ndim, 1)
            self.assertPreciseEqual(expect, flat)

    @skip_on_cudasim('CUDA Array Interface is not supported in the simulator')
    def test_ravel_stride_f(self):
        ary = np.arange(60)
        reshaped = np.asfortranarray(ary.reshape(2, 5, 2, 3))
        dary = cuda.to_device(reshaped)
        darystride = dary[::2, ::2, ::2, ::2]
        dary_data = dary.__cuda_array_interface__['data'][0]
        ddarystride_data = darystride.__cuda_array_interface__['data'][0]
        self.assertEqual(dary_data, ddarystride_data)
        with self.assertRaises(NotImplementedError):
            darystride.ravel()

    def test_reshape_c(self):
        ary = np.arange(10)
        expect = ary.reshape(2, 5)
        dary = cuda.to_device(ary)
        dary_reshaped = dary.reshape(2, 5)
        got = dary_reshaped.copy_to_host()
        self.assertPreciseEqual(expect, got)

    def test_reshape_f(self):
        ary = np.arange(10)
        expect = ary.reshape(2, 5, order='F')
        dary = cuda.to_device(ary)
        dary_reshaped = dary.reshape(2, 5, order='F')
        got = dary_reshaped.copy_to_host()
        self.assertPreciseEqual(expect, got)


if __name__ == '__main__':
    unittest.main()
