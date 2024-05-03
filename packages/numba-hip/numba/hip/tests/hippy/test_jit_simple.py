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
# Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

"""Very simple HIP JIT compilation examples.

Note:
    If you run this test via `pytest -v --durations=0 <test-file-name>.py`,
    the rest results will be listed together with the execution time 
    per test.
"""
import os

from numba import hip

hip.pose_as_cuda()
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase

DUMP_IR = bool(os.environ.get("NUMBA_HIP_TESTS_DUMP_IR", False))

# Disable low occupancy warnings for our simple kernels
from numba import config

config.CUDA_LOW_OCCUPANCY_WARNINGS = False

# Numba user code imports

import numpy as np
import math
from math import sin

runtimes = ""


class TestJitSimple(CUDATestCase):

    def test_00_compile_llvm_ir_for_empty_device_fun(self):

        def empty():
            pass

        ir, restype = cuda.compile_llvm_ir_for_current_device(
            pyfunc=empty, sig=(), device=True, to_bc=False
        )
        self.assertIn("test_00_compile_llvm_ir_for_empty_device_fun", ir)
        # with open("empty.ll","w") as outfile:
        #     outfile.write(ir)

    def test_01_compile_llvm_ir_for_syncthreads(self):

        # compile_llvm_ir_for_current_device

        def syncthreads():

            cuda.syncthreads()
            # syncthreads()

        ir, restype = cuda.compile_llvm_ir_for_current_device(
            pyfunc=syncthreads,
            sig=(),
            device=True,
            to_bc=False,
            name="GENERIC_OP",
        )
        self.assertIn("GENERIC_OP", ir)
        if DUMP_IR:
            with open("syncthreads.ll", "w") as outfile:
                outfile.write(ir)

    def test_03_jit_device_for_syncthreads(self):
        # jit - device function

        @cuda.jit(device=True)
        def syncthreads():

            cuda.syncthreads()

        self.assertIsInstance(syncthreads, cuda.dispatcher.HIPDispatcher)
        # print(syncthreads_jit)

    def test_04_jit_kernel_syncthreads(self):
        # jit + run - kernel

        @cuda.jit(device=False)
        def syncthreads():

            cuda.syncthreads()

        self.assertIsInstance(syncthreads, cuda.dispatcher.HIPDispatcher)

        threadsperblock = (16, 16)
        blockspergrid = (1, 1)
        syncthreads[blockspergrid, threadsperblock]()

    def test_05_compile_llvm_ir_for_one_of_each(self):

        def mydevicefun():
            x, y = cuda.grid(2)
            dim_x, dim_y = cuda.gridsize(2)
            cuda.syncthreads()
            ws = cuda.warpsize
            lane = cuda.laneid
            cuda.cos(5)
            cuda.cos(5.0)
            math.cos(5)
            sin(6)
            math.gamma(2)
            math.radians(2)
            math.degrees(2)
            x = cuda.threadIdx.x
            lA = cuda.local.array(shape=(4, 4), dtype=np.float32)
            sA = cuda.shared.array(shape=(4, 4), dtype=np.int64)
            lA[x] = 5

        ir, restype = cuda.compile_llvm_ir_for_current_device(
            pyfunc=mydevicefun, sig=(), device=True, to_bc=False, name="mydevicefun"
        )
        self.assertIn("mydevicefun", ir)
        if DUMP_IR:
            with open("one_of_each.ll", "w") as outfile:
                outfile.write(ir)

    def test_06_compile_llvm_ir_device_fun_with_args(self):
        def device_fun_with_args(arr, a):
            arr[cuda.threadIdx.x] *= a

        ir, restype = cuda.compile_llvm_ir_for_current_device(
            pyfunc=device_fun_with_args,
            sig="void(float64[:],float64)",
            device=True,
            to_bc=False,
            name="mydevicefun",  # TODO Fix this again
        )
        self.assertIn("device_fun_with_args", ir)
        if DUMP_IR:
            with open("device_fun_with_args.ll", "w") as outfile:
                outfile.write(ir)

    def test_07_reduction_like(self):
        import numpy as np
        from numba.types import int32

        a = cuda.to_device(np.arange(1024))
        nelem = len(a)

        def array_sum(data):
            tid = cuda.threadIdx.x  # TODO that's the issue
            # tid = cuda.get_threadIdx.x()
            size = len(data)
            i = cuda.grid(1)
            shr = cuda.shared.array(nelem, int32)
            # below lines must both present to create the error
            if tid < size:
                shr[tid] = data[i]  #: trouble maker

        ir, restype = cuda.compile_llvm_ir_for_current_device(
            pyfunc=array_sum,
            sig="void(float64[:])",
            device=True,
            to_bc=False,
            name="mydevicefun",
        )


if __name__ == "__main__":
    unittest.main()
