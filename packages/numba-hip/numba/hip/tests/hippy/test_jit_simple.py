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

"""Very simple HIP JIT compilation examples.
"""

import math
from numba import hip as cuda
from numba.hip.testing import unittest, HIPTestCase as CUDATestCase


class TestJitSimple(CUDATestCase):

    def test_compile_llvm_ir_for_empty_device_fun(self):

        def empty():
            pass

        for _ in range(0, 1):
            ir, restype = cuda.compile_llvm_ir_for_current_device(
                pyfunc=empty, sig=(), device=True, to_bc=False
            )
        self.assertIn("test_compile_llvm_ir_for_empty_device_fun", ir.decode("utf-8"))
        # with open("empty.ll","w") as outfile:
        #     outfile.write(ir.decode("utf-8"))

    def test_compile_llvm_ir_for_grid(self):

        def grid():
            x, y = cuda.grid(2)
            dim_x, dim_y = cuda.gridsize(2)
            cuda.syncthreads()
            cuda.get_threadIdx.x()
            ws = cuda.warpsize

        for _ in range(0, 2):
            ir, restype = cuda.compile_llvm_ir_for_current_device(
                pyfunc=grid, sig=(), device=True, to_bc=False
            )
        self.assertIn("grid", ir.decode("utf-8"))
        with open("grid.ll", "w") as outfile:
            outfile.write(ir.decode("utf-8"))

    def test_compile_llvm_ir_for_syncthreads(self):

        print(f"{id(cuda.syncthreads)=}")  # same id, same object

        # compile_llvm_ir_for_current_device

        def syncthreads():

            cuda.syncthreads()
            # syncthreads()

        for _ in range(0, 2):
            ir, restype = cuda.compile_llvm_ir_for_current_device(
                pyfunc=syncthreads,
                sig=(),
                device=True,
                to_bc=False,
                name="GENERIC_OP",
            )
        self.assertIn("GENERIC_OP", ir.decode("utf-8"))
        with open("syncthreads.ll", "w") as outfile:
            outfile.write(ir.decode("utf-8"))

    def test_jit_device_syncthreads(self):
        # jit - device function

        @cuda.jit(device=True)
        def syncthreads_jit():

            cuda.syncthreads()

        print(syncthreads_jit)

    def test_jit_kernel_syncthreads(self):
        # jit + run - kernel

        @cuda.jit(device=False)
        def syncthreads_kernel():

            cuda.syncthreads()

        threadsperblock = (16, 16)
        blockspergrid = (1, 1)
        syncthreads_kernel[blockspergrid, threadsperblock]()


if __name__ == "__main__":
    unittest.main()
