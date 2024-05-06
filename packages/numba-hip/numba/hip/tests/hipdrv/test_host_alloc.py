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

import numpy as np
from numba.hip.hipdrv import driver
from numba import hip as cuda
from numba.hip.testing import unittest, ContextResettingTestCase


class TestHostAlloc(ContextResettingTestCase):

    @unittest.skip(
        "TODO(HIP/AMD) hipHostMalloc with CU_MEMHOSTALLOC_DEVICEMAP seems to not have any effect"
    )
    def test_host_alloc_driver(self):
        n = 32
        mem = cuda.current_context().memhostalloc(n, mapped=True)
        # via driver
        # mapped -> flags |= enums.CU_MEMHOSTALLOC_DEVICEMAP

        dtype = np.dtype(np.uint8)
        ary = np.ndarray(shape=n // dtype.itemsize, dtype=dtype, buffer=mem)

        magic = 0xAB
        driver.device_memset(mem, magic, n)
        # TODO(HIP/AMD) mapped=True implies that this should be done by the runtime
        self.assertTrue(np.all(ary == magic))

        ary.fill(n)

        recv = np.empty_like(ary)

        driver.device_to_host(recv, mem, ary.size)

        self.assertTrue(np.all(ary == recv))
        self.assertTrue(np.all(recv == n))

    @unittest.skip("TODO(HIP/AMD) memcpyD2H to pinned array seems not to have any effect")
    def test_host_alloc_pinned(self):
        ary = cuda.pinned_array(10, dtype=np.uint32)
        ary.fill(123)
        self.assertTrue(all(ary == 123))
        devary = cuda.to_device(ary)
        driver.device_memset(devary, 0, driver.device_memory_size(devary))
        self.assertTrue(all(ary == 123))
        devary.copy_to_host(ary)
        self.assertTrue(all(ary == 0))

    @unittest.skip(
        "TODO(HIP/AMD) hipHostMalloc with CU_MEMHOSTALLOC_DEVICEMAP seems to not have any effect"
    )
    def test_host_alloc_mapped(self):
        ary = cuda.mapped_array(10, dtype=np.uint32)
        ary.fill(123)
        self.assertTrue(all(ary == 123))
        driver.device_memset(ary, 0, driver.device_memory_size(ary))
        self.assertTrue(all(ary == 0))
        self.assertTrue(sum(ary != 0) == 0)

    def test_host_operators(self):
        for ary in [
            cuda.mapped_array(10, dtype=np.uint32),
            cuda.pinned_array(10, dtype=np.uint32),
        ]:
            ary[:] = range(10)
            self.assertTrue(sum(ary + 1) == 55)
            self.assertTrue(sum((ary + 1) * 2 - 1) == 100)
            self.assertTrue(sum(ary < 5) == 5)
            self.assertTrue(sum(ary <= 5) == 6)
            self.assertTrue(sum(ary > 6) == 3)
            self.assertTrue(sum(ary >= 6) == 4)
            self.assertTrue(sum(ary**2) == 285)
            self.assertTrue(sum(ary // 2) == 20)
            self.assertTrue(sum(ary / 2.0) == 22.5)
            self.assertTrue(sum(ary % 2) == 5)


if __name__ == "__main__":
    unittest.main()
