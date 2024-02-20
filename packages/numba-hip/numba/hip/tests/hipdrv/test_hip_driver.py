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

import textwrap

from ctypes import byref, c_int, c_void_p, sizeof

from hip import hip as _hip, hiprtc as _hiprtc # via 'hip-python'

from numba.hip.hipdrv.driver import (host_to_device, device_to_host, driver,
                                       launch_kernel)
from numba.hip.hipdrv import devices, driver as _driver
from numba.hip.testing import unittest, HIPTestCase
# from numba.hip.testing import skip_on_cudasim # TODO HIP enable simulator

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, _hip.hipError_t) and err != _hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, _hiprtc.hiprtcResult)
        and err != _hiprtc.hiprtcResult.HIPRTC_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result

kernel_hip = textwrap.dedent("""\
extern "C" __global__ void set_thread_idx(int* arr) {
  arr[threadIdx.x] = threadIdx.x;
}
""").encode("utf-8")

class HipProgram:
    def __init__(self, name: str, source: bytes):
        self.source = source
        self.name = name.encode("utf-8")
        self.prog = None
        self.code = None
        self.code_size = None

    def compile(self,amdgpu_arch: str):
        self.prog = hip_check(
            _hiprtc.hiprtcCreateProgram(self.source, self.name, 0, [], [])
        )
        cflags = [b"--offload-arch=" + amdgpu_arch.encode("utf-8")]
        (err,) = _hiprtc.hiprtcCompileProgram(self.prog, len(cflags), cflags)
        if err != _hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(_hiprtc.hiprtcGetProgramLogSize(self.prog))
            log = bytearray(log_size)
            hip_check(_hiprtc.hiprtcGetProgramLog(self.prog, log))
            raise RuntimeError(log.decode())
        self.code_size = hip_check(_hiprtc.hiprtcGetCodeSize(self.prog))
        self.code = bytearray(self.code_size)
        hip_check(_hiprtc.hiprtcGetCode(self.prog, self.code))
        return self.code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.prog != None:
            hip_check(_hiprtc.hiprtcDestroyProgram(self.prog.createRef()))

# @skip_on_cudasim('CUDA Driver API unsupported in the simulator')
class TestCudaDriver(HIPTestCase):
    def setUp(self):
        super().setUp()
        self.assertTrue(len(devices.gpus) > 0)
        self.context = devices.get_context()
        device: _driver.Device = self.context.device
        with HipProgram("kernel.hip",kernel_hip) as prog:
            self.amdgpu_codeobj = prog.compile(device.amdgpu_arch)

    def tearDown(self):
        super().tearDown()
        del self.context

    def test_cuda_driver_basic(self):
        # TODO do something similar wih HIP

        module = self.context.create_module_from_codeobj(self.amdgpu_codeobj)
        function = module.get_function('set_thread_idx')

        array = (c_int * 100)()

        memory = self.context.memalloc(sizeof(array))
        host_to_device(memory, array, sizeof(array))

        ptr = memory.device_ctypes_pointer
        stream = 0

        if _driver.USE_NV_BINDING:
            ptr = c_void_p(int(ptr))
            stream = _driver.binding.CUstream(stream)

        launch_kernel(function.handle,  # Kernel
                      1,   1, 1,        # gx, gy, gz
                      100, 1, 1,        # bx, by, bz
                      0,                # dynamic shared mem
                      stream,           # stream
                      [ptr])            # arguments

        device_to_host(array, memory, sizeof(array))
        for i, v in enumerate(array):
            self.assertEqual(i, v)

        module.unload()

    def test_cuda_driver_stream_operations(self):
        module = self.context.create_module_from_codeobj(self.amdgpu_codeobj)
        function = module.get_function("set_thread_idx")

        array = (c_int * 100)()

        stream = self.context.create_stream()

        with stream.auto_synchronize():
            memory = self.context.memalloc(sizeof(array))
            host_to_device(memory, array, sizeof(array), stream=stream)

            ptr = memory.device_ctypes_pointer
            if _driver.USE_NV_BINDING:
                ptr = c_void_p(int(ptr))

            launch_kernel(function.handle,  # Kernel
                          1,   1, 1,        # gx, gy, gz
                          100, 1, 1,        # bx, by, bz
                          0,                # dynamic shared mem
                          stream.handle,    # stream
                          [ptr])            # arguments

        device_to_host(array, memory, sizeof(array), stream=stream)

        for i, v in enumerate(array):
            self.assertEqual(i, v)

    def test_cuda_driver_default_stream(self):
        # Test properties of the default stream
        ds = self.context.get_default_stream()
        self.assertIn("Default HIP stream", repr(ds))
        self.assertEqual(0, int(ds))
        # bool(stream) is the check that is done in memcpy to decide if async
        # version should be used. So the default (0) stream should be true-ish
        # even though 0 is usually false-ish in Python.
        self.assertTrue(ds)
        self.assertFalse(ds.external)

    def test_cuda_driver_legacy_default_stream(self):
        # Test properties of the legacy default stream
        ds = self.context.get_legacy_default_stream()
        # self.assertIn("Legacy default HIP stream", repr(ds))
        # self.assertEqual(1, int(ds))
        self.assertIn("Default HIP stream", repr(ds))
        self.assertEqual(0, int(ds))
        self.assertTrue(ds)
        self.assertFalse(ds.external)

    def test_cuda_driver_per_thread_default_stream(self):
        # Test properties of the per-thread default stream
        ds = self.context.get_per_thread_default_stream()
        self.assertIn("Per-thread default HIP stream", repr(ds))
        self.assertEqual(2, int(ds))
        self.assertTrue(ds)
        self.assertFalse(ds.external)

    def test_cuda_driver_stream(self):
        # Test properties of non-default streams
        s = self.context.create_stream()
        self.assertIn("HIP stream", repr(s))
        self.assertNotIn("Default", repr(s))
        self.assertNotIn("External", repr(s))
        self.assertNotEqual(0, int(s))
        self.assertTrue(s)
        self.assertFalse(s.external)

    def test_cuda_driver_external_stream(self):
        # Test properties of a stream created from an external stream object.
        # We use the driver API directly to create a stream, to emulate an
        # external library creating a stream
        if _driver.USE_NV_BINDING:
            handle = driver.cuStreamCreate(0)
            ptr = int(handle)
        else:
            raise NotImplementedError()
        s = self.context.create_external_stream(ptr)

        self.assertIn("External HIP stream", repr(s))
        # Ensure neither "Default" nor "default"
        self.assertNotIn("efault", repr(s))
        self.assertEqual(ptr, int(s))
        self.assertTrue(s)
        self.assertTrue(s.external)

    def test_cuda_driver_occupancy(self):
        module = self.context.create_module_from_codeobj(self.amdgpu_codeobj)
        function = module.get_function('set_thread_idx')

        value = self.context.get_active_blocks_per_multiprocessor(function,
                                                                  128, 128)
        self.assertTrue(value > 0)

        def b2d(bs): # is ignored
            return bs

        grid, block = self.context.get_max_potential_block_size(function, b2d,
                                                                128, 128)
        self.assertTrue(grid > 0)
        self.assertTrue(block > 0)


class TestDevice(HIPTestCase):
    def test_device_get_uuid(self):
        # A device UUID looks like:
        #
        #     GPU-e6489c45-5b68-3b03-bab7-0e7c8e809643
        #
        # To test, we construct an RE that matches this form and verify that
        # the returned UUID matches.
        #
        # Device UUIDs may not conform to parts of the UUID specification (RFC
        # 4122) pertaining to versions and variants, so we do not extract and
        # validate the values of these bits.

        h = '[0-9a-f]{%d}'
        h4 = h % 4
        h8 = h % 8
        h12 = h % 12
        uuid_format = f'^GPU-{h8}-{h4}-{h4}-{h4}-{h12}$'

        dev: _driver.Device = devices.get_context().device
        self.assertRegex(dev.uuid, uuid_format)


if __name__ == '__main__':
    unittest.main()
