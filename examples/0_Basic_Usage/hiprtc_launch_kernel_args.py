# MIT License
# 
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# [literalinclude-begin]
import ctypes
import array
import random
import math

from hip import hip, hiprtc

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hiprtc.hiprtcResult)
        and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result

source = b"""\
extern "C" __global__ void scale_vector(float factor, int n, short unused1, int unused2, float unused3, float *x) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid == 0 ) {
    printf("tid: %d, factor: %f, x*: %lu, n: %lu, unused1: %d, unused2: %d, unused3: %f\\n",tid,factor,x,n,(int) unused1,unused2,unused3);
  }
  if (tid < n) {
     x[tid] *= factor;
  }
}
"""

prog = hip_check(hiprtc.hiprtcCreateProgram(source, b"scale_vector", 0, [], []))

props = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(props,0))
arch = props.gcnArchName

print(f"Compiling kernel for {arch}")

cflags = [b"--offload-arch="+arch]
err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
    log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
    log = bytearray(log_size)
    hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
    raise RuntimeError(log.decode())
code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
code = bytearray(code_size)
hip_check(hiprtc.hiprtcGetCode(prog, code))
module = hip_check(hip.hipModuleLoadData(code))
kernel = hip_check(hip.hipModuleGetFunction(module, b"scale_vector"))

# kernel launch

## inputs
n = 100
x_h = array.array("f",[random.random() for i in range(0,n)])
num_bytes = x_h.itemsize * len(x_h)
x_d = hip_check(hip.hipMalloc(num_bytes))
print(f"{hex(int(x_d))=}")

## upload host data
hip_check(hip.hipMemcpy(x_d,x_h,num_bytes,hip.hipMemcpyKind.hipMemcpyHostToDevice))

factor = 1.23

## expected result
x_expected = [a*factor for a in x_h]

block = hip.dim3(x=32)
grid = hip.dim3(math.ceil(n/block.x))

## launch
hip_check(
    hip.hipModuleLaunchKernel(
        kernel,
        *grid,
        *block,
        sharedMemBytes=0,
        stream=None,
        kernelParams=None,
        extra=( 
          ctypes.c_float(factor), # 4 bytes
          ctypes.c_int(n),  # 8 bytes
          ctypes.c_short(5), # unused1, 10 bytes
          ctypes.c_int(2), # unused2, 16 bytes (+2 padding bytes)
          ctypes.c_float(5.6), # unused3 20 bytes
          x_d, # 32 bytes (+4 padding bytes)
        )
    )
)

# copy result back
hip_check(hip.hipMemcpy(x_h,x_d,num_bytes,hip.hipMemcpyKind.hipMemcpyDeviceToHost))

for i,x_h_i in enumerate(x_h):
    if not math.isclose(x_h_i,x_expected[i],rel_tol=1e-6):
        raise RuntimeError(f"values do not match, {x_h[i]=} vs. {x_expected[i]=}, {i=}")

hip_check(hip.hipFree(x_d))

hip_check(hip.hipModuleUnload(module))
hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))

print("ok")
