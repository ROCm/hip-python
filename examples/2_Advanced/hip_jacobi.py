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
import time
import math
import ctypes

import numpy as np

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

class GpuOffload:

    def __init__(self,source: bytes, kernel_names: list):
        self.source = source
        self.kernel_names = kernel_names

    def _get_arch(self) -> bytes:
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,0))
        return props.gcnArchName

    def compile_kernels(self):
        prog = hip_check(hiprtc.hiprtcCreateProgram(self.source, b"program", 0, [], []))
        cflags = [b"--offload-arch="+self._get_arch()]
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
        for kernel_name in self.kernel_names:
            setattr(self,"_"+kernel_name.decode(),hip_check(hip.hipModuleGetFunction(module, kernel_name)))
        
    def launch_kernel(self,kernel_name,grid,block,sharedmem=0,stream=None):
        kernel = getattr(self,"_"+kernel_name)

        def parametrized_launcher(*args):
            return hip.hipModuleLaunchKernel(
                kernel,
                *grid,
                *block,
                sharedMemBytes=sharedmem,
                stream=stream,
                kernelParams=None,
                extra=args
            )
        return parametrized_launcher

def init_cpu(n: int):
    """Generate random matix operator and righthand-side.
    """
    np.random.seed(0)
    A_h = np.random.rand(n,n).astype(dtype=np.float32) / n # in range [0,1]
    np.fill_diagonal(A_h, 1.0)
    b_h = np.random.rand(n).astype(dtype=np.float32) / n # in range [0,1]
    x_h = np.zeros(shape=(n,),dtype=np.float32)
    return (A_h, b_h, x_h)

def jacobi_cpu(x,A,b,n: int,iter: int):
    D = np.diag(A.diagonal())
    D_inv = np.diag(np.reciprocal(A.diagonal()))
    R = A - D
    x_next = np.zeros(shape=(n,),dtype=np.float32)
    for _ in range(0,iter):
        x_next = np.dot(D_inv,(b - np.dot(R,x))) # np.dot does matvec
        x, x_next = x_next, x
    return x

HIP_SOURCE = rb"""
  // GPU Jacobi (Unoptimized)
  extern "C" __global__ void jacobi_gpu_unoptimized(float* x, float* A, float* b, int n)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // every global thread handles one row

    float sigma = 0.0;
    for (int j=0; j<n; j++)
    {
      if (idx != j)
        sigma += A[idx*n + j] * x[j];
    }

    // Access on globale memory
    x[idx] = (b[idx] - sigma) / A[idx*n + idx];
    //printf("x[%d]=%f\n", idx,x[idx]);
  }

  // GPU Jacobi Optimized Version
  extern "C" __global__ 
  //__attribute__((amdgpu_flat_work_group_size(DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE)))
  //__attribute__((amdgpu_waves_per_eu(4)))
  //__launch_bounds__(DEFAULT_BLOCK_SIZE, 3)
  void jacobi_gpu(float* x, float* A, float* b, int n)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sigma = 0.0;
    const int s_align = 64;

    for (int i=0; i<n; i+=s_align){
      float tmp[s_align];

      for(int j=0; j<s_align; j++)
        tmp[j] =  A[idx*n + i+j];

      for(int j=0; j<s_align; j++){
      if (idx != i+j)
        sigma += tmp[j] * x[i+j];
      }
    }

    // Access on global memory
    x[idx] = (b[idx] - sigma) / A[idx*n + idx];
    //printf("x[%d]=%f\n", idx,x[idx]);
  }
  """

if __name__ in ("__test__","__main__"):
    DEBUG = False
    DEFAULT_BLOCK_SIZE = 64 # must be divisor of n!
    KERNEL_NAME = "jacobi_gpu" # "jacobi_gpu", "jacobi_gpu_unoptimized"

    n, iter = 512, 1000
    assert (n % DEFAULT_BLOCK_SIZE == 0)

    A_h, b_h, x_h = init_cpu(n)

    if DEBUG:
        print(f"{A_h=}")
        print(f"{b_h=}")

    # Run CPU version
    time_cpu = time.time()
    x_h = jacobi_cpu(x_h, A_h, b_h, n, iter)
    time_cpu = time.time() - time_cpu
    if DEBUG:
        print(f"{x_h=}")

    # Run GPU version
    x_h[:] = 0 # reset

    print(f"\nJacobi solver: {n=}, {iter=}")
    print(f"Device kernel: '{KERNEL_NAME}', block_size={DEFAULT_BLOCK_SIZE}\n")
    print("------------------------------------------------------")
    print(f"Host residum (||A*x_h-b||_2) : {np.linalg.norm(A_h*x_h-b_h,2):.9f}")
    print(f"Host compute time            : {time_cpu:.3f} sec")
    print("------------------------------------------------------")

    A_d = hip_check(hip.hipMalloc(A_h.size*A_h.itemsize))
    x_d = hip_check(hip.hipMalloc(x_h.size*x_h.itemsize))
    b_d = hip_check(hip.hipMalloc(b_h.size*b_h.itemsize))

    time_h2d = time.time()
    hip_check(hip.hipMemcpy(A_d, A_h, A_h.size*A_h.itemsize, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(x_d, x_h, x_h.size*x_h.itemsize, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(b_d, b_h, b_h.size*b_h.itemsize, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    time_h2d = time.time() - time_h2d

    gpu_offload = GpuOffload(source=HIP_SOURCE,kernel_names=(b"jacobi_gpu",b"jacobi_gpu_unoptimized"))
    time_compile = time.time()
    gpu_offload.compile_kernels()
    time_compile = time.time() - time_compile

    block = hip.dim3(DEFAULT_BLOCK_SIZE)
    grid  = hip.dim3(n/block[0])

    tstart = hip_check(hip.hipEventCreate())
    tstop = hip_check(hip.hipEventCreate())

    hip_check(hip.hipEventRecord(tstart, None))

    for i in range(0,iter):
        hip_check(gpu_offload.launch_kernel(KERNEL_NAME, grid, block)(x_d, A_d, b_d, ctypes.c_int(n)))
    
    hip_check(hip.hipEventRecord(tstop, None))
    hip_check(hip.hipEventSynchronize(tstop))
    time_compute = hip_check(hip.hipEventElapsedTime(tstart, tstop))/10**3

    # Copy data back to host, reuse x_h
    time_d2h = time.time()
    hip_check(hip.hipMemcpy(x_h, int(x_d), x_d.size*x_d.itemsize, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    time_d2h = time.time() - time_d2h

    print("\n------------------------------------------------------")
    print(f"Device residum (||A*x_d-b||_2) : {np.linalg.norm(A_h*x_h-b_h,2):.9f}")
    print(f"Device total time              : {time_compute+time_compile+time_h2d+time_d2h:.3f} sec")
    print(f"  > Device compute time        : {time_compute:.3f} sec")
    print(f"  > Copy to device             : {time_h2d:.3f} sec")
    print(f"  > Copy from device           : {time_d2h:.3f} sec")
    print(f"  > Device kernel compilation  : {time_compile:.3f} sec")
    print("------------------------------------------------------")
    # Clean up
    hip_check(hip.hipEventDestroy(tstart))
    hip_check(hip.hipEventDestroy(tstop))

    hip_check(hip.hipFree(A_d))
    hip_check(hip.hipFree(x_d))
    hip_check(hip.hipFree(b_d))
    print("ok")