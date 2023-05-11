# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import ctypes
import random
import array

from cuda import cuda

def cuda_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, cuda.cudaError_t) and err != cuda.cudaError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

# inputs
n = 100
x_h = array.array("i",[int(random.random()*10) for i in range(0,n)])
num_bytes = x_h.itemsize * len(x_h)
x_d = cuda_check(cuda.cudaMalloc(num_bytes))

stream = cuda_check(cuda.cudaStreamCreate())
cuda_check(cuda.cudaMemcpyAsync(x_d,x_h,num_bytes,cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,stream))
cuda_check(cuda.cudaMemsetAsync(x_d,0,num_bytes,stream))
cuda_check(cuda.cudaMemcpyAsync(x_h,x_d,num_bytes,cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,stream))
cuda_check(cuda.cudaStreamSynchronize(stream))
cuda_check(cuda.cudaStreamDestroy(stream))

# deallocate device data 
cuda_check(cuda.cudaFree(x_d))

for i,x in enumerate(x_h):
    if x != 0:
        raise ValueError(f"expected '0' for element {i}, is: '{x}'")
print("ok")
