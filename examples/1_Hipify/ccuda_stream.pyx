# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cimport cuda.ccudart as ccudart

cdef ccudart.cudaError_t err
cdef ccudart.cudaStream_t stream
DEF num_bytes = 4*100
cdef char[num_bytes] x_h
cdef void* x_d
cdef int x

def cuda_check(ccudart.cudaError_t err):
    IF HIP_PYTHON:
        success_status = ccudart.cudaSuccess
    ELSE:
        success_status = ccudart.cudaError.cudaSuccess
    if err != success_status:
        raise RuntimeError(f"reason: {err}")

IF HIP_PYTHON:
    print("using HIP Python wrapper for CUDA Python")

cuda_check(ccudart.cudaStreamCreate(&stream))
cuda_check(ccudart.cudaMalloc(&x_d, num_bytes))
cuda_check(ccudart.cudaMemcpyAsync(x_d,x_h, num_bytes, ccudart.cudaMemcpyHostToDevice, stream))
cuda_check(ccudart.cudaMemsetAsync(x_d, 0, num_bytes, stream))
cuda_check(ccudart.cudaMemcpyAsync(x_h, x_d, num_bytes, ccudart.cudaMemcpyDeviceToHost, stream))
cuda_check(ccudart.cudaStreamSynchronize(stream))
cuda_check(ccudart.cudaStreamDestroy(stream))

# deallocate device data
cuda_check(ccudart.cudaFree(x_d))

for i in range(0,round(num_bytes/4)):
    x = (<int*>&x_h[4*i])[0]
    if x != 0:
        raise ValueError(f"expected '0' for element {i}, is: '{x}'")
print("ok")
