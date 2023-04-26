import ctypes
import math
import numpy as np

from hip import hip
from hip import hipblas

def hip_check(call_result):
    if isinstance(call_result,tuple):
        err = call_result[0]
        result = call_result[1:]
    else:
        err = call_result
        result = None
    if isinstance(err,hip.hipStatus_t) and err != hip.hipStatus_t.hipSuccess:
        raise RuntimeError(str(err))
    elif isinstance(err,hipblas.hipblasStatus_t) and err != hipblas.hipblasStatus_t.HIPBLAS_STATUS_SUCCESS:
        raise RuntimeError(str(err))
    return result

num_elements = 100
num_bytes = num_elements * np.dtype(np.float32).itemsize

# input data on host
alpha = ctypes.c_float(1.2)
x_h = np.random.rand(num_bytes).astype(dtype=np.float32)
y_h = np.random.rand(num_bytes).astype(dtype=np.float32)

# device vectors
x_d = hip_check(hip.hipMalloc(num_bytes))
y_d = hip_check(hip.hipMalloc(num_bytes))

# copy input data to device
hip_check(hip.hipMemcpy(x_d,x_h,num_bytes,hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(y_d,y_h,num_bytes,hip.hipMemcpyKind.hipMemcpyHostToDevice))

# call hipblasSaxpy + initialization & destruction of handle
handle = hip_check(hipblas.hipblasHandleCreate())
hip_check(hipblas.hipblasSaxpy(handle, num_elements, ctypes.addressof(alpha), x_d, 0, y_d, 0))
hip_check(hipblas.hipblasDestroy(handle))

# copy result (stored in y_d) back to host (store in y_h)
hip_check(hip.hipMemcpy(y_h,y_d,num_bytes,hip.hipMemcpyKind.hipMemcpyDeviceToHost))

# compare to expected result
y_expected = alpha*x_h + y_h
print(f"y_h == y_expected: {np.allclose(y_expected,x_h)}")

# clean up
hip_check(hip.hipFree(x_d))
hip_check(hip.hipFree(y_d))


