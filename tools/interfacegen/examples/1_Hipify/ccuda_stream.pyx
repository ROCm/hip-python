# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cimport cuda.ccudart as ccudart

cdef ccudart.cudaError_t err
cdef ccudart.cudaStream_t stream

def check_err(ccudart.cudaError_t err):
    if err != ccudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"reason: {err.value}")

err = ccudart.cudaStreamCreate(&stream)
err = ccudart.cudaStreamDestroy(stream)
