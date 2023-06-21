# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

"""Enum constant hallucination example.

Demonstrates the enum constant hallucination feature of HIP Python's enum types
that can be used for porting CUDA applications with enum constants
that have no HIP equivalent.

The hallucinated enum types act as instances of the original type,
they all have the same value, which does not conflict with the values
of the original enum constants.

This script:

* Will fail while initializing error_kinds if you just run it as it is.
    
* Will pass if you specify environment variable HIP_PYTHON_ENUM_HALLUCINATE_MEMBERS=1 before running
"""

# [literalinclude-begin]
from cuda.cudart import cudaError_t

error_kinds = ( # some of those do not exist in HIP
    cudaError_t.cudaErrorInitializationError,
    cudaError_t.cudaErrorInsufficientDriver,
    cudaError_t.cudaErrorInvalidDeviceFunction,
    cudaError_t.cudaErrorInvalidDevice,
    cudaError_t.cudaErrorStartupFailure, # no HIP equivalent
    cudaError_t.cudaErrorInvalidKernelImage,
    cudaError_t.cudaErrorAlreadyAcquired,
    cudaError_t.cudaErrorOperatingSystem,
    cudaError_t.cudaErrorNotPermitted, # no HIP equivalent
    cudaError_t.cudaErrorNotSupported,
    cudaError_t.cudaErrorSystemNotReady, # no HIP equivalent
    cudaError_t.cudaErrorSystemDriverMismatch, # no HIP equivalent
    cudaError_t.cudaErrorCompatNotSupportedOnDevice, # no HIP equivalent
    cudaError_t.cudaErrorDeviceUninitialized,
    cudaError_t.cudaErrorTimeout, # no HIP equivalent
    cudaError_t.cudaErrorUnknown,
    cudaError_t.cudaErrorApiFailureBase, # no HIP equivalent
)

for err in error_kinds:
    assert isinstance(err,cudaError_t)
    assert (err != cudaError_t.cudaSuccess)
print("ok")
