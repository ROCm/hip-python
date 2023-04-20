# AMD_COPYRIGHT
from libc.stdint cimport *
import enum

from . cimport chipblas
hipblasVersionMajor = chipblas.hipblasVersionMajor

hipblaseVersionMinor = chipblas.hipblaseVersionMinor

hipblasVersionMinor = chipblas.hipblasVersionMinor

hipblasVersionPatch = chipblas.hipblasVersionPatch

cdef class hipblasBfloat16:
    pass

cdef class hipblasComplex:
    pass

cdef class hipblasDoubleComplex:
    pass

class hipblasStatus_t(enum.IntEnum):
    HIPBLAS_STATUS_SUCCESS = chipblas.HIPBLAS_STATUS_SUCCESS
    HIPBLAS_STATUS_NOT_INITIALIZED = chipblas.HIPBLAS_STATUS_NOT_INITIALIZED
    HIPBLAS_STATUS_ALLOC_FAILED = chipblas.HIPBLAS_STATUS_ALLOC_FAILED
    HIPBLAS_STATUS_INVALID_VALUE = chipblas.HIPBLAS_STATUS_INVALID_VALUE
    HIPBLAS_STATUS_MAPPING_ERROR = chipblas.HIPBLAS_STATUS_MAPPING_ERROR
    HIPBLAS_STATUS_EXECUTION_FAILED = chipblas.HIPBLAS_STATUS_EXECUTION_FAILED
    HIPBLAS_STATUS_INTERNAL_ERROR = chipblas.HIPBLAS_STATUS_INTERNAL_ERROR
    HIPBLAS_STATUS_NOT_SUPPORTED = chipblas.HIPBLAS_STATUS_NOT_SUPPORTED
    HIPBLAS_STATUS_ARCH_MISMATCH = chipblas.HIPBLAS_STATUS_ARCH_MISMATCH
    HIPBLAS_STATUS_HANDLE_IS_NULLPTR = chipblas.HIPBLAS_STATUS_HANDLE_IS_NULLPTR
    HIPBLAS_STATUS_INVALID_ENUM = chipblas.HIPBLAS_STATUS_INVALID_ENUM
    HIPBLAS_STATUS_UNKNOWN = chipblas.HIPBLAS_STATUS_UNKNOWN

class hipblasOperation_t(enum.IntEnum):
    HIPBLAS_OP_N = chipblas.HIPBLAS_OP_N
    HIPBLAS_OP_T = chipblas.HIPBLAS_OP_T
    HIPBLAS_OP_C = chipblas.HIPBLAS_OP_C

class hipblasPointerMode_t(enum.IntEnum):
    HIPBLAS_POINTER_MODE_HOST = chipblas.HIPBLAS_POINTER_MODE_HOST
    HIPBLAS_POINTER_MODE_DEVICE = chipblas.HIPBLAS_POINTER_MODE_DEVICE

class hipblasFillMode_t(enum.IntEnum):
    HIPBLAS_FILL_MODE_UPPER = chipblas.HIPBLAS_FILL_MODE_UPPER
    HIPBLAS_FILL_MODE_LOWER = chipblas.HIPBLAS_FILL_MODE_LOWER
    HIPBLAS_FILL_MODE_FULL = chipblas.HIPBLAS_FILL_MODE_FULL

class hipblasDiagType_t(enum.IntEnum):
    HIPBLAS_DIAG_NON_UNIT = chipblas.HIPBLAS_DIAG_NON_UNIT
    HIPBLAS_DIAG_UNIT = chipblas.HIPBLAS_DIAG_UNIT

class hipblasSideMode_t(enum.IntEnum):
    HIPBLAS_SIDE_LEFT = chipblas.HIPBLAS_SIDE_LEFT
    HIPBLAS_SIDE_RIGHT = chipblas.HIPBLAS_SIDE_RIGHT
    HIPBLAS_SIDE_BOTH = chipblas.HIPBLAS_SIDE_BOTH

class hipblasDatatype_t(enum.IntEnum):
    HIPBLAS_R_16F = chipblas.HIPBLAS_R_16F
    HIPBLAS_R_32F = chipblas.HIPBLAS_R_32F
    HIPBLAS_R_64F = chipblas.HIPBLAS_R_64F
    HIPBLAS_C_16F = chipblas.HIPBLAS_C_16F
    HIPBLAS_C_32F = chipblas.HIPBLAS_C_32F
    HIPBLAS_C_64F = chipblas.HIPBLAS_C_64F
    HIPBLAS_R_8I = chipblas.HIPBLAS_R_8I
    HIPBLAS_R_8U = chipblas.HIPBLAS_R_8U
    HIPBLAS_R_32I = chipblas.HIPBLAS_R_32I
    HIPBLAS_R_32U = chipblas.HIPBLAS_R_32U
    HIPBLAS_C_8I = chipblas.HIPBLAS_C_8I
    HIPBLAS_C_8U = chipblas.HIPBLAS_C_8U
    HIPBLAS_C_32I = chipblas.HIPBLAS_C_32I
    HIPBLAS_C_32U = chipblas.HIPBLAS_C_32U
    HIPBLAS_R_16B = chipblas.HIPBLAS_R_16B
    HIPBLAS_C_16B = chipblas.HIPBLAS_C_16B

class hipblasGemmAlgo_t(enum.IntEnum):
    HIPBLAS_GEMM_DEFAULT = chipblas.HIPBLAS_GEMM_DEFAULT

class hipblasAtomicsMode_t(enum.IntEnum):
    HIPBLAS_ATOMICS_NOT_ALLOWED = chipblas.HIPBLAS_ATOMICS_NOT_ALLOWED
    HIPBLAS_ATOMICS_ALLOWED = chipblas.HIPBLAS_ATOMICS_ALLOWED

class hipblasInt8Datatype_t(enum.IntEnum):
    HIPBLAS_INT8_DATATYPE_DEFAULT = chipblas.HIPBLAS_INT8_DATATYPE_DEFAULT
    HIPBLAS_INT8_DATATYPE_INT8 = chipblas.HIPBLAS_INT8_DATATYPE_INT8
    HIPBLAS_INT8_DATATYPE_PACK_INT8x4 = chipblas.HIPBLAS_INT8_DATATYPE_PACK_INT8x4