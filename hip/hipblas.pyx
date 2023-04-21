# AMD_COPYRIGHT
from libc cimport stdlib
from libc.stdint cimport *
import enum

from . cimport chipblas
hipblasVersionMajor = chipblas.hipblasVersionMajor

hipblaseVersionMinor = chipblas.hipblaseVersionMinor

hipblasVersionMinor = chipblas.hipblasVersionMinor

hipblasVersionPatch = chipblas.hipblasVersionPatch


cdef class hipblasHandle_t:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipblasHandle_t from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipblasHandle_t`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipblasHandle_t wrapper = hipblasHandle_t.__new__(hipblasHandle_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class hipblasBfloat16:
    cdef chipblas.hipblasBfloat16* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipblasBfloat16 from_ptr(chipblas.hipblasBfloat16 *_ptr, bint owner=False):
        """Factory function to create ``hipblasBfloat16`` objects from
        given ``chipblas.hipblasBfloat16`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipblasBfloat16 wrapper = hipblasBfloat16.__new__(hipblasBfloat16)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipblasBfloat16 new():
        """Factory function to create hipblasBfloat16 objects with
        newly allocated chipblas.hipblasBfloat16"""
        cdef chipblas.hipblasBfloat16 *_ptr = <chipblas.hipblasBfloat16 *>stdlib.malloc(sizeof(chipblas.hipblasBfloat16))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipblasBfloat16.from_ptr(_ptr, owner=True)
    def get_data(self, i):
        """Get value ``data`` of ``self._ptr[i]``.
        """
        return self._ptr[i].data
    def set_data(self, i, uint16_t value):
        """Set value ``data`` of ``self._ptr[i]``.
        """
        self._ptr[i].data = value
    @property
    def data(self):
        return self.get_data(0)
    @data.setter
    def data(self, uint16_t value):
        self.set_data(0,value)



cdef class hipblasComplex:
    cdef chipblas.hipblasComplex* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipblasComplex from_ptr(chipblas.hipblasComplex *_ptr, bint owner=False):
        """Factory function to create ``hipblasComplex`` objects from
        given ``chipblas.hipblasComplex`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipblasComplex wrapper = hipblasComplex.__new__(hipblasComplex)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipblasComplex new():
        """Factory function to create hipblasComplex objects with
        newly allocated chipblas.hipblasComplex"""
        cdef chipblas.hipblasComplex *_ptr = <chipblas.hipblasComplex *>stdlib.malloc(sizeof(chipblas.hipblasComplex))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipblasComplex.from_ptr(_ptr, owner=True)
    def get_x(self, i):
        """Get value ``x`` of ``self._ptr[i]``.
        """
        return self._ptr[i].x
    def set_x(self, i, float value):
        """Set value ``x`` of ``self._ptr[i]``.
        """
        self._ptr[i].x = value
    @property
    def x(self):
        return self.get_x(0)
    @x.setter
    def x(self, float value):
        self.set_x(0,value)
    def get_y(self, i):
        """Get value ``y`` of ``self._ptr[i]``.
        """
        return self._ptr[i].y
    def set_y(self, i, float value):
        """Set value ``y`` of ``self._ptr[i]``.
        """
        self._ptr[i].y = value
    @property
    def y(self):
        return self.get_y(0)
    @y.setter
    def y(self, float value):
        self.set_y(0,value)



cdef class hipblasDoubleComplex:
    cdef chipblas.hipblasDoubleComplex* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipblasDoubleComplex from_ptr(chipblas.hipblasDoubleComplex *_ptr, bint owner=False):
        """Factory function to create ``hipblasDoubleComplex`` objects from
        given ``chipblas.hipblasDoubleComplex`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipblasDoubleComplex wrapper = hipblasDoubleComplex.__new__(hipblasDoubleComplex)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef hipblasDoubleComplex new():
        """Factory function to create hipblasDoubleComplex objects with
        newly allocated chipblas.hipblasDoubleComplex"""
        cdef chipblas.hipblasDoubleComplex *_ptr = <chipblas.hipblasDoubleComplex *>stdlib.malloc(sizeof(chipblas.hipblasDoubleComplex))

        if _ptr is NULL:
            raise MemoryError
        # TODO init values, if present
        return hipblasDoubleComplex.from_ptr(_ptr, owner=True)
    def get_x(self, i):
        """Get value ``x`` of ``self._ptr[i]``.
        """
        return self._ptr[i].x
    def set_x(self, i, double value):
        """Set value ``x`` of ``self._ptr[i]``.
        """
        self._ptr[i].x = value
    @property
    def x(self):
        return self.get_x(0)
    @x.setter
    def x(self, double value):
        self.set_x(0,value)
    def get_y(self, i):
        """Get value ``y`` of ``self._ptr[i]``.
        """
        return self._ptr[i].y
    def set_y(self, i, double value):
        """Set value ``y`` of ``self._ptr[i]``.
        """
        self._ptr[i].y = value
    @property
    def y(self):
        return self.get_y(0)
    @y.setter
    def y(self, double value):
        self.set_y(0,value)


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
