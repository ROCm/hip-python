# AMD_COPYRIGHT
from libc cimport stdlib
from libc.stdint cimport *
import cython
import enum
#ctypedef int16_t __int16_t
#ctypedef uint16_t __uint16_t

from . cimport chipblas
hipblasVersionMajor = chipblas.hipblasVersionMajor

hipblaseVersionMinor = chipblas.hipblaseVersionMinor

hipblasVersionMinor = chipblas.hipblasVersionMinor

hipblasVersionPatch = chipblas.hipblasVersionPatch


cdef class __int16_t:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef __int16_t from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``__int16_t`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef __int16_t wrapper = __int16_t.__new__(__int16_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class __uint16_t:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef __uint16_t from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``__uint16_t`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef __uint16_t wrapper = __uint16_t.__new__(__uint16_t)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



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



cdef class hipblasHalf:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipblasHalf from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipblasHalf`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipblasHalf wrapper = hipblasHalf.__new__(hipblasHalf)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class hipblasInt8:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipblasInt8 from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipblasInt8`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipblasInt8 wrapper = hipblasInt8.__new__(hipblasInt8)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper



cdef class hipblasStride:
    cdef void* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef hipblasStride from_ptr(void *_ptr, bint owner=False):
        """Factory function to create ``hipblasStride`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipblasStride wrapper = hipblasStride.__new__(hipblasStride)
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
    cdef __allocate(chipblas.hipblasBfloat16** ptr):
        ptr[0] = <chipblas.hipblasBfloat16 *>stdlib.malloc(sizeof(chipblas.hipblasBfloat16))

        if ptr[0] is NULL:
            raise MemoryError
        # TODO init values, if present

    @staticmethod
    cdef hipblasBfloat16 new():
        """Factory function to create hipblasBfloat16 objects with
        newly allocated chipblas.hipblasBfloat16"""
        cdef chipblas.hipblasBfloat16 *ptr;
        hipblasBfloat16.__allocate(&ptr)
        return hipblasBfloat16.from_ptr(ptr, owner=True)
    
    def __init__(self):
       hipblasBfloat16.__allocate(&self._ptr)
       self.ptr_owner = True
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
    cdef __allocate(chipblas.hipblasComplex** ptr):
        ptr[0] = <chipblas.hipblasComplex *>stdlib.malloc(sizeof(chipblas.hipblasComplex))

        if ptr[0] is NULL:
            raise MemoryError
        # TODO init values, if present

    @staticmethod
    cdef hipblasComplex new():
        """Factory function to create hipblasComplex objects with
        newly allocated chipblas.hipblasComplex"""
        cdef chipblas.hipblasComplex *ptr;
        hipblasComplex.__allocate(&ptr)
        return hipblasComplex.from_ptr(ptr, owner=True)
    
    def __init__(self):
       hipblasComplex.__allocate(&self._ptr)
       self.ptr_owner = True
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
    cdef __allocate(chipblas.hipblasDoubleComplex** ptr):
        ptr[0] = <chipblas.hipblasDoubleComplex *>stdlib.malloc(sizeof(chipblas.hipblasDoubleComplex))

        if ptr[0] is NULL:
            raise MemoryError
        # TODO init values, if present

    @staticmethod
    cdef hipblasDoubleComplex new():
        """Factory function to create hipblasDoubleComplex objects with
        newly allocated chipblas.hipblasDoubleComplex"""
        cdef chipblas.hipblasDoubleComplex *ptr;
        hipblasDoubleComplex.__allocate(&ptr)
        return hipblasDoubleComplex.from_ptr(ptr, owner=True)
    
    def __init__(self):
       hipblasDoubleComplex.__allocate(&self._ptr)
       self.ptr_owner = True
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

@cython.embedsignature(True)
def hipblasCreate():
    """! \brief Create hipblas handle. */
    """
    handle = hipblasHandle_t.from_ptr(NULL,owner=True)
    hipblasCreate_____retval = hipblasStatus_t(chipblas.hipblasCreate(&handle._ptr))    # fully specified
    return (hipblasCreate_____retval,handle)


@cython.embedsignature(True)
def hipblasDestroy(hipblasHandle_t handle):
    """! \brief Destroys the library context created using hipblasCreate() */
    """
    hipblasDestroy_____retval = hipblasStatus_t(chipblas.hipblasDestroy(handle._ptr))    # fully specified
    return hipblasDestroy_____retval


@cython.embedsignature(True)
def hipblasSetStream(hipblasHandle_t handle, streamId):
    """! \brief Set stream for handle */
    """
    pass

@cython.embedsignature(True)
def hipblasGetStream(hipblasHandle_t handle):
    """! \brief Get stream[0] for handle */
    """
    pass

@cython.embedsignature(True)
def hipblasSetPointerMode(hipblasHandle_t handle, object mode):
    """! \brief Set hipblas pointer mode */
    """
    if not isinstance(mode,hipblasPointerMode_t):
        raise TypeError("argument 'mode' must be of type 'hipblasPointerMode_t'")
    hipblasSetPointerMode_____retval = hipblasStatus_t(chipblas.hipblasSetPointerMode(handle._ptr,mode.value))    # fully specified
    return hipblasSetPointerMode_____retval


@cython.embedsignature(True)
def hipblasGetPointerMode(hipblasHandle_t handle, mode):
    """! \brief Get hipblas pointer mode */
    """
    pass

@cython.embedsignature(True)
def hipblasSetInt8Datatype(hipblasHandle_t handle, object int8Type):
    """! \brief Set hipblas int8 Datatype */
    """
    if not isinstance(int8Type,hipblasInt8Datatype_t):
        raise TypeError("argument 'int8Type' must be of type 'hipblasInt8Datatype_t'")
    hipblasSetInt8Datatype_____retval = hipblasStatus_t(chipblas.hipblasSetInt8Datatype(handle._ptr,int8Type.value))    # fully specified
    return hipblasSetInt8Datatype_____retval


@cython.embedsignature(True)
def hipblasGetInt8Datatype(hipblasHandle_t handle, int8Type):
    """! \brief Get hipblas int8 Datatype*/
    """
    pass

@cython.embedsignature(True)
def hipblasSetVector(int n, int elemSize, x, int incx, y, int incy):
    """! \brief copy vector from host to device
        @param[in]
        n           [int]
                    number of elements in the vector
        @param[in]
        elemSize    [int]
                    Size of both vectors in bytes
        @param[in]
        x           pointer to vector on the host
        @param[in]
        incx        [int]
                    specifies the increment for the elements of the vector
        @param[out]
        y           pointer to vector on the device
        @param[in]
        incy        [int]
                    specifies the increment for the elements of the vector
    """
    pass

@cython.embedsignature(True)
def hipblasGetVector(int n, int elemSize, x, int incx, y, int incy):
    """! \brief copy vector from device to host
        @param[in]
        n           [int]
                    number of elements in the vector
        @param[in]
        elemSize    [int]
                    Size of both vectors in bytes
        @param[in]
        x           pointer to vector on the device
        @param[in]
        incx        [int]
                    specifies the increment for the elements of the vector
        @param[out]
        y           pointer to vector on the host
        @param[in]
        incy        [int]
                    specifies the increment for the elements of the vector
    """
    pass

@cython.embedsignature(True)
def hipblasSetMatrix(int rows, int cols, int elemSize, AP, int lda, BP, int ldb):
    """! \brief copy matrix from host to device
        @param[in]
        rows        [int]
                    number of rows in matrices
        @param[in]
        cols        [int]
                    number of columns in matrices
        @param[in]
        elemSize   [int]
                    number of bytes per element in the matrix
        @param[in]
        AP          pointer to matrix on the host
        @param[in]
        lda         [int]
                    specifies the leading dimension of A, lda >= rows
        @param[out]
        BP           pointer to matrix on the GPU
        @param[in]
        ldb         [int]
                    specifies the leading dimension of B, ldb >= rows
    """
    pass

@cython.embedsignature(True)
def hipblasGetMatrix(int rows, int cols, int elemSize, AP, int lda, BP, int ldb):
    """! \brief copy matrix from device to host
        @param[in]
        rows        [int]
                    number of rows in matrices
        @param[in]
        cols        [int]
                    number of columns in matrices
        @param[in]
        elemSize   [int]
                    number of bytes per element in the matrix
        @param[in]
        AP          pointer to matrix on the GPU
        @param[in]
        lda         [int]
                    specifies the leading dimension of A, lda >= rows
        @param[out]
        BP          pointer to matrix on the host
        @param[in]
        ldb         [int]
                    specifies the leading dimension of B, ldb >= rows
    """
    pass

@cython.embedsignature(True)
def hipblasSetVectorAsync(int n, int elemSize, x, int incx, y, int incy, stream):
    """! \brief asynchronously copy vector from host to device
        \details
        hipblasSetVectorAsync copies a vector from pinned host memory to device memory asynchronously.
        Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
        @param[in]
        n           [int]
                    number of elements in the vector
        @param[in]
        elemSize   [int]
                    number of bytes per element in the matrix
        @param[in]
        x           pointer to vector on the host
        @param[in]
        incx        [int]
                    specifies the increment for the elements of the vector
        @param[out]
        y           pointer to vector on the device
        @param[in]
        incy        [int]
                    specifies the increment for the elements of the vector
        @param[in]
        stream      specifies the stream into which this transfer request is queued
    """
    pass

@cython.embedsignature(True)
def hipblasGetVectorAsync(int n, int elemSize, x, int incx, y, int incy, stream):
    """! \brief asynchronously copy vector from device to host
        \details
        hipblasGetVectorAsync copies a vector from pinned host memory to device memory asynchronously.
        Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
        @param[in]
        n           [int]
                    number of elements in the vector
        @param[in]
        elemSize   [int]
                    number of bytes per element in the matrix
        @param[in]
        x           pointer to vector on the device
        @param[in]
        incx        [int]
                    specifies the increment for the elements of the vector
        @param[out]
        y           pointer to vector on the host
        @param[in]
        incy        [int]
                    specifies the increment for the elements of the vector
        @param[in]
        stream      specifies the stream into which this transfer request is queued
    """
    pass

@cython.embedsignature(True)
def hipblasSetMatrixAsync(int rows, int cols, int elemSize, AP, int lda, BP, int ldb, stream):
    """! \brief asynchronously copy matrix from host to device
        \details
        hipblasSetMatrixAsync copies a matrix from pinned host memory to device memory asynchronously.
        Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
        @param[in]
        rows        [int]
                    number of rows in matrices
        @param[in]
        cols        [int]
                    number of columns in matrices
        @param[in]
        elemSize   [int]
                    number of bytes per element in the matrix
        @param[in]
        AP           pointer to matrix on the host
        @param[in]
        lda         [int]
                    specifies the leading dimension of A, lda >= rows
        @param[out]
        BP           pointer to matrix on the GPU
        @param[in]
        ldb         [int]
                    specifies the leading dimension of B, ldb >= rows
        @param[in]
        stream      specifies the stream into which this transfer request is queued
    """
    pass

@cython.embedsignature(True)
def hipblasGetMatrixAsync(int rows, int cols, int elemSize, AP, int lda, BP, int ldb, stream):
    """! \brief asynchronously copy matrix from device to host
        \details
        hipblasGetMatrixAsync copies a matrix from device memory to pinned host memory asynchronously.
        Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
        @param[in]
        rows        [int]
                    number of rows in matrices
        @param[in]
        cols        [int]
                    number of columns in matrices
        @param[in]
        elemSize   [int]
                    number of bytes per element in the matrix
        @param[in]
        AP          pointer to matrix on the GPU
        @param[in]
        lda         [int]
                    specifies the leading dimension of A, lda >= rows
        @param[out]
        BP           pointer to matrix on the host
        @param[in]
        ldb         [int]
                    specifies the leading dimension of B, ldb >= rows
        @param[in]
        stream      specifies the stream into which this transfer request is queued
    """
    pass

@cython.embedsignature(True)
def hipblasSetAtomicsMode(hipblasHandle_t handle, object atomics_mode):
    """! \brief Set hipblasSetAtomicsMode*/
    """
    if not isinstance(atomics_mode,hipblasAtomicsMode_t):
        raise TypeError("argument 'atomics_mode' must be of type 'hipblasAtomicsMode_t'")
    hipblasSetAtomicsMode_____retval = hipblasStatus_t(chipblas.hipblasSetAtomicsMode(handle._ptr,atomics_mode.value))    # fully specified
    return hipblasSetAtomicsMode_____retval


@cython.embedsignature(True)
def hipblasGetAtomicsMode(hipblasHandle_t handle, atomics_mode):
    """! \brief Get hipblasSetAtomicsMode*/
    """
    pass

@cython.embedsignature(True)
def hipblasIsamax(hipblasHandle_t handle, int n, x, int incx, result):
    """! @{
        \brief BLAS Level 1 API

        \details
        amax finds the first index of the element of maximum magnitude of a vector x.

        - Supported precisions in rocBLAS : s,d,c,z.
        - Supported precisions in cuBLAS  : s,d,c,z.

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        result
                  device pointer or host pointer to store the amax index.
                  return is 0.0 if n, incx<=0.
    """
    pass

@cython.embedsignature(True)
def hipblasIdamax(hipblasHandle_t handle, int n, x, int incx, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasIcamax(hipblasHandle_t handle, int n, x, int incx, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasIzamax(hipblasHandle_t handle, int n, x, int incx, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasIsamin(hipblasHandle_t handle, int n, x, int incx, result):
    """! @{
        \brief BLAS Level 1 API

        \details
        amin finds the first index of the element of minimum magnitude of a vector x.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        result
                  device pointer or host pointer to store the amin index.
                  return is 0.0 if n, incx<=0.
    """
    pass

@cython.embedsignature(True)
def hipblasIdamin(hipblasHandle_t handle, int n, x, int incx, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasIcamin(hipblasHandle_t handle, int n, x, int incx, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasIzamin(hipblasHandle_t handle, int n, x, int incx, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasSasum(hipblasHandle_t handle, int n, x, int incx, result):
    """! @{
        \brief BLAS Level 1 API

        \details
        asum computes the sum of the magnitudes of elements of a real vector x,
             or the sum of magnitudes of the real and imaginary parts of elements if x is a complex vector.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x and y.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x. incx must be > 0.
        @param[inout]
        result
                  device pointer or host pointer to store the asum product.
                  return is 0.0 if n <= 0.
    """
    pass

@cython.embedsignature(True)
def hipblasDasum(hipblasHandle_t handle, int n, x, int incx, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasScasum(hipblasHandle_t handle, int n, x, int incx, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasDzasum(hipblasHandle_t handle, int n, x, int incx, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasHaxpy(hipblasHandle_t handle, int n, __uint16_t alpha, x, int incx, y, int incy):
    """! @{
        \brief BLAS Level 1 API

        \details
        axpy   computes constant alpha multiplied by vector x, plus vector y

            y := alpha * x + y

        - Supported precisions in rocBLAS : h,s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x and y.
        @param[in]
        alpha     device pointer or host pointer to specify the scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[out]
        y         device pointer storing vector y.
        @param[inout]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    pass

@cython.embedsignature(True)
def hipblasSaxpy(hipblasHandle_t handle, int n, x, int incx, y, int incy):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasDaxpy(hipblasHandle_t handle, int n, x, int incx, y, int incy):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCaxpy(hipblasHandle_t handle, int n, hipblasComplex alpha, x, int incx, y, int incy):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZaxpy(hipblasHandle_t handle, int n, hipblasDoubleComplex alpha, x, int incx, y, int incy):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasScopy(hipblasHandle_t handle, int n, x, int incx, y, int incy):
    """! @{
        \brief BLAS Level 1 API

        \details
        copy  copies each element x[i] into y[i], for  i = 1 , ... , n

            y := x,

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x to be copied to y.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[out]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    pass

@cython.embedsignature(True)
def hipblasDcopy(hipblasHandle_t handle, int n, x, int incx, y, int incy):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCcopy(hipblasHandle_t handle, int n, x, int incx, y, int incy):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZcopy(hipblasHandle_t handle, int n, x, int incx, y, int incy):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasHdot(hipblasHandle_t handle, int n, x, int incx, y, int incy, result):
    """! @{
        \brief BLAS Level 1 API

        \details
        dot(u)  performs the dot product of vectors x and y

            result = x * y;

        dotc  performs the dot product of the conjugate of complex vector x and complex vector y

            result = conjugate (x) * y;

        - Supported precisions in rocBLAS : h,bf,s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x and y.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of y.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        result
                  device pointer or host pointer to store the dot product.
                  return is 0.0 if n <= 0.
    """
    pass

@cython.embedsignature(True)
def hipblasBfdot(hipblasHandle_t handle, int n, x, int incx, y, int incy, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasSdot(hipblasHandle_t handle, int n, x, int incx, y, int incy, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasDdot(hipblasHandle_t handle, int n, x, int incx, y, int incy, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCdotc(hipblasHandle_t handle, int n, x, int incx, y, int incy, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCdotu(hipblasHandle_t handle, int n, x, int incx, y, int incy, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZdotc(hipblasHandle_t handle, int n, x, int incx, y, int incy, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZdotu(hipblasHandle_t handle, int n, x, int incx, y, int incy, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasSnrm2(hipblasHandle_t handle, int n, x, int incx, result):
    """! @{
        \brief BLAS Level 1 API

        \details
        nrm2 computes the euclidean norm of a real or complex vector

                  result := sqrt( x'*x ) for real vectors
                  result := sqrt( x**H*x ) for complex vectors

        - Supported precisions in rocBLAS : s,d,c,z,sc,dz
        - Supported precisions in cuBLAS  : s,d,sc,dz

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        result
                  device pointer or host pointer to store the nrm2 product.
                  return is 0.0 if n, incx<=0.
    """
    pass

@cython.embedsignature(True)
def hipblasDnrm2(hipblasHandle_t handle, int n, x, int incx, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasScnrm2(hipblasHandle_t handle, int n, x, int incx, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasDznrm2(hipblasHandle_t handle, int n, x, int incx, result):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasSrot(hipblasHandle_t handle, int n, x, int incx, y, int incy, c, s):
    """! @{
        \brief BLAS Level 1 API

        \details
        rot applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
            Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.

        - Supported precisions in rocBLAS : s,d,c,z,sc,dz
        - Supported precisions in cuBLAS  : s,d,c,z,cs,zd

        @param[in]
        handle  [hipblasHandle_t]
                handle to the hipblas library context queue.
        @param[in]
        n       [int]
                number of elements in the x and y vectors.
        @param[inout]
        x       device pointer storing vector x.
        @param[in]
        incx    [int]
                specifies the increment between elements of x.
        @param[inout]
        y       device pointer storing vector y.
        @param[in]
        incy    [int]
                specifies the increment between elements of y.
        @param[in]
        c       device pointer or host pointer storing scalar cosine component of the rotation matrix.
        @param[in]
        s       device pointer or host pointer storing scalar sine component of the rotation matrix.
    """
    pass

@cython.embedsignature(True)
def hipblasDrot(hipblasHandle_t handle, int n, x, int incx, y, int incy, c, s):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCrot(hipblasHandle_t handle, int n, x, int incx, y, int incy, c, s):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCsrot(hipblasHandle_t handle, int n, x, int incx, y, int incy, c, s):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZrot(hipblasHandle_t handle, int n, x, int incx, y, int incy, c, s):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZdrot(hipblasHandle_t handle, int n, x, int incx, y, int incy, c, s):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasSrotg(hipblasHandle_t handle, a, b, c, s):
    """! @{
        \brief BLAS Level 1 API

        \details
        rotg creates the Givens rotation matrix for the vector (a b).
             Scalars c and s and arrays a and b may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
             If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
             If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle  [hipblasHandle_t]
                handle to the hipblas library context queue.
        @param[inout]
        a       device pointer or host pointer to input vector element, overwritten with r.
        @param[inout]
        b       device pointer or host pointer to input vector element, overwritten with z.
        @param[inout]
        c       device pointer or host pointer to cosine element of Givens rotation.
        @param[inout]
        s       device pointer or host pointer sine element of Givens rotation.
    """
    pass

@cython.embedsignature(True)
def hipblasDrotg(hipblasHandle_t handle, a, b, c, s):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCrotg(hipblasHandle_t handle, a, b, c, s):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZrotg(hipblasHandle_t handle, a, b, c, s):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasSrotm(hipblasHandle_t handle, int n, x, int incx, y, int incy, param):
    """! @{
        \brief BLAS Level 1 API

        \details
        rotm applies the modified Givens rotation matrix defined by param to vectors x and y.

        - Supported precisions in rocBLAS : s,d
        - Supported precisions in cuBLAS  : s,d

        @param[in]
        handle  [hipblasHandle_t]
                handle to the hipblas library context queue.
        @param[in]
        n       [int]
                number of elements in the x and y vectors.
        @param[inout]
        x       device pointer storing vector x.
        @param[in]
        incx    [int]
                specifies the increment between elements of x.
        @param[inout]
        y       device pointer storing vector y.
        @param[in]
        incy    [int]
                specifies the increment between elements of y.
        @param[in]
        param   device vector or host vector of 5 elements defining the rotation.
                param[0] = flag
                param[1] = H11
                param[2] = H21
                param[3] = H12
                param[4] = H22
                The flag parameter defines the form of H:
                flag = -1 => H = ( H11 H12 H21 H22 )
                flag =  0 => H = ( 1.0 H12 H21 1.0 )
                flag =  1 => H = ( H11 1.0 -1.0 H22 )
                flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
                param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
    """
    pass

@cython.embedsignature(True)
def hipblasDrotm(hipblasHandle_t handle, int n, x, int incx, y, int incy, param):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasSrotmg(hipblasHandle_t handle, d1, d2, x1, y1, param):
    """! @{
        \brief BLAS Level 1 API

        \details
        rotmg creates the modified Givens rotation matrix for the vector (d1 * x1, d2 * y1).
              Parameters may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
              If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
              If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.

        - Supported precisions in rocBLAS : s,d
        - Supported precisions in cuBLAS  : s,d

        @param[in]
        handle  [hipblasHandle_t]
                handle to the hipblas library context queue.
        @param[inout]
        d1      device pointer or host pointer to input scalar that is overwritten.
        @param[inout]
        d2      device pointer or host pointer to input scalar that is overwritten.
        @param[inout]
        x1      device pointer or host pointer to input scalar that is overwritten.
        @param[in]
        y1      device pointer or host pointer to input scalar.
        @param[out]
        param   device vector or host vector of 5 elements defining the rotation.
                param[0] = flag
                param[1] = H11
                param[2] = H21
                param[3] = H12
                param[4] = H22
                The flag parameter defines the form of H:
                flag = -1 => H = ( H11 H12 H21 H22 )
                flag =  0 => H = ( 1.0 H12 H21 1.0 )
                flag =  1 => H = ( H11 1.0 -1.0 H22 )
                flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
                param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
    """
    pass

@cython.embedsignature(True)
def hipblasDrotmg(hipblasHandle_t handle, d1, d2, x1, y1, param):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasSscal(hipblasHandle_t handle, int n, x, int incx):
    """! @{
        \brief BLAS Level 1 API

        \details
        scal  scales each element of vector x with scalar alpha.

            x := alpha * x

        - Supported precisions in rocBLAS : s,d,c,z,cs,zd
        - Supported precisions in cuBLAS  : s,d,c,z,cs,zd

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x.
        @param[in]
        alpha     device pointer or host pointer for the scalar alpha.
        @param[inout]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
    """
    pass

@cython.embedsignature(True)
def hipblasDscal(hipblasHandle_t handle, int n, x, int incx):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCscal(hipblasHandle_t handle, int n, hipblasComplex alpha, x, int incx):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCsscal(hipblasHandle_t handle, int n, x, int incx):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZscal(hipblasHandle_t handle, int n, hipblasDoubleComplex alpha, x, int incx):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZdscal(hipblasHandle_t handle, int n, x, int incx):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasSswap(hipblasHandle_t handle, int n, x, int incx, y, int incy):
    """! @{
        \brief BLAS Level 1 API

        \details
        swap  interchanges vectors x and y.

            y := x; x := y

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x and y.
        @param[inout]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    pass

@cython.embedsignature(True)
def hipblasDswap(hipblasHandle_t handle, int n, x, int incx, y, int incy):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCswap(hipblasHandle_t handle, int n, x, int incx, y, int incy):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZswap(hipblasHandle_t handle, int n, x, int incx, y, int incy):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasSgbmv(hipblasHandle_t handle, object trans, int m, int n, int kl, int ku, AP, int lda, x, int incx, y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        gbmv performs one of the matrix-vector operations

            y := alpha*A*x    + beta*y,   or
            y := alpha*A**T*x + beta*y,   or
            y := alpha*A**H*x + beta*y,

        where alpha and beta are scalars, x and y are vectors and A is an
        m by n banded matrix with kl sub-diagonals and ku super-diagonals.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        trans     [hipblasOperation_t]
                  indicates whether matrix A is tranposed (conjugated) or not
        @param[in]
        m         [int]
                  number of rows of matrix A
        @param[in]
        n         [int]
                  number of columns of matrix A
        @param[in]
        kl        [int]
                  number of sub-diagonals of A
        @param[in]
        ku        [int]
                  number of super-diagonals of A
        @param[in]
        alpha     device pointer or host pointer to scalar alpha.
        @param[in]
            AP    device pointer storing banded matrix A.
                  Leading (kl + ku + 1) by n part of the matrix contains the coefficients
                  of the banded matrix. The leading diagonal resides in row (ku + 1) with
                  the first super-diagonal above on the RHS of row ku. The first sub-diagonal
                  resides below on the LHS of row ku + 2. This propogates up and down across
                  sub/super-diagonals.
                    Ex: (m = n = 7; ku = 2, kl = 2)
                    1 2 3 0 0 0 0             0 0 3 3 3 3 3
                    4 1 2 3 0 0 0             0 2 2 2 2 2 2
                    5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
                    0 5 4 1 2 3 0             4 4 4 4 4 4 0
                    0 0 5 4 1 2 0             5 5 5 5 5 0 0
                    0 0 0 5 4 1 2             0 0 0 0 0 0 0
                    0 0 0 0 5 4 1             0 0 0 0 0 0 0
                  Note that the empty elements which don't correspond to data will not
                  be referenced.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A. Must be >= (kl + ku + 1)
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        beta      device pointer or host pointer to scalar beta.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasDgbmv(hipblasHandle_t handle, object trans, int m, int n, int kl, int ku, AP, int lda, x, int incx, y, int incy):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasCgbmv(hipblasHandle_t handle, object trans, int m, int n, int kl, int ku, hipblasComplex alpha, AP, int lda, x, int incx, hipblasComplex beta, y, int incy):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasZgbmv(hipblasHandle_t handle, object trans, int m, int n, int kl, int ku, hipblasDoubleComplex alpha, AP, int lda, x, int incx, hipblasDoubleComplex beta, y, int incy):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasSgemv(hipblasHandle_t handle, object trans, int m, int n, AP, int lda, x, int incx, y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        gemv performs one of the matrix-vector operations

            y := alpha*A*x    + beta*y,   or
            y := alpha*A**T*x + beta*y,   or
            y := alpha*A**H*x + beta*y,

        where alpha and beta are scalars, x and y are vectors and A is an
        m by n matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        trans     [hipblasOperation_t]
                  indicates whether matrix A is tranposed (conjugated) or not
        @param[in]
        m         [int]
                  number of rows of matrix A
        @param[in]
        n         [int]
                  number of columns of matrix A
        @param[in]
        alpha     device pointer or host pointer to scalar alpha.
        @param[in]
        AP        device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        beta      device pointer or host pointer to scalar beta.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasDgemv(hipblasHandle_t handle, object trans, int m, int n, AP, int lda, x, int incx, y, int incy):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasCgemv(hipblasHandle_t handle, object trans, int m, int n, hipblasComplex alpha, AP, int lda, x, int incx, hipblasComplex beta, y, int incy):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasZgemv(hipblasHandle_t handle, object trans, int m, int n, hipblasDoubleComplex alpha, AP, int lda, x, int incx, hipblasDoubleComplex beta, y, int incy):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasSger(hipblasHandle_t handle, int m, int n, x, int incx, y, int incy, AP, int lda):
    """! @{
        \brief BLAS Level 2 API

        \details
        ger,geru,gerc performs the matrix-vector operations

            A := A + alpha*x*y**T , OR
            A := A + alpha*x*y**H for gerc

        where alpha is a scalar, x and y are vectors, and A is an
        m by n matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        m         [int]
                  the number of rows of the matrix A.
        @param[in]
        n         [int]
                  the number of columns of the matrix A.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
    """
    pass

@cython.embedsignature(True)
def hipblasDger(hipblasHandle_t handle, int m, int n, x, int incx, y, int incy, AP, int lda):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCgeru(hipblasHandle_t handle, int m, int n, hipblasComplex alpha, x, int incx, y, int incy, AP, int lda):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCgerc(hipblasHandle_t handle, int m, int n, hipblasComplex alpha, x, int incx, y, int incy, AP, int lda):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZgeru(hipblasHandle_t handle, int m, int n, hipblasDoubleComplex alpha, x, int incx, y, int incy, AP, int lda):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZgerc(hipblasHandle_t handle, int m, int n, hipblasDoubleComplex alpha, x, int incx, y, int incy, AP, int lda):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasChbmv(hipblasHandle_t handle, object uplo, int n, int k, hipblasComplex alpha, AP, int lda, x, int incx, hipblasComplex beta, y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        hbmv performs the matrix-vector operations

            y := alpha*A*x + beta*y

        where alpha and beta are scalars, x and y are n element vectors and A is an
        n by n Hermitian band matrix, with k super-diagonals.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is being supplied.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is being supplied.
        @param[in]
        n         [int]
                  the order of the matrix A.
        @param[in]
        k         [int]
                  the number of super-diagonals of the matrix A. Must be >= 0.
        @param[in]
        alpha     device pointer or host pointer to scalar alpha.
        @param[in]
        AP        device pointer storing matrix A. Of dimension (lda, n).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The leading (k + 1) by n part of A must contain the upper
                    triangular band part of the Hermitian matrix, with the leading
                    diagonal in row (k + 1), the first super-diagonal on the RHS
                    of row k, etc.
                    The top left k by x triangle of A will not be referenced.
                        Ex (upper, lda = n = 4, k = 1):
                        A                             Represented matrix
                        (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
                        (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
                        (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
                        (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)

                  if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The leading (k + 1) by n part of A must contain the lower
                    triangular band part of the Hermitian matrix, with the leading
                    diagonal in row (1), the first sub-diagonal on the LHS of
                    row 2, etc.
                    The bottom right k by k triangle of A will not be referenced.
                        Ex (lower, lda = 2, n = 4, k = 1):
                        A                               Represented matrix
                        (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
                        (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
                                                        (0, 0) (6, 8) (3, 0) (7,-7)
                                                        (0, 0) (0, 0) (7, 7) (4, 0)

                  As a Hermitian matrix, the imaginary part of the main diagonal
                  of A will not be referenced and is assumed to be == 0.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A. must be >= k + 1
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        beta      device pointer or host pointer to scalar beta.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZhbmv(hipblasHandle_t handle, object uplo, int n, int k, hipblasDoubleComplex alpha, AP, int lda, x, int incx, hipblasDoubleComplex beta, y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasChemv(hipblasHandle_t handle, object uplo, int n, hipblasComplex alpha, AP, int lda, x, int incx, hipblasComplex beta, y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        hemv performs one of the matrix-vector operations

            y := alpha*A*x + beta*y

        where alpha and beta are scalars, x and y are n element vectors and A is an
        n by n Hermitian matrix.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied.
                  HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied.
        @param[in]
        n         [int]
                  the order of the matrix A.
        @param[in]
        alpha     device pointer or host pointer to scalar alpha.
        @param[in]
        AP        device pointer storing matrix A. Of dimension (lda, n).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular part of A must contain
                    the upper triangular part of a Hermitian matrix. The lower
                    triangular part of A will not be referenced.
                  if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular part of A must contain
                    the lower triangular part of a Hermitian matrix. The upper
                    triangular part of A will not be referenced.
                  As a Hermitian matrix, the imaginary part of the main diagonal
                  of A will not be referenced and is assumed to be == 0.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A. must be >= max(1, n)
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        beta      device pointer or host pointer to scalar beta.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZhemv(hipblasHandle_t handle, object uplo, int n, hipblasDoubleComplex alpha, AP, int lda, x, int incx, hipblasDoubleComplex beta, y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasCher(hipblasHandle_t handle, object uplo, int n, x, int incx, AP, int lda):
    """! @{
        \brief BLAS Level 2 API

        \details
        her performs the matrix-vector operations

            A := A + alpha*x*x**H

        where alpha is a real scalar, x is a vector, and A is an
        n by n Hermitian matrix.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in A.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in A.
        @param[in]
        n         [int]
                  the number of rows and columns of matrix A, must be at least 0.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[inout]
        AP        device pointer storing the specified triangular portion of
                  the Hermitian matrix A. Of size (lda * n).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the Hermitian matrix A is supplied. The lower
                    triangluar portion will not be touched.
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the Hermitian matrix A is supplied. The upper
                    triangular portion will not be touched.
                Note that the imaginary part of the diagonal elements are not accessed and are assumed
                to be 0.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A. Must be at least max(1, n).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZher(hipblasHandle_t handle, object uplo, int n, x, int incx, AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasCher2(hipblasHandle_t handle, object uplo, int n, hipblasComplex alpha, x, int incx, y, int incy, AP, int lda):
    """! @{
        \brief BLAS Level 2 API

        \details
        her2 performs the matrix-vector operations

            A := A + alpha*x*y**H + conj(alpha)*y*x**H

        where alpha is a complex scalar, x and y are vectors, and A is an
        n by n Hermitian matrix.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied.
        @param[in]
        n         [int]
                  the number of rows and columns of matrix A, must be at least 0.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        AP         device pointer storing the specified triangular portion of
                  the Hermitian matrix A. Of size (lda, n).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the Hermitian matrix A is supplied. The lower triangular
                    portion of A will not be touched.
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the Hermitian matrix A is supplied. The upper triangular
                    portion of A will not be touched.
                Note that the imaginary part of the diagonal elements are not accessed and are assumed
                to be 0.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A. Must be at least max(lda, 1).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZher2(hipblasHandle_t handle, object uplo, int n, hipblasDoubleComplex alpha, x, int incx, y, int incy, AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasChpmv(hipblasHandle_t handle, object uplo, int n, hipblasComplex alpha, AP, x, int incx, hipblasComplex beta, y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        hpmv performs the matrix-vector operation

            y := alpha*A*x + beta*y

        where alpha and beta are scalars, x and y are n element vectors and A is an
        n by n Hermitian matrix, supplied in packed form (see description below).

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied in AP.
                  HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied in AP.
        @param[in]
        n         [int]
                  the order of the matrix A, must be >= 0.
        @param[in]
        alpha     device pointer or host pointer to scalar alpha.
        @param[in]
        AP        device pointer storing the packed version of the specified triangular portion of
                  the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the Hermitian matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(0,1)
                    AP(2) = A(1,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
                            (1, 0) (2, 1) (3, 2)
                            (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
                            (3,-2) (5, 1) (6, 0)
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the Hermitian matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(1,0)
                    AP(2) = A(2,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
                            (1, 0) (2, 1) (3, 2)
                            (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
                            (3,-2) (5, 1) (6, 0)
                Note that the imaginary part of the diagonal elements are not accessed and are assumed
                to be 0.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        beta      device pointer or host pointer to scalar beta.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZhpmv(hipblasHandle_t handle, object uplo, int n, hipblasDoubleComplex alpha, AP, x, int incx, hipblasDoubleComplex beta, y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasChpr(hipblasHandle_t handle, object uplo, int n, x, int incx, AP):
    """! @{
        \brief BLAS Level 2 API

        \details
        hpr performs the matrix-vector operations

            A := A + alpha*x*x**H

        where alpha is a real scalar, x is a vector, and A is an
        n by n Hermitian matrix, supplied in packed form.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
        @param[in]
        n         [int]
                  the number of rows and columns of matrix A, must be at least 0.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[inout]
        AP        device pointer storing the packed version of the specified triangular portion of
                  the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the Hermitian matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(0,1)
                    AP(2) = A(1,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
                            (1, 0) (2, 1) (4,9)
                            (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                            (4,-9) (5,-3) (6,0)
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the Hermitian matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(1,0)
                    AP(2) = A(2,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
                            (1, 0) (2, 1) (4,9)
                            (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                            (4,-9) (5,-3) (6,0)
                Note that the imaginary part of the diagonal elements are not accessed and are assumed
                to be 0.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZhpr(hipblasHandle_t handle, object uplo, int n, x, int incx, AP):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasChpr2(hipblasHandle_t handle, object uplo, int n, hipblasComplex alpha, x, int incx, y, int incy, AP):
    """! @{
        \brief BLAS Level 2 API

        \details
        hpr2 performs the matrix-vector operations

            A := A + alpha*x*y**H + conj(alpha)*y*x**H

        where alpha is a complex scalar, x and y are vectors, and A is an
        n by n Hermitian matrix, supplied in packed form.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
        @param[in]
        n         [int]
                  the number of rows and columns of matrix A, must be at least 0.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        AP        device pointer storing the packed version of the specified triangular portion of
                  the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the Hermitian matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(0,1)
                    AP(2) = A(1,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
                            (1, 0) (2, 1) (4,9)
                            (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                            (4,-9) (5,-3) (6,0)
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the Hermitian matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(1,0)
                    AP(2) = A(2,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
                            (1, 0) (2, 1) (4,9)
                            (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                            (4,-9) (5,-3) (6,0)
                Note that the imaginary part of the diagonal elements are not accessed and are assumed
                to be 0.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZhpr2(hipblasHandle_t handle, object uplo, int n, hipblasDoubleComplex alpha, x, int incx, y, int incy, AP):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasSsbmv(hipblasHandle_t handle, object uplo, int n, int k, AP, int lda, x, int incx, y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        sbmv performs the matrix-vector operation:

            y := alpha*A*x + beta*y,

        where alpha and beta are scalars, x and y are n element vectors and
        A should contain an upper or lower triangular n by n symmetric banded matrix.

        - Supported precisions in rocBLAS : s,d
        - Supported precisions in cuBLAS  : s,d

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
                  if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
        @param[in]
        n         [int]
        @param[in]
        k         [int]
                  specifies the number of sub- and super-diagonals
        @param[in]
        alpha
                  specifies the scalar alpha
        @param[in]
        AP         pointer storing matrix A on the GPU
        @param[in]
        lda       [int]
                  specifies the leading dimension of matrix A
        @param[in]
        x         pointer storing vector x on the GPU
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x
        @param[in]
        beta      specifies the scalar beta
        @param[out]
        y         pointer storing vector y on the GPU
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasDsbmv(hipblasHandle_t handle, object uplo, int n, int k, AP, int lda, x, int incx, y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasSspmv(hipblasHandle_t handle, object uplo, int n, AP, x, int incx, y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        spmv performs the matrix-vector operation:

            y := alpha*A*x + beta*y,

        where alpha and beta are scalars, x and y are n element vectors and
        A should contain an upper or lower triangular n by n packed symmetric matrix.

        - Supported precisions in rocBLAS : s,d
        - Supported precisions in cuBLAS  : s,d

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
                  if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
        @param[in]
        n         [int]
        @param[in]
        alpha
                  specifies the scalar alpha
        @param[in]
        AP         pointer storing matrix A on the GPU
        @param[in]
        x         pointer storing vector x on the GPU
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x
        @param[in]
        beta      specifies the scalar beta
        @param[out]
        y         pointer storing vector y on the GPU
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasDspmv(hipblasHandle_t handle, object uplo, int n, AP, x, int incx, y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasSspr(hipblasHandle_t handle, object uplo, int n, x, int incx, AP):
    """! @{
        \brief BLAS Level 2 API

        \details
        spr performs the matrix-vector operations

            A := A + alpha*x*x**T

        where alpha is a scalar, x is a vector, and A is an
        n by n symmetric matrix, supplied in packed form.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
        @param[in]
        n         [int]
                  the number of rows and columns of matrix A, must be at least 0.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[inout]
        AP        device pointer storing the packed version of the specified triangular portion of
                  the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the symmetric matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(0,1)
                    AP(2) = A(1,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
                            1 2 4 7
                            2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                            4 5 6 9
                            7 8 9 0
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the symmetric matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(1,0)
                    AP(2) = A(2,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
                            1 2 3 4
                            2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                            3 6 8 9
                            4 7 9 0
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasDspr(hipblasHandle_t handle, object uplo, int n, x, int incx, AP):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasCspr(hipblasHandle_t handle, object uplo, int n, hipblasComplex alpha, x, int incx, AP):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZspr(hipblasHandle_t handle, object uplo, int n, hipblasDoubleComplex alpha, x, int incx, AP):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasSspr2(hipblasHandle_t handle, object uplo, int n, x, int incx, y, int incy, AP):
    """! @{
        \brief BLAS Level 2 API

        \details
        spr2 performs the matrix-vector operation

            A := A + alpha*x*y**T + alpha*y*x**T

        where alpha is a scalar, x and y are vectors, and A is an
        n by n symmetric matrix, supplied in packed form.

        - Supported precisions in rocBLAS : s,d
        - Supported precisions in cuBLAS  : s,d

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
        @param[in]
        n         [int]
                  the number of rows and columns of matrix A, must be at least 0.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        AP        device pointer storing the packed version of the specified triangular portion of
                  the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the symmetric matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(0,1)
                    AP(2) = A(1,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
                            1 2 4 7
                            2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                            4 5 6 9
                            7 8 9 0
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the symmetric matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(1,0)
                    AP(n) = A(2,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
                            1 2 3 4
                            2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                            3 6 8 9
                            4 7 9 0
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasDspr2(hipblasHandle_t handle, object uplo, int n, x, int incx, y, int incy, AP):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasSsymv(hipblasHandle_t handle, object uplo, int n, AP, int lda, x, int incx, y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        symv performs the matrix-vector operation:

            y := alpha*A*x + beta*y,

        where alpha and beta are scalars, x and y are n element vectors and
        A should contain an upper or lower triangular n by n symmetric matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
                  if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
        @param[in]
        n         [int]
        @param[in]
        alpha
                  specifies the scalar alpha
        @param[in]
        AP         pointer storing matrix A on the GPU
        @param[in]
        lda       [int]
                  specifies the leading dimension of A
        @param[in]
        x         pointer storing vector x on the GPU
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x
        @param[in]
        beta      specifies the scalar beta
        @param[out]
        y         pointer storing vector y on the GPU
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasDsymv(hipblasHandle_t handle, object uplo, int n, AP, int lda, x, int incx, y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasCsymv(hipblasHandle_t handle, object uplo, int n, hipblasComplex alpha, AP, int lda, x, int incx, hipblasComplex beta, y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZsymv(hipblasHandle_t handle, object uplo, int n, hipblasDoubleComplex alpha, AP, int lda, x, int incx, hipblasDoubleComplex beta, y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasSsyr(hipblasHandle_t handle, object uplo, int n, x, int incx, AP, int lda):
    """! @{
        \brief BLAS Level 2 API

        \details
        syr performs the matrix-vector operations

            A := A + alpha*x*x**T

        where alpha is a scalar, x is a vector, and A is an
        n by n symmetric matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
                  if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced

        @param[in]
        n         [int]
                  the number of rows and columns of matrix A.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[inout]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasDsyr(hipblasHandle_t handle, object uplo, int n, x, int incx, AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasCsyr(hipblasHandle_t handle, object uplo, int n, hipblasComplex alpha, x, int incx, AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZsyr(hipblasHandle_t handle, object uplo, int n, hipblasDoubleComplex alpha, x, int incx, AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasSsyr2(hipblasHandle_t handle, object uplo, int n, x, int incx, y, int incy, AP, int lda):
    """! @{
        \brief BLAS Level 2 API

        \details
        syr2 performs the matrix-vector operations

            A := A + alpha*x*y**T + alpha*y*x**T

        where alpha is a scalar, x and y are vectors, and A is an
        n by n symmetric matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : No support

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
                  if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced

        @param[in]
        n         [int]
                  the number of rows and columns of matrix A.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasDsyr2(hipblasHandle_t handle, object uplo, int n, x, int incx, y, int incy, AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasCsyr2(hipblasHandle_t handle, object uplo, int n, hipblasComplex alpha, x, int incx, y, int incy, AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZsyr2(hipblasHandle_t handle, object uplo, int n, hipblasDoubleComplex alpha, x, int incx, y, int incy, AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasStbmv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, int k, AP, int lda, x, int incx):
    """! @{
        \brief BLAS Level 2 API

        \details
        tbmv performs one of the matrix-vector operations

            x := A*x      or
            x := A**T*x   or
            x := A**H*x,

        x is a vectors and A is a banded m by m matrix (see description below).

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  HIPBLAS_FILL_MODE_UPPER: A is an upper banded triangular matrix.
                  HIPBLAS_FILL_MODE_LOWER: A is a  lower banded triangular matrix.
        @param[in]
        transA     [hipblasOperation_t]
                  indicates whether matrix A is tranposed (conjugated) or not.
        @param[in]
        diag      [hipblasDiagType_t]
                  HIPBLAS_DIAG_UNIT: The main diagonal of A is assumed to consist of only
                                         1's and is not referenced.
                  HIPBLAS_DIAG_NON_UNIT: No assumptions are made of A's main diagonal.
        @param[in]
        m         [int]
                  the number of rows and columns of the matrix represented by A.
        @param[in]
        k         [int]
                  if uplo == HIPBLAS_FILL_MODE_UPPER, k specifies the number of super-diagonals
                  of the matrix A.
                  if uplo == HIPBLAS_FILL_MODE_LOWER, k specifies the number of sub-diagonals
                  of the matrix A.
                  k must satisfy k > 0 && k < lda.
        @param[in]
        AP         device pointer storing banded triangular matrix A.
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The matrix represented is an upper banded triangular matrix
                    with the main diagonal and k super-diagonals, everything
                    else can be assumed to be 0.
                    The matrix is compacted so that the main diagonal resides on the k'th
                    row, the first super diagonal resides on the RHS of the k-1'th row, etc,
                    with the k'th diagonal on the RHS of the 0'th row.
                       Ex: (HIPBLAS_FILL_MODE_UPPER; m = 5; k = 2)
                          1 6 9 0 0              0 0 9 8 7
                          0 2 7 8 0              0 6 7 8 9
                          0 0 3 8 7     ---->    1 2 3 4 5
                          0 0 0 4 9              0 0 0 0 0
                          0 0 0 0 5              0 0 0 0 0
                  if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The matrix represnted is a lower banded triangular matrix
                    with the main diagonal and k sub-diagonals, everything else can be
                    assumed to be 0.
                    The matrix is compacted so that the main diagonal resides on the 0'th row,
                    working up to the k'th diagonal residing on the LHS of the k'th row.
                       Ex: (HIPBLAS_FILL_MODE_LOWER; m = 5; k = 2)
                          1 0 0 0 0              1 2 3 4 5
                          6 2 0 0 0              6 7 8 9 0
                          9 7 3 0 0     ---->    9 8 7 0 0
                          0 8 8 4 0              0 0 0 0 0
                          0 0 7 9 5              0 0 0 0 0
        @param[in]
        lda       [int]
                  specifies the leading dimension of A. lda must satisfy lda > k.
        @param[inout]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasDtbmv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, int k, AP, int lda, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasCtbmv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, int k, AP, int lda, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasZtbmv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, int k, AP, int lda, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasStbsv(hipblasHandle_t handle, object uplo, object transA, object diag, int n, int k, AP, int lda, x, int incx):
    """! @{
        \brief BLAS Level 2 API

        \details
        tbsv solves

             A*x = b or A**T*x = b or A**H*x = b,

        where x and b are vectors and A is a banded triangular matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA     [hipblasOperation_t]
                   HIPBLAS_OP_N: Solves A*x = b
                   HIPBLAS_OP_T: Solves A**T*x = b
                   HIPBLAS_OP_C: Solves A**H*x = b

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular (i.e. the diagonal elements
                                           of A are not used in computations).
                HIPBLAS_DIAG_NON_UNIT: A is not assumed to be unit triangular.

        @param[in]
        n         [int]
                  n specifies the number of rows of b. n >= 0.
        @param[in]
        k         [int]
                  if(uplo == HIPBLAS_FILL_MODE_UPPER)
                    k specifies the number of super-diagonals of A.
                  if(uplo == HIPBLAS_FILL_MODE_LOWER)
                    k specifies the number of sub-diagonals of A.
                  k >= 0.

        @param[in]
        AP         device pointer storing the matrix A in banded format.

        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
                  lda >= (k + 1).

        @param[inout]
        x         device pointer storing input vector b. Overwritten by the output vector x.

        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasDtbsv(hipblasHandle_t handle, object uplo, object transA, object diag, int n, int k, AP, int lda, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasCtbsv(hipblasHandle_t handle, object uplo, object transA, object diag, int n, int k, AP, int lda, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasZtbsv(hipblasHandle_t handle, object uplo, object transA, object diag, int n, int k, AP, int lda, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasStpmv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, x, int incx):
    """! @{
        \brief BLAS Level 2 API

        \details
        tpmv performs one of the matrix-vector operations

             x = A*x or x = A**T*x,

        where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix, supplied in the pack form.

        The vector x is overwritten.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA     [hipblasOperation_t]

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
                HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.

        @param[in]
        m       [int]
                m specifies the number of rows of A. m >= 0.

        @param[in]
        AP       device pointer storing matrix A,
                of dimension at leat ( m * ( m + 1 ) / 2 ).
              Before entry with uplo = HIPBLAS_FILL_MODE_UPPER, the array A
              must contain the upper triangular matrix packed sequentially,
              column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
              a_{0,1} and a_{1, 1} respectively, and so on.
              Before entry with uplo = HIPBLAS_FILL_MODE_LOWER, the array A
              must contain the lower triangular matrix packed sequentially,
              column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
              a_{1,0} and a_{2,0} respectively, and so on.
              Note that when DIAG = HIPBLAS_DIAG_UNIT, the diagonal elements of A are
              not referenced, but are assumed to be unity.

        @param[in]
        x       device pointer storing vector x.

        @param[in]
        incx    [int]
                specifies the increment for the elements of x. incx must not be zero.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasDtpmv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasCtpmv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasZtpmv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasStpsv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, x, int incx):
    """! @{
        \brief BLAS Level 2 API

        \details
        tpsv solves

             A*x = b or A**T*x = b, or A**H*x = b,

        where x and b are vectors and A is a triangular matrix stored in the packed format.

        The input vector b is overwritten by the output vector x.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_N: Solves A*x = b
                HIPBLAS_OP_T: Solves A**T*x = b
                HIPBLAS_OP_C: Solves A**H*x = b

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular (i.e. the diagonal elements
                                           of A are not used in computations).
                HIPBLAS_DIAG_NON_UNIT: A is not assumed to be unit triangular.

        @param[in]
        m         [int]
                  m specifies the number of rows of b. m >= 0.

        @param[in]
        AP        device pointer storing the packed version of matrix A,
                  of dimension >= (n * (n + 1) / 2)

        @param[inout]
        x         device pointer storing vector b on input, overwritten by x on output.

        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasDtpsv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasCtpsv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasZtpsv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasStrmv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, int lda, x, int incx):
    """! @{
        \brief BLAS Level 2 API

        \details
        trmv performs one of the matrix-vector operations

             x = A*x or x = A**T*x,

        where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix.

        The vector x is overwritten.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA     [hipblasOperation_t]

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
                HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.

        @param[in]
        m         [int]
                  m specifies the number of rows of A. m >= 0.

        @param[in]
        AP        device pointer storing matrix A,
                  of dimension ( lda, m )

        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
                  lda = max( 1, m ).

        @param[in]
        x         device pointer storing vector x.

        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasDtrmv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, int lda, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasCtrmv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, int lda, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasZtrmv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, int lda, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasStrsv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, int lda, x, int incx):
    """! @{
        \brief BLAS Level 2 API

        \details
        trsv solves

             A*x = b or A**T*x = b,

        where x and b are vectors and A is a triangular matrix.

        The vector x is overwritten on b.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA     [hipblasOperation_t]

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
                HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.

        @param[in]
        m         [int]
                  m specifies the number of rows of b. m >= 0.

        @param[in]
        AP        device pointer storing matrix A,
                  of dimension ( lda, m )

        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
                  lda = max( 1, m ).

        @param[in]
        x         device pointer storing vector x.

        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasDtrsv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, int lda, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasCtrsv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, int lda, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasZtrsv(hipblasHandle_t handle, object uplo, object transA, object diag, int m, AP, int lda, x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasHgemm(hipblasHandle_t handle, object transA, object transB, int m, int n, int k, __uint16_t alpha, AP, int lda, BP, int ldb, __uint16_t beta, CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details
        gemm performs one of the matrix-matrix operations

            C = alpha*op( A )*op( B ) + beta*C,

        where op( X ) is one of

            op( X ) = X      or
            op( X ) = X**T   or
            op( X ) = X**H,

        alpha and beta are scalars, and A, B and C are matrices, with
        op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.

        - Supported precisions in rocBLAS : h,s,d,c,z
        - Supported precisions in cuBLAS  : h,s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]

                  .
        @param[in]
        transA    [hipblasOperation_t]
                  specifies the form of op( A )
        @param[in]
        transB    [hipblasOperation_t]
                  specifies the form of op( B )
        @param[in]
        m         [int]
                  number or rows of matrices op( A ) and C
        @param[in]
        n         [int]
                  number of columns of matrices op( B ) and C
        @param[in]
        k         [int]
                  number of columns of matrix op( A ) and number of rows of matrix op( B )
        @param[in]
        alpha     device pointer or host pointer specifying the scalar alpha.
        @param[in]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
        @param[in]
        BP         device pointer storing matrix B.
        @param[in]
        ldb       [int]
                  specifies the leading dimension of B.
        @param[in]
        beta      device pointer or host pointer specifying the scalar beta.
        @param[in, out]
        CP         device pointer storing matrix C on the GPU.
        @param[in]
        ldc       [int]
                  specifies the leading dimension of C.
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasSgemm(hipblasHandle_t handle, object transA, object transB, int m, int n, int k, AP, int lda, BP, int ldb, CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasDgemm(hipblasHandle_t handle, object transA, object transB, int m, int n, int k, AP, int lda, BP, int ldb, CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasCgemm(hipblasHandle_t handle, object transA, object transB, int m, int n, int k, hipblasComplex alpha, AP, int lda, BP, int ldb, hipblasComplex beta, CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasZgemm(hipblasHandle_t handle, object transA, object transB, int m, int n, int k, hipblasDoubleComplex alpha, AP, int lda, BP, int ldb, hipblasDoubleComplex beta, CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasCherk(hipblasHandle_t handle, object uplo, object transA, int n, int k, AP, int lda, CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        herk performs one of the matrix-matrix operations for a Hermitian rank-k update

        C := alpha*op( A )*op( A )^H + beta*C

        where  alpha and beta are scalars, op(A) is an n by k matrix, and
        C is a n x n Hermitian matrix stored as either upper or lower.

            op( A ) = A,  and A is n by k if transA == HIPBLAS_OP_N
            op( A ) = A^H and A is k by n if transA == HIPBLAS_OP_C

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_C:  op(A) = A^H
                HIPBLAS_ON_N:  op(A) = A

        @param[in]
        n       [int]
                n specifies the number of rows and columns of C. n >= 0.

        @param[in]
        k       [int]
                k specifies the number of columns of op(A). k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and A need not be set before
                entry.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                Martrix dimension is ( lda, k ) when if transA = HIPBLAS_OP_N, otherwise (lda, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
                otherwise lda >= max( 1, k ).

        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.
                The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, n ).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasZherk(hipblasHandle_t handle, object uplo, object transA, int n, int k, AP, int lda, CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasCherkx(hipblasHandle_t handle, object uplo, object transA, int n, int k, hipblasComplex alpha, AP, int lda, BP, int ldb, CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        herkx performs one of the matrix-matrix operations for a Hermitian rank-k update

        C := alpha*op( A )*op( B )^H + beta*C

        where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
        C is a n x n Hermitian matrix stored as either upper or lower.
        This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermitian.


            op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
            op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == HIPBLAS_OP_C

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_C:  op( A ) = A^H, op( B ) = B^H
                HIPBLAS_OP_N:  op( A ) = A, op( B ) = B

        @param[in]
        n       [int]
                n specifies the number of rows and columns of C. n >= 0.

        @param[in]
        k       [int]
                k specifies the number of columns of op(A). k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and A need not be set before
                entry.

        @param[in]
        AP      pointer storing matrix A on the GPU.
                Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
                otherwise lda >= max( 1, k ).
        @param[in]
        BP       pointer storing matrix B on the GPU.
                Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        ldb     [int]
                ldb specifies the first dimension of B.
                if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
                otherwise ldb >= max( 1, k ).
        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.
                The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, n ).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasZherkx(hipblasHandle_t handle, object uplo, object transA, int n, int k, hipblasDoubleComplex alpha, AP, int lda, BP, int ldb, CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasCher2k(hipblasHandle_t handle, object uplo, object transA, int n, int k, hipblasComplex alpha, AP, int lda, BP, int ldb, CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        her2k performs one of the matrix-matrix operations for a Hermitian rank-2k update

        C := alpha*op( A )*op( B )^H + conj(alpha)*op( B )*op( A )^H + beta*C

        where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
        C is a n x n Hermitian matrix stored as either upper or lower.

            op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
            op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == HIPBLAS_OP_C

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_C:  op( A ) = A^H, op( B ) = B^H
                HIPBLAS_OP_N:  op( A ) = A, op( B ) = B

        @param[in]
        n       [int]
                n specifies the number of rows and columns of C. n >= 0.

        @param[in]
        k       [int]
                k specifies the number of columns of op(A). k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and A need not be set before
                entry.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
                otherwise lda >= max( 1, k ).
        @param[in]
        BP       pointer storing matrix B on the GPU.
                Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        ldb     [int]
                ldb specifies the first dimension of B.
                if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
                otherwise ldb >= max( 1, k ).
        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.
                The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, n ).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasZher2k(hipblasHandle_t handle, object uplo, object transA, int n, int k, hipblasDoubleComplex alpha, AP, int lda, BP, int ldb, CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasSsymm(hipblasHandle_t handle, object side, object uplo, int m, int n, AP, int lda, BP, int ldb, CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        symm performs one of the matrix-matrix operations:

        C := alpha*A*B + beta*C if side == HIPBLAS_SIDE_LEFT,
        C := alpha*B*A + beta*C if side == HIPBLAS_SIDE_RIGHT,

        where alpha and beta are scalars, B and C are m by n matrices, and
        A is a symmetric matrix stored as either upper or lower.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        side  [hipblasSideMode_t]
                HIPBLAS_SIDE_LEFT:      C := alpha*A*B + beta*C
                HIPBLAS_SIDE_RIGHT:     C := alpha*B*A + beta*C

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix

        @param[in]
        m       [int]
                m specifies the number of rows of B and C. m >= 0.

        @param[in]
        n       [int]
                n specifies the number of columns of B and C. n >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A and B are not referenced.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                A is m by m if side == HIPBLAS_SIDE_LEFT
                A is n by n if side == HIPBLAS_SIDE_RIGHT
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
                otherwise lda >= max( 1, n ).

        @param[in]
        BP       pointer storing matrix B on the GPU.
                Matrix dimension is m by n

        @param[in]
        ldb     [int]
                ldb specifies the first dimension of B. ldb >= max( 1, m )

        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.
                Matrix dimension is m by n

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, m )
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasDsymm(hipblasHandle_t handle, object side, object uplo, int m, int n, AP, int lda, BP, int ldb, CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasCsymm(hipblasHandle_t handle, object side, object uplo, int m, int n, hipblasComplex alpha, AP, int lda, BP, int ldb, hipblasComplex beta, CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZsymm(hipblasHandle_t handle, object side, object uplo, int m, int n, hipblasDoubleComplex alpha, AP, int lda, BP, int ldb, hipblasDoubleComplex beta, CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasSsyrk(hipblasHandle_t handle, object uplo, object transA, int n, int k, AP, int lda, CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        syrk performs one of the matrix-matrix operations for a symmetric rank-k update

        C := alpha*op( A )*op( A )^T + beta*C

        where  alpha and beta are scalars, op(A) is an n by k matrix, and
        C is a symmetric n x n matrix stored as either upper or lower.

            op( A ) = A, and A is n by k if transA == HIPBLAS_OP_N
            op( A ) = A^T and A is k by n if transA == HIPBLAS_OP_T

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_T: op(A) = A^T
                HIPBLAS_OP_N: op(A) = A
                HIPBLAS_OP_C: op(A) = A^T

                HIPBLAS_OP_C is not supported for complex types, see cherk
                and zherk.

        @param[in]
        n       [int]
                n specifies the number of rows and columns of C. n >= 0.

        @param[in]
        k       [int]
                k specifies the number of columns of op(A). k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and A need not be set before
                entry.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                Martrix dimension is ( lda, k ) when if transA = HIPBLAS_OP_N, otherwise (lda, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
                otherwise lda >= max( 1, k ).

        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, n ).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasDsyrk(hipblasHandle_t handle, object uplo, object transA, int n, int k, AP, int lda, CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasCsyrk(hipblasHandle_t handle, object uplo, object transA, int n, int k, hipblasComplex alpha, AP, int lda, hipblasComplex beta, CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasZsyrk(hipblasHandle_t handle, object uplo, object transA, int n, int k, hipblasDoubleComplex alpha, AP, int lda, hipblasDoubleComplex beta, CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasSsyr2k(hipblasHandle_t handle, object uplo, object transA, int n, int k, AP, int lda, BP, int ldb, CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        syr2k performs one of the matrix-matrix operations for a symmetric rank-2k update

        C := alpha*(op( A )*op( B )^T + op( B )*op( A )^T) + beta*C

        where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
        C is a symmetric n x n matrix stored as either upper or lower.

            op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
            op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == HIPBLAS_OP_T

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_T:      op( A ) = A^T, op( B ) = B^T
                HIPBLAS_OP_N:           op( A ) = A, op( B ) = B

        @param[in]
        n       [int]
                n specifies the number of rows and columns of C. n >= 0.

        @param[in]
        k       [int]
                k specifies the number of columns of op(A) and op(B). k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and A need not be set before
                entry.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
                otherwise lda >= max( 1, k ).
        @param[in]
        BP       pointer storing matrix B on the GPU.
                Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        ldb     [int]
                ldb specifies the first dimension of B.
                if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
                otherwise ldb >= max( 1, k ).
        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, n ).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasDsyr2k(hipblasHandle_t handle, object uplo, object transA, int n, int k, AP, int lda, BP, int ldb, CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasCsyr2k(hipblasHandle_t handle, object uplo, object transA, int n, int k, hipblasComplex alpha, AP, int lda, BP, int ldb, hipblasComplex beta, CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasZsyr2k(hipblasHandle_t handle, object uplo, object transA, int n, int k, hipblasDoubleComplex alpha, AP, int lda, BP, int ldb, hipblasDoubleComplex beta, CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasSsyrkx(hipblasHandle_t handle, object uplo, object transA, int n, int k, AP, int lda, BP, int ldb, CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        syrkx performs one of the matrix-matrix operations for a symmetric rank-k update

        C := alpha*op( A )*op( B )^T + beta*C

        where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
        C is a symmetric n x n matrix stored as either upper or lower.
        This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be symmetric.

            op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
            op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == HIPBLAS_OP_T

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_T:      op( A ) = A^T, op( B ) = B^T
                HIPBLAS_OP_N:           op( A ) = A, op( B ) = B

        @param[in]
        n       [int]
                n specifies the number of rows and columns of C. n >= 0.

        @param[in]
        k       [int]
                k specifies the number of columns of op(A) and op(B). k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and A need not be set before
                entry.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
                otherwise lda >= max( 1, k ).

        @param[in]
        BP       pointer storing matrix B on the GPU.
                Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        ldb     [int]
                ldb specifies the first dimension of B.
                if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
                otherwise ldb >= max( 1, k ).

        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, n ).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasDsyrkx(hipblasHandle_t handle, object uplo, object transA, int n, int k, AP, int lda, BP, int ldb, CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasCsyrkx(hipblasHandle_t handle, object uplo, object transA, int n, int k, hipblasComplex alpha, AP, int lda, BP, int ldb, hipblasComplex beta, CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasZsyrkx(hipblasHandle_t handle, object uplo, object transA, int n, int k, hipblasDoubleComplex alpha, AP, int lda, BP, int ldb, hipblasDoubleComplex beta, CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasSgeam(hipblasHandle_t handle, object transA, object transB, int m, int n, AP, int lda, BP, int ldb, CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details
        geam performs one of the matrix-matrix operations

            C = alpha*op( A ) + beta*op( B ),

        where op( X ) is one of

            op( X ) = X      or
            op( X ) = X**T   or
            op( X ) = X**H,

        alpha and beta are scalars, and A, B and C are matrices, with
        op( A ) an m by n matrix, op( B ) an m by n matrix, and C an m by n matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        transA    [hipblasOperation_t]
                  specifies the form of op( A )
        @param[in]
        transB    [hipblasOperation_t]
                  specifies the form of op( B )
        @param[in]
        m         [int]
                  matrix dimension m.
        @param[in]
        n         [int]
                  matrix dimension n.
        @param[in]
        alpha     device pointer or host pointer specifying the scalar alpha.
        @param[in]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
        @param[in]
        beta      device pointer or host pointer specifying the scalar beta.
        @param[in]
        BP         device pointer storing matrix B.
        @param[in]
        ldb       [int]
                  specifies the leading dimension of B.
        @param[in, out]
        CP         device pointer storing matrix C.
        @param[in]
        ldc       [int]
                  specifies the leading dimension of C.
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasDgeam(hipblasHandle_t handle, object transA, object transB, int m, int n, AP, int lda, BP, int ldb, CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasCgeam(hipblasHandle_t handle, object transA, object transB, int m, int n, hipblasComplex alpha, AP, int lda, hipblasComplex beta, BP, int ldb, CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasZgeam(hipblasHandle_t handle, object transA, object transB, int m, int n, hipblasDoubleComplex alpha, AP, int lda, hipblasDoubleComplex beta, BP, int ldb, CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasChemm(hipblasHandle_t handle, object side, object uplo, int n, int k, hipblasComplex alpha, AP, int lda, BP, int ldb, hipblasComplex beta, CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        hemm performs one of the matrix-matrix operations:

        C := alpha*A*B + beta*C if side == HIPBLAS_SIDE_LEFT,
        C := alpha*B*A + beta*C if side == HIPBLAS_SIDE_RIGHT,

        where alpha and beta are scalars, B and C are m by n matrices, and
        A is a Hermitian matrix stored as either upper or lower.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        side  [hipblasSideMode_t]
                HIPBLAS_SIDE_LEFT:      C := alpha*A*B + beta*C
                HIPBLAS_SIDE_RIGHT:     C := alpha*B*A + beta*C

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix

        @param[in]
        n       [int]
                n specifies the number of rows of B and C. n >= 0.

        @param[in]
        k       [int]
                n specifies the number of columns of B and C. k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A and B are not referenced.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                A is m by m if side == HIPBLAS_SIDE_LEFT
                A is n by n if side == HIPBLAS_SIDE_RIGHT
                Only the upper/lower triangular part is accessed.
                The imaginary component of the diagonal elements is not used.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
                otherwise lda >= max( 1, n ).

        @param[in]
        BP       pointer storing matrix B on the GPU.
                Matrix dimension is m by n

        @param[in]
        ldb     [int]
                ldb specifies the first dimension of B. ldb >= max( 1, m )

        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.
                Matrix dimension is m by n

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, m )
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZhemm(hipblasHandle_t handle, object side, object uplo, int n, int k, hipblasDoubleComplex alpha, AP, int lda, BP, int ldb, hipblasDoubleComplex beta, CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    pass

@cython.embedsignature(True)
def hipblasStrmm(hipblasHandle_t handle, object side, object uplo, object transA, object diag, int m, int n, AP, int lda, BP, int ldb):
    """! @{
        \brief BLAS Level 3 API

        \details

        trmm performs one of the matrix-matrix operations

        B := alpha*op( A )*B,   or   B := alpha*B*op( A )

        where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
        non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

            op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.


        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        side    [hipblasSideMode_t]
                Specifies whether op(A) multiplies B from the left or right as follows:
                HIPBLAS_SIDE_LEFT:       B := alpha*op( A )*B.
                HIPBLAS_SIDE_RIGHT:      B := alpha*B*op( A ).

        @param[in]
        uplo    [hipblasFillMode_t]
                Specifies whether the matrix A is an upper or lower triangular matrix as follows:
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA  [hipblasOperation_t]
                Specifies the form of op(A) to be used in the matrix multiplication as follows:
                HIPBLAS_OP_N: op(A) = A.
                HIPBLAS_OP_T: op(A) = A^T.
                HIPBLAS_OP_C:  op(A) = A^H.

        @param[in]
        diag    [hipblasDiagType_t]
                Specifies whether or not A is unit triangular as follows:
                HIPBLAS_DIAG_UNIT:      A is assumed to be unit triangular.
                HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.

        @param[in]
        m       [int]
                m specifies the number of rows of B. m >= 0.

        @param[in]
        n       [int]
                n specifies the number of columns of B. n >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and B need not be set before
                entry.

        @param[in]
        AP       Device pointer to matrix A on the GPU.
                A has dimension ( lda, k ), where k is m
                when  side == HIPBLAS_SIDE_LEFT  and
                is  n  when  side == HIPBLAS_SIDE_RIGHT.

            When uplo == HIPBLAS_FILL_MODE_UPPER the  leading  k by k
            upper triangular part of the array  A must contain the upper
            triangular matrix  and the strictly lower triangular part of
            A is not referenced.

            When uplo == HIPBLAS_FILL_MODE_LOWER the  leading  k by k
            lower triangular part of the array  A must contain the lower
            triangular matrix  and the strictly upper triangular part of
            A is not referenced.

            Note that when  diag == HIPBLAS_DIAG_UNIT  the diagonal elements of
            A  are not referenced either,  but are assumed to be  unity.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if side == HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
                if side == HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).

        @param[inout]
        BP       Device pointer to the first matrix B_0 on the GPU.
                On entry,  the leading  m by n part of the array  B must
               contain the matrix  B,  and  on exit  is overwritten  by the
               transformed matrix.

        @param[in]
        ldb    [int]
               ldb specifies the first dimension of B. ldb >= max( 1, m ).
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasDtrmm(hipblasHandle_t handle, object side, object uplo, object transA, object diag, int m, int n, AP, int lda, BP, int ldb):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasCtrmm(hipblasHandle_t handle, object side, object uplo, object transA, object diag, int m, int n, hipblasComplex alpha, AP, int lda, BP, int ldb):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasZtrmm(hipblasHandle_t handle, object side, object uplo, object transA, object diag, int m, int n, hipblasDoubleComplex alpha, AP, int lda, BP, int ldb):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasStrsm(hipblasHandle_t handle, object side, object uplo, object transA, object diag, int m, int n, AP, int lda, BP, int ldb):
    """! @{
        \brief BLAS Level 3 API

        \details

        trsm solves

            op(A)*X = alpha*B or  X*op(A) = alpha*B,

        where alpha is a scalar, X and B are m by n matrices,
        A is triangular matrix and op(A) is one of

            op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

        The matrix X is overwritten on B.

        Note about memory allocation:
        When trsm is launched with a k evenly divisible by the internal block size of 128,
        and is no larger than 10 of these blocks, the API takes advantage of utilizing pre-allocated
        memory found in the handle to increase overall performance. This memory can be managed by using
        the environment variable WORKBUF_TRSM_B_CHNK. When this variable is not set the device memory
        used for temporary storage will default to 1 MB and may result in chunking, which in turn may
        reduce performance. Under these circumstances it is recommended that WORKBUF_TRSM_B_CHNK be set
        to the desired chunk of right hand sides to be used at a time.

        (where k is m when HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT)

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        side    [hipblasSideMode_t]
                HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
                HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_N: op(A) = A.
                HIPBLAS_OP_T: op(A) = A^T.
                HIPBLAS_OP_C: op(A) = A^H.

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
                HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.

        @param[in]
        m       [int]
                m specifies the number of rows of B. m >= 0.

        @param[in]
        n       [int]
                n specifies the number of columns of B. n >= 0.

        @param[in]
        alpha
                device pointer or host pointer specifying the scalar alpha. When alpha is
                &zero then A is not referenced and B need not be set before
                entry.

        @param[in]
        AP       device pointer storing matrix A.
                of dimension ( lda, k ), where k is m
                when  HIPBLAS_SIDE_LEFT  and
                is  n  when  HIPBLAS_SIDE_RIGHT
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
                if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).

        @param[in,out]
        BP       device pointer storing matrix B.

        @param[in]
        ldb    [int]
               ldb specifies the first dimension of B. ldb >= max( 1, m ).
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasDtrsm(hipblasHandle_t handle, object side, object uplo, object transA, object diag, int m, int n, AP, int lda, BP, int ldb):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasCtrsm(hipblasHandle_t handle, object side, object uplo, object transA, object diag, int m, int n, hipblasComplex alpha, AP, int lda, BP, int ldb):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasZtrsm(hipblasHandle_t handle, object side, object uplo, object transA, object diag, int m, int n, hipblasDoubleComplex alpha, AP, int lda, BP, int ldb):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasStrtri(hipblasHandle_t handle, object uplo, object diag, int n, AP, int lda, invA, int ldinvA):
    """! @{
        \brief BLAS Level 3 API

        \details
        trtri  compute the inverse of a matrix A, namely, invA

            and write the result into invA;

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : No support

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
                  if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
        @param[in]
        diag      [hipblasDiagType_t]
                  = 'HIPBLAS_DIAG_NON_UNIT', A is non-unit triangular;
                  = 'HIPBLAS_DIAG_UNIT', A is unit triangular;
        @param[in]
        n         [int]
                  size of matrix A and invA
        @param[in]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
        @param[out]
        invA      device pointer storing matrix invA.
        @param[in]
        ldinvA    [int]
                  specifies the leading dimension of invA.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasDtrtri(hipblasHandle_t handle, object uplo, object diag, int n, AP, int lda, invA, int ldinvA):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasCtrtri(hipblasHandle_t handle, object uplo, object diag, int n, AP, int lda, invA, int ldinvA):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasZtrtri(hipblasHandle_t handle, object uplo, object diag, int n, AP, int lda, invA, int ldinvA):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    pass

@cython.embedsignature(True)
def hipblasSdgmm(hipblasHandle_t handle, object side, int m, int n, AP, int lda, x, int incx, CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details
        dgmm performs one of the matrix-matrix operations

            C = A * diag(x) if side == HIPBLAS_SIDE_RIGHT
            C = diag(x) * A if side == HIPBLAS_SIDE_LEFT

        where C and A are m by n dimensional matrices. diag( x ) is a diagonal matrix
        and x is vector of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension m
        if side == HIPBLAS_SIDE_LEFT.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        side      [hipblasSideMode_t]
                  specifies the side of diag(x)
        @param[in]
        m         [int]
                  matrix dimension m.
        @param[in]
        n         [int]
                  matrix dimension n.
        @param[in]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment between values of x
        @param[in, out]
        CP         device pointer storing matrix C.
        @param[in]
        ldc       [int]
                  specifies the leading dimension of C.
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")
    pass

@cython.embedsignature(True)
def hipblasDdgmm(hipblasHandle_t handle, object side, int m, int n, AP, int lda, x, int incx, CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")
    pass

@cython.embedsignature(True)
def hipblasCdgmm(hipblasHandle_t handle, object side, int m, int n, AP, int lda, x, int incx, CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")
    pass

@cython.embedsignature(True)
def hipblasZdgmm(hipblasHandle_t handle, object side, int m, int n, AP, int lda, x, int incx, CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")
    pass

@cython.embedsignature(True)
def hipblasSgetrf(hipblasHandle_t handle, const int n, A, const int lda, ipiv, info):
    """! @{
        \brief SOLVER API

        \details
        getrf computes the LU factorization of a general n-by-n matrix A
        using partial pivoting with row interchanges. The LU factorization can
        be done without pivoting if ipiv is passed as a nullptr.

        In the case that ipiv is not null, the factorization has the form:

        \f[
            A = PLU
        \f]

        where P is a permutation matrix, L is lower triangular with unit
        diagonal elements, and U is upper triangular.

        In the case that ipiv is null, the factorization is done without pivoting:

        \f[
            A = LU
        \f]

        - Supported precisions in rocSOLVER : s,d,c,z
        - Supported precisions in cuBLAS    : s,d,c,z

        @param[in]
        handle    hipblasHandle_t.
        @param[in]
        n         int. n >= 0.\n
                  The number of columns and rows of the matrix A.
        @param[inout]
        A         pointer to type. Array on the GPU of dimension lda*n.\n
                  On entry, the n-by-n matrix A to be factored.
                  On exit, the factors L and U from the factorization.
                  The unit diagonal elements of L are not stored.
        @param[in]
        lda       int. lda >= n.\n
                  Specifies the leading dimension of A.
        @param[out]
        ipiv      pointer to int. Array on the GPU of dimension n.\n
                  The vector of pivot indices. Elements of ipiv are 1-based indices.
                  For 1 <= i <= n, the row i of the
                  matrix was interchanged with row ipiv[i].
                  Matrix P of the factorization can be derived from ipiv.
                  The factorization here can be done without pivoting if ipiv is passed
                  in as a nullptr.
        @param[out]
        info      pointer to a int on the GPU.\n
                  If info = 0, successful exit.
                  If info = j > 0, U is singular. U[j,j] is the first zero pivot.
    """
    pass

@cython.embedsignature(True)
def hipblasDgetrf(hipblasHandle_t handle, const int n, A, const int lda, ipiv, info):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCgetrf(hipblasHandle_t handle, const int n, A, const int lda, ipiv, info):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZgetrf(hipblasHandle_t handle, const int n, A, const int lda, ipiv, info):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasSgetrs(hipblasHandle_t handle, object trans, const int n, const int nrhs, A, const int lda, ipiv, B, const int ldb, info):
    """! @{
        \brief SOLVER API

        \details
        getrs solves a system of n linear equations on n variables in its factorized form.

        It solves one of the following systems, depending on the value of trans:

        \f[
            \begin{array}{cl}
            A X = B & \: \text{not transposed,}\\
            A^T X = B & \: \text{transposed, or}\\
            A^H X = B & \: \text{conjugate transposed.}
            \end{array}
        \f]

        Matrix A is defined by its triangular factors as returned by \ref hipblasSgetrf "getrf".

        - Supported precisions in rocSOLVER : s,d,c,z
        - Supported precisions in cuBLAS    : s,d,c,z


        @param[in]
        handle      hipblasHandle_t.
        @param[in]
        trans       hipblasOperation_t.\n
                    Specifies the form of the system of equations.
        @param[in]
        n           int. n >= 0.\n
                    The order of the system, i.e. the number of columns and rows of A.
        @param[in]
        nrhs        int. nrhs >= 0.\n
                    The number of right hand sides, i.e., the number of columns
                    of the matrix B.
        @param[in]
        A           pointer to type. Array on the GPU of dimension lda*n.\n
                    The factors L and U of the factorization A = P*L*U returned by \ref hipblasSgetrf "getrf".
        @param[in]
        lda         int. lda >= n.\n
                    The leading dimension of A.
        @param[in]
        ipiv        pointer to int. Array on the GPU of dimension n.\n
                    The pivot indices returned by \ref hipblasSgetrf "getrf".
        @param[in,out]
        B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
                    On entry, the right hand side matrix B.
                    On exit, the solution matrix X.
        @param[in]
        ldb         int. ldb >= n.\n
                    The leading dimension of B.
        @param[out]
        info      pointer to a int on the host.\n
                  If info = 0, successful exit.
                  If info = j < 0, the j-th argument is invalid.
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasDgetrs(hipblasHandle_t handle, object trans, const int n, const int nrhs, A, const int lda, ipiv, B, const int ldb, info):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasCgetrs(hipblasHandle_t handle, object trans, const int n, const int nrhs, A, const int lda, ipiv, B, const int ldb, info):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasZgetrs(hipblasHandle_t handle, object trans, const int n, const int nrhs, A, const int lda, ipiv, B, const int ldb, info):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasSgels(hipblasHandle_t handle, object trans, const int m, const int n, const int nrhs, A, const int lda, B, const int ldb, info, deviceInfo):
    """! @{
        \brief GELS solves an overdetermined (or underdetermined) linear system defined by an m-by-n
        matrix A, and a corresponding matrix B, using the QR factorization computed by \ref hipblasSgeqrf "GEQRF" (or the LQ
        factorization computed by "GELQF").

        \details
        Depending on the value of trans, the problem solved by this function is either of the form

        \f[
            \begin{array}{cl}
            A X = B & \: \text{not transposed, or}\\
            A' X = B & \: \text{transposed if real, or conjugate transposed if complex}
            \end{array}
        \f]

        If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
        and a least-squares solution approximating X is found by minimizing

        \f[
            || B - A  X || \quad \text{(or} \: || B - A' X ||\text{)}
        \f]

        If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
        and a unique solution for X is chosen such that \f$|| X ||\f$ is minimal.

        - Supported precisions in rocSOLVER : s,d,c,z
        - Supported precisions in cuBLAS    : currently unsupported

        @param[in]
        handle      hipblasHandle_t.
        @param[in]
        trans       hipblasOperation_t.\n
                    Specifies the form of the system of equations.
        @param[in]
        m           int. m >= 0.\n
                    The number of rows of matrix A.
        @param[in]
        n           int. n >= 0.\n
                    The number of columns of matrix A.
        @param[in]
        nrhs        int. nrhs >= 0.\n
                    The number of columns of matrices B and X;
                    i.e., the columns on the right hand side.
        @param[inout]
        A           pointer to type. Array on the GPU of dimension lda*n.\n
                    On entry, the matrix A.
                    On exit, the QR (or LQ) factorization of A as returned by "GEQRF" (or "GELQF").
        @param[in]
        lda         int. lda >= m.\n
                    Specifies the leading dimension of matrix A.
        @param[inout]
        B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
                    On entry, the matrix B.
                    On exit, when info = 0, B is overwritten by the solution vectors (and the residuals in
                    the overdetermined cases) stored as columns.
        @param[in]
        ldb         int. ldb >= max(m,n).\n
                    Specifies the leading dimension of matrix B.
        @param[out]
        info        pointer to an int on the host.\n
                    If info = 0, successful exit.
                    If info = j < 0, the j-th argument is invalid.
        @param[out]
        deviceInfo  pointer to int on the GPU.\n
                    If info = 0, successful exit.
                    If info = i > 0, the solution could not be computed because input matrix A is
                    rank deficient; the i-th diagonal element of its triangular factor is zero.
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasDgels(hipblasHandle_t handle, object trans, const int m, const int n, const int nrhs, A, const int lda, B, const int ldb, info, deviceInfo):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasCgels(hipblasHandle_t handle, object trans, const int m, const int n, const int nrhs, A, const int lda, B, const int ldb, info, deviceInfo):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasZgels(hipblasHandle_t handle, object trans, const int m, const int n, const int nrhs, A, const int lda, B, const int ldb, info, deviceInfo):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    pass

@cython.embedsignature(True)
def hipblasSgeqrf(hipblasHandle_t handle, const int m, const int n, A, const int lda, ipiv, info):
    """! @{
        \brief SOLVER API

        \details
        geqrf computes a QR factorization of a general m-by-n matrix A.

        The factorization has the form

        \f[
            A = Q\left[\begin{array}{c}
            R\\
            0
            \end{array}\right]
        \f]

        where R is upper triangular (upper trapezoidal if m < n), and Q is
        a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        \f[
            Q = H_1H_2\cdots H_k, \quad \text{with} \: k = \text{min}(m,n)
        \f]

        Each Householder matrix \f$H_i\f$ is given by

        \f[
            H_i = I - \text{ipiv}[i] \cdot v_i v_i'
        \f]

        where the first i-1 elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

        - Supported precisions in rocSOLVER : s,d,c,z
        - Supported precisions in cuBLAS    : s,d,c,z

        @param[in]
        handle    hipblasHandle_t.
        @param[in]
        m         int. m >= 0.\n
                  The number of rows of the matrix A.
        @param[in]
        n         int. n >= 0.\n
                  The number of columns of the matrix A.
        @param[inout]
        A         pointer to type. Array on the GPU of dimension lda*n.\n
                  On entry, the m-by-n matrix to be factored.
                  On exit, the elements on and above the diagonal contain the
                  factor R; the elements below the diagonal are the last m - i elements
                  of Householder vector v_i.
        @param[in]
        lda       int. lda >= m.\n
                  Specifies the leading dimension of A.
        @param[out]
        ipiv      pointer to type. Array on the GPU of dimension min(m,n).\n
                  The Householder scalars.
        @param[out]
        info      pointer to a int on the host.\n
                  If info = 0, successful exit.
                  If info = j < 0, the j-th argument is invalid.
    """
    pass

@cython.embedsignature(True)
def hipblasDgeqrf(hipblasHandle_t handle, const int m, const int n, A, const int lda, ipiv, info):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasCgeqrf(hipblasHandle_t handle, const int m, const int n, A, const int lda, ipiv, info):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasZgeqrf(hipblasHandle_t handle, const int m, const int n, A, const int lda, ipiv, info):
    """
    """
    pass

@cython.embedsignature(True)
def hipblasGemmEx(hipblasHandle_t handle, object transA, object transB, int m, int n, int k, A, object aType, int lda, B, object bType, int ldb, C, object cType, int ldc, object computeType, object algo):
    """! \brief BLAS EX API

        \details
        gemmEx performs one of the matrix-matrix operations

            C = alpha*op( A )*op( B ) + beta*C,

        where op( X ) is one of

            op( X ) = X      or
            op( X ) = X**T   or
            op( X ) = X**H,

        alpha and beta are scalars, and A, B, and C are matrices, with
        op( A ) an m by k matrix, op( B ) a k by n matrix and C is a m by n matrix.

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

        Note for int8 users - For rocBLAS backend, please read rocblas_gemm_ex documentation on int8
        data layout requirements. hipBLAS makes the assumption that the data layout is in the preferred
        format for a given device as documented in rocBLAS.

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        transA    [hipblasOperation_t]
                  specifies the form of op( A ).
        @param[in]
        transB    [hipblasOperation_t]
                  specifies the form of op( B ).
        @param[in]
        m         [int]
                  matrix dimension m.
        @param[in]
        n         [int]
                  matrix dimension n.
        @param[in]
        k         [int]
                  matrix dimension k.
        @param[in]
        alpha     [const void *]
                  device pointer or host pointer specifying the scalar alpha. Same datatype as computeType.
        @param[in]
        A         [void *]
                  device pointer storing matrix A.
        @param[in]
        aType    [hipblasDatatype_t]
                  specifies the datatype of matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
        @param[in]
        B         [void *]
                  device pointer storing matrix B.
        @param[in]
        bType    [hipblasDatatype_t]
                  specifies the datatype of matrix B.
        @param[in]
        ldb       [int]
                  specifies the leading dimension of B.
        @param[in]
        beta      [const void *]
                  device pointer or host pointer specifying the scalar beta. Same datatype as computeType.
        @param[in]
        C         [void *]
                  device pointer storing matrix C.
        @param[in]
        cType    [hipblasDatatype_t]
                  specifies the datatype of matrix C.
        @param[in]
        ldc       [int]
                  specifies the leading dimension of C.
        @param[in]
        computeType
                  [hipblasDatatype_t]
                  specifies the datatype of computation.
        @param[in]
        algo      [hipblasGemmAlgo_t]
                  enumerant specifying the algorithm type.
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")                    
    if not isinstance(aType,hipblasDatatype_t):
        raise TypeError("argument 'aType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(bType,hipblasDatatype_t):
        raise TypeError("argument 'bType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(cType,hipblasDatatype_t):
        raise TypeError("argument 'cType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(computeType,hipblasDatatype_t):
        raise TypeError("argument 'computeType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(algo,hipblasGemmAlgo_t):
        raise TypeError("argument 'algo' must be of type 'hipblasGemmAlgo_t'")
    pass

@cython.embedsignature(True)
def hipblasTrsmEx(hipblasHandle_t handle, object side, object uplo, object transA, object diag, int m, int n, A, int lda, B, int ldb, invA, int invAsize, object computeType):
    """! BLAS EX API

        \details
        trsmEx solves

            op(A)*X = alpha*B or X*op(A) = alpha*B,

        where alpha is a scalar, X and B are m by n matrices,
        A is triangular matrix and op(A) is one of

            op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

        The matrix X is overwritten on B.

        This function gives the user the ability to reuse the invA matrix between runs.
        If invA == NULL, hipblasTrsmEx will automatically calculate invA on every run.

        Setting up invA:
        The accepted invA matrix consists of the packed 128x128 inverses of the diagonal blocks of
        matrix A, followed by any smaller diagonal block that remains.
        To set up invA it is recommended that hipblasTrtriBatched be used with matrix A as the input.

        Device memory of size 128 x k should be allocated for invA ahead of time, where k is m when
        HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT. The actual number of elements in invA
        should be passed as invAsize.

        To begin, hipblasTrtriBatched must be called on the full 128x128 sized diagonal blocks of
        matrix A. Below are the restricted parameters:
          - n = 128
          - ldinvA = 128
          - stride_invA = 128x128
          - batchCount = k / 128,

        Then any remaining block may be added:
          - n = k % 128
          - invA = invA + stride_invA * previousBatchCount
          - ldinvA = 128
          - batchCount = 1

        @param[in]
        handle  [hipblasHandle_t]
                handle to the hipblas library context queue.

        @param[in]
        side    [hipblasSideMode_t]
                HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
                HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a lower triangular matrix.

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_N: op(A) = A.
                HIPBLAS_OP_T: op(A) = A^T.
                HIPBLAS_ON_C: op(A) = A^H.

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
                HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.

        @param[in]
        m       [int]
                m specifies the number of rows of B. m >= 0.

        @param[in]
        n       [int]
                n specifies the number of columns of B. n >= 0.

        @param[in]
        alpha   [void *]
                device pointer or host pointer specifying the scalar alpha. When alpha is
                &zero then A is not referenced, and B need not be set before
                entry.

        @param[in]
        A       [void *]
                device pointer storing matrix A.
                of dimension ( lda, k ), where k is m
                when HIPBLAS_SIDE_LEFT and
                is n when HIPBLAS_SIDE_RIGHT
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
                if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).

        @param[in, out]
        B       [void *]
                device pointer storing matrix B.
                B is of dimension ( ldb, n ).
                Before entry, the leading m by n part of the array B must
                contain the right-hand side matrix B, and on exit is
                overwritten by the solution matrix X.

        @param[in]
        ldb    [int]
               ldb specifies the first dimension of B. ldb >= max( 1, m ).

        @param[in]
        invA    [void *]
                device pointer storing the inverse diagonal blocks of A.
                invA is of dimension ( ld_invA, k ), where k is m
                when HIPBLAS_SIDE_LEFT and
                is n when HIPBLAS_SIDE_RIGHT.
                ld_invA must be equal to 128.

        @param[in]
        invAsize [int]
                invAsize specifies the number of elements of device memory in invA.

        @param[in]
        computeType [hipblasDatatype_t]
                specifies the datatype of computation
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")                    
    if not isinstance(computeType,hipblasDatatype_t):
        raise TypeError("argument 'computeType' must be of type 'hipblasDatatype_t'")
    pass

@cython.embedsignature(True)
def hipblasAxpyEx(hipblasHandle_t handle, int n, object alphaType, x, object xType, int incx, y, object yType, int incy, object executionType):
    """! \brief BLAS EX API

        \details
        axpyEx computes constant alpha multiplied by vector x, plus vector y

            y := alpha * x + y

            - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x and y.
        @param[in]
        alpha     device pointer or host pointer to specify the scalar alpha.
        @param[in]
        alphaType [hipblasDatatype_t]
                  specifies the datatype of alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        xType [hipblasDatatype_t]
               specifies the datatype of vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        yType [hipblasDatatype_t]
              specifies the datatype of vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[in]
        executionType [hipblasDatatype_t]
                      specifies the datatype of computation.
    """
    if not isinstance(alphaType,hipblasDatatype_t):
        raise TypeError("argument 'alphaType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(xType,hipblasDatatype_t):
        raise TypeError("argument 'xType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(yType,hipblasDatatype_t):
        raise TypeError("argument 'yType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(executionType,hipblasDatatype_t):
        raise TypeError("argument 'executionType' must be of type 'hipblasDatatype_t'")
    pass

@cython.embedsignature(True)
def hipblasDotEx(hipblasHandle_t handle, int n, x, object xType, int incx, y, object yType, int incy, result, object resultType, object executionType):
    """! @{
        \brief BLAS EX API

        \details
        dotEx  performs the dot product of vectors x and y

            result = x * y;

        dotcEx  performs the dot product of the conjugate of complex vector x and complex vector y

            result = conjugate (x) * y;

            - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x and y.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        xType [hipblasDatatype_t]
               specifies the datatype of vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of y.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        yType [hipblasDatatype_t]
              specifies the datatype of vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        result
                  device pointer or host pointer to store the dot product.
                  return is 0.0 if n <= 0.
        @param[in]
        resultType [hipblasDatatype_t]
                    specifies the datatype of the result.
        @param[in]
        executionType [hipblasDatatype_t]
                      specifies the datatype of computation.
    """
    if not isinstance(xType,hipblasDatatype_t):
        raise TypeError("argument 'xType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(yType,hipblasDatatype_t):
        raise TypeError("argument 'yType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(resultType,hipblasDatatype_t):
        raise TypeError("argument 'resultType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(executionType,hipblasDatatype_t):
        raise TypeError("argument 'executionType' must be of type 'hipblasDatatype_t'")
    pass

@cython.embedsignature(True)
def hipblasDotcEx(hipblasHandle_t handle, int n, x, object xType, int incx, y, object yType, int incy, result, object resultType, object executionType):
    """
    """
    if not isinstance(xType,hipblasDatatype_t):
        raise TypeError("argument 'xType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(yType,hipblasDatatype_t):
        raise TypeError("argument 'yType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(resultType,hipblasDatatype_t):
        raise TypeError("argument 'resultType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(executionType,hipblasDatatype_t):
        raise TypeError("argument 'executionType' must be of type 'hipblasDatatype_t'")
    pass

@cython.embedsignature(True)
def hipblasNrm2Ex(hipblasHandle_t handle, int n, x, object xType, int incx, result, object resultType, object executionType):
    """! \brief BLAS_EX API

        \details
        nrm2Ex computes the euclidean norm of a real or complex vector

                  result := sqrt( x'*x ) for real vectors
                  result := sqrt( x**H*x ) for complex vectors

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.


        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        xType [hipblasDatatype_t]
               specifies the datatype of the vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        result
                  device pointer or host pointer to store the nrm2 product.
                  return is 0.0 if n, incx<=0.
        @param[in]
        resultType [hipblasDatatype_t]
                    specifies the datatype of the result.
        @param[in]
        executionType [hipblasDatatype_t]
                      specifies the datatype of computation.
    """
    if not isinstance(xType,hipblasDatatype_t):
        raise TypeError("argument 'xType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(resultType,hipblasDatatype_t):
        raise TypeError("argument 'resultType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(executionType,hipblasDatatype_t):
        raise TypeError("argument 'executionType' must be of type 'hipblasDatatype_t'")
    pass

@cython.embedsignature(True)
def hipblasRotEx(hipblasHandle_t handle, int n, x, object xType, int incx, y, object yType, int incy, c, s, object csType, object executionType):
    """! \brief BLAS EX API

        \details
        rotEx applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
            Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.

        In the case where cs_type is real:
            x := c * x + s * y
                y := c * y - s * x

        In the case where cs_type is complex, the imaginary part of c is ignored:
            x := real(c) * x + s * y
                y := real(c) * y - conj(s) * x

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

        @param[in]
        handle  [hipblasHandle_t]
                handle to the hipblas library context queue.
        @param[in]
        n       [int]
                number of elements in the x and y vectors.
        @param[inout]
        x       device pointer storing vector x.
        @param[in]
        xType [hipblasDatatype_t]
               specifies the datatype of vector x.
        @param[in]
        incx    [int]
                specifies the increment between elements of x.
        @param[inout]
        y       device pointer storing vector y.
        @param[in]
        yType [hipblasDatatype_t]
               specifies the datatype of vector y.
        @param[in]
        incy    [int]
                specifies the increment between elements of y.
        @param[in]
        c       device pointer or host pointer storing scalar cosine component of the rotation matrix.
        @param[in]
        s       device pointer or host pointer storing scalar sine component of the rotation matrix.
        @param[in]
        csType [hipblasDatatype_t]
                specifies the datatype of c and s.
        @param[in]
        executionType [hipblasDatatype_t]
                       specifies the datatype of computation.
    """
    if not isinstance(xType,hipblasDatatype_t):
        raise TypeError("argument 'xType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(yType,hipblasDatatype_t):
        raise TypeError("argument 'yType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(csType,hipblasDatatype_t):
        raise TypeError("argument 'csType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(executionType,hipblasDatatype_t):
        raise TypeError("argument 'executionType' must be of type 'hipblasDatatype_t'")
    pass

@cython.embedsignature(True)
def hipblasScalEx(hipblasHandle_t handle, int n, object alphaType, x, object xType, int incx, object executionType):
    """! \brief BLAS EX API

        \details
        scalEx  scales each element of vector x with scalar alpha.

            x := alpha * x

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x.
        @param[in]
        alpha     device pointer or host pointer for the scalar alpha.
        @param[in]
        alphaType [hipblasDatatype_t]
                   specifies the datatype of alpha.
        @param[inout]
        x         device pointer storing vector x.
        @param[in]
        xType [hipblasDatatype_t]
               specifies the datatype of vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        executionType [hipblasDatatype_t]
                       specifies the datatype of computation.
    """
    if not isinstance(alphaType,hipblasDatatype_t):
        raise TypeError("argument 'alphaType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(xType,hipblasDatatype_t):
        raise TypeError("argument 'xType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(executionType,hipblasDatatype_t):
        raise TypeError("argument 'executionType' must be of type 'hipblasDatatype_t'")
    pass

@cython.embedsignature(True)
def hipblasStatusToString(object status):
    """! HIPBLAS Auxiliary API

        \details
        hipblasStatusToString

        Returns string representing hipblasStatus_t value

        @param[in]
        status  [hipblasStatus_t]
                hipBLAS status to convert to string
    """
    if not isinstance(status,hipblasStatus_t):
        raise TypeError("argument 'status' must be of type 'hipblasStatus_t'")
    cdef const char * hipblasStatusToString_____retval = chipblas.hipblasStatusToString(status.value)    # fully specified
