# AMD_COPYRIGHT
from libc cimport stdlib
from libc.stdint cimport *
cimport cpython.long
cimport cpython.buffer
cimport hip._util.types
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
#ctypedef int16_t __int16_t
#ctypedef uint16_t __uint16_t
from .hip cimport ihipStream_t

from . cimport chipblas
ctypedef short __int16_t

ctypedef unsigned short __uint16_t

cdef class hipblasHandle_t:
    cdef void * _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipblasHandle_t from_ptr(void * ptr, bint owner=*)
    @staticmethod
    cdef hipblasHandle_t from_pyobj(object pyobj)


ctypedef uint16_t hipblasHalf

ctypedef int8_t hipblasInt8

ctypedef int64_t hipblasStride

cdef class hipblasBfloat16:
    cdef chipblas.hipblasBfloat16* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipblasBfloat16 from_ptr(chipblas.hipblasBfloat16* ptr, bint owner=*)
    @staticmethod
    cdef hipblasBfloat16 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chipblas.hipblasBfloat16** ptr)
    @staticmethod
    cdef hipblasBfloat16 new()


cdef class hipblasComplex:
    cdef chipblas.hipblasComplex* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipblasComplex from_ptr(chipblas.hipblasComplex* ptr, bint owner=*)
    @staticmethod
    cdef hipblasComplex from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chipblas.hipblasComplex** ptr)
    @staticmethod
    cdef hipblasComplex new()


cdef class hipblasDoubleComplex:
    cdef chipblas.hipblasDoubleComplex* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipblasDoubleComplex from_ptr(chipblas.hipblasDoubleComplex* ptr, bint owner=*)
    @staticmethod
    cdef hipblasDoubleComplex from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chipblas.hipblasDoubleComplex** ptr)
    @staticmethod
    cdef hipblasDoubleComplex new()
