# AMD_COPYRIGHT
from libc cimport stdlib
from libc.stdint cimport *
cimport cpython.long
cimport cpython.buffer
cimport hip._util.types
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
from .hip cimport ihipStream_t, float2, double2

from . cimport chipfft
cdef class hipfftHandle_t:
    cdef chipfft.hipfftHandle_t* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipfftHandle_t from_ptr(chipfft.hipfftHandle_t* ptr, bint owner=*)
    @staticmethod
    cdef hipfftHandle_t from_pyobj(object pyobj)
