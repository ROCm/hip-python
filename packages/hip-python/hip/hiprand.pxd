# AMD_COPYRIGHT
from libc cimport stdlib
from libc.stdint cimport *
cimport cpython.long
cimport cpython.buffer
cimport hip._util.types
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
from .hip cimport ihipStream_t

from . cimport chiprand
cdef class uint4:
    cdef chiprand.uint4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef uint4 from_ptr(chiprand.uint4* ptr, bint owner=*)
    @staticmethod
    cdef uint4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chiprand.uint4** ptr)
    @staticmethod
    cdef uint4 new()


cdef class rocrand_discrete_distribution_st:
    cdef chiprand.rocrand_discrete_distribution_st* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef rocrand_discrete_distribution_st from_ptr(chiprand.rocrand_discrete_distribution_st* ptr, bint owner=*)
    @staticmethod
    cdef rocrand_discrete_distribution_st from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chiprand.rocrand_discrete_distribution_st** ptr)
    @staticmethod
    cdef rocrand_discrete_distribution_st new()


cdef class rocrand_generator_base_type:
    cdef chiprand.rocrand_generator_base_type* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef rocrand_generator_base_type from_ptr(chiprand.rocrand_generator_base_type* ptr, bint owner=*)
    @staticmethod
    cdef rocrand_generator_base_type from_pyobj(object pyobj)
