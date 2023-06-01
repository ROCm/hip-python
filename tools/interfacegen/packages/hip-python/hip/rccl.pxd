# AMD_COPYRIGHT
from libc cimport stdlib
from libc.stdint cimport *
cimport cpython.long
cimport cpython.buffer
cimport hip._util.types
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
from .hip cimport ihipStream_t

from . cimport crccl
cdef class ncclComm:
    cdef crccl.ncclComm* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ncclComm from_ptr(crccl.ncclComm* ptr, bint owner=*)
    @staticmethod
    cdef ncclComm from_pyobj(object pyobj)


cdef class ncclUniqueId:
    cdef crccl.ncclUniqueId* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ncclUniqueId from_ptr(crccl.ncclUniqueId* ptr, bint owner=*)
    @staticmethod
    cdef ncclUniqueId from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(crccl.ncclUniqueId** ptr)
    @staticmethod
    cdef ncclUniqueId new()
