# AMD_COPYRIGHT
# c imports
from libc cimport stdlib
from libc.stdint cimport *
cimport cpython.long
cimport cpython.buffer
# python imports
import cython
import ctypes
import enum
cimport hip._util.types

from . cimport chiprtc
cdef class ihiprtcLinkState:
    cdef chiprtc.ihiprtcLinkState* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ihiprtcLinkState from_ptr(chiprtc.ihiprtcLinkState* ptr, bint owner=*)
    @staticmethod
    cdef ihiprtcLinkState from_pyobj(object pyobj)


cdef class _hiprtcProgram:
    cdef chiprtc._hiprtcProgram* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef _hiprtcProgram from_ptr(chiprtc._hiprtcProgram* ptr, bint owner=*)
    @staticmethod
    cdef _hiprtcProgram from_pyobj(object pyobj)
