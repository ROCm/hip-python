# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cdef class DataHandle:
    # intended use: <target>._ptr
    cdef void* _ptr
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef DataHandle from_ptr(void* ptr)

    @staticmethod
    cdef DataHandle from_pyobj(object pyobj)