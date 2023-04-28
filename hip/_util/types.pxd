# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cdef class DataHandle:
    cdef void* _ptr
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef DataHandle from_ptr(void* ptr)

    @staticmethod
    cdef DataHandle from_pyobj(object pyobj)

cdef class ListOfStr(DataHandle):
    cdef bint _owner
    cdef size_t _num_entries