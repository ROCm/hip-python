# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cdef class DataHandle:
    cdef void* _ptr
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef DataHandle from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)

    @staticmethod
    cdef DataHandle from_pyobj(object pyobj)

cdef class ListOfBytes(DataHandle):
    cdef bint _owner
    
    @staticmethod
    cdef ListOfBytes from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)
    
    @staticmethod
    cdef ListOfBytes from_pyobj(object pyobj)
