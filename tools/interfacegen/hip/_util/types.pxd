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

cdef class Array(DataHandle):
    cdef size_t _itemsize
    cdef dict __cuda_array_interface__

    @staticmethod
    cdef Array from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)

    @staticmethod
    cdef Array from_pyobj(object pyobj)

    cdef _set_ptr(self,void* ptr)
    
    cdef int _numpy_typestr_to_bytes(self, str typestr)
    
    cdef tuple _handle_int(self,size_t subscript, size_t shape_dim)
    
    cdef tuple _handle_slice(self,slice subscript,size_t shape_dim)

cdef class ListOfDataHandle(DataHandle):
    cdef bint _owner
    
    @staticmethod
    cdef ListOfDataHandle from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)
    
    @staticmethod
    cdef ListOfDataHandle from_pyobj(object pyobj)

cdef class ListOfBytes(DataHandle):
    cdef bint _owner
    
    @staticmethod
    cdef ListOfBytes from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)
    
    @staticmethod
    cdef ListOfBytes from_pyobj(object pyobj)

cdef class ListOfInt(DataHandle):
    cdef bint _owner
    
    @staticmethod
    cdef ListOfInt from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)
    
    @staticmethod
    cdef ListOfInt from_pyobj(object pyobj)

cdef class ListOfUnsigned(DataHandle):
    cdef bint _owner
    
    @staticmethod
    cdef ListOfUnsigned from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)
    
    @staticmethod
    cdef ListOfUnsigned from_pyobj(object pyobj)

cdef class ListOfUnsignedLong(DataHandle):
    cdef bint _owner
    
    @staticmethod
    cdef ListOfUnsignedLong from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)
    
    @staticmethod
    cdef ListOfUnsignedLong from_pyobj(object pyobj)
