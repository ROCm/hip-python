# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cdef class Pointer:
    cdef void* _ptr
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef Pointer from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)

    @staticmethod
    cdef Pointer from_pyobj(object pyobj)

cdef class DeviceArray(Pointer):
    cdef size_t _itemsize
    cdef dict __dict__

    @staticmethod
    cdef DeviceArray from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)

    @staticmethod
    cdef DeviceArray from_pyobj(object pyobj)

    cdef _set_ptr(self,void* ptr)
    
    cdef int _numpy_typestr_to_bytes(self, str typestr)
    
    cdef tuple _handle_int(self,size_t subscript, size_t shape_dim)
    
    cdef tuple _handle_slice(self,slice subscript,size_t shape_dim)

cdef class ListOfPointer(Pointer):
    cdef bint _owner
    
    @staticmethod
    cdef ListOfPointer from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)
    
    @staticmethod
    cdef ListOfPointer from_pyobj(object pyobj)

cdef class ListOfBytes(Pointer):
    cdef bint _owner
    
    @staticmethod
    cdef ListOfBytes from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)
    
    @staticmethod
    cdef ListOfBytes from_pyobj(object pyobj)

cdef class ListOfInt(Pointer):
    cdef bint _owner
    
    @staticmethod
    cdef ListOfInt from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)
    
    @staticmethod
    cdef ListOfInt from_pyobj(object pyobj)

cdef class ListOfUnsigned(Pointer):
    cdef bint _owner
    
    @staticmethod
    cdef ListOfUnsigned from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)
    
    @staticmethod
    cdef ListOfUnsigned from_pyobj(object pyobj)

cdef class ListOfUnsignedLong(Pointer):
    cdef bint _owner
    
    @staticmethod
    cdef ListOfUnsignedLong from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)
    
    @staticmethod
    cdef ListOfUnsignedLong from_pyobj(object pyobj)
