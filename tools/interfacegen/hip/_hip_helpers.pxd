# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cimport hip._util.types

cdef class HipModuleLaunchKernel_extra(hip._util.types.DataHandle):
    cdef bint _owner
    cdef void* _config[5]
    cdef size_t _offset
    
    cdef size_t _aligned_size(self, size_t size, size_t factor)
    
    @staticmethod
    cdef HipModuleLaunchKernel_extra from_ptr(void* ptr)
    
    cdef void init_from_pyobj(self, object pyobj)
    
    @staticmethod
    cdef HipModuleLaunchKernel_extra from_pyobj(object pyobj)
