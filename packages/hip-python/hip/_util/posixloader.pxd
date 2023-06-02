# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cimport posix.dlfcn

cdef void* open_library(const char* path)
cdef void* close_library(void* handle) nogil
cdef void* load_symbol(void* handle, const char* name) nogil