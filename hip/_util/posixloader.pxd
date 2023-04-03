cimport posix.dlfcn

cdef void* open_library(char* path)
cdef void* close_library(void* handle) nogil
cdef void* load_symbol(void* handle, char* name) nogil