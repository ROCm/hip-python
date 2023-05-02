# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cimport posix.dlfcn

cdef void* open_library(const char* path):
    """
    Note:
        Uses gil because of `bytes`, `RuntimeError`.
    """
    cdef void* handle = posix.dlfcn.dlopen(path, posix.dlfcn.RTLD_NOW)
    if handle == NULL:
        raise RuntimeError(f"failed to dlopen {path}")
    return handle

cdef void* close_library(void* handle) nogil:
    if handle == NULL:
        with gil:
            raise RuntimeError(f"handle is NULL")
    cdef int rtype = posix.dlfcn.dlclose(handle)
    if rtype != 0:
        with gil:
            raise RuntimeError(f"Failed to dclose given handle")
    return handle

cdef void* load_symbol(void* handle, const char* name) nogil:
    return posix.dlfcn.dlsym(handle, name)
