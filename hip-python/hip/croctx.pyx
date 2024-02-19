# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file has been autogenerated, do not modify.

cimport hip._util.posixloader as loader
cdef void* _lib_handle = NULL

DLL = b"libroctx64.so"

cdef int __init() except 1 nogil:
    global DLL
    global _lib_handle
    cdef char* dll = NULL
    if _lib_handle == NULL:
        with gil:
            dll = DLL
        return loader.open_library(&_lib_handle,dll)
    return 0

cdef int __init_symbol(void** result, const char* name) except 1 nogil:
    global _lib_handle
    cdef int init_result = 0
    if _lib_handle == NULL:
        init_result = __init()
        if init_result > 0:
            return init_result
    if result[0] == NULL:
        return loader.load_symbol(result,_lib_handle, name)
    return 0


cdef void* _roctx_version_major__funptr = NULL
# 
# Query the major version of the installed library.
# 
# Return the major version of the installed library. This can be used to check
# if it is compatible with this interface version.
# 
# \return Returns the major version number.
cdef unsigned int roctx_version_major():
    global _roctx_version_major__funptr
    if __init_symbol(&_roctx_version_major__funptr,"roctx_version_major") > 0:
        pass
    return (<unsigned int (*)() noexcept nogil> _roctx_version_major__funptr)()


cdef void* _roctx_version_minor__funptr = NULL
# 
# Query the minor version of the installed library.
# 
# Return the minor version of the installed library. This can be used to check
# if it is compatible with this interface version.
# 
# \return Returns the minor version number.
cdef unsigned int roctx_version_minor():
    global _roctx_version_minor__funptr
    if __init_symbol(&_roctx_version_minor__funptr,"roctx_version_minor") > 0:
        pass
    return (<unsigned int (*)() noexcept nogil> _roctx_version_minor__funptr)()


cdef void* _roctxMarkA__funptr = NULL
# 
# Mark an event.
# 
# \param[in] message The message associated with the event.
cdef void roctxMarkA(const char * message):
    global _roctxMarkA__funptr
    if __init_symbol(&_roctxMarkA__funptr,"roctxMarkA") > 0:
        pass
    (<void (*)(const char *) noexcept nogil> _roctxMarkA__funptr)(message)


cdef void* _roctxRangePushA__funptr = NULL
# 
# Start a new nested range.
# 
# Nested ranges are stacked and local to the current CPU thread.
# 
# \param[in] message The message associated with this range.
# 
# \return Returns the level this nested range is started at. Nested range
# levels are 0 based.
cdef int roctxRangePushA(const char * message):
    global _roctxRangePushA__funptr
    if __init_symbol(&_roctxRangePushA__funptr,"roctxRangePushA") > 0:
        pass
    return (<int (*)(const char *) noexcept nogil> _roctxRangePushA__funptr)(message)


cdef void* _roctxRangePop__funptr = NULL
# 
# Stop the current nested range.
# 
# Stop the current nested range, and pop it from the stack. If a nested range
# was active before the last one was started, it becomes again the current
# nested range.
# 
# \return Returns the level the stopped nested range was started at, or a
# negative value if there was no nested range active.
cdef int roctxRangePop():
    global _roctxRangePop__funptr
    if __init_symbol(&_roctxRangePop__funptr,"roctxRangePop") > 0:
        pass
    return (<int (*)() noexcept nogil> _roctxRangePop__funptr)()


cdef void* _roctxRangeStartA__funptr = NULL
# 
# Starts a process range.
# 
# Start/stop ranges can be started and stopped in different threads. Each
# timespan is assigned a unique range ID.
# 
# \param[in] message The message associated with this range.
# 
# \return Returns the ID of the new range.
cdef unsigned long roctxRangeStartA(const char * message):
    global _roctxRangeStartA__funptr
    if __init_symbol(&_roctxRangeStartA__funptr,"roctxRangeStartA") > 0:
        pass
    return (<unsigned long (*)(const char *) noexcept nogil> _roctxRangeStartA__funptr)(message)


cdef void* _roctxRangeStop__funptr = NULL
# 
# Stop a process range.
cdef void roctxRangeStop(unsigned long id):
    global _roctxRangeStop__funptr
    if __init_symbol(&_roctxRangeStop__funptr,"roctxRangeStop") > 0:
        pass
    (<void (*)(unsigned long) noexcept nogil> _roctxRangeStop__funptr)(id)
