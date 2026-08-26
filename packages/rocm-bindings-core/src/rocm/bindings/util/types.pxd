# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

from libc.string cimport memchr


cdef inline str to_str_n(const char* p, Py_ssize_t n):
    """Text up to the first NUL within ``n`` bytes, or all ``n`` bytes.

    For the fixed-size char fields of the ROCm structs, where the extent is
    the size of the buffer and says nothing about the length of the string in
    it. A vendor may or may not terminate the text, so both cases have to
    read the same, and neither may read beyond the field: ``memchr`` bounds
    the search the way ``strlen`` does not.

    Undecodable bytes become U+FFFD. Strict decoding would let a malformed
    vendor field raise ``UnicodeDecodeError`` out of a property getter, which
    is worse than a replacement character in a name or a version string.

    Note:
        The slice bound reaches Cython's own slice-decode path, which calls
        the UTF-8 codec directly with an explicit stop and so needs no
        intermediate ``bytes``.
    """
    cdef const char* nul = <const char*>memchr(p, 0, n)
    return p[:(n if nul == NULL else nul - p)].decode("utf-8", "replace")


cdef class Pointer:
    cdef void* _ptr
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    # Camel-case used by intent to make this orthogonal to get_<property>(self, i)
    # of auto-generated subclasses.
    cdef void* getPtr(self)

    cpdef Pointer createRef(self)

    @staticmethod
    cdef Pointer fromPtr(void* ptr)

    cdef void init_from_pyobj(self, object pyobj)

    @staticmethod
    cdef Pointer fromPyobj(object pyobj)

cdef class CStr(Pointer):
    cdef bint _is_ptr_owner
    # These buffer protocol related arrays
    # have to stay alive as long
    # as any buffer views the data,
    # so we store them as members.
    cdef Py_ssize_t[1] _shape

    @staticmethod
    cdef CStr fromPtr(void* ptr)

    @staticmethod
    cdef CStr fromPyobj(object pyobj)

    cdef Py_ssize_t get_or_determine_len(self)

    cdef const char* getElementPtr(self)

    cpdef void malloc(self, Py_ssize_t content_len)

    cpdef void free(self)

cdef class NDBuffer(Pointer):
    cdef size_t _itemsize  # itemsize is not part of the CUDA array interface
    cdef dict _cuda_array_interface
    cdef object _pybuffer_obj  # keeps a wrapped exporter alive
    cdef bytes _typestr_bytes  # NUL-terminated, outlives the Py_buffer
    cdef Py_ssize_t* _py_buffer_shape  # shape info for this Python buffer
    cdef int __view_count  # For counting the current number of views

    @staticmethod
    cdef NDBuffer fromPtr(void* ptr)

    @staticmethod
    cdef NDBuffer fromPyobj(object pyobj)

    cdef _set_ptr(self, void* ptr)

    cdef int _numpy_typestr_to_bytes(self, str typestr)

    cdef tuple _handle_int(self, size_t subscript, size_t shape_dim)

    cdef tuple _handle_slice(self, slice subscript, size_t shape_dim)

cdef class DeviceArray(NDBuffer):

    @staticmethod
    cdef DeviceArray fromPtr(void* ptr)

    @staticmethod
    cdef DeviceArray fromPyobj(object pyobj)

cdef class ListOfPointer(Pointer):
    cdef bint _is_ptr_owner
    cdef Py_ssize_t _len

    @staticmethod
    cdef ListOfPointer fromPtr(void* ptr)

    @staticmethod
    cdef ListOfPointer fromPyobj(object pyobj)

cdef class ListOfBytes(Pointer):
    cdef bint _is_ptr_owner
    cdef Py_ssize_t _len

    @staticmethod
    cdef ListOfBytes fromPtr(void* ptr)

    @staticmethod
    cdef ListOfBytes fromPyobj(object pyobj)

cdef class ListOfInt(Pointer):
    cdef bint _is_ptr_owner
    cdef Py_ssize_t _len

    @staticmethod
    cdef ListOfInt fromPtr(void* ptr)

    @staticmethod
    cdef ListOfInt fromPyobj(object pyobj)

cdef class ListOfLong(Pointer):
    cdef bint _is_ptr_owner
    cdef Py_ssize_t _len

    @staticmethod
    cdef ListOfLong fromPtr(void* ptr)

    @staticmethod
    cdef ListOfLong fromPyobj(object pyobj)

cdef class ListOfUnsigned(Pointer):
    cdef bint _is_ptr_owner
    cdef Py_ssize_t _len

    @staticmethod
    cdef ListOfUnsigned fromPtr(void* ptr)

    @staticmethod
    cdef ListOfUnsigned fromPyobj(object pyobj)

cdef class ListOfUnsignedLong(Pointer):
    cdef bint _is_ptr_owner
    cdef Py_ssize_t _len

    @staticmethod
    cdef ListOfUnsignedLong fromPtr(void* ptr)

    @staticmethod
    cdef ListOfUnsignedLong fromPyobj(object pyobj)

cdef class ListOfInt64(Pointer):
    cdef bint _is_ptr_owner
    cdef Py_ssize_t _len

    @staticmethod
    cdef ListOfInt64 fromPtr(void* ptr)

    @staticmethod
    cdef ListOfInt64 fromPyobj(object pyobj)

cdef class ListOfUInt64(Pointer):
    cdef bint _is_ptr_owner
    cdef Py_ssize_t _len

    @staticmethod
    cdef ListOfUInt64 fromPtr(void* ptr)

    @staticmethod
    cdef ListOfUInt64 fromPyobj(object pyobj)

cdef class PointerToInt(ListOfInt):

    @staticmethod
    cdef PointerToInt fromPtr(void* ptr)

    @staticmethod
    cdef PointerToInt fromPyobj(object pyobj)

cdef class PointerToLong(ListOfLong):

    @staticmethod
    cdef PointerToLong fromPtr(void* ptr)

    @staticmethod
    cdef PointerToLong fromPyobj(object pyobj)

cdef class PointerToUnsigned(ListOfUnsigned):

    @staticmethod
    cdef PointerToUnsigned fromPtr(void* ptr)

    @staticmethod
    cdef PointerToUnsigned fromPyobj(object pyobj)

cdef class PointerToUnsignedLong(ListOfUnsignedLong):

    @staticmethod
    cdef PointerToUnsignedLong fromPtr(void* ptr)

    @staticmethod
    cdef PointerToUnsignedLong fromPyobj(object pyobj)

cdef class PointerToInt64(ListOfInt64):

    @staticmethod
    cdef PointerToInt64 fromPtr(void* ptr)

    @staticmethod
    cdef PointerToInt64 fromPyobj(object pyobj)

cdef class PointerToUInt64(ListOfUInt64):

    @staticmethod
    cdef PointerToUInt64 fromPtr(void* ptr)

    @staticmethod
    cdef PointerToUInt64 fromPyobj(object pyobj)
