# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

cdef class Pointer:
    cdef void* _ptr
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    # Camel-case used by intent to make this orthogonal to array get_<property>(self,i)
    # of auto-generated subclasses.
    cdef void* getPtr(self)

    @staticmethod
    cdef Pointer from_ptr(void* ptr)

    cdef void init_from_pyobj(self, object pyobj)

    @staticmethod
    cdef Pointer from_pyobj(object pyobj)

cdef class CStr(Pointer):
    cdef bint _owner
    # These buffer protocol related arrays
    # have to stay alive as long
    # as any buffer views the data,
    # so we store them as members.
    cdef Py_ssize_t[1] _shape
    cdef Py_ssize_t[1] strides

    @staticmethod
    cdef CStr from_ptr(void* ptr)

    @staticmethod
    cdef CStr from_pyobj(object pyobj)

    cdef Py_ssize_t get_or_determine_len(self)

    cdef const char* get_element_ptr(self)

    cpdef void malloc(self,Py_ssize_t size_bytes)

    cpdef void free(self)

cdef class ImmortalCStr(CStr):

    @staticmethod
    cdef ImmortalCStr from_ptr(void* ptr)

    @staticmethod
    cdef ImmortalCStr from_pyobj(object pyobj)

cdef class DeviceArray(Pointer):
    cdef size_t _itemsize
    cdef dict __dict__

    @staticmethod
    cdef DeviceArray from_ptr(void* ptr)

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

    @staticmethod
    cdef ListOfPointer from_pyobj(object pyobj)

cdef class ListOfBytes(Pointer):
    cdef bint _owner

    @staticmethod
    cdef ListOfBytes from_ptr(void* ptr)

    @staticmethod
    cdef ListOfBytes from_pyobj(object pyobj)

cdef class ListOfInt(Pointer):
    cdef bint _owner

    @staticmethod
    cdef ListOfInt from_ptr(void* ptr)

    @staticmethod
    cdef ListOfInt from_pyobj(object pyobj)

cdef class ListOfUnsigned(Pointer):
    cdef bint _owner

    @staticmethod
    cdef ListOfUnsigned from_ptr(void* ptr)

    @staticmethod
    cdef ListOfUnsigned from_pyobj(object pyobj)

cdef class ListOfUnsignedLong(Pointer):
    cdef bint _owner

    @staticmethod
    cdef ListOfUnsignedLong from_ptr(void* ptr)

    @staticmethod
    cdef ListOfUnsignedLong from_pyobj(object pyobj)
