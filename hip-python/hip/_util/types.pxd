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

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

cdef class Pointer:
    cdef void* _ptr
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    # Camel-case used by intent to make this orthogonal to array get_<property>(self,i)
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

    cpdef void malloc(self,Py_ssize_t size_bytes)

    cpdef void free(self)

cdef class ImmortalCStr(CStr):

    @staticmethod
    cdef ImmortalCStr fromPtr(void* ptr)

    @staticmethod
    cdef ImmortalCStr fromPyobj(object pyobj)

cdef class NDBuffer(Pointer):
    cdef size_t _itemsize # itemsize is not part of the CUDA array interface
    cdef dict __dict__
    cdef Py_ssize_t* _py_buffer_shape # for providing shape information
                                      # to viewers of this Python buffer
    cdef int __view_count # For counting the current number of views

    @staticmethod
    cdef NDBuffer fromPtr(void* ptr)

    @staticmethod
    cdef NDBuffer fromPyobj(object pyobj)

    cdef _set_ptr(self,void* ptr)

    cdef int _numpy_typestr_to_bytes(self, str typestr)

    cdef tuple _handle_int(self,size_t subscript, size_t shape_dim)

    cdef tuple _handle_slice(self,slice subscript,size_t shape_dim)

cdef class DeviceArray(NDBuffer):

    @staticmethod
    cdef DeviceArray fromPtr(void* ptr)

    @staticmethod
    cdef DeviceArray fromPyobj(object pyobj)

cdef class ListOfPointer(Pointer):
    cdef bint _is_ptr_owner

    @staticmethod
    cdef ListOfPointer fromPtr(void* ptr)

    @staticmethod
    cdef ListOfPointer fromPyobj(object pyobj)

cdef class ListOfBytes(Pointer):
    cdef bint _is_ptr_owner

    @staticmethod
    cdef ListOfBytes fromPtr(void* ptr)

    @staticmethod
    cdef ListOfBytes fromPyobj(object pyobj)

cdef class ListOfInt(Pointer):
    cdef bint _is_ptr_owner

    @staticmethod
    cdef ListOfInt fromPtr(void* ptr)

    @staticmethod
    cdef ListOfInt fromPyobj(object pyobj)

cdef class ListOfUnsigned(Pointer):
    cdef bint _is_ptr_owner

    @staticmethod
    cdef ListOfUnsigned fromPtr(void* ptr)

    @staticmethod
    cdef ListOfUnsigned fromPyobj(object pyobj)

cdef class ListOfUnsignedLong(Pointer):
    cdef bint _is_ptr_owner

    @staticmethod
    cdef ListOfUnsignedLong fromPtr(void* ptr)

    @staticmethod
    cdef ListOfUnsignedLong fromPyobj(object pyobj)
