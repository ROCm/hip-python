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
from libc cimport stdlib
from libc cimport string
from libc.stdint cimport *
cimport cpython.long
cimport cpython.buffer
cimport hip._util.types
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
from .hip cimport ihipStream_t

from . cimport chipblas
cdef class hipblasBfloat16:
    cdef chipblas.hipblasBfloat16* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipblasBfloat16 from_ptr(chipblas.hipblasBfloat16* ptr, bint owner=*)
    @staticmethod
    cdef hipblasBfloat16 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chipblas.hipblasBfloat16** ptr)
    @staticmethod
    cdef hipblasBfloat16 new()
    @staticmethod
    cdef hipblasBfloat16 from_value(chipblas.hipblasBfloat16 other)


cdef class hipblasComplex:
    cdef chipblas.hipblasComplex* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipblasComplex from_ptr(chipblas.hipblasComplex* ptr, bint owner=*)
    @staticmethod
    cdef hipblasComplex from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chipblas.hipblasComplex** ptr)
    @staticmethod
    cdef hipblasComplex new()
    @staticmethod
    cdef hipblasComplex from_value(chipblas.hipblasComplex other)


cdef class hipblasDoubleComplex:
    cdef chipblas.hipblasDoubleComplex* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipblasDoubleComplex from_ptr(chipblas.hipblasDoubleComplex* ptr, bint owner=*)
    @staticmethod
    cdef hipblasDoubleComplex from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chipblas.hipblasDoubleComplex** ptr)
    @staticmethod
    cdef hipblasDoubleComplex new()
    @staticmethod
    cdef hipblasDoubleComplex from_value(chipblas.hipblasDoubleComplex other)
