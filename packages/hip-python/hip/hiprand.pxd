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

from . cimport chiprand
cdef class uint4:
    cdef chiprand.uint4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef uint4 from_ptr(chiprand.uint4* ptr, bint owner=*)
    @staticmethod
    cdef uint4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chiprand.uint4** ptr)
    @staticmethod
    cdef uint4 new()
    @staticmethod
    cdef uint4 from_value(chiprand.uint4 other)


cdef class rocrand_discrete_distribution_st:
    cdef chiprand.rocrand_discrete_distribution_st* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef rocrand_discrete_distribution_st from_ptr(chiprand.rocrand_discrete_distribution_st* ptr, bint owner=*)
    @staticmethod
    cdef rocrand_discrete_distribution_st from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chiprand.rocrand_discrete_distribution_st** ptr)
    @staticmethod
    cdef rocrand_discrete_distribution_st new()
    @staticmethod
    cdef rocrand_discrete_distribution_st from_value(chiprand.rocrand_discrete_distribution_st other)


cdef class rocrand_generator_base_type:
    cdef chiprand.rocrand_generator_base_type* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef rocrand_generator_base_type from_ptr(chiprand.rocrand_generator_base_type* ptr, bint owner=*)
    @staticmethod
    cdef rocrand_generator_base_type from_pyobj(object pyobj)
