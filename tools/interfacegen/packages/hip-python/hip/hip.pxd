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

from . cimport chip
cdef class hipDeviceArch_t:
    cdef chip.hipDeviceArch_t* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipDeviceArch_t from_ptr(chip.hipDeviceArch_t* ptr, bint owner=*)
    @staticmethod
    cdef hipDeviceArch_t from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipDeviceArch_t** ptr)
    @staticmethod
    cdef hipDeviceArch_t new()
    @staticmethod
    cdef hipDeviceArch_t from_value(chip.hipDeviceArch_t other)


cdef class hipUUID_t:
    cdef chip.hipUUID_t* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipUUID_t from_ptr(chip.hipUUID_t* ptr, bint owner=*)
    @staticmethod
    cdef hipUUID_t from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipUUID_t** ptr)
    @staticmethod
    cdef hipUUID_t new()
    @staticmethod
    cdef hipUUID_t from_value(chip.hipUUID_t other)


cdef class hipDeviceProp_t:
    cdef chip.hipDeviceProp_t* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipDeviceProp_t from_ptr(chip.hipDeviceProp_t* ptr, bint owner=*)
    @staticmethod
    cdef hipDeviceProp_t from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipDeviceProp_t** ptr)
    @staticmethod
    cdef hipDeviceProp_t new()
    @staticmethod
    cdef hipDeviceProp_t from_value(chip.hipDeviceProp_t other)


cdef class hipPointerAttribute_t_union_0:
    cdef chip.hipPointerAttribute_t_union_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipPointerAttribute_t_union_0 from_ptr(chip.hipPointerAttribute_t_union_0* ptr, bint owner=*)
    @staticmethod
    cdef hipPointerAttribute_t_union_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipPointerAttribute_t_union_0** ptr)
    @staticmethod
    cdef hipPointerAttribute_t_union_0 new()
    @staticmethod
    cdef hipPointerAttribute_t_union_0 from_value(chip.hipPointerAttribute_t_union_0 other)


cdef class hipPointerAttribute_t:
    cdef chip.hipPointerAttribute_t* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipPointerAttribute_t from_ptr(chip.hipPointerAttribute_t* ptr, bint owner=*)
    @staticmethod
    cdef hipPointerAttribute_t from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipPointerAttribute_t** ptr)
    @staticmethod
    cdef hipPointerAttribute_t new()
    @staticmethod
    cdef hipPointerAttribute_t from_value(chip.hipPointerAttribute_t other)


cdef class hipChannelFormatDesc:
    cdef chip.hipChannelFormatDesc* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipChannelFormatDesc from_ptr(chip.hipChannelFormatDesc* ptr, bint owner=*)
    @staticmethod
    cdef hipChannelFormatDesc from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipChannelFormatDesc** ptr)
    @staticmethod
    cdef hipChannelFormatDesc new()
    @staticmethod
    cdef hipChannelFormatDesc from_value(chip.hipChannelFormatDesc other)


cdef class HIP_ARRAY_DESCRIPTOR:
    cdef chip.HIP_ARRAY_DESCRIPTOR* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef HIP_ARRAY_DESCRIPTOR from_ptr(chip.HIP_ARRAY_DESCRIPTOR* ptr, bint owner=*)
    @staticmethod
    cdef HIP_ARRAY_DESCRIPTOR from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.HIP_ARRAY_DESCRIPTOR** ptr)
    @staticmethod
    cdef HIP_ARRAY_DESCRIPTOR new()
    @staticmethod
    cdef HIP_ARRAY_DESCRIPTOR from_value(chip.HIP_ARRAY_DESCRIPTOR other)


cdef class HIP_ARRAY3D_DESCRIPTOR:
    cdef chip.HIP_ARRAY3D_DESCRIPTOR* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef HIP_ARRAY3D_DESCRIPTOR from_ptr(chip.HIP_ARRAY3D_DESCRIPTOR* ptr, bint owner=*)
    @staticmethod
    cdef HIP_ARRAY3D_DESCRIPTOR from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.HIP_ARRAY3D_DESCRIPTOR** ptr)
    @staticmethod
    cdef HIP_ARRAY3D_DESCRIPTOR new()
    @staticmethod
    cdef HIP_ARRAY3D_DESCRIPTOR from_value(chip.HIP_ARRAY3D_DESCRIPTOR other)


cdef class hipArray:
    cdef chip.hipArray* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipArray from_ptr(chip.hipArray* ptr, bint owner=*)
    @staticmethod
    cdef hipArray from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipArray** ptr)
    @staticmethod
    cdef hipArray new()
    @staticmethod
    cdef hipArray from_value(chip.hipArray other)


cdef class hip_Memcpy2D:
    cdef chip.hip_Memcpy2D* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hip_Memcpy2D from_ptr(chip.hip_Memcpy2D* ptr, bint owner=*)
    @staticmethod
    cdef hip_Memcpy2D from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hip_Memcpy2D** ptr)
    @staticmethod
    cdef hip_Memcpy2D new()
    @staticmethod
    cdef hip_Memcpy2D from_value(chip.hip_Memcpy2D other)


cdef class hipMipmappedArray:
    cdef chip.hipMipmappedArray* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipMipmappedArray from_ptr(chip.hipMipmappedArray* ptr, bint owner=*)
    @staticmethod
    cdef hipMipmappedArray from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipMipmappedArray** ptr)
    @staticmethod
    cdef hipMipmappedArray new()
    @staticmethod
    cdef hipMipmappedArray from_value(chip.hipMipmappedArray other)


cdef class HIP_TEXTURE_DESC_st:
    cdef chip.HIP_TEXTURE_DESC_st* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef HIP_TEXTURE_DESC_st from_ptr(chip.HIP_TEXTURE_DESC_st* ptr, bint owner=*)
    @staticmethod
    cdef HIP_TEXTURE_DESC_st from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.HIP_TEXTURE_DESC_st** ptr)
    @staticmethod
    cdef HIP_TEXTURE_DESC_st new()
    @staticmethod
    cdef HIP_TEXTURE_DESC_st from_value(chip.HIP_TEXTURE_DESC_st other)


cdef class hipResourceDesc_union_0_struct_0:
    cdef chip.hipResourceDesc_union_0_struct_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipResourceDesc_union_0_struct_0 from_ptr(chip.hipResourceDesc_union_0_struct_0* ptr, bint owner=*)
    @staticmethod
    cdef hipResourceDesc_union_0_struct_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipResourceDesc_union_0_struct_0** ptr)
    @staticmethod
    cdef hipResourceDesc_union_0_struct_0 new()
    @staticmethod
    cdef hipResourceDesc_union_0_struct_0 from_value(chip.hipResourceDesc_union_0_struct_0 other)


cdef class hipResourceDesc_union_0_struct_1:
    cdef chip.hipResourceDesc_union_0_struct_1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipResourceDesc_union_0_struct_1 from_ptr(chip.hipResourceDesc_union_0_struct_1* ptr, bint owner=*)
    @staticmethod
    cdef hipResourceDesc_union_0_struct_1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipResourceDesc_union_0_struct_1** ptr)
    @staticmethod
    cdef hipResourceDesc_union_0_struct_1 new()
    @staticmethod
    cdef hipResourceDesc_union_0_struct_1 from_value(chip.hipResourceDesc_union_0_struct_1 other)


cdef class hipResourceDesc_union_0_struct_2:
    cdef chip.hipResourceDesc_union_0_struct_2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipResourceDesc_union_0_struct_2 from_ptr(chip.hipResourceDesc_union_0_struct_2* ptr, bint owner=*)
    @staticmethod
    cdef hipResourceDesc_union_0_struct_2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipResourceDesc_union_0_struct_2** ptr)
    @staticmethod
    cdef hipResourceDesc_union_0_struct_2 new()
    @staticmethod
    cdef hipResourceDesc_union_0_struct_2 from_value(chip.hipResourceDesc_union_0_struct_2 other)


cdef class hipResourceDesc_union_0_struct_3:
    cdef chip.hipResourceDesc_union_0_struct_3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipResourceDesc_union_0_struct_3 from_ptr(chip.hipResourceDesc_union_0_struct_3* ptr, bint owner=*)
    @staticmethod
    cdef hipResourceDesc_union_0_struct_3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipResourceDesc_union_0_struct_3** ptr)
    @staticmethod
    cdef hipResourceDesc_union_0_struct_3 new()
    @staticmethod
    cdef hipResourceDesc_union_0_struct_3 from_value(chip.hipResourceDesc_union_0_struct_3 other)


cdef class hipResourceDesc_union_0:
    cdef chip.hipResourceDesc_union_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipResourceDesc_union_0 from_ptr(chip.hipResourceDesc_union_0* ptr, bint owner=*)
    @staticmethod
    cdef hipResourceDesc_union_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipResourceDesc_union_0** ptr)
    @staticmethod
    cdef hipResourceDesc_union_0 new()
    @staticmethod
    cdef hipResourceDesc_union_0 from_value(chip.hipResourceDesc_union_0 other)


cdef class hipResourceDesc:
    cdef chip.hipResourceDesc* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipResourceDesc from_ptr(chip.hipResourceDesc* ptr, bint owner=*)
    @staticmethod
    cdef hipResourceDesc from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipResourceDesc** ptr)
    @staticmethod
    cdef hipResourceDesc new()
    @staticmethod
    cdef hipResourceDesc from_value(chip.hipResourceDesc other)


cdef class HIP_RESOURCE_DESC_st_union_0_struct_0:
    cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_0 from_ptr(chip.HIP_RESOURCE_DESC_st_union_0_struct_0* ptr, bint owner=*)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.HIP_RESOURCE_DESC_st_union_0_struct_0** ptr)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_0 new()
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_0 from_value(chip.HIP_RESOURCE_DESC_st_union_0_struct_0 other)


cdef class HIP_RESOURCE_DESC_st_union_0_struct_1:
    cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_1 from_ptr(chip.HIP_RESOURCE_DESC_st_union_0_struct_1* ptr, bint owner=*)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.HIP_RESOURCE_DESC_st_union_0_struct_1** ptr)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_1 new()
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_1 from_value(chip.HIP_RESOURCE_DESC_st_union_0_struct_1 other)


cdef class HIP_RESOURCE_DESC_st_union_0_struct_2:
    cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_2 from_ptr(chip.HIP_RESOURCE_DESC_st_union_0_struct_2* ptr, bint owner=*)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.HIP_RESOURCE_DESC_st_union_0_struct_2** ptr)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_2 new()
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_2 from_value(chip.HIP_RESOURCE_DESC_st_union_0_struct_2 other)


cdef class HIP_RESOURCE_DESC_st_union_0_struct_3:
    cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_3 from_ptr(chip.HIP_RESOURCE_DESC_st_union_0_struct_3* ptr, bint owner=*)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.HIP_RESOURCE_DESC_st_union_0_struct_3** ptr)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_3 new()
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_3 from_value(chip.HIP_RESOURCE_DESC_st_union_0_struct_3 other)


cdef class HIP_RESOURCE_DESC_st_union_0_struct_4:
    cdef chip.HIP_RESOURCE_DESC_st_union_0_struct_4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_4 from_ptr(chip.HIP_RESOURCE_DESC_st_union_0_struct_4* ptr, bint owner=*)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.HIP_RESOURCE_DESC_st_union_0_struct_4** ptr)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_4 new()
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0_struct_4 from_value(chip.HIP_RESOURCE_DESC_st_union_0_struct_4 other)


cdef class HIP_RESOURCE_DESC_st_union_0:
    cdef chip.HIP_RESOURCE_DESC_st_union_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0 from_ptr(chip.HIP_RESOURCE_DESC_st_union_0* ptr, bint owner=*)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.HIP_RESOURCE_DESC_st_union_0** ptr)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0 new()
    @staticmethod
    cdef HIP_RESOURCE_DESC_st_union_0 from_value(chip.HIP_RESOURCE_DESC_st_union_0 other)


cdef class HIP_RESOURCE_DESC_st:
    cdef chip.HIP_RESOURCE_DESC_st* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef HIP_RESOURCE_DESC_st from_ptr(chip.HIP_RESOURCE_DESC_st* ptr, bint owner=*)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.HIP_RESOURCE_DESC_st** ptr)
    @staticmethod
    cdef HIP_RESOURCE_DESC_st new()
    @staticmethod
    cdef HIP_RESOURCE_DESC_st from_value(chip.HIP_RESOURCE_DESC_st other)


cdef class hipResourceViewDesc:
    cdef chip.hipResourceViewDesc* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipResourceViewDesc from_ptr(chip.hipResourceViewDesc* ptr, bint owner=*)
    @staticmethod
    cdef hipResourceViewDesc from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipResourceViewDesc** ptr)
    @staticmethod
    cdef hipResourceViewDesc new()
    @staticmethod
    cdef hipResourceViewDesc from_value(chip.hipResourceViewDesc other)


cdef class HIP_RESOURCE_VIEW_DESC_st:
    cdef chip.HIP_RESOURCE_VIEW_DESC_st* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef HIP_RESOURCE_VIEW_DESC_st from_ptr(chip.HIP_RESOURCE_VIEW_DESC_st* ptr, bint owner=*)
    @staticmethod
    cdef HIP_RESOURCE_VIEW_DESC_st from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.HIP_RESOURCE_VIEW_DESC_st** ptr)
    @staticmethod
    cdef HIP_RESOURCE_VIEW_DESC_st new()
    @staticmethod
    cdef HIP_RESOURCE_VIEW_DESC_st from_value(chip.HIP_RESOURCE_VIEW_DESC_st other)


cdef class hipPitchedPtr:
    cdef chip.hipPitchedPtr* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipPitchedPtr from_ptr(chip.hipPitchedPtr* ptr, bint owner=*)
    @staticmethod
    cdef hipPitchedPtr from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipPitchedPtr** ptr)
    @staticmethod
    cdef hipPitchedPtr new()
    @staticmethod
    cdef hipPitchedPtr from_value(chip.hipPitchedPtr other)


cdef class hipExtent:
    cdef chip.hipExtent* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExtent from_ptr(chip.hipExtent* ptr, bint owner=*)
    @staticmethod
    cdef hipExtent from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExtent** ptr)
    @staticmethod
    cdef hipExtent new()
    @staticmethod
    cdef hipExtent from_value(chip.hipExtent other)


cdef class hipPos:
    cdef chip.hipPos* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipPos from_ptr(chip.hipPos* ptr, bint owner=*)
    @staticmethod
    cdef hipPos from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipPos** ptr)
    @staticmethod
    cdef hipPos new()
    @staticmethod
    cdef hipPos from_value(chip.hipPos other)


cdef class hipMemcpy3DParms:
    cdef chip.hipMemcpy3DParms* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipMemcpy3DParms from_ptr(chip.hipMemcpy3DParms* ptr, bint owner=*)
    @staticmethod
    cdef hipMemcpy3DParms from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipMemcpy3DParms** ptr)
    @staticmethod
    cdef hipMemcpy3DParms new()
    @staticmethod
    cdef hipMemcpy3DParms from_value(chip.hipMemcpy3DParms other)


cdef class HIP_MEMCPY3D:
    cdef chip.HIP_MEMCPY3D* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef HIP_MEMCPY3D from_ptr(chip.HIP_MEMCPY3D* ptr, bint owner=*)
    @staticmethod
    cdef HIP_MEMCPY3D from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.HIP_MEMCPY3D** ptr)
    @staticmethod
    cdef HIP_MEMCPY3D new()
    @staticmethod
    cdef HIP_MEMCPY3D from_value(chip.HIP_MEMCPY3D other)


cdef class uchar1:
    cdef chip.uchar1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef uchar1 from_ptr(chip.uchar1* ptr, bint owner=*)
    @staticmethod
    cdef uchar1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.uchar1** ptr)
    @staticmethod
    cdef uchar1 new()
    @staticmethod
    cdef uchar1 from_value(chip.uchar1 other)


cdef class uchar2:
    cdef chip.uchar2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef uchar2 from_ptr(chip.uchar2* ptr, bint owner=*)
    @staticmethod
    cdef uchar2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.uchar2** ptr)
    @staticmethod
    cdef uchar2 new()
    @staticmethod
    cdef uchar2 from_value(chip.uchar2 other)


cdef class uchar3:
    cdef chip.uchar3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef uchar3 from_ptr(chip.uchar3* ptr, bint owner=*)
    @staticmethod
    cdef uchar3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.uchar3** ptr)
    @staticmethod
    cdef uchar3 new()
    @staticmethod
    cdef uchar3 from_value(chip.uchar3 other)


cdef class uchar4:
    cdef chip.uchar4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef uchar4 from_ptr(chip.uchar4* ptr, bint owner=*)
    @staticmethod
    cdef uchar4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.uchar4** ptr)
    @staticmethod
    cdef uchar4 new()
    @staticmethod
    cdef uchar4 from_value(chip.uchar4 other)


cdef class char1:
    cdef chip.char1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef char1 from_ptr(chip.char1* ptr, bint owner=*)
    @staticmethod
    cdef char1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.char1** ptr)
    @staticmethod
    cdef char1 new()
    @staticmethod
    cdef char1 from_value(chip.char1 other)


cdef class char2:
    cdef chip.char2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef char2 from_ptr(chip.char2* ptr, bint owner=*)
    @staticmethod
    cdef char2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.char2** ptr)
    @staticmethod
    cdef char2 new()
    @staticmethod
    cdef char2 from_value(chip.char2 other)


cdef class char3:
    cdef chip.char3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef char3 from_ptr(chip.char3* ptr, bint owner=*)
    @staticmethod
    cdef char3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.char3** ptr)
    @staticmethod
    cdef char3 new()
    @staticmethod
    cdef char3 from_value(chip.char3 other)


cdef class char4:
    cdef chip.char4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef char4 from_ptr(chip.char4* ptr, bint owner=*)
    @staticmethod
    cdef char4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.char4** ptr)
    @staticmethod
    cdef char4 new()
    @staticmethod
    cdef char4 from_value(chip.char4 other)


cdef class ushort1:
    cdef chip.ushort1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ushort1 from_ptr(chip.ushort1* ptr, bint owner=*)
    @staticmethod
    cdef ushort1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.ushort1** ptr)
    @staticmethod
    cdef ushort1 new()
    @staticmethod
    cdef ushort1 from_value(chip.ushort1 other)


cdef class ushort2:
    cdef chip.ushort2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ushort2 from_ptr(chip.ushort2* ptr, bint owner=*)
    @staticmethod
    cdef ushort2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.ushort2** ptr)
    @staticmethod
    cdef ushort2 new()
    @staticmethod
    cdef ushort2 from_value(chip.ushort2 other)


cdef class ushort3:
    cdef chip.ushort3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ushort3 from_ptr(chip.ushort3* ptr, bint owner=*)
    @staticmethod
    cdef ushort3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.ushort3** ptr)
    @staticmethod
    cdef ushort3 new()
    @staticmethod
    cdef ushort3 from_value(chip.ushort3 other)


cdef class ushort4:
    cdef chip.ushort4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ushort4 from_ptr(chip.ushort4* ptr, bint owner=*)
    @staticmethod
    cdef ushort4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.ushort4** ptr)
    @staticmethod
    cdef ushort4 new()
    @staticmethod
    cdef ushort4 from_value(chip.ushort4 other)


cdef class short1:
    cdef chip.short1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef short1 from_ptr(chip.short1* ptr, bint owner=*)
    @staticmethod
    cdef short1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.short1** ptr)
    @staticmethod
    cdef short1 new()
    @staticmethod
    cdef short1 from_value(chip.short1 other)


cdef class short2:
    cdef chip.short2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef short2 from_ptr(chip.short2* ptr, bint owner=*)
    @staticmethod
    cdef short2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.short2** ptr)
    @staticmethod
    cdef short2 new()
    @staticmethod
    cdef short2 from_value(chip.short2 other)


cdef class short3:
    cdef chip.short3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef short3 from_ptr(chip.short3* ptr, bint owner=*)
    @staticmethod
    cdef short3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.short3** ptr)
    @staticmethod
    cdef short3 new()
    @staticmethod
    cdef short3 from_value(chip.short3 other)


cdef class short4:
    cdef chip.short4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef short4 from_ptr(chip.short4* ptr, bint owner=*)
    @staticmethod
    cdef short4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.short4** ptr)
    @staticmethod
    cdef short4 new()
    @staticmethod
    cdef short4 from_value(chip.short4 other)


cdef class uint1:
    cdef chip.uint1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef uint1 from_ptr(chip.uint1* ptr, bint owner=*)
    @staticmethod
    cdef uint1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.uint1** ptr)
    @staticmethod
    cdef uint1 new()
    @staticmethod
    cdef uint1 from_value(chip.uint1 other)


cdef class uint2:
    cdef chip.uint2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef uint2 from_ptr(chip.uint2* ptr, bint owner=*)
    @staticmethod
    cdef uint2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.uint2** ptr)
    @staticmethod
    cdef uint2 new()
    @staticmethod
    cdef uint2 from_value(chip.uint2 other)


cdef class uint3:
    cdef chip.uint3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef uint3 from_ptr(chip.uint3* ptr, bint owner=*)
    @staticmethod
    cdef uint3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.uint3** ptr)
    @staticmethod
    cdef uint3 new()
    @staticmethod
    cdef uint3 from_value(chip.uint3 other)


cdef class uint4:
    cdef chip.uint4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef uint4 from_ptr(chip.uint4* ptr, bint owner=*)
    @staticmethod
    cdef uint4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.uint4** ptr)
    @staticmethod
    cdef uint4 new()
    @staticmethod
    cdef uint4 from_value(chip.uint4 other)


cdef class int1:
    cdef chip.int1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef int1 from_ptr(chip.int1* ptr, bint owner=*)
    @staticmethod
    cdef int1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.int1** ptr)
    @staticmethod
    cdef int1 new()
    @staticmethod
    cdef int1 from_value(chip.int1 other)


cdef class int2:
    cdef chip.int2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef int2 from_ptr(chip.int2* ptr, bint owner=*)
    @staticmethod
    cdef int2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.int2** ptr)
    @staticmethod
    cdef int2 new()
    @staticmethod
    cdef int2 from_value(chip.int2 other)


cdef class int3:
    cdef chip.int3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef int3 from_ptr(chip.int3* ptr, bint owner=*)
    @staticmethod
    cdef int3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.int3** ptr)
    @staticmethod
    cdef int3 new()
    @staticmethod
    cdef int3 from_value(chip.int3 other)


cdef class int4:
    cdef chip.int4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef int4 from_ptr(chip.int4* ptr, bint owner=*)
    @staticmethod
    cdef int4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.int4** ptr)
    @staticmethod
    cdef int4 new()
    @staticmethod
    cdef int4 from_value(chip.int4 other)


cdef class ulong1:
    cdef chip.ulong1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ulong1 from_ptr(chip.ulong1* ptr, bint owner=*)
    @staticmethod
    cdef ulong1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.ulong1** ptr)
    @staticmethod
    cdef ulong1 new()
    @staticmethod
    cdef ulong1 from_value(chip.ulong1 other)


cdef class ulong2:
    cdef chip.ulong2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ulong2 from_ptr(chip.ulong2* ptr, bint owner=*)
    @staticmethod
    cdef ulong2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.ulong2** ptr)
    @staticmethod
    cdef ulong2 new()
    @staticmethod
    cdef ulong2 from_value(chip.ulong2 other)


cdef class ulong3:
    cdef chip.ulong3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ulong3 from_ptr(chip.ulong3* ptr, bint owner=*)
    @staticmethod
    cdef ulong3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.ulong3** ptr)
    @staticmethod
    cdef ulong3 new()
    @staticmethod
    cdef ulong3 from_value(chip.ulong3 other)


cdef class ulong4:
    cdef chip.ulong4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ulong4 from_ptr(chip.ulong4* ptr, bint owner=*)
    @staticmethod
    cdef ulong4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.ulong4** ptr)
    @staticmethod
    cdef ulong4 new()
    @staticmethod
    cdef ulong4 from_value(chip.ulong4 other)


cdef class long1:
    cdef chip.long1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef long1 from_ptr(chip.long1* ptr, bint owner=*)
    @staticmethod
    cdef long1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.long1** ptr)
    @staticmethod
    cdef long1 new()
    @staticmethod
    cdef long1 from_value(chip.long1 other)


cdef class long2:
    cdef chip.long2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef long2 from_ptr(chip.long2* ptr, bint owner=*)
    @staticmethod
    cdef long2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.long2** ptr)
    @staticmethod
    cdef long2 new()
    @staticmethod
    cdef long2 from_value(chip.long2 other)


cdef class long3:
    cdef chip.long3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef long3 from_ptr(chip.long3* ptr, bint owner=*)
    @staticmethod
    cdef long3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.long3** ptr)
    @staticmethod
    cdef long3 new()
    @staticmethod
    cdef long3 from_value(chip.long3 other)


cdef class long4:
    cdef chip.long4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef long4 from_ptr(chip.long4* ptr, bint owner=*)
    @staticmethod
    cdef long4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.long4** ptr)
    @staticmethod
    cdef long4 new()
    @staticmethod
    cdef long4 from_value(chip.long4 other)


cdef class ulonglong1:
    cdef chip.ulonglong1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ulonglong1 from_ptr(chip.ulonglong1* ptr, bint owner=*)
    @staticmethod
    cdef ulonglong1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.ulonglong1** ptr)
    @staticmethod
    cdef ulonglong1 new()
    @staticmethod
    cdef ulonglong1 from_value(chip.ulonglong1 other)


cdef class ulonglong2:
    cdef chip.ulonglong2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ulonglong2 from_ptr(chip.ulonglong2* ptr, bint owner=*)
    @staticmethod
    cdef ulonglong2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.ulonglong2** ptr)
    @staticmethod
    cdef ulonglong2 new()
    @staticmethod
    cdef ulonglong2 from_value(chip.ulonglong2 other)


cdef class ulonglong3:
    cdef chip.ulonglong3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ulonglong3 from_ptr(chip.ulonglong3* ptr, bint owner=*)
    @staticmethod
    cdef ulonglong3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.ulonglong3** ptr)
    @staticmethod
    cdef ulonglong3 new()
    @staticmethod
    cdef ulonglong3 from_value(chip.ulonglong3 other)


cdef class ulonglong4:
    cdef chip.ulonglong4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ulonglong4 from_ptr(chip.ulonglong4* ptr, bint owner=*)
    @staticmethod
    cdef ulonglong4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.ulonglong4** ptr)
    @staticmethod
    cdef ulonglong4 new()
    @staticmethod
    cdef ulonglong4 from_value(chip.ulonglong4 other)


cdef class longlong1:
    cdef chip.longlong1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef longlong1 from_ptr(chip.longlong1* ptr, bint owner=*)
    @staticmethod
    cdef longlong1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.longlong1** ptr)
    @staticmethod
    cdef longlong1 new()
    @staticmethod
    cdef longlong1 from_value(chip.longlong1 other)


cdef class longlong2:
    cdef chip.longlong2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef longlong2 from_ptr(chip.longlong2* ptr, bint owner=*)
    @staticmethod
    cdef longlong2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.longlong2** ptr)
    @staticmethod
    cdef longlong2 new()
    @staticmethod
    cdef longlong2 from_value(chip.longlong2 other)


cdef class longlong3:
    cdef chip.longlong3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef longlong3 from_ptr(chip.longlong3* ptr, bint owner=*)
    @staticmethod
    cdef longlong3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.longlong3** ptr)
    @staticmethod
    cdef longlong3 new()
    @staticmethod
    cdef longlong3 from_value(chip.longlong3 other)


cdef class longlong4:
    cdef chip.longlong4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef longlong4 from_ptr(chip.longlong4* ptr, bint owner=*)
    @staticmethod
    cdef longlong4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.longlong4** ptr)
    @staticmethod
    cdef longlong4 new()
    @staticmethod
    cdef longlong4 from_value(chip.longlong4 other)


cdef class float1:
    cdef chip.float1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef float1 from_ptr(chip.float1* ptr, bint owner=*)
    @staticmethod
    cdef float1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.float1** ptr)
    @staticmethod
    cdef float1 new()
    @staticmethod
    cdef float1 from_value(chip.float1 other)


cdef class float2:
    cdef chip.float2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef float2 from_ptr(chip.float2* ptr, bint owner=*)
    @staticmethod
    cdef float2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.float2** ptr)
    @staticmethod
    cdef float2 new()
    @staticmethod
    cdef float2 from_value(chip.float2 other)


cdef class float3:
    cdef chip.float3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef float3 from_ptr(chip.float3* ptr, bint owner=*)
    @staticmethod
    cdef float3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.float3** ptr)
    @staticmethod
    cdef float3 new()
    @staticmethod
    cdef float3 from_value(chip.float3 other)


cdef class float4:
    cdef chip.float4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef float4 from_ptr(chip.float4* ptr, bint owner=*)
    @staticmethod
    cdef float4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.float4** ptr)
    @staticmethod
    cdef float4 new()
    @staticmethod
    cdef float4 from_value(chip.float4 other)


cdef class double1:
    cdef chip.double1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef double1 from_ptr(chip.double1* ptr, bint owner=*)
    @staticmethod
    cdef double1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.double1** ptr)
    @staticmethod
    cdef double1 new()
    @staticmethod
    cdef double1 from_value(chip.double1 other)


cdef class double2:
    cdef chip.double2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef double2 from_ptr(chip.double2* ptr, bint owner=*)
    @staticmethod
    cdef double2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.double2** ptr)
    @staticmethod
    cdef double2 new()
    @staticmethod
    cdef double2 from_value(chip.double2 other)


cdef class double3:
    cdef chip.double3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef double3 from_ptr(chip.double3* ptr, bint owner=*)
    @staticmethod
    cdef double3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.double3** ptr)
    @staticmethod
    cdef double3 new()
    @staticmethod
    cdef double3 from_value(chip.double3 other)


cdef class double4:
    cdef chip.double4* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef double4 from_ptr(chip.double4* ptr, bint owner=*)
    @staticmethod
    cdef double4 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.double4** ptr)
    @staticmethod
    cdef double4 new()
    @staticmethod
    cdef double4 from_value(chip.double4 other)


cdef class __hip_texture:
    cdef chip.__hip_texture* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef __hip_texture from_ptr(chip.__hip_texture* ptr, bint owner=*)
    @staticmethod
    cdef __hip_texture from_pyobj(object pyobj)


cdef class textureReference:
    cdef chip.textureReference* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef textureReference from_ptr(chip.textureReference* ptr, bint owner=*)
    @staticmethod
    cdef textureReference from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.textureReference** ptr)
    @staticmethod
    cdef textureReference new()
    @staticmethod
    cdef textureReference from_value(chip.textureReference other)


cdef class hipTextureDesc:
    cdef chip.hipTextureDesc* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipTextureDesc from_ptr(chip.hipTextureDesc* ptr, bint owner=*)
    @staticmethod
    cdef hipTextureDesc from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipTextureDesc** ptr)
    @staticmethod
    cdef hipTextureDesc new()
    @staticmethod
    cdef hipTextureDesc from_value(chip.hipTextureDesc other)


cdef class __hip_surface:
    cdef chip.__hip_surface* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef __hip_surface from_ptr(chip.__hip_surface* ptr, bint owner=*)
    @staticmethod
    cdef __hip_surface from_pyobj(object pyobj)


cdef class surfaceReference:
    cdef chip.surfaceReference* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef surfaceReference from_ptr(chip.surfaceReference* ptr, bint owner=*)
    @staticmethod
    cdef surfaceReference from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.surfaceReference** ptr)
    @staticmethod
    cdef surfaceReference new()
    @staticmethod
    cdef surfaceReference from_value(chip.surfaceReference other)


cdef class ihipCtx_t:
    cdef chip.ihipCtx_t* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ihipCtx_t from_ptr(chip.ihipCtx_t* ptr, bint owner=*)
    @staticmethod
    cdef ihipCtx_t from_pyobj(object pyobj)


cdef class ihipStream_t:
    cdef chip.ihipStream_t* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ihipStream_t from_ptr(chip.ihipStream_t* ptr, bint owner=*)
    @staticmethod
    cdef ihipStream_t from_pyobj(object pyobj)


cdef class hipIpcMemHandle_st:
    cdef chip.hipIpcMemHandle_st* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipIpcMemHandle_st from_ptr(chip.hipIpcMemHandle_st* ptr, bint owner=*)
    @staticmethod
    cdef hipIpcMemHandle_st from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipIpcMemHandle_st** ptr)
    @staticmethod
    cdef hipIpcMemHandle_st new()
    @staticmethod
    cdef hipIpcMemHandle_st from_value(chip.hipIpcMemHandle_st other)


cdef class hipIpcEventHandle_st:
    cdef chip.hipIpcEventHandle_st* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipIpcEventHandle_st from_ptr(chip.hipIpcEventHandle_st* ptr, bint owner=*)
    @staticmethod
    cdef hipIpcEventHandle_st from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipIpcEventHandle_st** ptr)
    @staticmethod
    cdef hipIpcEventHandle_st new()
    @staticmethod
    cdef hipIpcEventHandle_st from_value(chip.hipIpcEventHandle_st other)


cdef class ihipModule_t:
    cdef chip.ihipModule_t* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ihipModule_t from_ptr(chip.ihipModule_t* ptr, bint owner=*)
    @staticmethod
    cdef ihipModule_t from_pyobj(object pyobj)


cdef class ihipModuleSymbol_t:
    cdef chip.ihipModuleSymbol_t* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ihipModuleSymbol_t from_ptr(chip.ihipModuleSymbol_t* ptr, bint owner=*)
    @staticmethod
    cdef ihipModuleSymbol_t from_pyobj(object pyobj)


cdef class ihipMemPoolHandle_t:
    cdef chip.ihipMemPoolHandle_t* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ihipMemPoolHandle_t from_ptr(chip.ihipMemPoolHandle_t* ptr, bint owner=*)
    @staticmethod
    cdef ihipMemPoolHandle_t from_pyobj(object pyobj)


cdef class hipFuncAttributes:
    cdef chip.hipFuncAttributes* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipFuncAttributes from_ptr(chip.hipFuncAttributes* ptr, bint owner=*)
    @staticmethod
    cdef hipFuncAttributes from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipFuncAttributes** ptr)
    @staticmethod
    cdef hipFuncAttributes new()
    @staticmethod
    cdef hipFuncAttributes from_value(chip.hipFuncAttributes other)


cdef class ihipEvent_t:
    cdef chip.ihipEvent_t* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ihipEvent_t from_ptr(chip.ihipEvent_t* ptr, bint owner=*)
    @staticmethod
    cdef ihipEvent_t from_pyobj(object pyobj)


cdef class hipMemLocation:
    cdef chip.hipMemLocation* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipMemLocation from_ptr(chip.hipMemLocation* ptr, bint owner=*)
    @staticmethod
    cdef hipMemLocation from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipMemLocation** ptr)
    @staticmethod
    cdef hipMemLocation new()
    @staticmethod
    cdef hipMemLocation from_value(chip.hipMemLocation other)


cdef class hipMemAccessDesc:
    cdef chip.hipMemAccessDesc* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipMemAccessDesc from_ptr(chip.hipMemAccessDesc* ptr, bint owner=*)
    @staticmethod
    cdef hipMemAccessDesc from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipMemAccessDesc** ptr)
    @staticmethod
    cdef hipMemAccessDesc new()
    @staticmethod
    cdef hipMemAccessDesc from_value(chip.hipMemAccessDesc other)


cdef class hipMemPoolProps:
    cdef chip.hipMemPoolProps* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipMemPoolProps from_ptr(chip.hipMemPoolProps* ptr, bint owner=*)
    @staticmethod
    cdef hipMemPoolProps from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipMemPoolProps** ptr)
    @staticmethod
    cdef hipMemPoolProps new()
    @staticmethod
    cdef hipMemPoolProps from_value(chip.hipMemPoolProps other)


cdef class hipMemPoolPtrExportData:
    cdef chip.hipMemPoolPtrExportData* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipMemPoolPtrExportData from_ptr(chip.hipMemPoolPtrExportData* ptr, bint owner=*)
    @staticmethod
    cdef hipMemPoolPtrExportData from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipMemPoolPtrExportData** ptr)
    @staticmethod
    cdef hipMemPoolPtrExportData new()
    @staticmethod
    cdef hipMemPoolPtrExportData from_value(chip.hipMemPoolPtrExportData other)


cdef class dim3:
    cdef chip.dim3* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef dim3 from_ptr(chip.dim3* ptr, bint owner=*)
    @staticmethod
    cdef dim3 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.dim3** ptr)
    @staticmethod
    cdef dim3 new()
    @staticmethod
    cdef dim3 from_value(chip.dim3 other)


cdef class hipLaunchParams_t:
    cdef chip.hipLaunchParams_t* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipLaunchParams_t from_ptr(chip.hipLaunchParams_t* ptr, bint owner=*)
    @staticmethod
    cdef hipLaunchParams_t from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipLaunchParams_t** ptr)
    @staticmethod
    cdef hipLaunchParams_t new()
    @staticmethod
    cdef hipLaunchParams_t from_value(chip.hipLaunchParams_t other)


cdef class hipFunctionLaunchParams_t:
    cdef chip.hipFunctionLaunchParams_t* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipFunctionLaunchParams_t from_ptr(chip.hipFunctionLaunchParams_t* ptr, bint owner=*)
    @staticmethod
    cdef hipFunctionLaunchParams_t from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipFunctionLaunchParams_t** ptr)
    @staticmethod
    cdef hipFunctionLaunchParams_t new()
    @staticmethod
    cdef hipFunctionLaunchParams_t from_value(chip.hipFunctionLaunchParams_t other)


cdef class hipExternalMemoryHandleDesc_st_union_0_struct_0:
    cdef chip.hipExternalMemoryHandleDesc_st_union_0_struct_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalMemoryHandleDesc_st_union_0_struct_0 from_ptr(chip.hipExternalMemoryHandleDesc_st_union_0_struct_0* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalMemoryHandleDesc_st_union_0_struct_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalMemoryHandleDesc_st_union_0_struct_0** ptr)
    @staticmethod
    cdef hipExternalMemoryHandleDesc_st_union_0_struct_0 new()
    @staticmethod
    cdef hipExternalMemoryHandleDesc_st_union_0_struct_0 from_value(chip.hipExternalMemoryHandleDesc_st_union_0_struct_0 other)


cdef class hipExternalMemoryHandleDesc_st_union_0:
    cdef chip.hipExternalMemoryHandleDesc_st_union_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalMemoryHandleDesc_st_union_0 from_ptr(chip.hipExternalMemoryHandleDesc_st_union_0* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalMemoryHandleDesc_st_union_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalMemoryHandleDesc_st_union_0** ptr)
    @staticmethod
    cdef hipExternalMemoryHandleDesc_st_union_0 new()
    @staticmethod
    cdef hipExternalMemoryHandleDesc_st_union_0 from_value(chip.hipExternalMemoryHandleDesc_st_union_0 other)


cdef class hipExternalMemoryHandleDesc_st:
    cdef chip.hipExternalMemoryHandleDesc_st* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalMemoryHandleDesc_st from_ptr(chip.hipExternalMemoryHandleDesc_st* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalMemoryHandleDesc_st from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalMemoryHandleDesc_st** ptr)
    @staticmethod
    cdef hipExternalMemoryHandleDesc_st new()
    @staticmethod
    cdef hipExternalMemoryHandleDesc_st from_value(chip.hipExternalMemoryHandleDesc_st other)


cdef class hipExternalMemoryBufferDesc_st:
    cdef chip.hipExternalMemoryBufferDesc_st* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalMemoryBufferDesc_st from_ptr(chip.hipExternalMemoryBufferDesc_st* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalMemoryBufferDesc_st from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalMemoryBufferDesc_st** ptr)
    @staticmethod
    cdef hipExternalMemoryBufferDesc_st new()
    @staticmethod
    cdef hipExternalMemoryBufferDesc_st from_value(chip.hipExternalMemoryBufferDesc_st other)


cdef class hipExternalSemaphoreHandleDesc_st_union_0_struct_0:
    cdef chip.hipExternalSemaphoreHandleDesc_st_union_0_struct_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st_union_0_struct_0 from_ptr(chip.hipExternalSemaphoreHandleDesc_st_union_0_struct_0* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st_union_0_struct_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalSemaphoreHandleDesc_st_union_0_struct_0** ptr)
    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st_union_0_struct_0 new()
    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st_union_0_struct_0 from_value(chip.hipExternalSemaphoreHandleDesc_st_union_0_struct_0 other)


cdef class hipExternalSemaphoreHandleDesc_st_union_0:
    cdef chip.hipExternalSemaphoreHandleDesc_st_union_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st_union_0 from_ptr(chip.hipExternalSemaphoreHandleDesc_st_union_0* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st_union_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalSemaphoreHandleDesc_st_union_0** ptr)
    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st_union_0 new()
    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st_union_0 from_value(chip.hipExternalSemaphoreHandleDesc_st_union_0 other)


cdef class hipExternalSemaphoreHandleDesc_st:
    cdef chip.hipExternalSemaphoreHandleDesc_st* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st from_ptr(chip.hipExternalSemaphoreHandleDesc_st* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalSemaphoreHandleDesc_st** ptr)
    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st new()
    @staticmethod
    cdef hipExternalSemaphoreHandleDesc_st from_value(chip.hipExternalSemaphoreHandleDesc_st other)


cdef class hipExternalSemaphoreSignalParams_st_struct_0_struct_0:
    cdef chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_0 from_ptr(chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_0* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_0** ptr)
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_0 new()
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_0 from_value(chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_0 other)


cdef class hipExternalSemaphoreSignalParams_st_struct_0_struct_1:
    cdef chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_1 from_ptr(chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_1* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_1** ptr)
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_1 new()
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0_struct_1 from_value(chip.hipExternalSemaphoreSignalParams_st_struct_0_struct_1 other)


cdef class hipExternalSemaphoreSignalParams_st_struct_0:
    cdef chip.hipExternalSemaphoreSignalParams_st_struct_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0 from_ptr(chip.hipExternalSemaphoreSignalParams_st_struct_0* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalSemaphoreSignalParams_st_struct_0** ptr)
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0 new()
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st_struct_0 from_value(chip.hipExternalSemaphoreSignalParams_st_struct_0 other)


cdef class hipExternalSemaphoreSignalParams_st:
    cdef chip.hipExternalSemaphoreSignalParams_st* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st from_ptr(chip.hipExternalSemaphoreSignalParams_st* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalSemaphoreSignalParams_st** ptr)
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st new()
    @staticmethod
    cdef hipExternalSemaphoreSignalParams_st from_value(chip.hipExternalSemaphoreSignalParams_st other)


cdef class hipExternalSemaphoreWaitParams_st_struct_0_struct_0:
    cdef chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_0 from_ptr(chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_0* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_0** ptr)
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_0 new()
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_0 from_value(chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_0 other)


cdef class hipExternalSemaphoreWaitParams_st_struct_0_struct_1:
    cdef chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_1 from_ptr(chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_1* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_1** ptr)
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_1 new()
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0_struct_1 from_value(chip.hipExternalSemaphoreWaitParams_st_struct_0_struct_1 other)


cdef class hipExternalSemaphoreWaitParams_st_struct_0:
    cdef chip.hipExternalSemaphoreWaitParams_st_struct_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0 from_ptr(chip.hipExternalSemaphoreWaitParams_st_struct_0* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalSemaphoreWaitParams_st_struct_0** ptr)
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0 new()
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st_struct_0 from_value(chip.hipExternalSemaphoreWaitParams_st_struct_0 other)


cdef class hipExternalSemaphoreWaitParams_st:
    cdef chip.hipExternalSemaphoreWaitParams_st* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st from_ptr(chip.hipExternalSemaphoreWaitParams_st* ptr, bint owner=*)
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipExternalSemaphoreWaitParams_st** ptr)
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st new()
    @staticmethod
    cdef hipExternalSemaphoreWaitParams_st from_value(chip.hipExternalSemaphoreWaitParams_st other)


cdef class _hipGraphicsResource:
    cdef chip._hipGraphicsResource* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef _hipGraphicsResource from_ptr(chip._hipGraphicsResource* ptr, bint owner=*)
    @staticmethod
    cdef _hipGraphicsResource from_pyobj(object pyobj)


cdef class ihipGraph:
    cdef chip.ihipGraph* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ihipGraph from_ptr(chip.ihipGraph* ptr, bint owner=*)
    @staticmethod
    cdef ihipGraph from_pyobj(object pyobj)


cdef class hipGraphNode:
    cdef chip.hipGraphNode* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipGraphNode from_ptr(chip.hipGraphNode* ptr, bint owner=*)
    @staticmethod
    cdef hipGraphNode from_pyobj(object pyobj)


cdef class hipGraphExec:
    cdef chip.hipGraphExec* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipGraphExec from_ptr(chip.hipGraphExec* ptr, bint owner=*)
    @staticmethod
    cdef hipGraphExec from_pyobj(object pyobj)


cdef class hipUserObject:
    cdef chip.hipUserObject* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipUserObject from_ptr(chip.hipUserObject* ptr, bint owner=*)
    @staticmethod
    cdef hipUserObject from_pyobj(object pyobj)


cdef class hipHostFn_t:
    cdef chip.hipHostFn_t _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipHostFn_t from_ptr(chip.hipHostFn_t ptr, bint owner=*)
    @staticmethod
    cdef hipHostFn_t from_pyobj(object pyobj)


cdef class hipHostNodeParams:
    cdef chip.hipHostNodeParams* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipHostNodeParams from_ptr(chip.hipHostNodeParams* ptr, bint owner=*)
    @staticmethod
    cdef hipHostNodeParams from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipHostNodeParams** ptr)
    @staticmethod
    cdef hipHostNodeParams new()
    @staticmethod
    cdef hipHostNodeParams from_value(chip.hipHostNodeParams other)


cdef class hipKernelNodeParams:
    cdef chip.hipKernelNodeParams* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipKernelNodeParams from_ptr(chip.hipKernelNodeParams* ptr, bint owner=*)
    @staticmethod
    cdef hipKernelNodeParams from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipKernelNodeParams** ptr)
    @staticmethod
    cdef hipKernelNodeParams new()
    @staticmethod
    cdef hipKernelNodeParams from_value(chip.hipKernelNodeParams other)


cdef class hipMemsetParams:
    cdef chip.hipMemsetParams* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipMemsetParams from_ptr(chip.hipMemsetParams* ptr, bint owner=*)
    @staticmethod
    cdef hipMemsetParams from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipMemsetParams** ptr)
    @staticmethod
    cdef hipMemsetParams new()
    @staticmethod
    cdef hipMemsetParams from_value(chip.hipMemsetParams other)


cdef class hipMemAllocNodeParams:
    cdef chip.hipMemAllocNodeParams* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipMemAllocNodeParams from_ptr(chip.hipMemAllocNodeParams* ptr, bint owner=*)
    @staticmethod
    cdef hipMemAllocNodeParams from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipMemAllocNodeParams** ptr)
    @staticmethod
    cdef hipMemAllocNodeParams new()
    @staticmethod
    cdef hipMemAllocNodeParams from_value(chip.hipMemAllocNodeParams other)


cdef class hipAccessPolicyWindow:
    cdef chip.hipAccessPolicyWindow* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipAccessPolicyWindow from_ptr(chip.hipAccessPolicyWindow* ptr, bint owner=*)
    @staticmethod
    cdef hipAccessPolicyWindow from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipAccessPolicyWindow** ptr)
    @staticmethod
    cdef hipAccessPolicyWindow new()
    @staticmethod
    cdef hipAccessPolicyWindow from_value(chip.hipAccessPolicyWindow other)


cdef class hipKernelNodeAttrValue:
    cdef chip.hipKernelNodeAttrValue* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipKernelNodeAttrValue from_ptr(chip.hipKernelNodeAttrValue* ptr, bint owner=*)
    @staticmethod
    cdef hipKernelNodeAttrValue from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipKernelNodeAttrValue** ptr)
    @staticmethod
    cdef hipKernelNodeAttrValue new()
    @staticmethod
    cdef hipKernelNodeAttrValue from_value(chip.hipKernelNodeAttrValue other)


cdef class hipMemAllocationProp_struct_0:
    cdef chip.hipMemAllocationProp_struct_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipMemAllocationProp_struct_0 from_ptr(chip.hipMemAllocationProp_struct_0* ptr, bint owner=*)
    @staticmethod
    cdef hipMemAllocationProp_struct_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipMemAllocationProp_struct_0** ptr)
    @staticmethod
    cdef hipMemAllocationProp_struct_0 new()
    @staticmethod
    cdef hipMemAllocationProp_struct_0 from_value(chip.hipMemAllocationProp_struct_0 other)


cdef class hipMemAllocationProp:
    cdef chip.hipMemAllocationProp* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipMemAllocationProp from_ptr(chip.hipMemAllocationProp* ptr, bint owner=*)
    @staticmethod
    cdef hipMemAllocationProp from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipMemAllocationProp** ptr)
    @staticmethod
    cdef hipMemAllocationProp new()
    @staticmethod
    cdef hipMemAllocationProp from_value(chip.hipMemAllocationProp other)


cdef class ihipMemGenericAllocationHandle:
    cdef chip.ihipMemGenericAllocationHandle* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef ihipMemGenericAllocationHandle from_ptr(chip.ihipMemGenericAllocationHandle* ptr, bint owner=*)
    @staticmethod
    cdef ihipMemGenericAllocationHandle from_pyobj(object pyobj)


cdef class hipArrayMapInfo_union_0:
    cdef chip.hipArrayMapInfo_union_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipArrayMapInfo_union_0 from_ptr(chip.hipArrayMapInfo_union_0* ptr, bint owner=*)
    @staticmethod
    cdef hipArrayMapInfo_union_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipArrayMapInfo_union_0** ptr)
    @staticmethod
    cdef hipArrayMapInfo_union_0 new()
    @staticmethod
    cdef hipArrayMapInfo_union_0 from_value(chip.hipArrayMapInfo_union_0 other)


cdef class hipArrayMapInfo_union_1_struct_0:
    cdef chip.hipArrayMapInfo_union_1_struct_0* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipArrayMapInfo_union_1_struct_0 from_ptr(chip.hipArrayMapInfo_union_1_struct_0* ptr, bint owner=*)
    @staticmethod
    cdef hipArrayMapInfo_union_1_struct_0 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipArrayMapInfo_union_1_struct_0** ptr)
    @staticmethod
    cdef hipArrayMapInfo_union_1_struct_0 new()
    @staticmethod
    cdef hipArrayMapInfo_union_1_struct_0 from_value(chip.hipArrayMapInfo_union_1_struct_0 other)


cdef class hipArrayMapInfo_union_1_struct_1:
    cdef chip.hipArrayMapInfo_union_1_struct_1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipArrayMapInfo_union_1_struct_1 from_ptr(chip.hipArrayMapInfo_union_1_struct_1* ptr, bint owner=*)
    @staticmethod
    cdef hipArrayMapInfo_union_1_struct_1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipArrayMapInfo_union_1_struct_1** ptr)
    @staticmethod
    cdef hipArrayMapInfo_union_1_struct_1 new()
    @staticmethod
    cdef hipArrayMapInfo_union_1_struct_1 from_value(chip.hipArrayMapInfo_union_1_struct_1 other)


cdef class hipArrayMapInfo_union_1:
    cdef chip.hipArrayMapInfo_union_1* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipArrayMapInfo_union_1 from_ptr(chip.hipArrayMapInfo_union_1* ptr, bint owner=*)
    @staticmethod
    cdef hipArrayMapInfo_union_1 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipArrayMapInfo_union_1** ptr)
    @staticmethod
    cdef hipArrayMapInfo_union_1 new()
    @staticmethod
    cdef hipArrayMapInfo_union_1 from_value(chip.hipArrayMapInfo_union_1 other)


cdef class hipArrayMapInfo_union_2:
    cdef chip.hipArrayMapInfo_union_2* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipArrayMapInfo_union_2 from_ptr(chip.hipArrayMapInfo_union_2* ptr, bint owner=*)
    @staticmethod
    cdef hipArrayMapInfo_union_2 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipArrayMapInfo_union_2** ptr)
    @staticmethod
    cdef hipArrayMapInfo_union_2 new()
    @staticmethod
    cdef hipArrayMapInfo_union_2 from_value(chip.hipArrayMapInfo_union_2 other)


cdef class hipArrayMapInfo:
    cdef chip.hipArrayMapInfo* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipArrayMapInfo from_ptr(chip.hipArrayMapInfo* ptr, bint owner=*)
    @staticmethod
    cdef hipArrayMapInfo from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chip.hipArrayMapInfo** ptr)
    @staticmethod
    cdef hipArrayMapInfo new()
    @staticmethod
    cdef hipArrayMapInfo from_value(chip.hipArrayMapInfo other)


cdef class hipStreamCallback_t:
    cdef chip.hipStreamCallback_t _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipStreamCallback_t from_ptr(chip.hipStreamCallback_t ptr, bint owner=*)
    @staticmethod
    cdef hipStreamCallback_t from_pyobj(object pyobj)
