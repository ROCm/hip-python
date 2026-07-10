# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
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

# ``cimport``-able declarations for the high-level ``cuda.bindings.cufile``
# interop layer (see ``cufile.pyx``). This mirrors how cuda-python's
# ``cufile.pxd`` sits over its ``cycufile.pxd``: here the ``cpdef`` layer sits
# over the hand-written ``cycufile.pxd`` C-level aliases.

from libc.stdint cimport intptr_t

cimport cuda.bindings.cycufile as cycufile


cdef class Descr:
    # Owns (or views) a contiguous array of ``CUfileDescr_t`` == hipFileDescr_t.
    cdef cycufile.hipFileDescr* _ptr
    cdef size_t _n
    cdef bint _owner


cdef class IOParams:
    # Owns (or views) a contiguous array of ``CUfileIOParams_t``.
    cdef cycufile.hipFileIOParams* _ptr
    cdef size_t _n
    cdef bint _owner


cdef class IOEvents:
    # Owns (or views) a contiguous array of ``CUfileIOEvents_t``.
    cdef cycufile.hipFileIOEvents* _ptr
    cdef size_t _n
    cdef bint _owner


cpdef intptr_t handle_register(intptr_t descr) except? 0
cpdef handle_deregister(intptr_t fh)
cpdef buf_register(intptr_t buf_ptr_base, size_t length, int flags)
cpdef buf_deregister(intptr_t buf_ptr_base)
cpdef read(intptr_t fh, intptr_t buf_ptr_base, size_t size, long file_offset, long buf_ptr_offset)
cpdef write(intptr_t fh, intptr_t buf_ptr_base, size_t size, long file_offset, long buf_ptr_offset)
cpdef driver_open()
cpdef driver_close()
cpdef use_count()
cpdef driver_get_properties(intptr_t props)
cpdef driver_set_poll_mode(bint poll, size_t poll_threshold_size)
cpdef driver_set_max_direct_io_size(size_t max_direct_io_size)
cpdef driver_set_max_cache_size(size_t max_cache_size)
cpdef driver_set_max_pinned_mem_size(size_t max_pinned_size)
cpdef intptr_t batch_io_set_up(unsigned int nr) except? 0
cpdef batch_io_submit(intptr_t batch_idp, unsigned int nr, intptr_t iocbp, unsigned int flags)
cpdef batch_io_get_status(intptr_t batch_idp, unsigned int min_nr, intptr_t nr, intptr_t iocbp, intptr_t timeout)
cpdef batch_io_cancel(intptr_t batch_idp)
cpdef batch_io_destroy(intptr_t batch_idp)
cpdef read_async(intptr_t fh, intptr_t buf_ptr_base, intptr_t size_p, intptr_t file_offset_p, intptr_t buf_ptr_offset_p, intptr_t bytes_read_p, intptr_t stream)
cpdef write_async(intptr_t fh, intptr_t buf_ptr_base, intptr_t size_p, intptr_t file_offset_p, intptr_t buf_ptr_offset_p, intptr_t bytes_written_p, intptr_t stream)
cpdef stream_register(intptr_t stream, unsigned int flags)
cpdef stream_deregister(intptr_t stream)
cpdef int get_version() except? 0
cpdef get_parameter_size_t(int param)
cpdef get_parameter_bool(int param)
cpdef str get_parameter_string(int param, int len)
cpdef set_parameter_size_t(int param, size_t value)
cpdef set_parameter_bool(int param, bint value)
cpdef set_parameter_string(int param, intptr_t desc_str)
cpdef str op_status_error(int status)
