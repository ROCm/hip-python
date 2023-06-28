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

from libc.stdint cimport *
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
from .chip cimport hipStream_t, float2, double2
cdef extern from "hipfft/hipfft.h":

    cdef int HIPFFT_FORWARD

    cdef int HIPFFT_BACKWARD

    cdef enum hipfftResult_t:
        HIPFFT_SUCCESS
        HIPFFT_INVALID_PLAN
        HIPFFT_ALLOC_FAILED
        HIPFFT_INVALID_TYPE
        HIPFFT_INVALID_VALUE
        HIPFFT_INTERNAL_ERROR
        HIPFFT_EXEC_FAILED
        HIPFFT_SETUP_FAILED
        HIPFFT_INVALID_SIZE
        HIPFFT_UNALIGNED_DATA
        HIPFFT_INCOMPLETE_PARAMETER_LIST
        HIPFFT_INVALID_DEVICE
        HIPFFT_PARSE_ERROR
        HIPFFT_NO_WORKSPACE
        HIPFFT_NOT_IMPLEMENTED
        HIPFFT_NOT_SUPPORTED

    ctypedef hipfftResult_t hipfftResult

    cdef enum hipfftType_t:
        HIPFFT_R2C
        HIPFFT_C2R
        HIPFFT_C2C
        HIPFFT_D2Z
        HIPFFT_Z2D
        HIPFFT_Z2Z

    ctypedef hipfftType_t hipfftType

    cdef enum hipfftLibraryPropertyType_t:
        HIPFFT_MAJOR_VERSION
        HIPFFT_MINOR_VERSION
        HIPFFT_PATCH_LEVEL

    ctypedef hipfftLibraryPropertyType_t hipfftLibraryPropertyType

    cdef struct hipfftHandle_t:
        pass

    ctypedef hipfftHandle_t * hipfftHandle

    ctypedef float2 hipfftComplex

    ctypedef double2 hipfftDoubleComplex

    ctypedef float hipfftReal

    ctypedef double hipfftDoubleReal

# @brief Create a new one-dimensional FFT plan.
# 
# @details Allocate and initialize a new one-dimensional FFT plan.
# 
# @param[out] plan Pointer to the FFT plan handle.
# @param[in] nx FFT length.
# @param[in] type FFT type.
# @param[in] batch Number of batched transforms to compute.
cdef hipfftResult_t hipfftPlan1d(hipfftHandle* plan,int nx,hipfftType_t type,int batch) nogil


# @brief Create a new two-dimensional FFT plan.
# 
# @details Allocate and initialize a new two-dimensional FFT plan.
# Two-dimensional data should be stored in C ordering (row-major
# format), so that indexes in y-direction (j index) vary the
# fastest.
# 
# @param[out] plan Pointer to the FFT plan handle.
# @param[in] nx Number of elements in the x-direction (slow index).
# @param[in] ny Number of elements in the y-direction (fast index).
# @param[in] type FFT type.
cdef hipfftResult_t hipfftPlan2d(hipfftHandle* plan,int nx,int ny,hipfftType_t type) nogil


# @brief Create a new three-dimensional FFT plan.
# 
# @details Allocate and initialize a new three-dimensional FFT plan.
# Three-dimensional data should be stored in C ordering (row-major
# format), so that indexes in z-direction (k index) vary the
# fastest.
# 
# @param[out] plan Pointer to the FFT plan handle.
# @param[in] nx Number of elements in the x-direction (slowest index).
# @param[in] ny Number of elements in the y-direction.
# @param[in] nz Number of elements in the z-direction (fastest index).
# @param[in] type FFT type.
cdef hipfftResult_t hipfftPlan3d(hipfftHandle* plan,int nx,int ny,int nz,hipfftType_t type) nogil


#  @brief Create a new batched rank-dimensional FFT plan with advanced data layout.
# 
# @details Allocate and initialize a new batched rank-dimensional
#  FFT plan. The number of elements to transform in each direction of
#  the input data is specified in n.
# 
#  The batch parameter tells hipFFT how many transforms to perform.
#  The distance between the first elements of two consecutive batches
#  of the input and output data are specified with the idist and odist
#  parameters.
# 
#  The inembed and onembed parameters define the input and output data
#  layouts. The number of elements in the data is assumed to be larger
#  than the number of elements in the transform. Strided data layouts
#  are also supported. Strides along the fastest direction in the input
#  and output data are specified via the istride and ostride parameters.
# 
#  If both inembed and onembed parameters are set to NULL, all the
#  advanced data layout parameters are ignored and reverted to default
#  values, i.e., the batched transform is performed with non-strided data
#  access and the number of data/transform elements are assumed to be
#  equivalent.
# 
#  @param[out] plan Pointer to the FFT plan handle.
#  @param[in] rank Dimension of transform (1, 2, or 3).
#  @param[in] n Number of elements to transform in the x/y/z directions.
#  @param[in] inembed Number of elements in the input data in the x/y/z directions.
#  @param[in] istride Distance between two successive elements in the input data.
#  @param[in] idist Distance between input batches.
#  @param[in] onembed Number of elements in the output data in the x/y/z directions.
#  @param[in] ostride Distance between two successive elements in the output data.
#  @param[in] odist Distance between output batches.
#  @param[in] type FFT type.
#  @param[in] batch Number of batched transforms to perform.
cdef hipfftResult_t hipfftPlanMany(hipfftHandle* plan,int rank,int * n,int * inembed,int istride,int idist,int * onembed,int ostride,int odist,hipfftType_t type,int batch) nogil


# @brief Allocate a new plan.
cdef hipfftResult_t hipfftCreate(hipfftHandle* plan) nogil


# @brief Set scaling factor.
# 
# @details hipFFT multiplies each element of the result by the given factor at the end of the transform.
# 
# The supplied factor must be a finite number.  That is, it must neither be infinity nor NaN.
# 
# This function must be called after the plan is allocated using
# ::hipfftCreate, but before the plan is initialized by any of the
# "MakePlan" functions.
#
cdef hipfftResult_t hipfftExtPlanScaleFactor(hipfftHandle plan,double scalefactor) nogil


# @brief Initialize a new one-dimensional FFT plan.
# 
# @details Assumes that the plan has been created already, and
# modifies the plan associated with the plan handle.
# 
# @param[in] plan Handle of the FFT plan.
# @param[in] nx FFT length.
# @param[in] type FFT type.
# @param[in] batch Number of batched transforms to compute.
cdef hipfftResult_t hipfftMakePlan1d(hipfftHandle plan,int nx,hipfftType_t type,int batch,unsigned long * workSize) nogil


# @brief Initialize a new two-dimensional FFT plan.
# 
# @details Assumes that the plan has been created already, and
# modifies the plan associated with the plan handle.
# Two-dimensional data should be stored in C ordering (row-major
# format), so that indexes in y-direction (j index) vary the
# fastest.
# 
# @param[in] plan Handle of the FFT plan.
# @param[in] nx Number of elements in the x-direction (slow index).
# @param[in] ny Number of elements in the y-direction (fast index).
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftMakePlan2d(hipfftHandle plan,int nx,int ny,hipfftType_t type,unsigned long * workSize) nogil


# @brief Initialize a new two-dimensional FFT plan.
# 
# @details Assumes that the plan has been created already, and
# modifies the plan associated with the plan handle.
# Three-dimensional data should be stored in C ordering (row-major
# format), so that indexes in z-direction (k index) vary the
# fastest.
# 
# @param[in] plan Handle of the FFT plan.
# @param[in] nx Number of elements in the x-direction (slowest index).
# @param[in] ny Number of elements in the y-direction.
# @param[in] nz Number of elements in the z-direction (fastest index).
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftMakePlan3d(hipfftHandle plan,int nx,int ny,int nz,hipfftType_t type,unsigned long * workSize) nogil


# @brief Initialize a new batched rank-dimensional FFT plan with advanced data layout.
# 
# @details Assumes that the plan has been created already, and
# modifies the plan associated with the plan handle. The number
# of elements to transform in each direction of the input data
# in the FFT plan is specified in n.
# 
# The batch parameter tells hipFFT how many transforms to perform.
# The distance between the first elements of two consecutive batches
# of the input and output data are specified with the idist and odist
# parameters.
# 
# The inembed and onembed parameters define the input and output data
# layouts. The number of elements in the data is assumed to be larger
# than the number of elements in the transform. Strided data layouts
# are also supported. Strides along the fastest direction in the input
# and output data are specified via the istride and ostride parameters.
# 
# If both inembed and onembed parameters are set to NULL, all the
# advanced data layout parameters are ignored and reverted to default
# values, i.e., the batched transform is performed with non-strided data
# access and the number of data/transform elements are assumed to be
# equivalent.
# 
# @param[out] plan Pointer to the FFT plan handle.
# @param[in] rank Dimension of transform (1, 2, or 3).
# @param[in] n Number of elements to transform in the x/y/z directions.
# @param[in] inembed Number of elements in the input data in the x/y/z directions.
# @param[in] istride Distance between two successive elements in the input data.
# @param[in] idist Distance between input batches.
# @param[in] onembed Number of elements in the output data in the x/y/z directions.
# @param[in] ostride Distance between two successive elements in the output data.
# @param[in] odist Distance between output batches.
# @param[in] type FFT type.
# @param[in] batch Number of batched transforms to perform.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftMakePlanMany(hipfftHandle plan,int rank,int * n,int * inembed,int istride,int idist,int * onembed,int ostride,int odist,hipfftType_t type,int batch,unsigned long * workSize) nogil



cdef hipfftResult_t hipfftMakePlanMany64(hipfftHandle plan,int rank,long long * n,long long * inembed,long long istride,long long idist,long long * onembed,long long ostride,long long odist,hipfftType_t type,long long batch,unsigned long * workSize) nogil


# @brief Return an estimate of the work area size required for a 1D plan.
# 
# @param[in] nx Number of elements in the x-direction.
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftEstimate1d(int nx,hipfftType_t type,int batch,unsigned long * workSize) nogil


# @brief Return an estimate of the work area size required for a 2D plan.
# 
# @param[in] nx Number of elements in the x-direction.
# @param[in] ny Number of elements in the y-direction.
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftEstimate2d(int nx,int ny,hipfftType_t type,unsigned long * workSize) nogil


# @brief Return an estimate of the work area size required for a 3D plan.
# 
# @param[in] nx Number of elements in the x-direction.
# @param[in] ny Number of elements in the y-direction.
# @param[in] nz Number of elements in the z-direction.
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftEstimate3d(int nx,int ny,int nz,hipfftType_t type,unsigned long * workSize) nogil


# @brief Return an estimate of the work area size required for a rank-dimensional plan.
# 
# @param[in] rank Dimension of FFT transform (1, 2, or 3).
# @param[in] n Number of elements in the x/y/z directions.
# @param[in] inembed
# @param[in] istride
# @param[in] idist Distance between input batches.
# @param[in] onembed
# @param[in] ostride
# @param[in] odist Distance between output batches.
# @param[in] type FFT type.
# @param[in] batch Number of batched transforms to perform.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftEstimateMany(int rank,int * n,int * inembed,int istride,int idist,int * onembed,int ostride,int odist,hipfftType_t type,int batch,unsigned long * workSize) nogil


# @brief Return size of the work area size required for a 1D plan.
# 
# @param[in] plan Pointer to the FFT plan.
# @param[in] nx Number of elements in the x-direction.
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftGetSize1d(hipfftHandle plan,int nx,hipfftType_t type,int batch,unsigned long * workSize) nogil


# @brief Return size of the work area size required for a 2D plan.
# 
# @param[in] plan Pointer to the FFT plan.
# @param[in] nx Number of elements in the x-direction.
# @param[in] ny Number of elements in the y-direction.
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftGetSize2d(hipfftHandle plan,int nx,int ny,hipfftType_t type,unsigned long * workSize) nogil


# @brief Return size of the work area size required for a 3D plan.
# 
# @param[in] plan Pointer to the FFT plan.
# @param[in] nx Number of elements in the x-direction.
# @param[in] ny Number of elements in the y-direction.
# @param[in] nz Number of elements in the z-direction.
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftGetSize3d(hipfftHandle plan,int nx,int ny,int nz,hipfftType_t type,unsigned long * workSize) nogil


# @brief Return size of the work area size required for a rank-dimensional plan.
# 
# @param[in] plan Pointer to the FFT plan.
# @param[in] rank Dimension of FFT transform (1, 2, or 3).
# @param[in] n Number of elements in the x/y/z directions.
# @param[in] inembed
# @param[in] istride
# @param[in] idist Distance between input batches.
# @param[in] onembed
# @param[in] ostride
# @param[in] odist Distance between output batches.
# @param[in] type FFT type.
# @param[in] batch Number of batched transforms to perform.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftGetSizeMany(hipfftHandle plan,int rank,int * n,int * inembed,int istride,int idist,int * onembed,int ostride,int odist,hipfftType_t type,int batch,unsigned long * workSize) nogil



cdef hipfftResult_t hipfftGetSizeMany64(hipfftHandle plan,int rank,long long * n,long long * inembed,long long istride,long long idist,long long * onembed,long long ostride,long long odist,hipfftType_t type,long long batch,unsigned long * workSize) nogil


# @brief Return size of the work area size required for a rank-dimensional plan.
# 
# @param[in] plan Pointer to the FFT plan.
cdef hipfftResult_t hipfftGetSize(hipfftHandle plan,unsigned long * workSize) nogil


# @brief Set the plan's auto-allocation flag.  The plan will allocate its own workarea.
# 
# @param[in] plan Pointer to the FFT plan.
# @param[in] autoAllocate 0 to disable auto-allocation, non-zero to enable.
cdef hipfftResult_t hipfftSetAutoAllocation(hipfftHandle plan,int autoAllocate) nogil


# @brief Set the plan's work area.
# 
# @param[in] plan Pointer to the FFT plan.
# @param[in] workArea Pointer to the work area (on device).
cdef hipfftResult_t hipfftSetWorkArea(hipfftHandle plan,void * workArea) nogil


# @brief Execute a (float) complex-to-complex FFT.
# 
# @details If the input and output buffers are equal, an in-place
# transform is performed.
# 
# @param plan The FFT plan.
# @param idata Input data (on device).
# @param odata Output data (on device).
# @param direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
cdef hipfftResult_t hipfftExecC2C(hipfftHandle plan,float2 * idata,float2 * odata,int direction) nogil


# @brief Execute a (float) real-to-complex FFT.
# 
# @details If the input and output buffers are equal, an in-place
# transform is performed.
# 
# @param plan The FFT plan.
# @param idata Input data (on device).
# @param odata Output data (on device).
cdef hipfftResult_t hipfftExecR2C(hipfftHandle plan,float * idata,float2 * odata) nogil


# @brief Execute a (float) complex-to-real FFT.
# 
# @details If the input and output buffers are equal, an in-place
# transform is performed.
# 
# @param plan The FFT plan.
# @param idata Input data (on device).
# @param odata Output data (on device).
cdef hipfftResult_t hipfftExecC2R(hipfftHandle plan,float2 * idata,float * odata) nogil


# @brief Execute a (double) complex-to-complex FFT.
# 
# @details If the input and output buffers are equal, an in-place
# transform is performed.
# 
# @param plan The FFT plan.
# @param idata Input data (on device).
# @param odata Output data (on device).
# @param direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
cdef hipfftResult_t hipfftExecZ2Z(hipfftHandle plan,double2 * idata,double2 * odata,int direction) nogil


# @brief Execute a (double) real-to-complex FFT.
# 
# @details If the input and output buffers are equal, an in-place
# transform is performed.
# 
# @param plan The FFT plan.
# @param idata Input data (on device).
# @param odata Output data (on device).
cdef hipfftResult_t hipfftExecD2Z(hipfftHandle plan,double * idata,double2 * odata) nogil


# @brief Execute a (double) complex-to-real FFT.
# 
# @details If the input and output buffers are equal, an in-place
# transform is performed.
# 
# @param plan The FFT plan.
# @param idata Input data (on device).
# @param odata Output data (on device).
cdef hipfftResult_t hipfftExecZ2D(hipfftHandle plan,double2 * idata,double * odata) nogil


#  @brief Set HIP stream to execute plan on.
# 
# @details Associates a HIP stream with a hipFFT plan.  All kernels
# launched by this plan are associated with the provided stream.
# 
# @param plan The FFT plan.
# @param stream The HIP stream.
cdef hipfftResult_t hipfftSetStream(hipfftHandle plan,hipStream_t stream) nogil


# @brief Destroy and deallocate an existing plan.
cdef hipfftResult_t hipfftDestroy(hipfftHandle plan) nogil


# @brief Get rocFFT/cuFFT version.
# 
# @param[out] version cuFFT/rocFFT version (returned value).
cdef hipfftResult_t hipfftGetVersion(int * version) nogil


# @brief Get library property.
# 
# @param[in] type Property type.
# @param[out] value Returned value.
cdef hipfftResult_t hipfftGetProperty(hipfftLibraryPropertyType_t type,int * value) nogil
