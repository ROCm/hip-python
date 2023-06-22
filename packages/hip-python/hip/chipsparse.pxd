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
from .chip cimport *
cdef extern from "hipsparse/hipsparse.h":

    ctypedef void * hipsparseHandle_t

    ctypedef void * hipsparseMatDescr_t

    ctypedef void * hipsparseHybMat_t

    ctypedef void * hipsparseColorInfo_t

    cdef struct bsrsv2Info:
        pass

    ctypedef bsrsv2Info * bsrsv2Info_t

    cdef struct bsrsm2Info:
        pass

    ctypedef bsrsm2Info * bsrsm2Info_t

    cdef struct bsrilu02Info:
        pass

    ctypedef bsrilu02Info * bsrilu02Info_t

    cdef struct bsric02Info:
        pass

    ctypedef bsric02Info * bsric02Info_t

    cdef struct csrsv2Info:
        pass

    ctypedef csrsv2Info * csrsv2Info_t

    cdef struct csrsm2Info:
        pass

    ctypedef csrsm2Info * csrsm2Info_t

    cdef struct csrilu02Info:
        pass

    ctypedef csrilu02Info * csrilu02Info_t

    cdef struct csric02Info:
        pass

    ctypedef csric02Info * csric02Info_t

    cdef struct csrgemm2Info:
        pass

    ctypedef csrgemm2Info * csrgemm2Info_t

    cdef struct pruneInfo:
        pass

    ctypedef pruneInfo * pruneInfo_t

    cdef struct csru2csrInfo:
        pass

    ctypedef csru2csrInfo * csru2csrInfo_t

    ctypedef enum hipsparseStatus_t:
        HIPSPARSE_STATUS_SUCCESS
        HIPSPARSE_STATUS_NOT_INITIALIZED
        HIPSPARSE_STATUS_ALLOC_FAILED
        HIPSPARSE_STATUS_INVALID_VALUE
        HIPSPARSE_STATUS_ARCH_MISMATCH
        HIPSPARSE_STATUS_MAPPING_ERROR
        HIPSPARSE_STATUS_EXECUTION_FAILED
        HIPSPARSE_STATUS_INTERNAL_ERROR
        HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
        HIPSPARSE_STATUS_ZERO_PIVOT
        HIPSPARSE_STATUS_NOT_SUPPORTED
        HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES

    ctypedef enum hipsparsePointerMode_t:
        HIPSPARSE_POINTER_MODE_HOST
        HIPSPARSE_POINTER_MODE_DEVICE

    ctypedef enum hipsparseAction_t:
        HIPSPARSE_ACTION_SYMBOLIC
        HIPSPARSE_ACTION_NUMERIC

    ctypedef enum hipsparseMatrixType_t:
        HIPSPARSE_MATRIX_TYPE_GENERAL
        HIPSPARSE_MATRIX_TYPE_SYMMETRIC
        HIPSPARSE_MATRIX_TYPE_HERMITIAN
        HIPSPARSE_MATRIX_TYPE_TRIANGULAR

    ctypedef enum hipsparseFillMode_t:
        HIPSPARSE_FILL_MODE_LOWER
        HIPSPARSE_FILL_MODE_UPPER

    ctypedef enum hipsparseDiagType_t:
        HIPSPARSE_DIAG_TYPE_NON_UNIT
        HIPSPARSE_DIAG_TYPE_UNIT

    ctypedef enum hipsparseIndexBase_t:
        HIPSPARSE_INDEX_BASE_ZERO
        HIPSPARSE_INDEX_BASE_ONE

    ctypedef enum hipsparseOperation_t:
        HIPSPARSE_OPERATION_NON_TRANSPOSE
        HIPSPARSE_OPERATION_TRANSPOSE
        HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE

    ctypedef enum hipsparseHybPartition_t:
        HIPSPARSE_HYB_PARTITION_AUTO
        HIPSPARSE_HYB_PARTITION_USER
        HIPSPARSE_HYB_PARTITION_MAX

    ctypedef enum hipsparseSolvePolicy_t:
        HIPSPARSE_SOLVE_POLICY_NO_LEVEL
        HIPSPARSE_SOLVE_POLICY_USE_LEVEL

    ctypedef enum hipsparseSideMode_t:
        HIPSPARSE_SIDE_LEFT
        HIPSPARSE_SIDE_RIGHT

    ctypedef enum hipsparseDirection_t:
        HIPSPARSE_DIRECTION_ROW
        HIPSPARSE_DIRECTION_COLUMN

# \ingroup aux_module
# \brief Create a hipsparse handle
# 
# \details
# \p hipsparseCreate creates the hipSPARSE library context. It must be
# initialized before any other hipSPARSE API function is invoked and must be passed to
# all subsequent library function calls. The handle should be destroyed at the end
# using hipsparseDestroy().
cdef hipsparseStatus_t hipsparseCreate(void ** handle) nogil


# \ingroup aux_module
# \brief Destroy a hipsparse handle
# 
# \details
# \p hipsparseDestroy destroys the hipSPARSE library context and releases all
# resources used by the hipSPARSE library.
cdef hipsparseStatus_t hipsparseDestroy(void * handle) nogil


# \ingroup aux_module
# \brief Get hipSPARSE version
# 
# \details
# \p hipsparseGetVersion gets the hipSPARSE library version number.
# - patch = version % 100
# - minor = version / 100 % 1000
# - major = version / 100000
cdef hipsparseStatus_t hipsparseGetVersion(void * handle,int * version) nogil


# \ingroup aux_module
# \brief Get hipSPARSE git revision
# 
# \details
# \p hipsparseGetGitRevision gets the hipSPARSE library git commit revision (SHA-1).
cdef hipsparseStatus_t hipsparseGetGitRevision(void * handle,char * rev) nogil


# \ingroup aux_module
# \brief Specify user defined HIP stream
# 
# \details
# \p hipsparseSetStream specifies the stream to be used by the hipSPARSE library
# context and all subsequent function calls.
cdef hipsparseStatus_t hipsparseSetStream(void * handle,hipStream_t streamId) nogil


# \ingroup aux_module
# \brief Get current stream from library context
# 
# \details
# \p hipsparseGetStream gets the hipSPARSE library context stream which is currently
# used for all subsequent function calls.
cdef hipsparseStatus_t hipsparseGetStream(void * handle,hipStream_t* streamId) nogil


# \ingroup aux_module
# \brief Specify pointer mode
# 
# \details
# \p hipsparseSetPointerMode specifies the pointer mode to be used by the hipSPARSE
# library context and all subsequent function calls. By default, all values are passed
# by reference on the host. Valid pointer modes are \ref HIPSPARSE_POINTER_MODE_HOST
# or \p HIPSPARSE_POINTER_MODE_DEVICE.
cdef hipsparseStatus_t hipsparseSetPointerMode(void * handle,hipsparsePointerMode_t mode) nogil


# \ingroup aux_module
# \brief Get current pointer mode from library context
# 
# \details
# \p hipsparseGetPointerMode gets the hipSPARSE library context pointer mode which
# is currently used for all subsequent function calls.
cdef hipsparseStatus_t hipsparseGetPointerMode(void * handle,hipsparsePointerMode_t * mode) nogil


# \ingroup aux_module
# \brief Create a matrix descriptor
# \details
# \p hipsparseCreateMatDescr creates a matrix descriptor. It initializes
# \ref hipsparseMatrixType_t to \ref HIPSPARSE_MATRIX_TYPE_GENERAL and
# \ref hipsparseIndexBase_t to \ref HIPSPARSE_INDEX_BASE_ZERO. It should be destroyed
# at the end using hipsparseDestroyMatDescr().
cdef hipsparseStatus_t hipsparseCreateMatDescr(void ** descrA) nogil


# \ingroup aux_module
# \brief Destroy a matrix descriptor
# 
# \details
# \p hipsparseDestroyMatDescr destroys a matrix descriptor and releases all
# resources used by the descriptor.
cdef hipsparseStatus_t hipsparseDestroyMatDescr(void * descrA) nogil


# \ingroup aux_module
# \brief Copy a matrix descriptor
# \details
# \p hipsparseCopyMatDescr copies a matrix descriptor. Both, source and destination
# matrix descriptors must be initialized prior to calling \p hipsparseCopyMatDescr.
cdef hipsparseStatus_t hipsparseCopyMatDescr(void * dest,void *const src) nogil


# \ingroup aux_module
# \brief Specify the matrix type of a matrix descriptor
# 
# \details
# \p hipsparseSetMatType sets the matrix type of a matrix descriptor. Valid
# matrix types are \ref HIPSPARSE_MATRIX_TYPE_GENERAL,
# \ref HIPSPARSE_MATRIX_TYPE_SYMMETRIC, \ref HIPSPARSE_MATRIX_TYPE_HERMITIAN or
# \ref HIPSPARSE_MATRIX_TYPE_TRIANGULAR.
cdef hipsparseStatus_t hipsparseSetMatType(void * descrA,hipsparseMatrixType_t type) nogil


# \ingroup aux_module
# \brief Get the matrix type of a matrix descriptor
# 
# \details
# \p hipsparseGetMatType returns the matrix type of a matrix descriptor.
cdef hipsparseMatrixType_t hipsparseGetMatType(void *const descrA) nogil


# \ingroup aux_module
# \brief Specify the matrix fill mode of a matrix descriptor
# 
# \details
# \p hipsparseSetMatFillMode sets the matrix fill mode of a matrix descriptor.
# Valid fill modes are \ref HIPSPARSE_FILL_MODE_LOWER or
# \ref HIPSPARSE_FILL_MODE_UPPER.
cdef hipsparseStatus_t hipsparseSetMatFillMode(void * descrA,hipsparseFillMode_t fillMode) nogil


# \ingroup aux_module
# \brief Get the matrix fill mode of a matrix descriptor
# 
# \details
# \p hipsparseGetMatFillMode returns the matrix fill mode of a matrix descriptor.
cdef hipsparseFillMode_t hipsparseGetMatFillMode(void *const descrA) nogil


# \ingroup aux_module
# \brief Specify the matrix diagonal type of a matrix descriptor
# 
# \details
# \p hipsparseSetMatDiagType sets the matrix diagonal type of a matrix
# descriptor. Valid diagonal types are \ref HIPSPARSE_DIAG_TYPE_UNIT or
# \ref HIPSPARSE_DIAG_TYPE_NON_UNIT.
cdef hipsparseStatus_t hipsparseSetMatDiagType(void * descrA,hipsparseDiagType_t diagType) nogil


# \ingroup aux_module
# \brief Get the matrix diagonal type of a matrix descriptor
# 
# \details
# \p hipsparseGetMatDiagType returns the matrix diagonal type of a matrix
# descriptor.
cdef hipsparseDiagType_t hipsparseGetMatDiagType(void *const descrA) nogil


# \ingroup aux_module
# \brief Specify the index base of a matrix descriptor
# 
# \details
# \p hipsparseSetMatIndexBase sets the index base of a matrix descriptor. Valid
# options are \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
cdef hipsparseStatus_t hipsparseSetMatIndexBase(void * descrA,hipsparseIndexBase_t base) nogil


# \ingroup aux_module
# \brief Get the index base of a matrix descriptor
# 
# \details
# \p hipsparseGetMatIndexBase returns the index base of a matrix descriptor.
cdef hipsparseIndexBase_t hipsparseGetMatIndexBase(void *const descrA) nogil


# \ingroup aux_module
# \brief Create a \p HYB matrix structure
# 
# \details
# \p hipsparseCreateHybMat creates a structure that holds the matrix in \p HYB
# storage format. It should be destroyed at the end using hipsparseDestroyHybMat().
cdef hipsparseStatus_t hipsparseCreateHybMat(void ** hybA) nogil


# \ingroup aux_module
# \brief Destroy a \p HYB matrix structure
# 
# \details
# \p hipsparseDestroyHybMat destroys a \p HYB structure.
cdef hipsparseStatus_t hipsparseDestroyHybMat(void * hybA) nogil


# \ingroup aux_module
# \brief Create a bsrsv2 info structure
# 
# \details
# \p hipsparseCreateBsrsv2Info creates a structure that holds the bsrsv2 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyBsrsv2Info().
cdef hipsparseStatus_t hipsparseCreateBsrsv2Info(bsrsv2Info_t* info) nogil


# \ingroup aux_module
# \brief Destroy a bsrsv2 info structure
# 
# \details
# \p hipsparseDestroyBsrsv2Info destroys a bsrsv2 info structure.
cdef hipsparseStatus_t hipsparseDestroyBsrsv2Info(bsrsv2Info_t info) nogil


# \ingroup aux_module
# \brief Create a bsrsm2 info structure
# 
# \details
# \p hipsparseCreateBsrsm2Info creates a structure that holds the bsrsm2 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyBsrsm2Info().
cdef hipsparseStatus_t hipsparseCreateBsrsm2Info(bsrsm2Info_t* info) nogil


# \ingroup aux_module
# \brief Destroy a bsrsm2 info structure
# 
# \details
# \p hipsparseDestroyBsrsm2Info destroys a bsrsm2 info structure.
cdef hipsparseStatus_t hipsparseDestroyBsrsm2Info(bsrsm2Info_t info) nogil


# \ingroup aux_module
# \brief Create a bsrilu02 info structure
# 
# \details
# \p hipsparseCreateBsrilu02Info creates a structure that holds the bsrilu02 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyBsrilu02Info().
cdef hipsparseStatus_t hipsparseCreateBsrilu02Info(bsrilu02Info_t* info) nogil


# \ingroup aux_module
# \brief Destroy a bsrilu02 info structure
# 
# \details
# \p hipsparseDestroyBsrilu02Info destroys a bsrilu02 info structure.
cdef hipsparseStatus_t hipsparseDestroyBsrilu02Info(bsrilu02Info_t info) nogil


# \ingroup aux_module
# \brief Create a bsric02 info structure
# 
# \details
# \p hipsparseCreateBsric02Info creates a structure that holds the bsric02 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyBsric02Info().
cdef hipsparseStatus_t hipsparseCreateBsric02Info(bsric02Info_t* info) nogil


# \ingroup aux_module
# \brief Destroy a bsric02 info structure
# 
# \details
# \p hipsparseDestroyBsric02Info destroys a bsric02 info structure.
cdef hipsparseStatus_t hipsparseDestroyBsric02Info(bsric02Info_t info) nogil


# \ingroup aux_module
# \brief Create a csrsv2 info structure
# 
# \details
# \p hipsparseCreateCsrsv2Info creates a structure that holds the csrsv2 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyCsrsv2Info().
cdef hipsparseStatus_t hipsparseCreateCsrsv2Info(csrsv2Info_t* info) nogil


# \ingroup aux_module
# \brief Destroy a csrsv2 info structure
# 
# \details
# \p hipsparseDestroyCsrsv2Info destroys a csrsv2 info structure.
cdef hipsparseStatus_t hipsparseDestroyCsrsv2Info(csrsv2Info_t info) nogil


# \ingroup aux_module
# \brief Create a csrsm2 info structure
# 
# \details
# \p hipsparseCreateCsrsm2Info creates a structure that holds the csrsm2 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyCsrsm2Info().
cdef hipsparseStatus_t hipsparseCreateCsrsm2Info(csrsm2Info_t* info) nogil


# \ingroup aux_module
# \brief Destroy a csrsm2 info structure
# 
# \details
# \p hipsparseDestroyCsrsm2Info destroys a csrsm2 info structure.
cdef hipsparseStatus_t hipsparseDestroyCsrsm2Info(csrsm2Info_t info) nogil


# \ingroup aux_module
# \brief Create a csrilu02 info structure
# 
# \details
# \p hipsparseCreateCsrilu02Info creates a structure that holds the csrilu02 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyCsrilu02Info().
cdef hipsparseStatus_t hipsparseCreateCsrilu02Info(csrilu02Info_t* info) nogil


# \ingroup aux_module
# \brief Destroy a csrilu02 info structure
# 
# \details
# \p hipsparseDestroyCsrilu02Info destroys a csrilu02 info structure.
cdef hipsparseStatus_t hipsparseDestroyCsrilu02Info(csrilu02Info_t info) nogil


# \ingroup aux_module
# \brief Create a csric02 info structure
# 
# \details
# \p hipsparseCreateCsric02Info creates a structure that holds the csric02 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyCsric02Info().
cdef hipsparseStatus_t hipsparseCreateCsric02Info(csric02Info_t* info) nogil


# \ingroup aux_module
# \brief Destroy a csric02 info structure
# 
# \details
# \p hipsparseDestroyCsric02Info destroys a csric02 info structure.
cdef hipsparseStatus_t hipsparseDestroyCsric02Info(csric02Info_t info) nogil


# \ingroup aux_module
# \brief Create a csru2csr info structure
# 
# \details
# \p hipsparseCreateCsru2csrInfo creates a structure that holds the csru2csr info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyCsru2csrInfo().
cdef hipsparseStatus_t hipsparseCreateCsru2csrInfo(csru2csrInfo_t* info) nogil


# \ingroup aux_module
# \brief Destroy a csru2csr info structure
# 
# \details
# \p hipsparseDestroyCsru2csrInfo destroys a csru2csr info structure.
cdef hipsparseStatus_t hipsparseDestroyCsru2csrInfo(csru2csrInfo_t info) nogil


# \ingroup aux_module
# \brief Create a color info structure
# 
# \details
# \p hipsparseCreateColorInfo creates a structure that holds the color info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyColorInfo().
cdef hipsparseStatus_t hipsparseCreateColorInfo(void ** info) nogil


# \ingroup aux_module
# \brief Destroy a color info structure
# 
# \details
# \p hipsparseDestroyColorInfo destroys a color info structure.
cdef hipsparseStatus_t hipsparseDestroyColorInfo(void * info) nogil


# \ingroup aux_module
# \brief Create a csrgemm2 info structure
# 
# \details
# \p hipsparseCreateCsrgemm2Info creates a structure that holds the csrgemm2 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyCsrgemm2Info().
cdef hipsparseStatus_t hipsparseCreateCsrgemm2Info(csrgemm2Info_t* info) nogil


# \ingroup aux_module
# \brief Destroy a csrgemm2 info structure
# 
# \details
# \p hipsparseDestroyCsrgemm2Info destroys a csrgemm2 info structure.
cdef hipsparseStatus_t hipsparseDestroyCsrgemm2Info(csrgemm2Info_t info) nogil


# \ingroup aux_module
# \brief Create a prune info structure
# 
# \details
# \p hipsparseCreatePruneInfo creates a structure that holds the prune info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyPruneInfo().
cdef hipsparseStatus_t hipsparseCreatePruneInfo(pruneInfo_t* info) nogil


# \ingroup aux_module
# \brief Destroy a prune info structure
# 
# \details
# \p hipsparseDestroyPruneInfo destroys a prune info structure.
cdef hipsparseStatus_t hipsparseDestroyPruneInfo(pruneInfo_t info) nogil


#  \ingroup level1_module
# \brief Scale a sparse vector and add it to a dense vector.
# 
# \details
# \p hipsparseXaxpyi multiplies the sparse vector \f$x\f$ with scalar \f$\alpha\f$ and
# adds the result to the dense vector \f$y\f$, such that
# 
# \f[
#     y := y + \alpha \cdot x
# \f]
# 
# \code{.c}
#     for(i = 0; i < nnz; ++i)
#     {
#         y[x_ind[i]] = y[x_ind[i]] + alpha * x_val[i];
#     }
# \endcode
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSaxpyi(void * handle,int nnz,const float * alpha,const float * xVal,const int * xInd,float * y,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseDaxpyi(void * handle,int nnz,const double * alpha,const double * xVal,const int * xInd,double * y,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseCaxpyi(void * handle,int nnz,float2 * alpha,float2 * xVal,const int * xInd,float2 * y,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseZaxpyi(void * handle,int nnz,double2 * alpha,double2 * xVal,const int * xInd,double2 * y,hipsparseIndexBase_t idxBase) nogil


#  \ingroup level1_module
# \brief Compute the dot product of a sparse vector with a dense vector.
# 
# \details
# \p hipsparseXdoti computes the dot product of the sparse vector \f$x\f$ with the
# dense vector \f$y\f$, such that
# \f[
#   \text{result} := y^T x
# \f]
# 
# \code{.c}
#     for(i = 0; i < nnz; ++i)
#     {
#         result += x_val[i] * y[x_ind[i]];
#     }
# \endcode
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSdoti(void * handle,int nnz,const float * xVal,const int * xInd,const float * y,float * result,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseDdoti(void * handle,int nnz,const double * xVal,const int * xInd,const double * y,double * result,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseCdoti(void * handle,int nnz,float2 * xVal,const int * xInd,float2 * y,float2 * result,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseZdoti(void * handle,int nnz,double2 * xVal,const int * xInd,double2 * y,double2 * result,hipsparseIndexBase_t idxBase) nogil


#  \ingroup level1_module
# \brief Compute the dot product of a complex conjugate sparse vector with a dense
# vector.
# 
# \details
# \p hipsparseXdotci computes the dot product of the complex conjugate sparse vector
# \f$x\f$ with the dense vector \f$y\f$, such that
# \f[
#   \text{result} := \bar{x}^H y
# \f]
# 
# \code{.c}
#     for(i = 0; i < nnz; ++i)
#     {
#         result += conj(x_val[i]) * y[x_ind[i]];
#     }
# \endcode
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseCdotci(void * handle,int nnz,float2 * xVal,const int * xInd,float2 * y,float2 * result,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseZdotci(void * handle,int nnz,double2 * xVal,const int * xInd,double2 * y,double2 * result,hipsparseIndexBase_t idxBase) nogil


#  \ingroup level1_module
# \brief Gather elements from a dense vector and store them into a sparse vector.
# 
# \details
# \p hipsparseXgthr gathers the elements that are listed in \p x_ind from the dense
# vector \f$y\f$ and stores them in the sparse vector \f$x\f$.
# 
# \code{.c}
#     for(i = 0; i < nnz; ++i)
#     {
#         x_val[i] = y[x_ind[i]];
#     }
# \endcode
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSgthr(void * handle,int nnz,const float * y,float * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseDgthr(void * handle,int nnz,const double * y,double * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseCgthr(void * handle,int nnz,float2 * y,float2 * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseZgthr(void * handle,int nnz,double2 * y,double2 * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil


#  \ingroup level1_module
# \brief Gather and zero out elements from a dense vector and store them into a sparse
# vector.
# 
# \details
# \p hipsparseXgthrz gathers the elements that are listed in \p x_ind from the dense
# vector \f$y\f$ and stores them in the sparse vector \f$x\f$. The gathered elements
# in \f$y\f$ are replaced by zero.
# 
# \code{.c}
#     for(i = 0; i < nnz; ++i)
#     {
#         x_val[i]    = y[x_ind[i]];
#         y[x_ind[i]] = 0;
#     }
# \endcode
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSgthrz(void * handle,int nnz,float * y,float * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseDgthrz(void * handle,int nnz,double * y,double * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseCgthrz(void * handle,int nnz,float2 * y,float2 * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseZgthrz(void * handle,int nnz,double2 * y,double2 * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil


#  \ingroup level1_module
# \brief Apply Givens rotation to a dense and a sparse vector.
# 
# \details
# \p hipsparseXroti applies the Givens rotation matrix \f$G\f$ to the sparse vector
# \f$x\f$ and the dense vector \f$y\f$, where
# \f[
#   G = \begin{pmatrix} c & s \\ -s & c \end{pmatrix}
# \f]
# 
# \code{.c}
#     for(i = 0; i < nnz; ++i)
#     {
#         x_tmp = x_val[i];
#         y_tmp = y[x_ind[i]];
# 
#         x_val[i]    = c * x_tmp + s * y_tmp;
#         y[x_ind[i]] = c * y_tmp - s * x_tmp;
#     }
# \endcode
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSroti(void * handle,int nnz,float * xVal,const int * xInd,float * y,const float * c,const float * s,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseDroti(void * handle,int nnz,double * xVal,const int * xInd,double * y,const double * c,const double * s,hipsparseIndexBase_t idxBase) nogil


#  \ingroup level1_module
# \brief Scatter elements from a dense vector across a sparse vector.
# 
# \details
# \p hipsparseXsctr scatters the elements that are listed in \p x_ind from the sparse
# vector \f$x\f$ into the dense vector \f$y\f$. Indices of \f$y\f$ that are not listed
# in \p x_ind remain unchanged.
# 
# \code{.c}
#     for(i = 0; i < nnz; ++i)
#     {
#         y[x_ind[i]] = x_val[i];
#     }
# \endcode
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSsctr(void * handle,int nnz,const float * xVal,const int * xInd,float * y,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseDsctr(void * handle,int nnz,const double * xVal,const int * xInd,double * y,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseCsctr(void * handle,int nnz,float2 * xVal,const int * xInd,float2 * y,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseZsctr(void * handle,int nnz,double2 * xVal,const int * xInd,double2 * y,hipsparseIndexBase_t idxBase) nogil


#  \ingroup level2_module
# \brief Sparse matrix vector multiplication using CSR storage format
# 
# \details
# \p hipsparseXcsrmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
# matrix, defined in CSR storage format, and the dense vector \f$x\f$ and adds the
# result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
# such that
# \f[
#   y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# 
# \code{.c}
#     for(i = 0; i < m; ++i)
#     {
#         y[i] = beta * y[i];
# 
#         for(j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j)
#         {
#             y[i] = y[i] + alpha * csr_val[j] * x[csr_col_ind[j]];
#         }
#     }
# \endcode
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# 
# \note
# Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
cdef hipsparseStatus_t hipsparseScsrmv(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * x,const float * beta,float * y) nogil



cdef hipsparseStatus_t hipsparseDcsrmv(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * x,const double * beta,double * y) nogil



cdef hipsparseStatus_t hipsparseCcsrmv(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * x,float2 * beta,float2 * y) nogil



cdef hipsparseStatus_t hipsparseZcsrmv(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * x,double2 * beta,double2 * y) nogil


#  \ingroup level2_module
# \brief Sparse triangular solve using CSR storage format
# 
# \details
# \p hipsparseXcsrsv2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
# structural or numerical zero has been found during hipsparseScsrsv2_solve(),
# hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() or hipsparseZcsrsv2_solve()
# computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position,
# using same index base as the CSR matrix.
# 
# \p position can be in host or device memory. If no zero pivot has been found,
# \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
# \note \p hipsparseXcsrsv2_zeroPivot is a blocking function. It might influence
# performance negatively.
cdef hipsparseStatus_t hipsparseXcsrsv2_zeroPivot(void * handle,csrsv2Info_t info,int * position) nogil


#  \ingroup level2_module
# \brief Sparse triangular solve using CSR storage format
# 
# \details
# \p hipsparseXcsrsv2_bufferSize returns the size of the temporary storage buffer that
# is required by hipsparseScsrsv2_analysis(), hipsparseDcsrsv2_analysis(),
# hipsparseCcsrsv2_analysis(), hipsparseZcsrsv2_analysis(), hipsparseScsrsv2_solve(),
# hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve(). The
# temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseScsrsv2_bufferSize(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDcsrsv2_bufferSize(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCcsrsv2_bufferSize(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZcsrsv2_bufferSize(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,int * pBufferSizeInBytes) nogil


#  \ingroup level2_module
# \brief Sparse triangular solve using CSR storage format
# 
# \details
# \p hipsparseXcsrsv2_bufferSizeExt returns the size of the temporary storage buffer that
# is required by hipsparseScsrsv2_analysis(), hipsparseDcsrsv2_analysis(),
# hipsparseCcsrsv2_analysis(), hipsparseZcsrsv2_analysis(), hipsparseScsrsv2_solve(),
# hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve(). The
# temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseScsrsv2_bufferSizeExt(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseDcsrsv2_bufferSizeExt(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseCcsrsv2_bufferSizeExt(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseZcsrsv2_bufferSizeExt(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,unsigned long * pBufferSize) nogil


#  \ingroup level2_module
# \brief Sparse triangular solve using CSR storage format
# 
# \details
# \p hipsparseXcsrsv2_analysis performs the analysis step for hipsparseScsrsv2_solve(),
# hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve().
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseScsrsv2_analysis(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDcsrsv2_analysis(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCcsrsv2_analysis(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZcsrsv2_analysis(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


#  \ingroup level2_module
# \brief Sparse triangular solve using CSR storage format
# 
# \details
# \p hipsparseXcsrsv2_solve solves a sparse triangular linear system of a sparse
# \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution vector
# \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
# \f[
#   op(A) \cdot y = \alpha \cdot x,
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# 
# \p hipsparseXcsrsv2_solve requires a user allocated temporary buffer. Its size is
# returned by hipsparseXcsrsv2_bufferSize() or hipsparseXcsrsv2_bufferSizeExt().
# Furthermore, analysis meta data is required. It can be obtained by
# hipsparseXcsrsv2_analysis(). \p hipsparseXcsrsv2_solve reports the first zero pivot
# (either numerical or structural zero). The zero pivot status can be checked calling
# hipsparseXcsrsv2_zeroPivot(). If
# \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
# reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
# 
# \note
# The sparse CSR matrix has to be sorted. This can be achieved by calling
# hipsparseXcsrsort().
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# 
# \note
# Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE and
# \p trans == \ref HIPSPARSE_OPERATION_TRANSPOSE is supported.
cdef hipsparseStatus_t hipsparseScsrsv2_solve(void * handle,hipsparseOperation_t transA,int m,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,const float * f,float * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDcsrsv2_solve(void * handle,hipsparseOperation_t transA,int m,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,const double * f,double * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCcsrsv2_solve(void * handle,hipsparseOperation_t transA,int m,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,float2 * f,float2 * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZcsrsv2_solve(void * handle,hipsparseOperation_t transA,int m,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,double2 * f,double2 * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


#  \ingroup level2_module
# \brief Sparse matrix vector multiplication using HYB storage format
# 
# \details
# \p hipsparseXhybmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
# matrix, defined in HYB storage format, and the dense vector \f$x\f$ and adds the
# result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
# such that
# \f[
#   y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# 
# \note
# Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
cdef hipsparseStatus_t hipsparseShybmv(void * handle,hipsparseOperation_t transA,const float * alpha,void *const descrA,void *const hybA,const float * x,const float * beta,float * y) nogil



cdef hipsparseStatus_t hipsparseDhybmv(void * handle,hipsparseOperation_t transA,const double * alpha,void *const descrA,void *const hybA,const double * x,const double * beta,double * y) nogil



cdef hipsparseStatus_t hipsparseChybmv(void * handle,hipsparseOperation_t transA,float2 * alpha,void *const descrA,void *const hybA,float2 * x,float2 * beta,float2 * y) nogil



cdef hipsparseStatus_t hipsparseZhybmv(void * handle,hipsparseOperation_t transA,double2 * alpha,void *const descrA,void *const hybA,double2 * x,double2 * beta,double2 * y) nogil


#  \ingroup level2_module
# \brief Sparse matrix vector multiplication using BSR storage format
# 
# \details
# \p hipsparseXbsrmv multiplies the scalar \f$\alpha\f$ with a sparse
# \f$(mb \cdot \text{block_dim}) \times (nb \cdot \text{block_dim})\f$
# matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
# result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
# such that
# \f[
#   y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# 
# \note
# Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
cdef hipsparseStatus_t hipsparseSbsrmv(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nb,int nnzb,const float * alpha,void *const descrA,const float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,const float * x,const float * beta,float * y) nogil



cdef hipsparseStatus_t hipsparseDbsrmv(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nb,int nnzb,const double * alpha,void *const descrA,const double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,const double * x,const double * beta,double * y) nogil



cdef hipsparseStatus_t hipsparseCbsrmv(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nb,int nnzb,float2 * alpha,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,float2 * x,float2 * beta,float2 * y) nogil



cdef hipsparseStatus_t hipsparseZbsrmv(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nb,int nnzb,double2 * alpha,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,double2 * x,double2 * beta,double2 * y) nogil


#  \ingroup level2_module
# \brief Sparse matrix vector multiplication with mask operation using BSR storage format
# 
# \details
# \p hipsparseXbsrxmv multiplies the scalar \f$\alpha\f$ with a sparse
# \f$(mb \cdot \text{block_dim}) \times (nb \cdot \text{block_dim})\f$
# modified matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
# result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
# such that
# \f[
#   y := \left( \alpha \cdot op(A) \cdot x + \beta \cdot y \right)\left( \text{mask} \right),
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# 
# The \f$\text{mask}\f$ is defined as an array of block row indices.
# The input sparse matrix is defined with a modified BSR storage format where the beginning and the end of each row
# is defined with two arrays, \p bsr_row_ptr and \p bsr_end_ptr (both of size \p mb), rather the usual \p bsr_row_ptr of size \p mb + 1.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# 
# \note
# Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
# Currently, \p block_dim == 1 is not supported.
cdef hipsparseStatus_t hipsparseSbsrxmv(void * handle,hipsparseDirection_t dir,hipsparseOperation_t trans,int sizeOfMask,int mb,int nb,int nnzb,const float * alpha,void *const descr,const float * bsrVal,const int * bsrMaskPtr,const int * bsrRowPtr,const int * bsrEndPtr,const int * bsrColInd,int blockDim,const float * x,const float * beta,float * y) nogil



cdef hipsparseStatus_t hipsparseDbsrxmv(void * handle,hipsparseDirection_t dir,hipsparseOperation_t trans,int sizeOfMask,int mb,int nb,int nnzb,const double * alpha,void *const descr,const double * bsrVal,const int * bsrMaskPtr,const int * bsrRowPtr,const int * bsrEndPtr,const int * bsrColInd,int blockDim,const double * x,const double * beta,double * y) nogil



cdef hipsparseStatus_t hipsparseCbsrxmv(void * handle,hipsparseDirection_t dir,hipsparseOperation_t trans,int sizeOfMask,int mb,int nb,int nnzb,float2 * alpha,void *const descr,float2 * bsrVal,const int * bsrMaskPtr,const int * bsrRowPtr,const int * bsrEndPtr,const int * bsrColInd,int blockDim,float2 * x,float2 * beta,float2 * y) nogil



cdef hipsparseStatus_t hipsparseZbsrxmv(void * handle,hipsparseDirection_t dir,hipsparseOperation_t trans,int sizeOfMask,int mb,int nb,int nnzb,double2 * alpha,void *const descr,double2 * bsrVal,const int * bsrMaskPtr,const int * bsrRowPtr,const int * bsrEndPtr,const int * bsrColInd,int blockDim,double2 * x,double2 * beta,double2 * y) nogil


#  \ingroup level2_module
# \brief Sparse triangular solve using BSR storage format
# 
# \details
# \p hipsparseXbsrsv2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
# structural or numerical zero has been found during hipsparseXbsrsv2_analysis() or
# hipsparseXbsrsv2_solve() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
# is stored in \p position, using same index base as the BSR matrix.
# 
# \p position can be in host or device memory. If no zero pivot has been found,
# \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
# \note \p hipsparseXbsrsv2_zeroPivot is a blocking function. It might influence
# performance negatively.
cdef hipsparseStatus_t hipsparseXbsrsv2_zeroPivot(void * handle,bsrsv2Info_t info,int * position) nogil


#  \ingroup level2_module
# \brief Sparse triangular solve using BSR storage format
# 
# \details
# \p hipsparseXbsrsv2_bufferSize returns the size of the temporary storage buffer that
# is required by hipsparseXbsrsv2_analysis() and hipsparseXbsrsv2_solve(). The
# temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseSbsrsv2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDbsrsv2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCbsrsv2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZbsrsv2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,int * pBufferSizeInBytes) nogil


#  \ingroup level2_module
# \brief Sparse triangular solve using BSR storage format
# 
# \details
# \p hipsparseXbsrsv2_bufferSizeExt returns the size of the temporary storage buffer that
# is required by hipsparseXbsrsv2_analysis() and hipsparseXbsrsv2_solve(). The
# temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseSbsrsv2_bufferSizeExt(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseDbsrsv2_bufferSizeExt(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseCbsrsv2_bufferSizeExt(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseZbsrsv2_bufferSizeExt(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,unsigned long * pBufferSize) nogil


#  \ingroup level2_module
# \brief Sparse triangular solve using BSR storage format
# 
# \details
# \p hipsparseXbsrsv2_analysis performs the analysis step for hipsparseXbsrsv2_solve().
# 
# \note
# If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSbsrsv2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,const float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDbsrsv2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,const double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCbsrsv2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZbsrsv2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


#  \ingroup level2_module
# \brief Sparse triangular solve using BSR storage format
# 
# \details
# \p hipsparseXbsrsv2_solve solves a sparse triangular linear system of a sparse
# \f$m \times m\f$ matrix, defined in BSR storage format, a dense solution vector
# \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
# \f[
#   op(A) \cdot y = \alpha \cdot x,
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# 
# \p hipsparseXbsrsv2_solve requires a user allocated temporary buffer. Its size is
# returned by hipsparseXbsrsv2_bufferSize() or hipsparseXbsrsv2_bufferSizeExt().
# Furthermore, analysis meta data is required. It can be obtained by
# hipsparseXbsrsv2_analysis(). \p hipsparseXbsrsv2_solve reports the first zero pivot
# (either numerical or structural zero). The zero pivot status can be checked calling
# hipsparseXbsrsv2_zeroPivot(). If
# \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
# reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
# 
# \note
# The sparse BSR matrix has to be sorted.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# 
# \note
# Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE and
# \p trans == \ref HIPSPARSE_OPERATION_TRANSPOSE is supported.
cdef hipsparseStatus_t hipsparseSbsrsv2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,const float * alpha,void *const descrA,const float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,const float * f,float * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDbsrsv2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,const double * alpha,void *const descrA,const double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,const double * f,double * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCbsrsv2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,float2 * alpha,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,float2 * f,float2 * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZbsrsv2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,double2 * alpha,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,double2 * f,double2 * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


# \ingroup level2_module
# \brief Dense matrix sparse vector multiplication
# 
# \details
# \p hipsparseXgemvi_bufferSize returns the size of the temporary storage buffer
# required by hipsparseXgemvi(). The temporary storage buffer must be allocated by the
# user.
cdef hipsparseStatus_t hipsparseSgemvi_bufferSize(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,int * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseDgemvi_bufferSize(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,int * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseCgemvi_bufferSize(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,int * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseZgemvi_bufferSize(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,int * pBufferSize) nogil


# \ingroup level2_module
# \brief Dense matrix sparse vector multiplication
# 
# \details
# \p hipsparseXgemvi multiplies the scalar \f$\alpha\f$ with a dense \f$m \times n\f$
# matrix \f$A\f$ and the sparse vector \f$x\f$ and adds the result to the dense vector
# \f$y\f$ that is multiplied by the scalar \f$\beta\f$, such that
# \f[
#   y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# 
# \p hipsparseXgemvi requires a user allocated temporary buffer. Its size is returned
# by hipsparseXgemvi_bufferSize().
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# 
# \note
# Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
cdef hipsparseStatus_t hipsparseSgemvi(void * handle,hipsparseOperation_t transA,int m,int n,const float * alpha,const float * A,int lda,int nnz,const float * x,const int * xInd,const float * beta,float * y,hipsparseIndexBase_t idxBase,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDgemvi(void * handle,hipsparseOperation_t transA,int m,int n,const double * alpha,const double * A,int lda,int nnz,const double * x,const int * xInd,const double * beta,double * y,hipsparseIndexBase_t idxBase,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCgemvi(void * handle,hipsparseOperation_t transA,int m,int n,float2 * alpha,float2 * A,int lda,int nnz,float2 * x,const int * xInd,float2 * beta,float2 * y,hipsparseIndexBase_t idxBase,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZgemvi(void * handle,hipsparseOperation_t transA,int m,int n,double2 * alpha,double2 * A,int lda,int nnz,double2 * x,const int * xInd,double2 * beta,double2 * y,hipsparseIndexBase_t idxBase,void * pBuffer) nogil


# \ingroup level3_module
# \brief Sparse matrix dense matrix multiplication using BSR storage format
# 
# \details
# \p hipsparseXbsrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$mb \times kb\f$
# matrix \f$A\f$, defined in BSR storage format, and the dense \f$k \times n\f$
# matrix \f$B\f$ (where \f$k = block\_dim \times kb\f$) and adds the result to the dense
# \f$m \times n\f$ matrix \f$C\f$ (where \f$m = block\_dim \times mb\f$) that
# is multiplied by the scalar \f$\beta\f$, such that
# \f[
#   C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#   \end{array}
#   \right.
# \f]
# and
# \f[
#   op(B) = \left\{
#   \begin{array}{ll}
#       B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
#   \end{array}
#   \right.
# \f]
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# 
# \note
# Currently, only \p trans_A == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
cdef hipsparseStatus_t hipsparseSbsrmm(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transB,int mb,int n,int kb,int nnzb,const float * alpha,void *const descrA,const float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,const float * B,int ldb,const float * beta,float * C,int ldc) nogil



cdef hipsparseStatus_t hipsparseDbsrmm(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transB,int mb,int n,int kb,int nnzb,const double * alpha,void *const descrA,const double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,const double * B,int ldb,const double * beta,double * C,int ldc) nogil



cdef hipsparseStatus_t hipsparseCbsrmm(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transB,int mb,int n,int kb,int nnzb,float2 * alpha,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,float2 * B,int ldb,float2 * beta,float2 * C,int ldc) nogil



cdef hipsparseStatus_t hipsparseZbsrmm(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transB,int mb,int n,int kb,int nnzb,double2 * alpha,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,double2 * B,int ldb,double2 * beta,double2 * C,int ldc) nogil


#  \ingroup level3_module
# \brief Sparse matrix dense matrix multiplication using CSR storage format
# 
# \details
# \p hipsparseXcsrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
# matrix \f$A\f$, defined in CSR storage format, and the dense \f$k \times n\f$
# matrix \f$B\f$ and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
# is multiplied by the scalar \f$\beta\f$, such that
# \f[
#   C := \alpha \cdot op(A) \cdot B + \beta \cdot C,
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# 
# \code{.c}
#     for(i = 0; i < ldc; ++i)
#     {
#         for(j = 0; j < n; ++j)
#         {
#             C[i][j] = beta * C[i][j];
# 
#             for(k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
#             {
#                 C[i][j] += alpha * csr_val[k] * B[csr_col_ind[k]][j];
#             }
#         }
#     }
# \endcode
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseScsrmm(void * handle,hipsparseOperation_t transA,int m,int n,int k,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * B,int ldb,const float * beta,float * C,int ldc) nogil



cdef hipsparseStatus_t hipsparseDcsrmm(void * handle,hipsparseOperation_t transA,int m,int n,int k,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * B,int ldb,const double * beta,double * C,int ldc) nogil



cdef hipsparseStatus_t hipsparseCcsrmm(void * handle,hipsparseOperation_t transA,int m,int n,int k,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * B,int ldb,float2 * beta,float2 * C,int ldc) nogil



cdef hipsparseStatus_t hipsparseZcsrmm(void * handle,hipsparseOperation_t transA,int m,int n,int k,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * B,int ldb,double2 * beta,double2 * C,int ldc) nogil


#  \ingroup level3_module
# \brief Sparse matrix dense matrix multiplication using CSR storage format
# 
# \details
# \p hipsparseXcsrmm2 multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
# matrix \f$A\f$, defined in CSR storage format, and the dense \f$k \times n\f$
# matrix \f$B\f$ and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
# is multiplied by the scalar \f$\beta\f$, such that
# \f[
#   C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# and
# \f[
#   op(B) = \left\{
#   \begin{array}{ll}
#       B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# 
# \code{.c}
#     for(i = 0; i < ldc; ++i)
#     {
#         for(j = 0; j < n; ++j)
#         {
#             C[i][j] = beta * C[i][j];
# 
#             for(k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
#             {
#                 C[i][j] += alpha * csr_val[k] * B[csr_col_ind[k]][j];
#             }
#         }
#     }
# \endcode
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseScsrmm2(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * B,int ldb,const float * beta,float * C,int ldc) nogil



cdef hipsparseStatus_t hipsparseDcsrmm2(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * B,int ldb,const double * beta,double * C,int ldc) nogil



cdef hipsparseStatus_t hipsparseCcsrmm2(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * B,int ldb,float2 * beta,float2 * C,int ldc) nogil



cdef hipsparseStatus_t hipsparseZcsrmm2(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * B,int ldb,double2 * beta,double2 * C,int ldc) nogil


#  \ingroup level3_module
# \brief Sparse triangular system solve using BSR storage format
# 
# \details
# \p hipsparseXbsrsm2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
# structural or numerical zero has been found during hipsparseXbsrsm2_analysis() or
# hipsparseXbsrsm2_solve() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
# is stored in \p position, using same index base as the BSR matrix.
# 
# \p position can be in host or device memory. If no zero pivot has been found,
# \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
# \note \p hipsparseXbsrsm2_zeroPivot is a blocking function. It might influence
# performance negatively.
cdef hipsparseStatus_t hipsparseXbsrsm2_zeroPivot(void * handle,bsrsm2Info_t info,int * position) nogil


#  \ingroup level3_module
# \brief Sparse triangular system solve using BSR storage format
# 
# \details
# \p hipsparseXbsrsm2_buffer_size returns the size of the temporary storage buffer that
# is required by hipsparseXbsrsm2_analysis() and hipsparseXbsrsm2_solve(). The
# temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseSbsrsm2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDbsrsm2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCbsrsm2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZbsrsm2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,int * pBufferSizeInBytes) nogil


#  \ingroup level3_module
# \brief Sparse triangular system solve using BSR storage format
# 
# \details
# \p hipsparseXbsrsm2_analysis performs the analysis step for hipsparseXbsrsm2_solve().
# 
# \note
# If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSbsrsm2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,const float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDbsrsm2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,const double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCbsrsm2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZbsrsm2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


#  \ingroup level3_module
# \brief Sparse triangular system solve using BSR storage format
# 
# \details
# \p hipsparseXbsrsm2_solve solves a sparse triangular linear system of a sparse
# \f$m \times m\f$ matrix, defined in BSR storage format, a dense solution matrix
# \f$X\f$ and the right-hand side matrix \f$B\f$ that is multiplied by \f$\alpha\f$, such that
# \f[
#   op(A) \cdot op(X) = \alpha \cdot op(B),
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# ,
# \f[
#   op(X) = \left\{
#   \begin{array}{ll}
#       X,   & \text{if trans_X == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       X^T, & \text{if trans_X == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       X^H, & \text{if trans_X == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# 
# \p hipsparseXbsrsm2_solve requires a user allocated temporary buffer. Its size is
# returned by hipsparseXbsrsm2_bufferSize(). Furthermore, analysis meta data is
# required. It can be obtained by hipsparseXbsrsm2_analysis(). \p hipsparseXbsrsm2_solve
# reports the first zero pivot (either numerical or structural zero). The zero pivot
# status can be checked calling hipsparseXbsrsm2_zeroPivot(). If
# \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
# reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
# 
# \note
# The sparse BSR matrix has to be sorted.
# 
# \note
# Operation type of B and X must match, if \f$op(B)=B, op(X)=X\f$.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# 
# \note
# Currently, only \p trans_A != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE and
# \p trans_X != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE is supported.
cdef hipsparseStatus_t hipsparseSbsrsm2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,const float * alpha,void *const descrA,const float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,const float * B,int ldb,float * X,int ldx,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDbsrsm2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,const double * alpha,void *const descrA,const double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,const double * B,int ldb,double * X,int ldx,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCbsrsm2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,float2 * alpha,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,float2 * B,int ldb,float2 * X,int ldx,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZbsrsm2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,double2 * alpha,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,double2 * B,int ldb,double2 * X,int ldx,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


#  \ingroup level3_module
# \brief Sparse triangular system solve using CSR storage format
# 
# \details
# \p hipsparseXcsrsm2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
# structural or numerical zero has been found during hipsparseXcsrsm2_analysis() or
# hipsparseXcsrsm2_solve() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
# is stored in \p position, using same index base as the CSR matrix.
# 
# \p position can be in host or device memory. If no zero pivot has been found,
# \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
# \note \p hipsparseXcsrsm2_zeroPivot is a blocking function. It might influence
# performance negatively.
cdef hipsparseStatus_t hipsparseXcsrsm2_zeroPivot(void * handle,csrsm2Info_t info,int * position) nogil


#  \ingroup level3_module
# \brief Sparse triangular system solve using CSR storage format
# 
# \details
# \p hipsparseXcsrsm2_bufferSizeExt returns the size of the temporary storage buffer
# that is required by hipsparseXcsrsm2_analysis() and hipsparseXcsrsm2_solve(). The
# temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseScsrsm2_bufferSizeExt(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseDcsrsm2_bufferSizeExt(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseCcsrsm2_bufferSizeExt(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseZcsrsm2_bufferSizeExt(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,unsigned long * pBufferSize) nogil


#  \ingroup level3_module
# \brief Sparse triangular system solve using CSR storage format
# 
# \details
# \p hipsparseXcsrsm2_analysis performs the analysis step for hipsparseXcsrsm2_solve().
# 
# \note
# If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseScsrsm2_analysis(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDcsrsm2_analysis(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCcsrsm2_analysis(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZcsrsm2_analysis(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


#  \ingroup level3_module
# \brief Sparse triangular system solve using CSR storage format
# 
# \details
# \p hipsparseXcsrsm2_solve solves a sparse triangular linear system of a sparse
# \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution matrix
# \f$X\f$ and the right-hand side matrix \f$B\f$ that is multiplied by \f$\alpha\f$, such that
# \f[
#   op(A) \cdot op(X) = \alpha \cdot op(B),
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# ,
# \f[
#   op(B) = \left\{
#   \begin{array}{ll}
#       B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# and
# \f[
#   op(X) = \left\{
#   \begin{array}{ll}
#       X,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       X^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       X^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# 
# \p hipsparseXcsrsm2_solve requires a user allocated temporary buffer. Its size is
# returned by hipsparseXcsrsm2_bufferSizeExt(). Furthermore, analysis meta data is
# required. It can be obtained by hipsparseXcsrsm2_analysis().
# \p hipsparseXcsrsm2_solve reports the first zero pivot (either numerical or structural
# zero). The zero pivot status can be checked calling hipsparseXcsrsm2_zeroPivot(). If
# \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
# reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
# 
# \note
# The sparse CSR matrix has to be sorted. This can be achieved by calling
# hipsparseXcsrsort().
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# 
# \note
# Currently, only \p trans_A != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE and
# \p trans_B != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE is supported.
cdef hipsparseStatus_t hipsparseScsrsm2_solve(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDcsrsm2_solve(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCcsrsm2_solve(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZcsrsm2_solve(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


#  \ingroup level3_module
# \brief Dense matrix sparse matrix multiplication using CSR storage format
# 
# \details
# \p hipsparseXgemmi multiplies the scalar \f$\alpha\f$ with a dense \f$m \times k\f$
# matrix \f$A\f$ and the sparse \f$k \times n\f$ matrix \f$B\f$, defined in CSR
# storage format and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
# is multiplied by the scalar \f$\beta\f$, such that
# \f[
#   C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# and
# \f[
#   op(B) = \left\{
#   \begin{array}{ll}
#       B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSgemmi(void * handle,int m,int n,int k,int nnz,const float * alpha,const float * A,int lda,const float * cscValB,const int * cscColPtrB,const int * cscRowIndB,const float * beta,float * C,int ldc) nogil



cdef hipsparseStatus_t hipsparseDgemmi(void * handle,int m,int n,int k,int nnz,const double * alpha,const double * A,int lda,const double * cscValB,const int * cscColPtrB,const int * cscRowIndB,const double * beta,double * C,int ldc) nogil



cdef hipsparseStatus_t hipsparseCgemmi(void * handle,int m,int n,int k,int nnz,float2 * alpha,float2 * A,int lda,float2 * cscValB,const int * cscColPtrB,const int * cscRowIndB,float2 * beta,float2 * C,int ldc) nogil



cdef hipsparseStatus_t hipsparseZgemmi(void * handle,int m,int n,int k,int nnz,double2 * alpha,double2 * A,int lda,double2 * cscValB,const int * cscColPtrB,const int * cscRowIndB,double2 * beta,double2 * C,int ldc) nogil


#  \ingroup extra_module
# \brief Sparse matrix sparse matrix addition using CSR storage format
# 
# \details
# \p hipsparseXcsrgeamNnz computes the total CSR non-zero elements and the CSR row
# offsets, that point to the start of every row of the sparse CSR matrix, of the
# resulting matrix C. It is assumed that \p csr_row_ptr_C has been allocated with
# size \p m + 1.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# \note
# Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
cdef hipsparseStatus_t hipsparseXcsrgeamNnz(void * handle,int m,int n,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr) nogil


#  \ingroup extra_module
# \brief Sparse matrix sparse matrix addition using CSR storage format
# 
# \details
# \p hipsparseXcsrgeam multiplies the scalar \f$\alpha\f$ with the sparse
# \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
# scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
# storage format, and adds both resulting matrices to obtain the sparse
# \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
# \f[
#   C := \alpha \cdot A + \beta \cdot B.
# \f]
# 
# It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
# \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
# \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
# the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgeamNnz().
# 
# \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
# \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
# \note This function is non blocking and executed asynchronously with respect to the
#       host. It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseScsrgeam(void * handle,int m,int n,const float * alpha,void *const descrA,int nnzA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * beta,void *const descrB,int nnzB,const float * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,float * csrValC,int * csrRowPtrC,int * csrColIndC) nogil



cdef hipsparseStatus_t hipsparseDcsrgeam(void * handle,int m,int n,const double * alpha,void *const descrA,int nnzA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * beta,void *const descrB,int nnzB,const double * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,double * csrValC,int * csrRowPtrC,int * csrColIndC) nogil



cdef hipsparseStatus_t hipsparseCcsrgeam(void * handle,int m,int n,float2 * alpha,void *const descrA,int nnzA,float2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,float2 * beta,void *const descrB,int nnzB,float2 * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,float2 * csrValC,int * csrRowPtrC,int * csrColIndC) nogil



cdef hipsparseStatus_t hipsparseZcsrgeam(void * handle,int m,int n,double2 * alpha,void *const descrA,int nnzA,double2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,double2 * beta,void *const descrB,int nnzB,double2 * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,double2 * csrValC,int * csrRowPtrC,int * csrColIndC) nogil


#  \ingroup extra_module
# \brief Sparse matrix sparse matrix multiplication using CSR storage format
# 
# \details
# \p hipsparseXcsrgeam2_bufferSizeExt returns the size of the temporary storage buffer
# that is required by hipsparseXcsrgeam2Nnz() and hipsparseXcsrgeam2(). The temporary
# storage buffer must be allocated by the user.
# 
# \note
# Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
cdef hipsparseStatus_t hipsparseScsrgeam2_bufferSizeExt(void * handle,int m,int n,const float * alpha,void *const descrA,int nnzA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * beta,void *const descrB,int nnzB,const float * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,const float * csrSortedValC,const int * csrSortedRowPtrC,const int * csrSortedColIndC,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDcsrgeam2_bufferSizeExt(void * handle,int m,int n,const double * alpha,void *const descrA,int nnzA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * beta,void *const descrB,int nnzB,const double * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,const double * csrSortedValC,const int * csrSortedRowPtrC,const int * csrSortedColIndC,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCcsrgeam2_bufferSizeExt(void * handle,int m,int n,float2 * alpha,void *const descrA,int nnzA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * beta,void *const descrB,int nnzB,float2 * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,float2 * csrSortedValC,const int * csrSortedRowPtrC,const int * csrSortedColIndC,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZcsrgeam2_bufferSizeExt(void * handle,int m,int n,double2 * alpha,void *const descrA,int nnzA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * beta,void *const descrB,int nnzB,double2 * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,double2 * csrSortedValC,const int * csrSortedRowPtrC,const int * csrSortedColIndC,unsigned long * pBufferSizeInBytes) nogil


#  \ingroup extra_module
# \brief Sparse matrix sparse matrix addition using CSR storage format
# 
# \details
# \p hipsparseXcsrgeam2Nnz computes the total CSR non-zero elements and the CSR row
# offsets, that point to the start of every row of the sparse CSR matrix, of the
# resulting matrix C. It is assumed that \p csr_row_ptr_C has been allocated with
# size \p m + 1.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# \note
# Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
cdef hipsparseStatus_t hipsparseXcsrgeam2Nnz(void * handle,int m,int n,void *const descrA,int nnzA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,void *const descrB,int nnzB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,int * csrSortedRowPtrC,int * nnzTotalDevHostPtr,void * workspace) nogil


#  \ingroup extra_module
# \brief Sparse matrix sparse matrix addition using CSR storage format
# 
# \details
# \p hipsparseXcsrgeam2 multiplies the scalar \f$\alpha\f$ with the sparse
# \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
# scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
# storage format, and adds both resulting matrices to obtain the sparse
# \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
# \f[
#   C := \alpha \cdot A + \beta \cdot B.
# \f]
# 
# It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
# \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
# \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
# the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgeam2Nnz().
# 
# \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
# \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
# \note This function is non blocking and executed asynchronously with respect to the
#       host. It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseScsrgeam2(void * handle,int m,int n,const float * alpha,void *const descrA,int nnzA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * beta,void *const descrB,int nnzB,const float * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,float * csrSortedValC,int * csrSortedRowPtrC,int * csrSortedColIndC,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDcsrgeam2(void * handle,int m,int n,const double * alpha,void *const descrA,int nnzA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * beta,void *const descrB,int nnzB,const double * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,double * csrSortedValC,int * csrSortedRowPtrC,int * csrSortedColIndC,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCcsrgeam2(void * handle,int m,int n,float2 * alpha,void *const descrA,int nnzA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * beta,void *const descrB,int nnzB,float2 * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,float2 * csrSortedValC,int * csrSortedRowPtrC,int * csrSortedColIndC,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZcsrgeam2(void * handle,int m,int n,double2 * alpha,void *const descrA,int nnzA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * beta,void *const descrB,int nnzB,double2 * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,double2 * csrSortedValC,int * csrSortedRowPtrC,int * csrSortedColIndC,void * pBuffer) nogil


#  \ingroup extra_module
# \brief Sparse matrix sparse matrix multiplication using CSR storage format
# 
# \details
# \p hipsparseXcsrgemmNnz computes the total CSR non-zero elements and the CSR row
# offsets, that point to the start of every row of the sparse CSR matrix, of the
# resulting multiplied matrix C. It is assumed that \p csr_row_ptr_C has been allocated
# with size \p m + 1.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# 
# \note
# Please note, that for matrix products with more than 8192 intermediate products per
# row, additional temporary storage buffer is allocated by the algorithm.
# 
# \note
# Currently, only \p trans_A == \p trans_B == \ref HIPSPARSE_OPERATION_NONE is
# supported.
# 
# \note
# Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
cdef hipsparseStatus_t hipsparseXcsrgemmNnz(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr) nogil


#  \ingroup extra_module
# \brief Sparse matrix sparse matrix multiplication using CSR storage format
# 
# \details
# \p hipsparseXcsrgemm multiplies the sparse \f$m \times k\f$ matrix \f$A\f$, defined in
# CSR storage format with the sparse \f$k \times n\f$ matrix \f$B\f$, defined in CSR
# storage format, and stores the result in the sparse \f$m \times n\f$ matrix \f$C\f$,
# defined in CSR storage format, such that
# \f[
#   C := op(A) \cdot op(B),
# \f]
# with
# \f[
#   op(A) = \left\{
#   \begin{array}{ll}
#       A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# and
# \f[
#   op(B) = \left\{
#   \begin{array}{ll}
#       B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#       B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
#       B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#   \end{array}
#   \right.
# \f]
# 
# It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
# \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
# \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
# the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgemmNnz().
# 
# \note Currently, only \p trans_A == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
# \note Currently, only \p trans_B == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
# \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
# \note This function is non blocking and executed asynchronously with respect to the
#       host. It may return before the actual computation has finished.
# \note Please note, that for matrix products with more than 4096 non-zero entries per
# row, additional temporary storage buffer is allocated by the algorithm.
cdef hipsparseStatus_t hipsparseScsrgemm(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,void *const descrA,int nnzA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const float * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,float * csrValC,const int * csrRowPtrC,int * csrColIndC) nogil



cdef hipsparseStatus_t hipsparseDcsrgemm(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,void *const descrA,int nnzA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const double * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,double * csrValC,const int * csrRowPtrC,int * csrColIndC) nogil



cdef hipsparseStatus_t hipsparseCcsrgemm(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,void *const descrA,int nnzA,float2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,float2 * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,float2 * csrValC,const int * csrRowPtrC,int * csrColIndC) nogil



cdef hipsparseStatus_t hipsparseZcsrgemm(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,void *const descrA,int nnzA,double2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,double2 * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,double2 * csrValC,const int * csrRowPtrC,int * csrColIndC) nogil


#  \ingroup extra_module
# \brief Sparse matrix sparse matrix multiplication using CSR storage format
# 
# \details
# \p hipsparseXcsrgemm2_bufferSizeExt returns the size of the temporary storage buffer
# that is required by hipsparseXcsrgemm2Nnz() and hipsparseXcsrgemm2(). The temporary
# storage buffer must be allocated by the user.
# 
# \note
# Please note, that for matrix products with more than 4096 non-zero entries per row,
# additional temporary storage buffer is allocated by the algorithm.
# 
# \note
# Please note, that for matrix products with more than 8192 intermediate products per
# row, additional temporary storage buffer is allocated by the algorithm.
# 
# \note
# Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
cdef hipsparseStatus_t hipsparseScsrgemm2_bufferSizeExt(void * handle,int m,int n,int k,const float * alpha,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,const float * beta,void *const descrD,int nnzD,const int * csrRowPtrD,const int * csrColIndD,csrgemm2Info_t info,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDcsrgemm2_bufferSizeExt(void * handle,int m,int n,int k,const double * alpha,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,const double * beta,void *const descrD,int nnzD,const int * csrRowPtrD,const int * csrColIndD,csrgemm2Info_t info,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCcsrgemm2_bufferSizeExt(void * handle,int m,int n,int k,float2 * alpha,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,float2 * beta,void *const descrD,int nnzD,const int * csrRowPtrD,const int * csrColIndD,csrgemm2Info_t info,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZcsrgemm2_bufferSizeExt(void * handle,int m,int n,int k,double2 * alpha,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,double2 * beta,void *const descrD,int nnzD,const int * csrRowPtrD,const int * csrColIndD,csrgemm2Info_t info,unsigned long * pBufferSizeInBytes) nogil


#  \ingroup extra_module
# \brief Sparse matrix sparse matrix multiplication using CSR storage format
# 
# \details
# \p hipsparseXcsrgemm2Nnz computes the total CSR non-zero elements and the CSR row
# offsets, that point to the start of every row of the sparse CSR matrix, of the
# resulting multiplied matrix C. It is assumed that \p csr_row_ptr_C has been allocated
# with size \p m + 1.
# The required buffer size can be obtained by hipsparseXcsrgemm2_bufferSizeExt().
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
# 
# \note
# Please note, that for matrix products with more than 8192 intermediate products per
# row, additional temporary storage buffer is allocated by the algorithm.
# 
# \note
# Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
cdef hipsparseStatus_t hipsparseXcsrgemm2Nnz(void * handle,int m,int n,int k,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,void *const descrD,int nnzD,const int * csrRowPtrD,const int * csrColIndD,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr,csrgemm2Info_t info,void * pBuffer) nogil


#  \ingroup extra_module
# \brief Sparse matrix sparse matrix multiplication using CSR storage format
# 
# \details
# \p hipsparseXcsrgemm2 multiplies the scalar \f$\alpha\f$ with the sparse
# \f$m \times k\f$ matrix \f$A\f$, defined in CSR storage format, and the sparse
# \f$k \times n\f$ matrix \f$B\f$, defined in CSR storage format, and adds the result
# to the sparse \f$m \times n\f$ matrix \f$D\f$ that is multiplied by \f$\beta\f$. The
# final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$, defined in CSR
# storage format, such
# that
# \f[
#   C := \alpha \cdot A \cdot B + \beta \cdot D
# \f]
# 
# It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
# \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
# \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
# the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgemm2Nnz(). The
# required buffer size for the computation can be obtained by
# hipsparseXcsrgemm2_bufferSizeExt().
# 
# \note If \f$\alpha == 0\f$, then \f$C = \beta \cdot D\f$ will be computed.
# \note If \f$\beta == 0\f$, then \f$C = \alpha \cdot A \cdot B\f$ will be computed.
# \note \f$\alpha == beta == 0\f$ is invalid.
# \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
# \note This function is non blocking and executed asynchronously with respect to the
#       host. It may return before the actual computation has finished.
# \note Please note, that for matrix products with more than 4096 non-zero entries per
# row, additional temporary storage buffer is allocated by the algorithm.
cdef hipsparseStatus_t hipsparseScsrgemm2(void * handle,int m,int n,int k,const float * alpha,void *const descrA,int nnzA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const float * csrValB,const int * csrRowPtrB,const int * csrColIndB,const float * beta,void *const descrD,int nnzD,const float * csrValD,const int * csrRowPtrD,const int * csrColIndD,void *const descrC,float * csrValC,const int * csrRowPtrC,int * csrColIndC,csrgemm2Info_t info,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDcsrgemm2(void * handle,int m,int n,int k,const double * alpha,void *const descrA,int nnzA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const double * csrValB,const int * csrRowPtrB,const int * csrColIndB,const double * beta,void *const descrD,int nnzD,const double * csrValD,const int * csrRowPtrD,const int * csrColIndD,void *const descrC,double * csrValC,const int * csrRowPtrC,int * csrColIndC,csrgemm2Info_t info,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCcsrgemm2(void * handle,int m,int n,int k,float2 * alpha,void *const descrA,int nnzA,float2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,float2 * csrValB,const int * csrRowPtrB,const int * csrColIndB,float2 * beta,void *const descrD,int nnzD,float2 * csrValD,const int * csrRowPtrD,const int * csrColIndD,void *const descrC,float2 * csrValC,const int * csrRowPtrC,int * csrColIndC,csrgemm2Info_t info,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZcsrgemm2(void * handle,int m,int n,int k,double2 * alpha,void *const descrA,int nnzA,double2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,double2 * csrValB,const int * csrRowPtrB,const int * csrColIndB,double2 * beta,void *const descrD,int nnzD,double2 * csrValD,const int * csrRowPtrD,const int * csrColIndD,void *const descrC,double2 * csrValC,const int * csrRowPtrC,int * csrColIndC,csrgemm2Info_t info,void * pBuffer) nogil


# \ingroup precond_module
# \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
# format
# 
# \details
# \p hipsparseXbsrilu02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
# structural or numerical zero has been found during hipsparseXbsrilu02_analysis() or
# hipsparseXbsrilu02() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is
# stored in \p position, using same index base as the BSR matrix.
# 
# \p position can be in host or device memory. If no zero pivot has been found,
# \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
# \note
# If a zero pivot is found, \p position \f$=j\f$ means that either the diagonal block
# \f$A_{j,j}\f$ is missing (structural zero) or the diagonal block \f$A_{j,j}\f$ is not
# invertible (numerical zero).
# 
# \note \p hipsparseXbsrilu02_zeroPivot is a blocking function. It might influence
# performance negatively.
cdef hipsparseStatus_t hipsparseXbsrilu02_zeroPivot(void * handle,bsrilu02Info_t info,int * position) nogil


# \ingroup precond_module
# \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
# format
# 
# \details
# \p hipsparseXbsrilu02_numericBoost enables the user to replace a numerical value in
# an incomplete LU factorization. \p tol is used to determine whether a numerical value
# is replaced by \p boost_val, such that \f$A_{j,j} = \text{boost_val}\f$ if
# \f$\text{tol} \ge \left|A_{j,j}\right|\f$.
# 
# \note The boost value is enabled by setting \p enable_boost to 1 or disabled by
# setting \p enable_boost to 0.
# 
# \note \p tol and \p boost_val can be in host or device memory.
cdef hipsparseStatus_t hipsparseSbsrilu02_numericBoost(void * handle,bsrilu02Info_t info,int enable_boost,double * tol,float * boost_val) nogil



cdef hipsparseStatus_t hipsparseDbsrilu02_numericBoost(void * handle,bsrilu02Info_t info,int enable_boost,double * tol,double * boost_val) nogil



cdef hipsparseStatus_t hipsparseCbsrilu02_numericBoost(void * handle,bsrilu02Info_t info,int enable_boost,double * tol,float2 * boost_val) nogil



cdef hipsparseStatus_t hipsparseZbsrilu02_numericBoost(void * handle,bsrilu02Info_t info,int enable_boost,double * tol,double2 * boost_val) nogil


# \ingroup precond_module
# \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
# format
# 
# \details
# \p hipsparseXbsrilu02_bufferSize returns the size of the temporary storage buffer
# that is required by hipsparseXbsrilu02_analysis() and hipsparseXbsrilu02_solve().
# The temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseSbsrilu02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDbsrilu02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCbsrilu02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZbsrilu02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,int * pBufferSizeInBytes) nogil


# \ingroup precond_module
# \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
# format
# 
# \details
# \p hipsparseXbsrilu02_analysis performs the analysis step for hipsparseXbsrilu02().
# 
# \note
# If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSbsrilu02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDbsrilu02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCbsrilu02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZbsrilu02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


# \ingroup precond_module
# \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
# format
# 
# \details
# \p hipsparseXbsrilu02 computes the incomplete LU factorization with 0 fill-ins and no
# pivoting of a sparse \f$mb \times mb\f$ BSR matrix \f$A\f$, such that
# \f[
#   A \approx LU
# \f]
# 
# \p hipsparseXbsrilu02 requires a user allocated temporary buffer. Its size is
# returned by hipsparseXbsrilu02_bufferSize(). Furthermore, analysis meta data is
# required. It can be obtained by hipsparseXbsrilu02_analysis(). \p hipsparseXbsrilu02
# reports the first zero pivot (either numerical or structural zero). The zero pivot
# status can be obtained by calling hipsparseXbsrilu02_zeroPivot().
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSbsrilu02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float * bsrSortedValA_valM,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDbsrilu02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double * bsrSortedValA_valM,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCbsrilu02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float2 * bsrSortedValA_valM,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZbsrilu02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double2 * bsrSortedValA_valM,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


#  \ingroup precond_module
# \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
# storage format
# 
# \details
# \p hipsparseXcsrilu02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
# structural or numerical zero has been found during hipsparseXcsrilu02() computation.
# The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, using same
# index base as the CSR matrix.
# 
# \p position can be in host or device memory. If no zero pivot has been found,
# \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
# \note \p hipsparseXcsrilu02_zeroPivot is a blocking function. It might influence
# performance negatively.
cdef hipsparseStatus_t hipsparseXcsrilu02_zeroPivot(void * handle,csrilu02Info_t info,int * position) nogil


# \ingroup precond_module
# \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR storage
# format
# 
# \details
# \p hipsparseXcsrilu02_numericBoost enables the user to replace a numerical value in
# an incomplete LU factorization. \p tol is used to determine whether a numerical value
# is replaced by \p boost_val, such that \f$A_{j,j} = \text{boost_val}\f$ if
# \f$\text{tol} \ge \left|A_{j,j}\right|\f$.
# 
# \note The boost value is enabled by setting \p enable_boost to 1 or disabled by
# setting \p enable_boost to 0.
# 
# \note \p tol and \p boost_val can be in host or device memory.
cdef hipsparseStatus_t hipsparseScsrilu02_numericBoost(void * handle,csrilu02Info_t info,int enable_boost,double * tol,float * boost_val) nogil



cdef hipsparseStatus_t hipsparseDcsrilu02_numericBoost(void * handle,csrilu02Info_t info,int enable_boost,double * tol,double * boost_val) nogil



cdef hipsparseStatus_t hipsparseCcsrilu02_numericBoost(void * handle,csrilu02Info_t info,int enable_boost,double * tol,float2 * boost_val) nogil



cdef hipsparseStatus_t hipsparseZcsrilu02_numericBoost(void * handle,csrilu02Info_t info,int enable_boost,double * tol,double2 * boost_val) nogil


#  \ingroup precond_module
# \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
# storage format
# 
# \details
# \p hipsparseXcsrilu02_bufferSize returns the size of the temporary storage buffer
# that is required by hipsparseXcsrilu02_analysis() and hipsparseXcsrilu02_solve(). the
# temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseScsrilu02_bufferSize(void * handle,int m,int nnz,void *const descrA,float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDcsrilu02_bufferSize(void * handle,int m,int nnz,void *const descrA,double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCcsrilu02_bufferSize(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZcsrilu02_bufferSize(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,int * pBufferSizeInBytes) nogil


#  \ingroup precond_module
# \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
# storage format
# 
# \details
# \p hipsparseXcsrilu02_bufferSizeExt returns the size of the temporary storage buffer
# that is required by hipsparseXcsrilu02_analysis() and hipsparseXcsrilu02_solve(). the
# temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseScsrilu02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseDcsrilu02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseCcsrilu02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseZcsrilu02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,unsigned long * pBufferSize) nogil


#  \ingroup precond_module
# \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
# storage format
# 
# \details
# \p hipsparseXcsrilu02_analysis performs the analysis step for hipsparseXcsrilu02().
# 
# \note
# If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseScsrilu02_analysis(void * handle,int m,int nnz,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDcsrilu02_analysis(void * handle,int m,int nnz,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCcsrilu02_analysis(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZcsrilu02_analysis(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


#  \ingroup precond_module
# \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
# storage format
# 
# \details
# \p hipsparseXcsrilu02 computes the incomplete LU factorization with 0 fill-ins and no
# pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
# \f[
#   A \approx LU
# \f]
# 
# \p hipsparseXcsrilu02 requires a user allocated temporary buffer. Its size is returned
# by hipsparseXcsrilu02_bufferSize() or hipsparseXcsrilu02_bufferSizeExt(). Furthermore,
# analysis meta data is required. It can be obtained by hipsparseXcsrilu02_analysis().
# \p hipsparseXcsrilu02 reports the first zero pivot (either numerical or structural
# zero). The zero pivot status can be obtained by calling hipsparseXcsrilu02_zeroPivot().
# 
# \note
# The sparse CSR matrix has to be sorted. This can be achieved by calling
# hipsparseXcsrsort().
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseScsrilu02(void * handle,int m,int nnz,void *const descrA,float * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDcsrilu02(void * handle,int m,int nnz,void *const descrA,double * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCcsrilu02(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZcsrilu02(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


# \ingroup precond_module
# \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
# storage format
# 
# \details
# \p hipsparseXbsric02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
# structural or numerical zero has been found during hipsparseXbsric02_analysis() or
# hipsparseXbsric02() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is
# stored in \p position, using same index base as the BSR matrix.
# 
# \p position can be in host or device memory. If no zero pivot has been found,
# \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
# \note
# If a zero pivot is found, \p position=j means that either the diagonal block \p A(j,j)
# is missing (structural zero) or the diagonal block \p A(j,j) is not positive definite
# (numerical zero).
# 
# \note \p hipsparseXbsric02_zeroPivot is a blocking function. It might influence
# performance negatively.
cdef hipsparseStatus_t hipsparseXbsric02_zeroPivot(void * handle,bsric02Info_t info,int * position) nogil


# \ingroup precond_module
# \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
# storage format
# 
# \details
# \p hipsparseXbsric02_bufferSize returns the size of the temporary storage buffer
# that is required by hipsparseXbsric02_analysis() and hipsparseXbsric02(). The
# temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseSbsric02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDbsric02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCbsric02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZbsric02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,int * pBufferSizeInBytes) nogil


# \ingroup precond_module
# \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
# storage format
# 
# \details
# \p hipsparseXbsric02_analysis performs the analysis step for hipsparseXbsric02().
# 
# \note
# If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSbsric02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,const float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDbsric02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,const double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCbsric02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZbsric02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


# \ingroup precond_module
# \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
# storage format
# 
# \details
# \p hipsparseXbsric02 computes the incomplete Cholesky factorization with 0 fill-ins
# and no pivoting of a sparse \f$mb \times mb\f$ BSR matrix \f$A\f$, such that
# \f[
#   A \approx LL^T
# \f]
# 
# \p hipsparseXbsric02 requires a user allocated temporary buffer. Its size is returned
# by hipsparseXbsric02_bufferSize(). Furthermore, analysis meta data is required. It
# can be obtained by hipsparseXbsric02_analysis(). \p hipsparseXbsric02 reports the
# first zero pivot (either numerical or structural zero). The zero pivot status can be
# obtained by calling hipsparseXbsric02_zeroPivot().
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSbsric02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDbsric02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCbsric02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZbsric02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


#  \ingroup precond_module
# \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
# storage format
# 
# \details
# \p hipsparseXcsric02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
# structural or numerical zero has been found during hipsparseXcsric02_analysis() or
# hipsparseXcsric02() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
# is stored in \p position, using same index base as the CSR matrix.
# 
# \p position can be in host or device memory. If no zero pivot has been found,
# \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
# \note \p hipsparseXcsric02_zeroPivot is a blocking function. It might influence
# performance negatively.
cdef hipsparseStatus_t hipsparseXcsric02_zeroPivot(void * handle,csric02Info_t info,int * position) nogil


#  \ingroup precond_module
# \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
# storage format
# 
# \details
# \p hipsparseXcsric02_bufferSize returns the size of the temporary storage buffer
# that is required by hipsparseXcsric02_analysis() and hipsparseXcsric02().
cdef hipsparseStatus_t hipsparseScsric02_bufferSize(void * handle,int m,int nnz,void *const descrA,float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDcsric02_bufferSize(void * handle,int m,int nnz,void *const descrA,double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCcsric02_bufferSize(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,int * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZcsric02_bufferSize(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,int * pBufferSizeInBytes) nogil


#  \ingroup precond_module
# \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
# storage format
# 
# \details
# \p hipsparseXcsric02_bufferSizeExt returns the size of the temporary storage buffer
# that is required by hipsparseXcsric02_analysis() and hipsparseXcsric02().
cdef hipsparseStatus_t hipsparseScsric02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseDcsric02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseCcsric02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,unsigned long * pBufferSize) nogil



cdef hipsparseStatus_t hipsparseZcsric02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,unsigned long * pBufferSize) nogil


#  \ingroup precond_module
# \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
# storage format
# 
# \details
# \p hipsparseXcsric02_analysis performs the analysis step for hipsparseXcsric02().
# 
# \note
# If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseScsric02_analysis(void * handle,int m,int nnz,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDcsric02_analysis(void * handle,int m,int nnz,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCcsric02_analysis(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZcsric02_analysis(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


#  \ingroup precond_module
# \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
# storage format
# 
# \details
# \p hipsparseXcsric02 computes the incomplete Cholesky factorization with 0 fill-ins
# and no pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
# \f[
#   A \approx LL^T
# \f]
# 
# \p hipsparseXcsric02 requires a user allocated temporary buffer. Its size is returned
# by hipsparseXcsric02_bufferSize() or hipsparseXcsric02_bufferSizeExt(). Furthermore,
# analysis meta data is required. It can be obtained by hipsparseXcsric02_analysis().
# \p hipsparseXcsric02 reports the first zero pivot (either numerical or structural
# zero). The zero pivot status can be obtained by calling hipsparseXcsric02_zeroPivot().
# 
# \note
# The sparse CSR matrix has to be sorted. This can be achieved by calling
# hipsparseXcsrsort().
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseScsric02(void * handle,int m,int nnz,void *const descrA,float * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDcsric02(void * handle,int m,int nnz,void *const descrA,double * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCcsric02(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZcsric02(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil


#  \ingroup precond_module
# \brief Tridiagonal solver with pivoting
# 
# \details
# \p hipsparseXgtsv2_bufferSize returns the size of the temporary storage buffer
# that is required by hipsparseXgtsv2(). The temporary storage buffer must be
# allocated by the user.
cdef hipsparseStatus_t hipsparseSgtsv2_bufferSizeExt(void * handle,int m,int n,const float * dl,const float * d,const float * du,const float * B,int ldb,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDgtsv2_bufferSizeExt(void * handle,int m,int n,const double * dl,const double * d,const double * du,const double * B,int db,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCgtsv2_bufferSizeExt(void * handle,int m,int n,float2 * dl,float2 * d,float2 * du,float2 * B,int ldb,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZgtsv2_bufferSizeExt(void * handle,int m,int n,double2 * dl,double2 * d,double2 * du,double2 * B,int ldb,unsigned long * pBufferSizeInBytes) nogil


#  \ingroup precond_module
# \brief Tridiagonal solver with pivoting
# 
# \details
# \p hipsparseXgtsv2 solves a tridiagonal system for multiple right hand sides using pivoting.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSgtsv2(void * handle,int m,int n,const float * dl,const float * d,const float * du,float * B,int ldb,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDgtsv2(void * handle,int m,int n,const double * dl,const double * d,const double * du,double * B,int ldb,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCgtsv2(void * handle,int m,int n,float2 * dl,float2 * d,float2 * du,float2 * B,int ldb,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZgtsv2(void * handle,int m,int n,double2 * dl,double2 * d,double2 * du,double2 * B,int ldb,void * pBuffer) nogil


#  \ingroup precond_module
# \brief Tridiagonal solver (no pivoting)
# 
# \details
# \p hipsparseXgtsv2_nopivot_bufferSizeExt returns the size of the temporary storage
# buffer that is required by hipsparseXgtsv2_nopivot(). The temporary storage buffer
# must be allocated by the user.
cdef hipsparseStatus_t hipsparseSgtsv2_nopivot_bufferSizeExt(void * handle,int m,int n,const float * dl,const float * d,const float * du,const float * B,int ldb,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDgtsv2_nopivot_bufferSizeExt(void * handle,int m,int n,const double * dl,const double * d,const double * du,const double * B,int db,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCgtsv2_nopivot_bufferSizeExt(void * handle,int m,int n,float2 * dl,float2 * d,float2 * du,float2 * B,int ldb,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZgtsv2_nopivot_bufferSizeExt(void * handle,int m,int n,double2 * dl,double2 * d,double2 * du,double2 * B,int ldb,unsigned long * pBufferSizeInBytes) nogil


#  \ingroup precond_module
# \brief Tridiagonal solver (no pivoting)
# 
# \details
# \p hipsparseXgtsv2_nopivot solves a tridiagonal linear system for multiple right-hand sides
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSgtsv2_nopivot(void * handle,int m,int n,const float * dl,const float * d,const float * du,float * B,int ldb,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDgtsv2_nopivot(void * handle,int m,int n,const double * dl,const double * d,const double * du,double * B,int ldb,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCgtsv2_nopivot(void * handle,int m,int n,float2 * dl,float2 * d,float2 * du,float2 * B,int ldb,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZgtsv2_nopivot(void * handle,int m,int n,double2 * dl,double2 * d,double2 * du,double2 * B,int ldb,void * pBuffer) nogil


#  \ingroup precond_module
# \brief Strided Batch tridiagonal solver (no pivoting)
# 
# \details
# \p hipsparseXgtsv2StridedBatch_bufferSizeExt returns the size of the temporary storage
# buffer that is required by hipsparseXgtsv2StridedBatch(). The temporary storage buffer
# must be allocated by the user.
cdef hipsparseStatus_t hipsparseSgtsv2StridedBatch_bufferSizeExt(void * handle,int m,const float * dl,const float * d,const float * du,const float * x,int batchCount,int batchStride,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDgtsv2StridedBatch_bufferSizeExt(void * handle,int m,const double * dl,const double * d,const double * du,const double * x,int batchCount,int batchStride,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCgtsv2StridedBatch_bufferSizeExt(void * handle,int m,float2 * dl,float2 * d,float2 * du,float2 * x,int batchCount,int batchStride,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZgtsv2StridedBatch_bufferSizeExt(void * handle,int m,double2 * dl,double2 * d,double2 * du,double2 * x,int batchCount,int batchStride,unsigned long * pBufferSizeInBytes) nogil


#  \ingroup precond_module
# \brief Strided Batch tridiagonal solver (no pivoting)
# 
# \details
# \p hipsparseXgtsv2StridedBatch solves a batched tridiagonal linear system
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSgtsv2StridedBatch(void * handle,int m,const float * dl,const float * d,const float * du,float * x,int batchCount,int batchStride,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDgtsv2StridedBatch(void * handle,int m,const double * dl,const double * d,const double * du,double * x,int batchCount,int batchStride,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCgtsv2StridedBatch(void * handle,int m,float2 * dl,float2 * d,float2 * du,float2 * x,int batchCount,int batchStride,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZgtsv2StridedBatch(void * handle,int m,double2 * dl,double2 * d,double2 * du,double2 * x,int batchCount,int batchStride,void * pBuffer) nogil


#  \ingroup precond_module
# \brief Interleaved Batch tridiagonal solver
# 
# \details
# \p hipsparseXgtsvInterleavedBatch_bufferSizeExt returns the size of the temporary storage
# buffer that is required by hipsparseXgtsvInterleavedBatch(). The temporary storage buffer
# must be allocated by the user.
cdef hipsparseStatus_t hipsparseSgtsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,const float * dl,const float * d,const float * du,const float * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDgtsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,const double * dl,const double * d,const double * du,const double * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCgtsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,float2 * dl,float2 * d,float2 * du,float2 * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZgtsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,double2 * dl,double2 * d,double2 * du,double2 * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil


#  \ingroup precond_module
# \brief Interleaved Batch tridiagonal solver
# 
# \details
# \p hipsparseXgtsvInterleavedBatch solves a batched tridiagonal linear system
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSgtsvInterleavedBatch(void * handle,int algo,int m,float * dl,float * d,float * du,float * x,int batchCount,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDgtsvInterleavedBatch(void * handle,int algo,int m,double * dl,double * d,double * du,double * x,int batchCount,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCgtsvInterleavedBatch(void * handle,int algo,int m,float2 * dl,float2 * d,float2 * du,float2 * x,int batchCount,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZgtsvInterleavedBatch(void * handle,int algo,int m,double2 * dl,double2 * d,double2 * du,double2 * x,int batchCount,void * pBuffer) nogil


#  \ingroup precond_module
# \brief Interleaved Batch pentadiagonal solver
# 
# \details
# \p hipsparseXgpsvInterleavedBatch_bufferSizeExt returns the size of the temporary storage
# buffer that is required by hipsparseXgpsvInterleavedBatch(). The temporary storage buffer
# must be allocated by the user.
cdef hipsparseStatus_t hipsparseSgpsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,const float * ds,const float * dl,const float * d,const float * du,const float * dw,const float * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDgpsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,const double * ds,const double * dl,const double * d,const double * du,const double * dw,const double * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCgpsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,float2 * ds,float2 * dl,float2 * d,float2 * du,float2 * dw,float2 * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZgpsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,double2 * ds,double2 * dl,double2 * d,double2 * du,double2 * dw,double2 * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil


#  \ingroup precond_module
# \brief Interleaved Batch pentadiagonal solver
# 
# \details
# \p hipsparseXgpsvInterleavedBatch solves a batched pentadiagonal linear system
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSgpsvInterleavedBatch(void * handle,int algo,int m,float * ds,float * dl,float * d,float * du,float * dw,float * x,int batchCount,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDgpsvInterleavedBatch(void * handle,int algo,int m,double * ds,double * dl,double * d,double * du,double * dw,double * x,int batchCount,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCgpsvInterleavedBatch(void * handle,int algo,int m,float2 * ds,float2 * dl,float2 * d,float2 * du,float2 * dw,float2 * x,int batchCount,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZgpsvInterleavedBatch(void * handle,int algo,int m,double2 * ds,double2 * dl,double2 * d,double2 * du,double2 * dw,double2 * x,int batchCount,void * pBuffer) nogil


#  \ingroup conv_module
# \brief
# This function computes the number of nonzero elements per row or column and the total
# number of nonzero elements in a dense matrix.
# 
# \details
# The routine does support asynchronous execution if the pointer mode is set to device.
cdef hipsparseStatus_t hipsparseSnnz(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,const float * A,int lda,int * nnzPerRowColumn,int * nnzTotalDevHostPtr) nogil



cdef hipsparseStatus_t hipsparseDnnz(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,const double * A,int lda,int * nnzPerRowColumn,int * nnzTotalDevHostPtr) nogil



cdef hipsparseStatus_t hipsparseCnnz(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,float2 * A,int lda,int * nnzPerRowColumn,int * nnzTotalDevHostPtr) nogil



cdef hipsparseStatus_t hipsparseZnnz(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,double2 * A,int lda,int * nnzPerRowColumn,int * nnzTotalDevHostPtr) nogil


#  \ingroup conv_module
# \brief
# This function converts the matrix A in dense format into a sparse matrix in CSR format.
# All the parameters are assumed to have been pre-allocated by the user and the arrays
# are filled in based on nnz_per_row, which can be pre-computed with hipsparseXnnz().
# It is executed asynchronously with respect to the host and may return control to the
# application on the host before the entire result is ready.
cdef hipsparseStatus_t hipsparseSdense2csr(void * handle,int m,int n,void *const descr,const float * A,int ld,const int * nnz_per_rows,float * csr_val,int * csr_row_ptr,int * csr_col_ind) nogil



cdef hipsparseStatus_t hipsparseDdense2csr(void * handle,int m,int n,void *const descr,const double * A,int ld,const int * nnz_per_rows,double * csr_val,int * csr_row_ptr,int * csr_col_ind) nogil



cdef hipsparseStatus_t hipsparseCdense2csr(void * handle,int m,int n,void *const descr,float2 * A,int ld,const int * nnz_per_rows,float2 * csr_val,int * csr_row_ptr,int * csr_col_ind) nogil



cdef hipsparseStatus_t hipsparseZdense2csr(void * handle,int m,int n,void *const descr,double2 * A,int ld,const int * nnz_per_rows,double2 * csr_val,int * csr_row_ptr,int * csr_col_ind) nogil


#  \ingroup conv_module
# \brief
# This function computes the the size of the user allocated temporary storage buffer used when converting and pruning
# a dense matrix to a CSR matrix.
# 
# \details
# \p hipsparseXpruneDense2csr_bufferSizeExt returns the size of the temporary storage buffer
# that is required by hipsparseXpruneDense2csrNnz() and hipsparseXpruneDense2csr(). The
# temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseSpruneDense2csr_bufferSize(void * handle,int m,int n,const float * A,int lda,const float * threshold,void *const descr,const float * csrVal,const int * csrRowPtr,const int * csrColInd,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseDpruneDense2csr_bufferSize(void * handle,int m,int n,const double * A,int lda,const double * threshold,void *const descr,const double * csrVal,const int * csrRowPtr,const int * csrColInd,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseSpruneDense2csr_bufferSizeExt(void * handle,int m,int n,const float * A,int lda,const float * threshold,void *const descr,const float * csrVal,const int * csrRowPtr,const int * csrColInd,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseDpruneDense2csr_bufferSizeExt(void * handle,int m,int n,const double * A,int lda,const double * threshold,void *const descr,const double * csrVal,const int * csrRowPtr,const int * csrColInd,unsigned long * bufferSize) nogil


#  \ingroup conv_module
# \brief
# This function computes the number of nonzero elements per row and the total number of
# nonzero elements in a dense matrix once elements less than the threshold are pruned
# from the matrix.
# 
# \details
# The routine does support asynchronous execution if the pointer mode is set to device.
cdef hipsparseStatus_t hipsparseSpruneDense2csrNnz(void * handle,int m,int n,const float * A,int lda,const float * threshold,void *const descr,int * csrRowPtr,int * nnzTotalDevHostPtr,void * buffer) nogil



cdef hipsparseStatus_t hipsparseDpruneDense2csrNnz(void * handle,int m,int n,const double * A,int lda,const double * threshold,void *const descr,int * csrRowPtr,int * nnzTotalDevHostPtr,void * buffer) nogil


#  \ingroup conv_module
# \brief
# This function converts the matrix A in dense format into a sparse matrix in CSR format
# while pruning values that are less than the threshold. All the parameters are assumed
# to have been pre-allocated by the user.
# 
# \details
# The user first allocates \p csrRowPtr to have \p m+1 elements and then calls
# hipsparseXpruneDense2csrNnz() which fills in the \p csrRowPtr array and stores the
# number of elements that are larger than the pruning threshold in \p nnzTotalDevHostPtr.
# The user then allocates \p csrColInd and \p csrVal to have size \p nnzTotalDevHostPtr
# and completes the conversion by calling hipsparseXpruneDense2csr(). A temporary storage
# buffer is used by both hipsparseXpruneDense2csrNnz() and hipsparseXpruneDense2csr() and
# must be allocated by the user and whose size is determined by
# hipsparseXpruneDense2csr_bufferSizeExt(). The routine hipsparseXpruneDense2csr() is
# executed asynchronously with respect to the host and may return control to the
# application on the host before the entire result is ready.
cdef hipsparseStatus_t hipsparseSpruneDense2csr(void * handle,int m,int n,const float * A,int lda,const float * threshold,void *const descr,float * csrVal,const int * csrRowPtr,int * csrColInd,void * buffer) nogil



cdef hipsparseStatus_t hipsparseDpruneDense2csr(void * handle,int m,int n,const double * A,int lda,const double * threshold,void *const descr,double * csrVal,const int * csrRowPtr,int * csrColInd,void * buffer) nogil


#  \ingroup conv_module
# \brief
# This function computes the size of the user allocated temporary storage buffer used
# when converting and pruning by percentage a dense matrix to a CSR matrix.
# 
# \details
# When converting and pruning a dense matrix A to a CSR matrix by percentage the
# following steps are performed. First the user calls
# \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
# temporary storage buffer. Once determined, this buffer must be allocated by the user.
# Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
# \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
# by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
# at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
# 
# The pruning by percentage works by first sorting the absolute values of the dense
# matrix \p A. We then determine a position in this sorted array by
# \f[
#   pos = ceil(m*n*(percentage/100)) - 1
#   pos = min(pos, m*n-1)
#   pos = max(pos, 0)
#   threshold = sorted_A[pos]
# \f]
# Once we have this threshold we prune values in the dense matrix \p A as in
# \p hipsparseXpruneDense2csr. It is executed asynchronously with respect to the host
# and may return control to the application on the host before the entire result is
# ready.
cdef hipsparseStatus_t hipsparseSpruneDense2csrByPercentage_bufferSize(void * handle,int m,int n,const float * A,int lda,float percentage,void *const descr,const float * csrVal,const int * csrRowPtr,const int * csrColInd,pruneInfo_t info,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseDpruneDense2csrByPercentage_bufferSize(void * handle,int m,int n,const double * A,int lda,double percentage,void *const descr,const double * csrVal,const int * csrRowPtr,const int * csrColInd,pruneInfo_t info,unsigned long * bufferSize) nogil


#  \ingroup conv_module
# \brief
# This function computes the size of the user allocated temporary storage buffer used
# when converting and pruning by percentage a dense matrix to a CSR matrix.
# 
# \details
# When converting and pruning a dense matrix A to a CSR matrix by percentage the
# following steps are performed. First the user calls
# \p hipsparseXpruneDense2csrByPercentage_bufferSizeExt which determines the size of the
# temporary storage buffer. Once determined, this buffer must be allocated by the user.
# Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
# \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
# by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
# at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
# 
# The pruning by percentage works by first sorting the absolute values of the dense
# matrix \p A. We then determine a position in this sorted array by
# \f[
#   pos = ceil(m*n*(percentage/100)) - 1
#   pos = min(pos, m*n-1)
#   pos = max(pos, 0)
#   threshold = sorted_A[pos]
# \f]
# Once we have this threshold we prune values in the dense matrix \p A as in
# \p hipsparseXpruneDense2csr. It is executed asynchronously with respect to the host
# and may return control to the application on the host before the entire result is
# ready.
cdef hipsparseStatus_t hipsparseSpruneDense2csrByPercentage_bufferSizeExt(void * handle,int m,int n,const float * A,int lda,float percentage,void *const descr,const float * csrVal,const int * csrRowPtr,const int * csrColInd,pruneInfo_t info,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseDpruneDense2csrByPercentage_bufferSizeExt(void * handle,int m,int n,const double * A,int lda,double percentage,void *const descr,const double * csrVal,const int * csrRowPtr,const int * csrColInd,pruneInfo_t info,unsigned long * bufferSize) nogil


#  \ingroup conv_module
# \brief
# This function computes the number of nonzero elements per row and the total number of
# nonzero elements in a dense matrix when converting and pruning by percentage a dense
# matrix to a CSR matrix.
# 
# \details
# When converting and pruning a dense matrix A to a CSR matrix by percentage the
# following steps are performed. First the user calls
# \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
# temporary storage buffer. Once determined, this buffer must be allocated by the user.
# Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
# \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
# by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
# at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
# 
# The pruning by percentage works by first sorting the absolute values of the dense
# matrix \p A. We then determine a position in this sorted array by
# \f[
#   pos = ceil(m*n*(percentage/100)) - 1
#   pos = min(pos, m*n-1)
#   pos = max(pos, 0)
#   threshold = sorted_A[pos]
# \f]
# Once we have this threshold we prune values in the dense matrix \p A as in
# \p hipsparseXpruneDense2csr. The routine does support asynchronous execution if the
# pointer mode is set to device.
cdef hipsparseStatus_t hipsparseSpruneDense2csrNnzByPercentage(void * handle,int m,int n,const float * A,int lda,float percentage,void *const descr,int * csrRowPtr,int * nnzTotalDevHostPtr,pruneInfo_t info,void * buffer) nogil



cdef hipsparseStatus_t hipsparseDpruneDense2csrNnzByPercentage(void * handle,int m,int n,const double * A,int lda,double percentage,void *const descr,int * csrRowPtr,int * nnzTotalDevHostPtr,pruneInfo_t info,void * buffer) nogil


#  \ingroup conv_module
# \brief
# This function computes the number of nonzero elements per row and the total number of
# nonzero elements in a dense matrix when converting and pruning by percentage a dense
# matrix to a CSR matrix.
# 
# \details
# When converting and pruning a dense matrix A to a CSR matrix by percentage the
# following steps are performed. First the user calls
# \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
# temporary storage buffer. Once determined, this buffer must be allocated by the user.
# Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
# \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
# by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
# at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
# 
# The pruning by percentage works by first sorting the absolute values of the dense
# matrix \p A. We then determine a position in this sorted array by
# \f[
#   pos = ceil(m*n*(percentage/100)) - 1
#   pos = min(pos, m*n-1)
#   pos = max(pos, 0)
#   threshold = sorted_A[pos]
# \f]
# Once we have this threshold we prune values in the dense matrix \p A as in
# \p hipsparseXpruneDense2csr. The routine does support asynchronous execution if the
# pointer mode is set to device.
cdef hipsparseStatus_t hipsparseSpruneDense2csrByPercentage(void * handle,int m,int n,const float * A,int lda,float percentage,void *const descr,float * csrVal,const int * csrRowPtr,int * csrColInd,pruneInfo_t info,void * buffer) nogil



cdef hipsparseStatus_t hipsparseDpruneDense2csrByPercentage(void * handle,int m,int n,const double * A,int lda,double percentage,void *const descr,double * csrVal,const int * csrRowPtr,int * csrColInd,pruneInfo_t info,void * buffer) nogil


#  \ingroup conv_module
# \brief
# 
# This function converts the matrix A in dense format into a sparse matrix in CSC format.
# All the parameters are assumed to have been pre-allocated by the user and the arrays are filled in based on nnz_per_columns, which can be pre-computed with hipsparseXnnz().
# It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
cdef hipsparseStatus_t hipsparseSdense2csc(void * handle,int m,int n,void *const descr,const float * A,int ld,const int * nnz_per_columns,float * csc_val,int * csc_row_ind,int * csc_col_ptr) nogil



cdef hipsparseStatus_t hipsparseDdense2csc(void * handle,int m,int n,void *const descr,const double * A,int ld,const int * nnz_per_columns,double * csc_val,int * csc_row_ind,int * csc_col_ptr) nogil



cdef hipsparseStatus_t hipsparseCdense2csc(void * handle,int m,int n,void *const descr,float2 * A,int ld,const int * nnz_per_columns,float2 * csc_val,int * csc_row_ind,int * csc_col_ptr) nogil



cdef hipsparseStatus_t hipsparseZdense2csc(void * handle,int m,int n,void *const descr,double2 * A,int ld,const int * nnz_per_columns,double2 * csc_val,int * csc_row_ind,int * csc_col_ptr) nogil


#  \ingroup conv_module
# \brief
# This function converts the sparse matrix in CSR format into a dense matrix.
# It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
cdef hipsparseStatus_t hipsparseScsr2dense(void * handle,int m,int n,void *const descr,const float * csr_val,const int * csr_row_ptr,const int * csr_col_ind,float * A,int ld) nogil



cdef hipsparseStatus_t hipsparseDcsr2dense(void * handle,int m,int n,void *const descr,const double * csr_val,const int * csr_row_ptr,const int * csr_col_ind,double * A,int ld) nogil



cdef hipsparseStatus_t hipsparseCcsr2dense(void * handle,int m,int n,void *const descr,float2 * csr_val,const int * csr_row_ptr,const int * csr_col_ind,float2 * A,int ld) nogil



cdef hipsparseStatus_t hipsparseZcsr2dense(void * handle,int m,int n,void *const descr,double2 * csr_val,const int * csr_row_ptr,const int * csr_col_ind,double2 * A,int ld) nogil


#  \ingroup conv_module
# \brief
# This function converts the sparse matrix in CSC format into a dense matrix.
# It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
cdef hipsparseStatus_t hipsparseScsc2dense(void * handle,int m,int n,void *const descr,const float * csc_val,const int * csc_row_ind,const int * csc_col_ptr,float * A,int ld) nogil



cdef hipsparseStatus_t hipsparseDcsc2dense(void * handle,int m,int n,void *const descr,const double * csc_val,const int * csc_row_ind,const int * csc_col_ptr,double * A,int ld) nogil



cdef hipsparseStatus_t hipsparseCcsc2dense(void * handle,int m,int n,void *const descr,float2 * csc_val,const int * csc_row_ind,const int * csc_col_ptr,float2 * A,int ld) nogil



cdef hipsparseStatus_t hipsparseZcsc2dense(void * handle,int m,int n,void *const descr,double2 * csc_val,const int * csc_row_ind,const int * csc_col_ptr,double2 * A,int ld) nogil


#  \ingroup conv_module
# \brief
# This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
# BSR matrix given a sparse CSR matrix as input.
# 
# \details
# The routine does support asynchronous execution if the pointer mode is set to device.
cdef hipsparseStatus_t hipsparseXcsr2bsrNnz(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,const int * csrRowPtrA,const int * csrColIndA,int blockDim,void *const descrC,int * bsrRowPtrC,int * bsrNnzb) nogil


#  \ingroup conv_module
# Given a sparse CSR matrix and a non-negative tolerance, this function computes how many entries would be left
# in each row of the matrix if elements less than the tolerance were removed. It also computes the total number
# of remaining elements in the matrix.
cdef hipsparseStatus_t hipsparseSnnz_compress(void * handle,int m,void *const descrA,const float * csrValA,const int * csrRowPtrA,int * nnzPerRow,int * nnzC,float tol) nogil



cdef hipsparseStatus_t hipsparseDnnz_compress(void * handle,int m,void *const descrA,const double * csrValA,const int * csrRowPtrA,int * nnzPerRow,int * nnzC,double tol) nogil



cdef hipsparseStatus_t hipsparseCnnz_compress(void * handle,int m,void *const descrA,float2 * csrValA,const int * csrRowPtrA,int * nnzPerRow,int * nnzC,float2 tol) nogil



cdef hipsparseStatus_t hipsparseZnnz_compress(void * handle,int m,void *const descrA,double2 * csrValA,const int * csrRowPtrA,int * nnzPerRow,int * nnzC,double2 tol) nogil


#  \ingroup conv_module
# \brief Convert a sparse CSR matrix into a sparse COO matrix
# 
# \details
# \p hipsparseXcsr2coo converts the CSR array containing the row offsets, that point
# to the start of every row, into a COO array of row indices.
# 
# \note
# It can also be used to convert a CSC array containing the column offsets into a COO
# array of column indices.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseXcsr2coo(void * handle,const int * csrRowPtr,int nnz,int m,int * cooRowInd,hipsparseIndexBase_t idxBase) nogil


#  \ingroup conv_module
# \brief Convert a sparse CSR matrix into a sparse CSC matrix
# 
# \details
# \p hipsparseXcsr2csc converts a CSR matrix into a CSC matrix. \p hipsparseXcsr2csc
# can also be used to convert a CSC matrix into a CSR matrix. \p copy_values decides
# whether \p csc_val is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
# or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
# 
# \note
# The resulting matrix can also be seen as the transpose of the input matrix.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseScsr2csc(void * handle,int m,int n,int nnz,const float * csrSortedVal,const int * csrSortedRowPtr,const int * csrSortedColInd,float * cscSortedVal,int * cscSortedRowInd,int * cscSortedColPtr,hipsparseAction_t copyValues,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseDcsr2csc(void * handle,int m,int n,int nnz,const double * csrSortedVal,const int * csrSortedRowPtr,const int * csrSortedColInd,double * cscSortedVal,int * cscSortedRowInd,int * cscSortedColPtr,hipsparseAction_t copyValues,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseCcsr2csc(void * handle,int m,int n,int nnz,float2 * csrSortedVal,const int * csrSortedRowPtr,const int * csrSortedColInd,float2 * cscSortedVal,int * cscSortedRowInd,int * cscSortedColPtr,hipsparseAction_t copyValues,hipsparseIndexBase_t idxBase) nogil



cdef hipsparseStatus_t hipsparseZcsr2csc(void * handle,int m,int n,int nnz,double2 * csrSortedVal,const int * csrSortedRowPtr,const int * csrSortedColInd,double2 * cscSortedVal,int * cscSortedRowInd,int * cscSortedColPtr,hipsparseAction_t copyValues,hipsparseIndexBase_t idxBase) nogil


cdef extern from "hipsparse/hipsparse.h":

    ctypedef enum hipsparseCsr2CscAlg_t:
        HIPSPARSE_CSR2CSC_ALG1
        HIPSPARSE_CSR2CSC_ALG2

#  \ingroup conv_module
# \brief This function computes the size of the user allocated temporary storage buffer used
# when converting a sparse CSR matrix into a sparse CSC matrix.
# 
# \details
# \p hipsparseXcsr2cscEx2_bufferSize calculates the required user allocated temporary buffer needed
# by \p hipsparseXcsr2cscEx2 to convert a CSR matrix into a CSC matrix. \p hipsparseXcsr2cscEx2
# can also be used to convert a CSC matrix into a CSR matrix. \p copy_values decides
# whether \p csc_val is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
# or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
# 
# \note
# The resulting matrix can also be seen as the transpose of the input matrix.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseCsr2cscEx2_bufferSize(void * handle,int m,int n,int nnz,const void * csrVal,const int * csrRowPtr,const int * csrColInd,void * cscVal,int * cscColPtr,int * cscRowInd,hipDataType valType,hipsparseAction_t copyValues,hipsparseIndexBase_t idxBase,hipsparseCsr2CscAlg_t alg,unsigned long * bufferSize) nogil


#  \ingroup conv_module
# \brief Convert a sparse CSR matrix into a sparse CSC matrix
# 
# \details
# \p hipsparseXcsr2cscEx2 converts a CSR matrix into a CSC matrix. \p hipsparseXcsr2cscEx2
# can also be used to convert a CSC matrix into a CSR matrix. \p copy_values decides
# whether \p csc_val is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
# or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
# 
# \note
# The resulting matrix can also be seen as the transpose of the input matrix.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseCsr2cscEx2(void * handle,int m,int n,int nnz,const void * csrVal,const int * csrRowPtr,const int * csrColInd,void * cscVal,int * cscColPtr,int * cscRowInd,hipDataType valType,hipsparseAction_t copyValues,hipsparseIndexBase_t idxBase,hipsparseCsr2CscAlg_t alg,void * buffer) nogil


#  \ingroup conv_module
# \brief Convert a sparse CSR matrix into a sparse HYB matrix
# 
# \details
# \p hipsparseXcsr2hyb converts a CSR matrix into a HYB matrix. It is assumed
# that \p hyb has been initialized with hipsparseCreateHybMat().
# 
# \note
# This function requires a significant amount of storage for the HYB matrix,
# depending on the matrix structure.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseScsr2hyb(void * handle,int m,int n,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,void * hybA,int userEllWidth,hipsparseHybPartition_t partitionType) nogil



cdef hipsparseStatus_t hipsparseDcsr2hyb(void * handle,int m,int n,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,void * hybA,int userEllWidth,hipsparseHybPartition_t partitionType) nogil



cdef hipsparseStatus_t hipsparseCcsr2hyb(void * handle,int m,int n,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,void * hybA,int userEllWidth,hipsparseHybPartition_t partitionType) nogil



cdef hipsparseStatus_t hipsparseZcsr2hyb(void * handle,int m,int n,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,void * hybA,int userEllWidth,hipsparseHybPartition_t partitionType) nogil


#  \ingroup conv_module
# \brief Convert a sparse GEneral BSR matrix into a sparse GEneral BSC matrix
# 
# \details
# \p hipsparseXgebsr2gebsc_bufferSize returns the size of the temporary storage buffer
# required by hipsparseXgebsr2gebsc().
# The temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseSgebsr2gebsc_bufferSize(void * handle,int mb,int nb,int nnzb,const float * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil



cdef hipsparseStatus_t hipsparseDgebsr2gebsc_bufferSize(void * handle,int mb,int nb,int nnzb,const double * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil



cdef hipsparseStatus_t hipsparseCgebsr2gebsc_bufferSize(void * handle,int mb,int nb,int nnzb,float2 * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil



cdef hipsparseStatus_t hipsparseZgebsr2gebsc_bufferSize(void * handle,int mb,int nb,int nnzb,double2 * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil


#  \ingroup conv_module
# \brief Convert a sparse GEneral BSR matrix into a sparse GEneral BSC matrix
# 
# \details
# \p hipsparseXgebsr2gebsc converts a GEneral BSR matrix into a GEneral BSC matrix. \p hipsparseXgebsr2gebsc
# can also be used to convert a GEneral BSC matrix into a GEneral BSR matrix. \p copy_values decides
# whether \p bsc_val is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
# or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
# 
# \p hipsparseXgebsr2gebsc requires extra temporary storage buffer that has to be allocated
# by the user. Storage buffer size can be determined by hipsparseXgebsr2gebsc_bufferSize().
# 
# \note
# The resulting matrix can also be seen as the transpose of the input matrix.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSgebsr2gebsc(void * handle,int mb,int nb,int nnzb,const float * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,float * bsc_val,int * bsc_row_ind,int * bsc_col_ptr,hipsparseAction_t copy_values,hipsparseIndexBase_t idx_base,void * temp_buffer) nogil



cdef hipsparseStatus_t hipsparseDgebsr2gebsc(void * handle,int mb,int nb,int nnzb,const double * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,double * bsc_val,int * bsc_row_ind,int * bsc_col_ptr,hipsparseAction_t copy_values,hipsparseIndexBase_t idx_base,void * temp_buffer) nogil



cdef hipsparseStatus_t hipsparseCgebsr2gebsc(void * handle,int mb,int nb,int nnzb,float2 * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,float2 * bsc_val,int * bsc_row_ind,int * bsc_col_ptr,hipsparseAction_t copy_values,hipsparseIndexBase_t idx_base,void * temp_buffer) nogil



cdef hipsparseStatus_t hipsparseZgebsr2gebsc(void * handle,int mb,int nb,int nnzb,double2 * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,double2 * bsc_val,int * bsc_row_ind,int * bsc_col_ptr,hipsparseAction_t copy_values,hipsparseIndexBase_t idx_base,void * temp_buffer) nogil


#  \ingroup conv_module
# \brief
#  \details
#  \p hipsparseXcsr2gebsr_bufferSize returns the size of the temporary buffer that
#  is required by \p hipsparseXcsr2gebcsrNnz and \p hipsparseXcsr2gebcsr.
#  The temporary storage buffer must be allocated by the user.
# 
# This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
# GEneral BSR matrix given a sparse CSR matrix as input.
# 
# \details
# The routine does support asynchronous execution if the pointer mode is set to device.
cdef hipsparseStatus_t hipsparseScsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,const float * csr_val,const int * csr_row_ptr,const int * csr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil



cdef hipsparseStatus_t hipsparseDcsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,const double * csr_val,const int * csr_row_ptr,const int * csr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil



cdef hipsparseStatus_t hipsparseCcsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,float2 * csr_val,const int * csr_row_ptr,const int * csr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil



cdef hipsparseStatus_t hipsparseZcsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,double2 * csr_val,const int * csr_row_ptr,const int * csr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil


#  \ingroup conv_module
# \brief
# This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
# GEneral BSR matrix given a sparse CSR matrix as input.
#
cdef hipsparseStatus_t hipsparseXcsr2gebsrNnz(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,const int * csr_row_ptr,const int * csr_col_ind,void *const bsr_descr,int * bsr_row_ptr,int row_block_dim,int col_block_dim,int * bsr_nnz_devhost,void * p_buffer) nogil


#  \ingroup conv_module
# \brief Convert a sparse CSR matrix into a sparse GEneral BSR matrix
# 
# \details
# \p hipsparseXcsr2gebsr converts a CSR matrix into a GEneral BSR matrix. It is assumed,
# that \p bsr_val, \p bsr_col_ind and \p bsr_row_ptr are allocated. Allocation size
# for \p bsr_row_ptr is computed as \p mb+1 where \p mb is the number of block rows in
# the GEneral BSR matrix. Allocation size for \p bsr_val and \p bsr_col_ind is computed using
# \p csr2gebsr_nnz() which also fills in \p bsr_row_ptr.
cdef hipsparseStatus_t hipsparseScsr2gebsr(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,const float * csr_val,const int * csr_row_ptr,const int * csr_col_ind,void *const bsr_descr,float * bsr_val,int * bsr_row_ptr,int * bsr_col_ind,int row_block_dim,int col_block_dim,void * p_buffer) nogil



cdef hipsparseStatus_t hipsparseDcsr2gebsr(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,const double * csr_val,const int * csr_row_ptr,const int * csr_col_ind,void *const bsr_descr,double * bsr_val,int * bsr_row_ptr,int * bsr_col_ind,int row_block_dim,int col_block_dim,void * p_buffer) nogil



cdef hipsparseStatus_t hipsparseCcsr2gebsr(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,float2 * csr_val,const int * csr_row_ptr,const int * csr_col_ind,void *const bsr_descr,float2 * bsr_val,int * bsr_row_ptr,int * bsr_col_ind,int row_block_dim,int col_block_dim,void * p_buffer) nogil



cdef hipsparseStatus_t hipsparseZcsr2gebsr(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,double2 * csr_val,const int * csr_row_ptr,const int * csr_col_ind,void *const bsr_descr,double2 * bsr_val,int * bsr_row_ptr,int * bsr_col_ind,int row_block_dim,int col_block_dim,void * p_buffer) nogil


#  \ingroup conv_module
# \brief Convert a sparse CSR matrix into a sparse BSR matrix
# 
# \details
# \p hipsparseXcsr2bsr converts a CSR matrix into a BSR matrix. It is assumed,
# that \p bsr_val, \p bsr_col_ind and \p bsr_row_ptr are allocated. Allocation size
# for \p bsr_row_ptr is computed as \p mb+1 where \p mb is the number of block rows in
# the BSR matrix. Allocation size for \p bsr_val and \p bsr_col_ind is computed using
# \p csr2bsr_nnz() which also fills in \p bsr_row_ptr.
# 
# \p hipsparseXcsr2bsr requires extra temporary storage that is allocated internally if
# \p block_dim>16
cdef hipsparseStatus_t hipsparseScsr2bsr(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,int blockDim,void *const descrC,float * bsrValC,int * bsrRowPtrC,int * bsrColIndC) nogil



cdef hipsparseStatus_t hipsparseDcsr2bsr(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,int blockDim,void *const descrC,double * bsrValC,int * bsrRowPtrC,int * bsrColIndC) nogil



cdef hipsparseStatus_t hipsparseCcsr2bsr(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,float2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,int blockDim,void *const descrC,float2 * bsrValC,int * bsrRowPtrC,int * bsrColIndC) nogil



cdef hipsparseStatus_t hipsparseZcsr2bsr(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,double2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,int blockDim,void *const descrC,double2 * bsrValC,int * bsrRowPtrC,int * bsrColIndC) nogil


#  \ingroup conv_module
# \brief Convert a sparse BSR matrix into a sparse CSR matrix
# 
# \details
# \p hipsparseXbsr2csr converts a BSR matrix into a CSR matrix. It is assumed,
# that \p csr_val, \p csr_col_ind and \p csr_row_ptr are allocated. Allocation size
# for \p csr_row_ptr is computed by the number of block rows multiplied by the block
# dimension plus one. Allocation for \p csr_val and \p csr_col_ind is computed by the
# the number of blocks in the BSR matrix multiplied by the block dimension squared.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSbsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,const float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,void *const descrC,float * csrValC,int * csrRowPtrC,int * csrColIndC) nogil



cdef hipsparseStatus_t hipsparseDbsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,const double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,void *const descrC,double * csrValC,int * csrRowPtrC,int * csrColIndC) nogil



cdef hipsparseStatus_t hipsparseCbsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,void *const descrC,float2 * csrValC,int * csrRowPtrC,int * csrColIndC) nogil



cdef hipsparseStatus_t hipsparseZbsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,void *const descrC,double2 * csrValC,int * csrRowPtrC,int * csrColIndC) nogil


#  \ingroup conv_module
# \brief Convert a sparse general BSR matrix into a sparse CSR matrix
# 
# \details
# \p hipsparseXgebsr2csr converts a BSR matrix into a CSR matrix. It is assumed,
# that \p csr_val, \p csr_col_ind and \p csr_row_ptr are allocated. Allocation size
# for \p csr_row_ptr is computed by the number of block rows multiplied by the block
# dimension plus one. Allocation for \p csr_val and \p csr_col_ind is computed by the
# the number of blocks in the BSR matrix multiplied by the product of the block dimensions.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseSgebsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,const float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDim,int colBlockDim,void *const descrC,float * csrValC,int * csrRowPtrC,int * csrColIndC) nogil



cdef hipsparseStatus_t hipsparseDgebsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,const double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDim,int colBlockDim,void *const descrC,double * csrValC,int * csrRowPtrC,int * csrColIndC) nogil



cdef hipsparseStatus_t hipsparseCgebsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDim,int colBlockDim,void *const descrC,float2 * csrValC,int * csrRowPtrC,int * csrColIndC) nogil



cdef hipsparseStatus_t hipsparseZgebsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDim,int colBlockDim,void *const descrC,double2 * csrValC,int * csrRowPtrC,int * csrColIndC) nogil


# \ingroup conv_module
# \brief Convert a sparse CSR matrix into a compressed sparse CSR matrix
# 
# \details
# \p hipsparseXcsr2csr_compress converts a CSR matrix into a compressed CSR matrix by
# removing entries in the input CSR matrix that are below a non-negative threshold \p tol
# 
# \note
# In the case of complex matrices only the magnitude of the real part of \p tol is used.
cdef hipsparseStatus_t hipsparseScsr2csr_compress(void * handle,int m,int n,void *const descrA,const float * csrValA,const int * csrColIndA,const int * csrRowPtrA,int nnzA,const int * nnzPerRow,float * csrValC,int * csrColIndC,int * csrRowPtrC,float tol) nogil



cdef hipsparseStatus_t hipsparseDcsr2csr_compress(void * handle,int m,int n,void *const descrA,const double * csrValA,const int * csrColIndA,const int * csrRowPtrA,int nnzA,const int * nnzPerRow,double * csrValC,int * csrColIndC,int * csrRowPtrC,double tol) nogil



cdef hipsparseStatus_t hipsparseCcsr2csr_compress(void * handle,int m,int n,void *const descrA,float2 * csrValA,const int * csrColIndA,const int * csrRowPtrA,int nnzA,const int * nnzPerRow,float2 * csrValC,int * csrColIndC,int * csrRowPtrC,float2 tol) nogil



cdef hipsparseStatus_t hipsparseZcsr2csr_compress(void * handle,int m,int n,void *const descrA,double2 * csrValA,const int * csrColIndA,const int * csrRowPtrA,int nnzA,const int * nnzPerRow,double2 * csrValC,int * csrColIndC,int * csrRowPtrC,double2 tol) nogil


# \ingroup conv_module
# \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
# 
# \details
# \p hipsparseXpruneCsr2csr_bufferSize returns the size of the temporary buffer that
# is required by \p hipsparseXpruneCsr2csrNnz and hipsparseXpruneCsr2csr. The
# temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSize(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * threshold,void *const descrC,const float * csrValC,const int * csrRowPtrC,const int * csrColIndC,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSize(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * threshold,void *const descrC,const double * csrValC,const int * csrRowPtrC,const int * csrColIndC,unsigned long * bufferSize) nogil


# \ingroup conv_module
# \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
# 
# \details
# \p hipsparseXpruneCsr2csr_bufferSizeExt returns the size of the temporary buffer that
# is required by \p hipsparseXpruneCsr2csrNnz and hipsparseXpruneCsr2csr. The
# temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSizeExt(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * threshold,void *const descrC,const float * csrValC,const int * csrRowPtrC,const int * csrColIndC,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSizeExt(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * threshold,void *const descrC,const double * csrValC,const int * csrRowPtrC,const int * csrColIndC,unsigned long * bufferSize) nogil


# \ingroup conv_module
# \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
# 
# \details
# \p hipsparseXpruneCsr2csrNnz computes the number of nonzero elements per row and the total
# number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
# pruned from the matrix.
# 
# \note The routine does support asynchronous execution if the pointer mode is set to device.
cdef hipsparseStatus_t hipsparseSpruneCsr2csrNnz(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * threshold,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr,void * buffer) nogil



cdef hipsparseStatus_t hipsparseDpruneCsr2csrNnz(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * threshold,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr,void * buffer) nogil


# \ingroup conv_module
# \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
# 
# \details
# This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
# that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
# The user first calls hipsparseXpruneCsr2csr_bufferSize() to determine the size of the buffer used
# by hipsparseXpruneCsr2csrNnz() and hipsparseXpruneCsr2csr() which the user then allocates. The user then
# allocates \p csr_row_ptr_C to have \p m+1 elements and then calls hipsparseXpruneCsr2csrNnz() which fills
# in the \p csr_row_ptr_C array stores then number of elements that are larger than the pruning threshold
# in \p nnz_total_dev_host_ptr. The user then calls hipsparseXpruneCsr2csr() to complete the conversion. It
# is executed asynchronously with respect to the host and may return control to the application on the host
# before the entire result is ready.
cdef hipsparseStatus_t hipsparseSpruneCsr2csr(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * threshold,void *const descrC,float * csrValC,const int * csrRowPtrC,int * csrColIndC,void * buffer) nogil



cdef hipsparseStatus_t hipsparseDpruneCsr2csr(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * threshold,void *const descrC,double * csrValC,const int * csrRowPtrC,int * csrColIndC,void * buffer) nogil


# \ingroup conv_module
# \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
# 
# \details
# \p hipsparseXpruneCsr2csrByPercentage_bufferSize returns the size of the temporary buffer that
# is required by \p hipsparseXpruneCsr2csrNnzByPercentage.
# The temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSize(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,float percentage,void *const descrC,const float * csrValC,const int * csrRowPtrC,const int * csrColIndC,pruneInfo_t info,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSize(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,double percentage,void *const descrC,const double * csrValC,const int * csrRowPtrC,const int * csrColIndC,pruneInfo_t info,unsigned long * bufferSize) nogil


# \ingroup conv_module
# \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
# 
# \details
# \p hipsparseXpruneCsr2csrByPercentage_bufferSizeExt returns the size of the temporary buffer that
# is required by \p hipsparseXpruneCsr2csrNnzByPercentage.
# The temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSizeExt(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,float percentage,void *const descrC,const float * csrValC,const int * csrRowPtrC,const int * csrColIndC,pruneInfo_t info,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSizeExt(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,double percentage,void *const descrC,const double * csrValC,const int * csrRowPtrC,const int * csrColIndC,pruneInfo_t info,unsigned long * bufferSize) nogil


# \ingroup conv_module
# \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
# 
# \details
# \p hipsparseXpruneCsr2csrNnzByPercentage computes the number of nonzero elements per row and the total
# number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
# pruned from the matrix.
# 
# \note The routine does support asynchronous execution if the pointer mode is set to device.
cdef hipsparseStatus_t hipsparseSpruneCsr2csrNnzByPercentage(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,float percentage,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr,pruneInfo_t info,void * buffer) nogil



cdef hipsparseStatus_t hipsparseDpruneCsr2csrNnzByPercentage(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,double percentage,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr,pruneInfo_t info,void * buffer) nogil


# \ingroup conv_module
# \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
# 
# \details
# This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
# that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
# The user first calls hipsparseXpruneCsr2csr_bufferSize() to determine the size of the buffer used
# by hipsparseXpruneCsr2csrNnz() and hipsparseXpruneCsr2csr() which the user then allocates. The user then
# allocates \p csr_row_ptr_C to have \p m+1 elements and then calls hipsparseXpruneCsr2csrNnz() which fills
# in the \p csr_row_ptr_C array stores then number of elements that are larger than the pruning threshold
# in \p nnz_total_dev_host_ptr. The user then calls hipsparseXpruneCsr2csr() to complete the conversion. It
# is executed asynchronously with respect to the host and may return control to the application on the host
# before the entire result is ready.
cdef hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,float percentage,void *const descrC,float * csrValC,const int * csrRowPtrC,int * csrColIndC,pruneInfo_t info,void * buffer) nogil



cdef hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,double percentage,void *const descrC,double * csrValC,const int * csrRowPtrC,int * csrColIndC,pruneInfo_t info,void * buffer) nogil


#  \ingroup conv_module
# \brief Convert a sparse HYB matrix into a sparse CSR matrix
# 
# \details
# \p hipsparseXhyb2csr converts a HYB matrix into a CSR matrix.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseShyb2csr(void * handle,void *const descrA,void *const hybA,float * csrSortedValA,int * csrSortedRowPtrA,int * csrSortedColIndA) nogil



cdef hipsparseStatus_t hipsparseDhyb2csr(void * handle,void *const descrA,void *const hybA,double * csrSortedValA,int * csrSortedRowPtrA,int * csrSortedColIndA) nogil



cdef hipsparseStatus_t hipsparseChyb2csr(void * handle,void *const descrA,void *const hybA,float2 * csrSortedValA,int * csrSortedRowPtrA,int * csrSortedColIndA) nogil



cdef hipsparseStatus_t hipsparseZhyb2csr(void * handle,void *const descrA,void *const hybA,double2 * csrSortedValA,int * csrSortedRowPtrA,int * csrSortedColIndA) nogil


# \ingroup conv_module
# \brief Convert a sparse COO matrix into a sparse CSR matrix
# 
# \details
# \p hipsparseXcoo2csr converts the COO array containing the row indices into a
# CSR array of row offsets, that point to the start of every row.
# It is assumed that the COO row index array is sorted.
# 
# \note It can also be used, to convert a COO array containing the column indices into
# a CSC array of column offsets, that point to the start of every column. Then, it is
# assumed that the COO column index array is sorted, instead.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseXcoo2csr(void * handle,const int * cooRowInd,int nnz,int m,int * csrRowPtr,hipsparseIndexBase_t idxBase) nogil


#  \ingroup conv_module
# \brief Create the identity map
# 
# \details
# \p hipsparseCreateIdentityPermutation stores the identity map in \p p, such that
# \f$p = 0:1:(n-1)\f$.
# 
# \code{.c}
#     for(i = 0; i < n; ++i)
#     {
#         p[i] = i;
#     }
# \endcode
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseCreateIdentityPermutation(void * handle,int n,int * p) nogil


#  \ingroup conv_module
# \brief Sort a sparse CSR matrix
# 
# \details
# \p hipsparseXcsrsort_bufferSizeExt returns the size of the temporary storage buffer
# required by hipsparseXcsrsort(). The temporary storage buffer must be allocated by
# the user.
cdef hipsparseStatus_t hipsparseXcsrsort_bufferSizeExt(void * handle,int m,int n,int nnz,const int * csrRowPtr,const int * csrColInd,unsigned long * pBufferSizeInBytes) nogil


#  \ingroup conv_module
# \brief Sort a sparse CSR matrix
# 
# \details
# \p hipsparseXcsrsort sorts a matrix in CSR format. The sorted permutation vector
# \p perm can be used to obtain sorted \p csr_val array. In this case, \p perm must be
# initialized as the identity permutation, see hipsparseCreateIdentityPermutation().
# 
# \p hipsparseXcsrsort requires extra temporary storage buffer that has to be allocated by
# the user. Storage buffer size can be determined by hipsparseXcsrsort_bufferSizeExt().
# 
# \note
# \p perm can be \p NULL if a sorted permutation vector is not required.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseXcsrsort(void * handle,int m,int n,int nnz,void *const descrA,const int * csrRowPtr,int * csrColInd,int * P,void * pBuffer) nogil


#  \ingroup conv_module
# \brief Sort a sparse CSC matrix
# 
# \details
# \p hipsparseXcscsort_bufferSizeExt returns the size of the temporary storage buffer
# required by hipsparseXcscsort(). The temporary storage buffer must be allocated by
# the user.
cdef hipsparseStatus_t hipsparseXcscsort_bufferSizeExt(void * handle,int m,int n,int nnz,const int * cscColPtr,const int * cscRowInd,unsigned long * pBufferSizeInBytes) nogil


#  \ingroup conv_module
# \brief Sort a sparse CSC matrix
# 
# \details
# \p hipsparseXcscsort sorts a matrix in CSC format. The sorted permutation vector
# \p perm can be used to obtain sorted \p csc_val array. In this case, \p perm must be
# initialized as the identity permutation, see hipsparseCreateIdentityPermutation().
# 
# \p hipsparseXcscsort requires extra temporary storage buffer that has to be allocated by
# the user. Storage buffer size can be determined by hipsparseXcscsort_bufferSizeExt().
# 
# \note
# \p perm can be \p NULL if a sorted permutation vector is not required.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseXcscsort(void * handle,int m,int n,int nnz,void *const descrA,const int * cscColPtr,int * cscRowInd,int * P,void * pBuffer) nogil


#  \ingroup conv_module
# \brief Sort a sparse COO matrix
# 
# \details
# \p hipsparseXcoosort_bufferSizeExt returns the size of the temporary storage buffer
# required by hipsparseXcoosort(). The temporary storage buffer must be allocated by
# the user.
cdef hipsparseStatus_t hipsparseXcoosort_bufferSizeExt(void * handle,int m,int n,int nnz,const int * cooRows,const int * cooCols,unsigned long * pBufferSizeInBytes) nogil


#  \ingroup conv_module
# \brief Sort a sparse COO matrix by row
# 
# \details
# \p hipsparseXcoosortByRow sorts a matrix in COO format by row. The sorted
# permutation vector \p perm can be used to obtain sorted \p coo_val array. In this
# case, \p perm must be initialized as the identity permutation, see
# hipsparseCreateIdentityPermutation().
# 
# \p hipsparseXcoosortByRow requires extra temporary storage buffer that has to be
# allocated by the user. Storage buffer size can be determined by
# hipsparseXcoosort_bufferSizeExt().
# 
# \note
# \p perm can be \p NULL if a sorted permutation vector is not required.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseXcoosortByRow(void * handle,int m,int n,int nnz,int * cooRows,int * cooCols,int * P,void * pBuffer) nogil


#  \ingroup conv_module
# \brief Sort a sparse COO matrix by column
# 
# \details
# \p hipsparseXcoosortByColumn sorts a matrix in COO format by column. The sorted
# permutation vector \p perm can be used to obtain sorted \p coo_val array. In this
# case, \p perm must be initialized as the identity permutation, see
# hipsparseCreateIdentityPermutation().
# 
# \p hipsparseXcoosortByColumn requires extra temporary storage buffer that has to be
# allocated by the user. Storage buffer size can be determined by
# hipsparseXcoosort_bufferSizeExt().
# 
# \note
# \p perm can be \p NULL if a sorted permutation vector is not required.
# 
# \note
# This function is non blocking and executed asynchronously with respect to the host.
# It may return before the actual computation has finished.
cdef hipsparseStatus_t hipsparseXcoosortByColumn(void * handle,int m,int n,int nnz,int * cooRows,int * cooCols,int * P,void * pBuffer) nogil


#  \ingroup conv_module
# \brief
# This function computes the the size of the user allocated temporary storage buffer used when converting a sparse
# general BSR matrix to another sparse general BSR matrix.
# 
# \details
# \p hipsparseXgebsr2gebsr_bufferSize returns the size of the temporary storage buffer
# that is required by hipsparseXgebsr2gebsrNnz() and hipsparseXgebsr2gebsr().
# The temporary storage buffer must be allocated by the user.
cdef hipsparseStatus_t hipsparseSgebsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,const float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,int rowBlockDimC,int colBlockDimC,int * bufferSize) nogil



cdef hipsparseStatus_t hipsparseDgebsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,const double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,int rowBlockDimC,int colBlockDimC,int * bufferSize) nogil



cdef hipsparseStatus_t hipsparseCgebsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,int rowBlockDimC,int colBlockDimC,int * bufferSize) nogil



cdef hipsparseStatus_t hipsparseZgebsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,int rowBlockDimC,int colBlockDimC,int * bufferSize) nogil


#  \ingroup conv_module
# \brief This function is used when converting a general BSR sparse matrix \p A to another general BSR sparse matrix \p C.
# Specifically, this function determines the number of non-zero blocks that will exist in \p C (stored using either a host
# or device pointer), and computes the row pointer array for \p C.
# 
# \details
# The routine does support asynchronous execution.
cdef hipsparseStatus_t hipsparseXgebsr2gebsrNnz(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,void *const descrC,int * bsrRowPtrC,int rowBlockDimC,int colBlockDimC,int * nnzTotalDevHostPtr,void * buffer) nogil


#  \ingroup conv_module
# \brief
# This function converts the general BSR sparse matrix \p A to another general BSR sparse matrix \p C.
# 
# \details
# The conversion uses three steps. First, the user calls hipsparseXgebsr2gebsr_bufferSize() to determine the size of
# the required temporary storage buffer. The user then allocates this buffer. Secondly, the user then allocates \p mb_C+1
# integers for the row pointer array for \p C where \p mb_C=(m+row_block_dim_C-1)/row_block_dim_C. The user then calls
# hipsparseXgebsr2gebsrNnz() to fill in the row pointer array for \p C ( \p bsr_row_ptr_C ) and determine the number of
# non-zero blocks that will exist in \p C. Finally, the user allocates space for the colimn indices array of \p C to have
# \p nnzb_C elements and space for the values array of \p C to have \p nnzb_C*roc_block_dim_C*col_block_dim_C and then calls
# hipsparseXgebsr2gebsr() to complete the conversion.
cdef hipsparseStatus_t hipsparseSgebsr2gebsr(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,const float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,void *const descrC,float * bsrValC,int * bsrRowPtrC,int * bsrColIndC,int rowBlockDimC,int colBlockDimC,void * buffer) nogil



cdef hipsparseStatus_t hipsparseDgebsr2gebsr(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,const double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,void *const descrC,double * bsrValC,int * bsrRowPtrC,int * bsrColIndC,int rowBlockDimC,int colBlockDimC,void * buffer) nogil



cdef hipsparseStatus_t hipsparseCgebsr2gebsr(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,void *const descrC,float2 * bsrValC,int * bsrRowPtrC,int * bsrColIndC,int rowBlockDimC,int colBlockDimC,void * buffer) nogil



cdef hipsparseStatus_t hipsparseZgebsr2gebsr(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,void *const descrC,double2 * bsrValC,int * bsrRowPtrC,int * bsrColIndC,int rowBlockDimC,int colBlockDimC,void * buffer) nogil


#  \ingroup conv_module
# \brief
# This function calculates the amount of temporary storage required for
# hipsparseXcsru2csr() and hipsparseXcsr2csru().
cdef hipsparseStatus_t hipsparseScsru2csr_bufferSizeExt(void * handle,int m,int n,int nnz,float * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseDcsru2csr_bufferSizeExt(void * handle,int m,int n,int nnz,double * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseCcsru2csr_bufferSizeExt(void * handle,int m,int n,int nnz,float2 * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,unsigned long * pBufferSizeInBytes) nogil



cdef hipsparseStatus_t hipsparseZcsru2csr_bufferSizeExt(void * handle,int m,int n,int nnz,double2 * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,unsigned long * pBufferSizeInBytes) nogil


#  \ingroup conv_module
# \brief
# This function converts unsorted CSR format to sorted CSR format. The required
# temporary storage has to be allocated by the user.
cdef hipsparseStatus_t hipsparseScsru2csr(void * handle,int m,int n,int nnz,void *const descrA,float * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDcsru2csr(void * handle,int m,int n,int nnz,void *const descrA,double * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCcsru2csr(void * handle,int m,int n,int nnz,void *const descrA,float2 * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZcsru2csr(void * handle,int m,int n,int nnz,void *const descrA,double2 * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil


#  \ingroup conv_module
# \brief
# This function converts sorted CSR format to unsorted CSR format. The required
# temporary storage has to be allocated by the user.
cdef hipsparseStatus_t hipsparseScsr2csru(void * handle,int m,int n,int nnz,void *const descrA,float * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseDcsr2csru(void * handle,int m,int n,int nnz,void *const descrA,double * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseCcsr2csru(void * handle,int m,int n,int nnz,void *const descrA,float2 * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil



cdef hipsparseStatus_t hipsparseZcsr2csru(void * handle,int m,int n,int nnz,void *const descrA,double2 * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil


#  \ingroup reordering_module
# \brief Coloring of the adjacency graph of the matrix \f$A\f$ stored in the CSR format.
# 
# \details
# \p hipsparseXcsrcolor performs the coloring of the undirected graph represented by the (symmetric) sparsity pattern of the matrix \f$A\f$ stored in CSR format. Graph coloring is a way of coloring the nodes of a graph such that no two adjacent nodes are of the same color. The \p fraction_to_color is a parameter to only color a given percentage of the graph nodes, the remaining uncolored nodes receive distinct new colors. The optional \p reordering array is a permutation array such that unknowns of the same color are grouped. The matrix \f$A\f$ must be stored as a general matrix with a symmetric sparsity pattern, and if the matrix \f$A\f$ is non-symmetric then the user is responsible to provide the symmetric part \f$\frac{A+A^T}{2}\f$.
cdef hipsparseStatus_t hipsparseScsrcolor(void * handle,int m,int nnz,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * fractionToColor,int * ncolors,int * coloring,int * reordering,void * info) nogil



cdef hipsparseStatus_t hipsparseDcsrcolor(void * handle,int m,int nnz,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * fractionToColor,int * ncolors,int * coloring,int * reordering,void * info) nogil



cdef hipsparseStatus_t hipsparseCcsrcolor(void * handle,int m,int nnz,void *const descrA,float2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * fractionToColor,int * ncolors,int * coloring,int * reordering,void * info) nogil



cdef hipsparseStatus_t hipsparseZcsrcolor(void * handle,int m,int nnz,void *const descrA,double2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * fractionToColor,int * ncolors,int * coloring,int * reordering,void * info) nogil


cdef extern from "hipsparse/hipsparse.h":

    ctypedef void * hipsparseSpVecDescr_t

    ctypedef void * hipsparseSpMatDescr_t

    ctypedef void * hipsparseDnVecDescr_t

    ctypedef void * hipsparseDnMatDescr_t

    cdef struct hipsparseSpGEMMDescr:
        pass

    ctypedef hipsparseSpGEMMDescr * hipsparseSpGEMMDescr_t

    cdef struct hipsparseSpSVDescr:
        pass

    ctypedef hipsparseSpSVDescr * hipsparseSpSVDescr_t

    cdef struct hipsparseSpSMDescr:
        pass

    ctypedef hipsparseSpSMDescr * hipsparseSpSMDescr_t

    ctypedef enum hipsparseFormat_t:
        HIPSPARSE_FORMAT_CSR
        HIPSPARSE_FORMAT_CSC
        HIPSPARSE_FORMAT_COO
        HIPSPARSE_FORMAT_COO_AOS
        HIPSPARSE_FORMAT_BLOCKED_ELL

    ctypedef enum hipsparseOrder_t:
        HIPSPARSE_ORDER_ROW
        HIPSPARSE_ORDER_COLUMN
        HIPSPARSE_ORDER_COL

    ctypedef enum hipsparseIndexType_t:
        HIPSPARSE_INDEX_16U
        HIPSPARSE_INDEX_32I
        HIPSPARSE_INDEX_64I

    ctypedef enum hipsparseSpMVAlg_t:
        HIPSPARSE_MV_ALG_DEFAULT
        HIPSPARSE_COOMV_ALG
        HIPSPARSE_CSRMV_ALG1
        HIPSPARSE_CSRMV_ALG2
        HIPSPARSE_SPMV_ALG_DEFAULT
        HIPSPARSE_SPMV_COO_ALG1
        HIPSPARSE_SPMV_COO_ALG2
        HIPSPARSE_SPMV_CSR_ALG1
        HIPSPARSE_SPMV_CSR_ALG2

    ctypedef enum hipsparseSpMMAlg_t:
        HIPSPARSE_MM_ALG_DEFAULT
        HIPSPARSE_COOMM_ALG1
        HIPSPARSE_COOMM_ALG2
        HIPSPARSE_COOMM_ALG3
        HIPSPARSE_CSRMM_ALG1
        HIPSPARSE_SPMM_ALG_DEFAULT
        HIPSPARSE_SPMM_COO_ALG1
        HIPSPARSE_SPMM_COO_ALG2
        HIPSPARSE_SPMM_COO_ALG3
        HIPSPARSE_SPMM_COO_ALG4
        HIPSPARSE_SPMM_CSR_ALG1
        HIPSPARSE_SPMM_CSR_ALG2
        HIPSPARSE_SPMM_BLOCKED_ELL_ALG1
        HIPSPARSE_SPMM_CSR_ALG3

    ctypedef enum hipsparseSparseToDenseAlg_t:
        HIPSPARSE_SPARSETODENSE_ALG_DEFAULT

    ctypedef enum hipsparseDenseToSparseAlg_t:
        HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT

    ctypedef enum hipsparseSDDMMAlg_t:
        HIPSPARSE_SDDMM_ALG_DEFAULT

    ctypedef enum hipsparseSpSVAlg_t:
        HIPSPARSE_SPSV_ALG_DEFAULT

    ctypedef enum hipsparseSpSMAlg_t:
        HIPSPARSE_SPSM_ALG_DEFAULT

    ctypedef enum hipsparseSpMatAttribute_t:
        HIPSPARSE_SPMAT_FILL_MODE
        HIPSPARSE_SPMAT_DIAG_TYPE

    ctypedef enum hipsparseSpGEMMAlg_t:
        HIPSPARSE_SPGEMM_DEFAULT
        HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC
        HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC


cdef hipsparseStatus_t hipsparseCreateSpVec(void ** spVecDescr,long size,long nnz,void * indices,void * values,hipsparseIndexType_t idxType,hipsparseIndexBase_t idxBase,hipDataType valueType) nogil



cdef hipsparseStatus_t hipsparseDestroySpVec(void * spVecDescr) nogil



cdef hipsparseStatus_t hipsparseSpVecGet(void *const spVecDescr,long * size,long * nnz,void ** indices,void ** values,hipsparseIndexType_t * idxType,hipsparseIndexBase_t * idxBase,hipDataType * valueType) nogil



cdef hipsparseStatus_t hipsparseSpVecGetIndexBase(void *const spVecDescr,hipsparseIndexBase_t * idxBase) nogil



cdef hipsparseStatus_t hipsparseSpVecGetValues(void *const spVecDescr,void ** values) nogil



cdef hipsparseStatus_t hipsparseSpVecSetValues(void * spVecDescr,void * values) nogil



cdef hipsparseStatus_t hipsparseCreateCoo(void ** spMatDescr,long rows,long cols,long nnz,void * cooRowInd,void * cooColInd,void * cooValues,hipsparseIndexType_t cooIdxType,hipsparseIndexBase_t idxBase,hipDataType valueType) nogil



cdef hipsparseStatus_t hipsparseCreateCooAoS(void ** spMatDescr,long rows,long cols,long nnz,void * cooInd,void * cooValues,hipsparseIndexType_t cooIdxType,hipsparseIndexBase_t idxBase,hipDataType valueType) nogil



cdef hipsparseStatus_t hipsparseCreateCsr(void ** spMatDescr,long rows,long cols,long nnz,void * csrRowOffsets,void * csrColInd,void * csrValues,hipsparseIndexType_t csrRowOffsetsType,hipsparseIndexType_t csrColIndType,hipsparseIndexBase_t idxBase,hipDataType valueType) nogil



cdef hipsparseStatus_t hipsparseCreateCsc(void ** spMatDescr,long rows,long cols,long nnz,void * cscColOffsets,void * cscRowInd,void * cscValues,hipsparseIndexType_t cscColOffsetsType,hipsparseIndexType_t cscRowIndType,hipsparseIndexBase_t idxBase,hipDataType valueType) nogil



cdef hipsparseStatus_t hipsparseCreateBlockedEll(void ** spMatDescr,long rows,long cols,long ellBlockSize,long ellCols,void * ellColInd,void * ellValue,hipsparseIndexType_t ellIdxType,hipsparseIndexBase_t idxBase,hipDataType valueType) nogil



cdef hipsparseStatus_t hipsparseDestroySpMat(void * spMatDescr) nogil



cdef hipsparseStatus_t hipsparseCooGet(void *const spMatDescr,long * rows,long * cols,long * nnz,void ** cooRowInd,void ** cooColInd,void ** cooValues,hipsparseIndexType_t * idxType,hipsparseIndexBase_t * idxBase,hipDataType * valueType) nogil



cdef hipsparseStatus_t hipsparseCooAoSGet(void *const spMatDescr,long * rows,long * cols,long * nnz,void ** cooInd,void ** cooValues,hipsparseIndexType_t * idxType,hipsparseIndexBase_t * idxBase,hipDataType * valueType) nogil



cdef hipsparseStatus_t hipsparseCsrGet(void *const spMatDescr,long * rows,long * cols,long * nnz,void ** csrRowOffsets,void ** csrColInd,void ** csrValues,hipsparseIndexType_t * csrRowOffsetsType,hipsparseIndexType_t * csrColIndType,hipsparseIndexBase_t * idxBase,hipDataType * valueType) nogil



cdef hipsparseStatus_t hipsparseBlockedEllGet(void *const spMatDescr,long * rows,long * cols,long * ellBlockSize,long * ellCols,void ** ellColInd,void ** ellValue,hipsparseIndexType_t * ellIdxType,hipsparseIndexBase_t * idxBase,hipDataType * valueType) nogil



cdef hipsparseStatus_t hipsparseCsrSetPointers(void * spMatDescr,void * csrRowOffsets,void * csrColInd,void * csrValues) nogil



cdef hipsparseStatus_t hipsparseCscSetPointers(void * spMatDescr,void * cscColOffsets,void * cscRowInd,void * cscValues) nogil



cdef hipsparseStatus_t hipsparseCooSetPointers(void * spMatDescr,void * cooRowInd,void * cooColInd,void * cooValues) nogil



cdef hipsparseStatus_t hipsparseSpMatGetSize(void * spMatDescr,long * rows,long * cols,long * nnz) nogil



cdef hipsparseStatus_t hipsparseSpMatGetFormat(void *const spMatDescr,hipsparseFormat_t * format) nogil



cdef hipsparseStatus_t hipsparseSpMatGetIndexBase(void *const spMatDescr,hipsparseIndexBase_t * idxBase) nogil



cdef hipsparseStatus_t hipsparseSpMatGetValues(void * spMatDescr,void ** values) nogil



cdef hipsparseStatus_t hipsparseSpMatSetValues(void * spMatDescr,void * values) nogil



cdef hipsparseStatus_t hipsparseSpMatGetStridedBatch(void * spMatDescr,int * batchCount) nogil



cdef hipsparseStatus_t hipsparseSpMatSetStridedBatch(void * spMatDescr,int batchCount) nogil



cdef hipsparseStatus_t hipsparseCooSetStridedBatch(void * spMatDescr,int batchCount,long batchStride) nogil



cdef hipsparseStatus_t hipsparseCsrSetStridedBatch(void * spMatDescr,int batchCount,long offsetsBatchStride,long columnsValuesBatchStride) nogil



cdef hipsparseStatus_t hipsparseSpMatGetAttribute(void * spMatDescr,hipsparseSpMatAttribute_t attribute,void * data,unsigned long dataSize) nogil



cdef hipsparseStatus_t hipsparseSpMatSetAttribute(void * spMatDescr,hipsparseSpMatAttribute_t attribute,const void * data,unsigned long dataSize) nogil



cdef hipsparseStatus_t hipsparseCreateDnVec(void ** dnVecDescr,long size,void * values,hipDataType valueType) nogil



cdef hipsparseStatus_t hipsparseDestroyDnVec(void * dnVecDescr) nogil



cdef hipsparseStatus_t hipsparseDnVecGet(void *const dnVecDescr,long * size,void ** values,hipDataType * valueType) nogil



cdef hipsparseStatus_t hipsparseDnVecGetValues(void *const dnVecDescr,void ** values) nogil



cdef hipsparseStatus_t hipsparseDnVecSetValues(void * dnVecDescr,void * values) nogil



cdef hipsparseStatus_t hipsparseCreateDnMat(void ** dnMatDescr,long rows,long cols,long ld,void * values,hipDataType valueType,hipsparseOrder_t order) nogil



cdef hipsparseStatus_t hipsparseDestroyDnMat(void * dnMatDescr) nogil



cdef hipsparseStatus_t hipsparseDnMatGet(void *const dnMatDescr,long * rows,long * cols,long * ld,void ** values,hipDataType * valueType,hipsparseOrder_t * order) nogil



cdef hipsparseStatus_t hipsparseDnMatGetValues(void *const dnMatDescr,void ** values) nogil



cdef hipsparseStatus_t hipsparseDnMatSetValues(void * dnMatDescr,void * values) nogil



cdef hipsparseStatus_t hipsparseDnMatGetStridedBatch(void * dnMatDescr,int * batchCount,long * batchStride) nogil



cdef hipsparseStatus_t hipsparseDnMatSetStridedBatch(void * dnMatDescr,int batchCount,long batchStride) nogil



cdef hipsparseStatus_t hipsparseAxpby(void * handle,const void * alpha,void * vecX,const void * beta,void * vecY) nogil



cdef hipsparseStatus_t hipsparseGather(void * handle,void * vecY,void * vecX) nogil



cdef hipsparseStatus_t hipsparseScatter(void * handle,void * vecX,void * vecY) nogil



cdef hipsparseStatus_t hipsparseRot(void * handle,const void * c_coeff,const void * s_coeff,void * vecX,void * vecY) nogil



cdef hipsparseStatus_t hipsparseSparseToDense_bufferSize(void * handle,void * matA,void * matB,hipsparseSparseToDenseAlg_t alg,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseSparseToDense(void * handle,void * matA,void * matB,hipsparseSparseToDenseAlg_t alg,void * externalBuffer) nogil



cdef hipsparseStatus_t hipsparseDenseToSparse_bufferSize(void * handle,void * matA,void * matB,hipsparseDenseToSparseAlg_t alg,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseDenseToSparse_analysis(void * handle,void * matA,void * matB,hipsparseDenseToSparseAlg_t alg,void * externalBuffer) nogil



cdef hipsparseStatus_t hipsparseDenseToSparse_convert(void * handle,void * matA,void * matB,hipsparseDenseToSparseAlg_t alg,void * externalBuffer) nogil



cdef hipsparseStatus_t hipsparseSpVV_bufferSize(void * handle,hipsparseOperation_t opX,void * vecX,void * vecY,void * result,hipDataType computeType,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseSpVV(void * handle,hipsparseOperation_t opX,void * vecX,void * vecY,void * result,hipDataType computeType,void * externalBuffer) nogil



cdef hipsparseStatus_t hipsparseSpMV_bufferSize(void * handle,hipsparseOperation_t opA,const void * alpha,void *const matA,void *const vecX,const void * beta,void *const vecY,hipDataType computeType,hipsparseSpMVAlg_t alg,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseSpMV_preprocess(void * handle,hipsparseOperation_t opA,const void * alpha,void *const matA,void *const vecX,const void * beta,void *const vecY,hipDataType computeType,hipsparseSpMVAlg_t alg,void * externalBuffer) nogil



cdef hipsparseStatus_t hipsparseSpMV(void * handle,hipsparseOperation_t opA,const void * alpha,void *const matA,void *const vecX,const void * beta,void *const vecY,hipDataType computeType,hipsparseSpMVAlg_t alg,void * externalBuffer) nogil



cdef hipsparseStatus_t hipsparseSpMM_bufferSize(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const matA,void *const matB,const void * beta,void *const matC,hipDataType computeType,hipsparseSpMMAlg_t alg,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseSpMM_preprocess(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const matA,void *const matB,const void * beta,void *const matC,hipDataType computeType,hipsparseSpMMAlg_t alg,void * externalBuffer) nogil



cdef hipsparseStatus_t hipsparseSpMM(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const matA,void *const matB,const void * beta,void *const matC,hipDataType computeType,hipsparseSpMMAlg_t alg,void * externalBuffer) nogil



cdef hipsparseStatus_t hipsparseSpGEMM_createDescr(hipsparseSpGEMMDescr_t* descr) nogil



cdef hipsparseStatus_t hipsparseSpGEMM_destroyDescr(hipsparseSpGEMMDescr_t descr) nogil



cdef hipsparseStatus_t hipsparseSpGEMM_workEstimation(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void * matA,void * matB,const void * beta,void * matC,hipDataType computeType,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr,unsigned long * bufferSize1,void * externalBuffer1) nogil



cdef hipsparseStatus_t hipsparseSpGEMM_compute(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void * matA,void * matB,const void * beta,void * matC,hipDataType computeType,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr,unsigned long * bufferSize2,void * externalBuffer2) nogil



cdef hipsparseStatus_t hipsparseSpGEMM_copy(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void * matA,void * matB,const void * beta,void * matC,hipDataType computeType,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr) nogil



cdef hipsparseStatus_t hipsparseSpGEMMreuse_workEstimation(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,void * matA,void * matB,void * matC,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr,unsigned long * bufferSize1,void * externalBuffer1) nogil



cdef hipsparseStatus_t hipsparseSpGEMMreuse_nnz(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,void * matA,void * matB,void * matC,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr,unsigned long * bufferSize2,void * externalBuffer2,unsigned long * bufferSize3,void * externalBuffer3,unsigned long * bufferSize4,void * externalBuffer4) nogil



cdef hipsparseStatus_t hipsparseSpGEMMreuse_compute(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void * matA,void * matB,const void * beta,void * matC,hipDataType computeType,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr) nogil



cdef hipsparseStatus_t hipsparseSpGEMMreuse_copy(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,void * matA,void * matB,void * matC,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr,unsigned long * bufferSize5,void * externalBuffer5) nogil



cdef hipsparseStatus_t hipsparseSDDMM(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const A,void *const B,const void * beta,void * C,hipDataType computeType,hipsparseSDDMMAlg_t alg,void * tempBuffer) nogil



cdef hipsparseStatus_t hipsparseSDDMM_bufferSize(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const A,void *const B,const void * beta,void * C,hipDataType computeType,hipsparseSDDMMAlg_t alg,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseSDDMM_preprocess(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const A,void *const B,const void * beta,void * C,hipDataType computeType,hipsparseSDDMMAlg_t alg,void * tempBuffer) nogil



cdef hipsparseStatus_t hipsparseSpSV_createDescr(hipsparseSpSVDescr_t* descr) nogil



cdef hipsparseStatus_t hipsparseSpSV_destroyDescr(hipsparseSpSVDescr_t descr) nogil



cdef hipsparseStatus_t hipsparseSpSV_bufferSize(void * handle,hipsparseOperation_t opA,const void * alpha,void *const matA,void *const x,void *const y,hipDataType computeType,hipsparseSpSVAlg_t alg,hipsparseSpSVDescr_t spsvDescr,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseSpSV_analysis(void * handle,hipsparseOperation_t opA,const void * alpha,void *const matA,void *const x,void *const y,hipDataType computeType,hipsparseSpSVAlg_t alg,hipsparseSpSVDescr_t spsvDescr,void * externalBuffer) nogil



cdef hipsparseStatus_t hipsparseSpSV_solve(void * handle,hipsparseOperation_t opA,const void * alpha,void *const matA,void *const x,void *const y,hipDataType computeType,hipsparseSpSVAlg_t alg,hipsparseSpSVDescr_t spsvDescr,void * externalBuffer) nogil



cdef hipsparseStatus_t hipsparseSpSM_createDescr(hipsparseSpSMDescr_t* descr) nogil



cdef hipsparseStatus_t hipsparseSpSM_destroyDescr(hipsparseSpSMDescr_t descr) nogil



cdef hipsparseStatus_t hipsparseSpSM_bufferSize(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const matA,void *const matB,void *const matC,hipDataType computeType,hipsparseSpSMAlg_t alg,hipsparseSpSMDescr_t spsmDescr,unsigned long * bufferSize) nogil



cdef hipsparseStatus_t hipsparseSpSM_analysis(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const matA,void *const matB,void *const matC,hipDataType computeType,hipsparseSpSMAlg_t alg,hipsparseSpSMDescr_t spsmDescr,void * externalBuffer) nogil



cdef hipsparseStatus_t hipsparseSpSM_solve(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const matA,void *const matB,void *const matC,hipDataType computeType,hipsparseSpSMAlg_t alg,hipsparseSpSMDescr_t spsmDescr,void * externalBuffer) nogil
