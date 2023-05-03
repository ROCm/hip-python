# AMD_COPYRIGHT
from libc.stdint cimport *
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
from .chip cimport hipStream_t
cdef extern from "hipblas/hipblas.h":

    cdef int hipblasVersionMajor

    cdef int hipblaseVersionMinor

    cdef int hipblasVersionMinor

    cdef int hipblasVersionPatch

    ctypedef short __int16_t

    ctypedef unsigned short __uint16_t

    ctypedef void * hipblasHandle_t

    ctypedef uint16_t hipblasHalf

    ctypedef int8_t hipblasInt8

    ctypedef int64_t hipblasStride

    cdef struct hipblasBfloat16:
        unsigned short data

    cdef struct hipblasComplex:
        float x
        float y

    cdef struct hipblasDoubleComplex:
        double x
        double y

    ctypedef enum hipblasStatus_t:
        HIPBLAS_STATUS_SUCCESS
        HIPBLAS_STATUS_NOT_INITIALIZED
        HIPBLAS_STATUS_ALLOC_FAILED
        HIPBLAS_STATUS_INVALID_VALUE
        HIPBLAS_STATUS_MAPPING_ERROR
        HIPBLAS_STATUS_EXECUTION_FAILED
        HIPBLAS_STATUS_INTERNAL_ERROR
        HIPBLAS_STATUS_NOT_SUPPORTED
        HIPBLAS_STATUS_ARCH_MISMATCH
        HIPBLAS_STATUS_HANDLE_IS_NULLPTR
        HIPBLAS_STATUS_INVALID_ENUM
        HIPBLAS_STATUS_UNKNOWN

    ctypedef enum hipblasOperation_t:
        HIPBLAS_OP_N
        HIPBLAS_OP_T
        HIPBLAS_OP_C

    ctypedef enum hipblasPointerMode_t:
        HIPBLAS_POINTER_MODE_HOST
        HIPBLAS_POINTER_MODE_DEVICE

    ctypedef enum hipblasFillMode_t:
        HIPBLAS_FILL_MODE_UPPER
        HIPBLAS_FILL_MODE_LOWER
        HIPBLAS_FILL_MODE_FULL

    ctypedef enum hipblasDiagType_t:
        HIPBLAS_DIAG_NON_UNIT
        HIPBLAS_DIAG_UNIT

    ctypedef enum hipblasSideMode_t:
        HIPBLAS_SIDE_LEFT
        HIPBLAS_SIDE_RIGHT
        HIPBLAS_SIDE_BOTH

    ctypedef enum hipblasDatatype_t:
        HIPBLAS_R_16F
        HIPBLAS_R_32F
        HIPBLAS_R_64F
        HIPBLAS_C_16F
        HIPBLAS_C_32F
        HIPBLAS_C_64F
        HIPBLAS_R_8I
        HIPBLAS_R_8U
        HIPBLAS_R_32I
        HIPBLAS_R_32U
        HIPBLAS_C_8I
        HIPBLAS_C_8U
        HIPBLAS_C_32I
        HIPBLAS_C_32U
        HIPBLAS_R_16B
        HIPBLAS_C_16B

    ctypedef enum hipblasGemmAlgo_t:
        HIPBLAS_GEMM_DEFAULT

    ctypedef enum hipblasAtomicsMode_t:
        HIPBLAS_ATOMICS_NOT_ALLOWED
        HIPBLAS_ATOMICS_ALLOWED

    ctypedef enum hipblasInt8Datatype_t:
        HIPBLAS_INT8_DATATYPE_DEFAULT
        HIPBLAS_INT8_DATATYPE_INT8
        HIPBLAS_INT8_DATATYPE_PACK_INT8x4

# ! \brief Create hipblas handle. */
cdef hipblasStatus_t hipblasCreate(void ** handle) nogil


# ! \brief Destroys the library context created using hipblasCreate() */
cdef hipblasStatus_t hipblasDestroy(void * handle) nogil


# ! \brief Set stream for handle */
cdef hipblasStatus_t hipblasSetStream(void * handle,hipStream_t streamId) nogil


# ! \brief Get stream[0] for handle */
cdef hipblasStatus_t hipblasGetStream(void * handle,hipStream_t* streamId) nogil


# ! \brief Set hipblas pointer mode */
cdef hipblasStatus_t hipblasSetPointerMode(void * handle,hipblasPointerMode_t mode) nogil


# ! \brief Get hipblas pointer mode */
cdef hipblasStatus_t hipblasGetPointerMode(void * handle,hipblasPointerMode_t * mode) nogil


# ! \brief Set hipblas int8 Datatype */
cdef hipblasStatus_t hipblasSetInt8Datatype(void * handle,hipblasInt8Datatype_t int8Type) nogil


# ! \brief Get hipblas int8 Datatype*/
cdef hipblasStatus_t hipblasGetInt8Datatype(void * handle,hipblasInt8Datatype_t * int8Type) nogil


# ! \brief copy vector from host to device
#     @param[in]
#     n           [int]
#                 number of elements in the vector
#     @param[in]
#     elemSize    [int]
#                 Size of both vectors in bytes
#     @param[in]
#     x           pointer to vector on the host
#     @param[in]
#     incx        [int]
#                 specifies the increment for the elements of the vector
#     @param[out]
#     y           pointer to vector on the device
#     @param[in]
#     incy        [int]
#                 specifies the increment for the elements of the vector
cdef hipblasStatus_t hipblasSetVector(int n,int elemSize,const void * x,int incx,void * y,int incy) nogil


# ! \brief copy vector from device to host
#     @param[in]
#     n           [int]
#                 number of elements in the vector
#     @param[in]
#     elemSize    [int]
#                 Size of both vectors in bytes
#     @param[in]
#     x           pointer to vector on the device
#     @param[in]
#     incx        [int]
#                 specifies the increment for the elements of the vector
#     @param[out]
#     y           pointer to vector on the host
#     @param[in]
#     incy        [int]
#                 specifies the increment for the elements of the vector
cdef hipblasStatus_t hipblasGetVector(int n,int elemSize,const void * x,int incx,void * y,int incy) nogil


# ! \brief copy matrix from host to device
#     @param[in]
#     rows        [int]
#                 number of rows in matrices
#     @param[in]
#     cols        [int]
#                 number of columns in matrices
#     @param[in]
#     elemSize   [int]
#                 number of bytes per element in the matrix
#     @param[in]
#     AP          pointer to matrix on the host
#     @param[in]
#     lda         [int]
#                 specifies the leading dimension of A, lda >= rows
#     @param[out]
#     BP           pointer to matrix on the GPU
#     @param[in]
#     ldb         [int]
#                 specifies the leading dimension of B, ldb >= rows
cdef hipblasStatus_t hipblasSetMatrix(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb) nogil


# ! \brief copy matrix from device to host
#     @param[in]
#     rows        [int]
#                 number of rows in matrices
#     @param[in]
#     cols        [int]
#                 number of columns in matrices
#     @param[in]
#     elemSize   [int]
#                 number of bytes per element in the matrix
#     @param[in]
#     AP          pointer to matrix on the GPU
#     @param[in]
#     lda         [int]
#                 specifies the leading dimension of A, lda >= rows
#     @param[out]
#     BP          pointer to matrix on the host
#     @param[in]
#     ldb         [int]
#                 specifies the leading dimension of B, ldb >= rows
cdef hipblasStatus_t hipblasGetMatrix(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb) nogil


# ! \brief asynchronously copy vector from host to device
#     \details
#     hipblasSetVectorAsync copies a vector from pinned host memory to device memory asynchronously.
#     Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
#     @param[in]
#     n           [int]
#                 number of elements in the vector
#     @param[in]
#     elemSize   [int]
#                 number of bytes per element in the matrix
#     @param[in]
#     x           pointer to vector on the host
#     @param[in]
#     incx        [int]
#                 specifies the increment for the elements of the vector
#     @param[out]
#     y           pointer to vector on the device
#     @param[in]
#     incy        [int]
#                 specifies the increment for the elements of the vector
#     @param[in]
#     stream      specifies the stream into which this transfer request is queued
cdef hipblasStatus_t hipblasSetVectorAsync(int n,int elemSize,const void * x,int incx,void * y,int incy,hipStream_t stream) nogil


# ! \brief asynchronously copy vector from device to host
#     \details
#     hipblasGetVectorAsync copies a vector from pinned host memory to device memory asynchronously.
#     Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
#     @param[in]
#     n           [int]
#                 number of elements in the vector
#     @param[in]
#     elemSize   [int]
#                 number of bytes per element in the matrix
#     @param[in]
#     x           pointer to vector on the device
#     @param[in]
#     incx        [int]
#                 specifies the increment for the elements of the vector
#     @param[out]
#     y           pointer to vector on the host
#     @param[in]
#     incy        [int]
#                 specifies the increment for the elements of the vector
#     @param[in]
#     stream      specifies the stream into which this transfer request is queued
cdef hipblasStatus_t hipblasGetVectorAsync(int n,int elemSize,const void * x,int incx,void * y,int incy,hipStream_t stream) nogil


# ! \brief asynchronously copy matrix from host to device
#     \details
#     hipblasSetMatrixAsync copies a matrix from pinned host memory to device memory asynchronously.
#     Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
#     @param[in]
#     rows        [int]
#                 number of rows in matrices
#     @param[in]
#     cols        [int]
#                 number of columns in matrices
#     @param[in]
#     elemSize   [int]
#                 number of bytes per element in the matrix
#     @param[in]
#     AP           pointer to matrix on the host
#     @param[in]
#     lda         [int]
#                 specifies the leading dimension of A, lda >= rows
#     @param[out]
#     BP           pointer to matrix on the GPU
#     @param[in]
#     ldb         [int]
#                 specifies the leading dimension of B, ldb >= rows
#     @param[in]
#     stream      specifies the stream into which this transfer request is queued
cdef hipblasStatus_t hipblasSetMatrixAsync(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb,hipStream_t stream) nogil


# ! \brief asynchronously copy matrix from device to host
#     \details
#     hipblasGetMatrixAsync copies a matrix from device memory to pinned host memory asynchronously.
#     Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
#     @param[in]
#     rows        [int]
#                 number of rows in matrices
#     @param[in]
#     cols        [int]
#                 number of columns in matrices
#     @param[in]
#     elemSize   [int]
#                 number of bytes per element in the matrix
#     @param[in]
#     AP          pointer to matrix on the GPU
#     @param[in]
#     lda         [int]
#                 specifies the leading dimension of A, lda >= rows
#     @param[out]
#     BP           pointer to matrix on the host
#     @param[in]
#     ldb         [int]
#                 specifies the leading dimension of B, ldb >= rows
#     @param[in]
#     stream      specifies the stream into which this transfer request is queued
cdef hipblasStatus_t hipblasGetMatrixAsync(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb,hipStream_t stream) nogil


# ! \brief Set hipblasSetAtomicsMode*/
cdef hipblasStatus_t hipblasSetAtomicsMode(void * handle,hipblasAtomicsMode_t atomics_mode) nogil


# ! \brief Get hipblasSetAtomicsMode*/
cdef hipblasStatus_t hipblasGetAtomicsMode(void * handle,hipblasAtomicsMode_t * atomics_mode) nogil


# ! @{
#     \brief BLAS Level 1 API
# 
#     \details
#     amax finds the first index of the element of maximum magnitude of a vector x.
# 
#     - Supported precisions in rocBLAS : s,d,c,z.
#     - Supported precisions in cuBLAS  : s,d,c,z.
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     n         [int]
#               the number of elements in x.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of y.
#     @param[inout]
#     result
#               device pointer or host pointer to store the amax index.
#               return is 0.0 if n, incx<=0.
cdef hipblasStatus_t hipblasIsamax(void * handle,int n,const float * x,int incx,int * result) nogil



cdef hipblasStatus_t hipblasIdamax(void * handle,int n,const double * x,int incx,int * result) nogil



cdef hipblasStatus_t hipblasIcamax(void * handle,int n,hipblasComplex * x,int incx,int * result) nogil



cdef hipblasStatus_t hipblasIzamax(void * handle,int n,hipblasDoubleComplex * x,int incx,int * result) nogil


# ! @{
#     \brief BLAS Level 1 API
# 
#     \details
#     amin finds the first index of the element of minimum magnitude of a vector x.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     n         [int]
#               the number of elements in x.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of y.
#     @param[inout]
#     result
#               device pointer or host pointer to store the amin index.
#               return is 0.0 if n, incx<=0.
cdef hipblasStatus_t hipblasIsamin(void * handle,int n,const float * x,int incx,int * result) nogil



cdef hipblasStatus_t hipblasIdamin(void * handle,int n,const double * x,int incx,int * result) nogil



cdef hipblasStatus_t hipblasIcamin(void * handle,int n,hipblasComplex * x,int incx,int * result) nogil



cdef hipblasStatus_t hipblasIzamin(void * handle,int n,hipblasDoubleComplex * x,int incx,int * result) nogil


# ! @{
#     \brief BLAS Level 1 API
# 
#     \details
#     asum computes the sum of the magnitudes of elements of a real vector x,
#          or the sum of magnitudes of the real and imaginary parts of elements if x is a complex vector.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     n         [int]
#               the number of elements in x and y.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x. incx must be > 0.
#     @param[inout]
#     result
#               device pointer or host pointer to store the asum product.
#               return is 0.0 if n <= 0.
#
cdef hipblasStatus_t hipblasSasum(void * handle,int n,const float * x,int incx,float * result) nogil



cdef hipblasStatus_t hipblasDasum(void * handle,int n,const double * x,int incx,double * result) nogil



cdef hipblasStatus_t hipblasScasum(void * handle,int n,hipblasComplex * x,int incx,float * result) nogil



cdef hipblasStatus_t hipblasDzasum(void * handle,int n,hipblasDoubleComplex * x,int incx,double * result) nogil


# ! @{
#     \brief BLAS Level 1 API
# 
#     \details
#     axpy   computes constant alpha multiplied by vector x, plus vector y
# 
#         y := alpha * x + y
# 
#     - Supported precisions in rocBLAS : h,s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     n         [int]
#               the number of elements in x and y.
#     @param[in]
#     alpha     device pointer or host pointer to specify the scalar alpha.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[out]
#     y         device pointer storing vector y.
#     @param[inout]
#     incy      [int]
#               specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasHaxpy(void * handle,int n,const unsigned short * alpha,const unsigned short * x,int incx,unsigned short * y,int incy) nogil



cdef hipblasStatus_t hipblasSaxpy(void * handle,int n,const float * alpha,const float * x,int incx,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDaxpy(void * handle,int n,const double * alpha,const double * x,int incx,double * y,int incy) nogil



cdef hipblasStatus_t hipblasCaxpy(void * handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZaxpy(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil


# ! @{
#     \brief BLAS Level 1 API
# 
#     \details
#     copy  copies each element x[i] into y[i], for  i = 1 , ... , n
# 
#         y := x,
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     n         [int]
#               the number of elements in x to be copied to y.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[out]
#     y         device pointer storing vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasScopy(void * handle,int n,const float * x,int incx,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDcopy(void * handle,int n,const double * x,int incx,double * y,int incy) nogil



cdef hipblasStatus_t hipblasCcopy(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZcopy(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil


# ! @{
#     \brief BLAS Level 1 API
# 
#     \details
#     dot(u)  performs the dot product of vectors x and y
# 
#         result = x * y;
# 
#     dotc  performs the dot product of the conjugate of complex vector x and complex vector y
# 
#         result = conjugate (x) * y;
# 
#     - Supported precisions in rocBLAS : h,bf,s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     n         [int]
#               the number of elements in x and y.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of y.
#     @param[in]
#     y         device pointer storing vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#     @param[inout]
#     result
#               device pointer or host pointer to store the dot product.
#               return is 0.0 if n <= 0.
#
cdef hipblasStatus_t hipblasHdot(void * handle,int n,const unsigned short * x,int incx,const unsigned short * y,int incy,unsigned short * result) nogil



cdef hipblasStatus_t hipblasBfdot(void * handle,int n,hipblasBfloat16 * x,int incx,hipblasBfloat16 * y,int incy,hipblasBfloat16 * result) nogil



cdef hipblasStatus_t hipblasSdot(void * handle,int n,const float * x,int incx,const float * y,int incy,float * result) nogil



cdef hipblasStatus_t hipblasDdot(void * handle,int n,const double * x,int incx,const double * y,int incy,double * result) nogil



cdef hipblasStatus_t hipblasCdotc(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * result) nogil



cdef hipblasStatus_t hipblasCdotu(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * result) nogil



cdef hipblasStatus_t hipblasZdotc(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * result) nogil



cdef hipblasStatus_t hipblasZdotu(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * result) nogil


# ! @{
#     \brief BLAS Level 1 API
# 
#     \details
#     nrm2 computes the euclidean norm of a real or complex vector
# 
#               result := sqrt( x'*x ) for real vectors
#               result := sqrt( x**H*x ) for complex vectors
# 
#     - Supported precisions in rocBLAS : s,d,c,z,sc,dz
#     - Supported precisions in cuBLAS  : s,d,sc,dz
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     n         [int]
#               the number of elements in x.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of y.
#     @param[inout]
#     result
#               device pointer or host pointer to store the nrm2 product.
#               return is 0.0 if n, incx<=0.
cdef hipblasStatus_t hipblasSnrm2(void * handle,int n,const float * x,int incx,float * result) nogil



cdef hipblasStatus_t hipblasDnrm2(void * handle,int n,const double * x,int incx,double * result) nogil



cdef hipblasStatus_t hipblasScnrm2(void * handle,int n,hipblasComplex * x,int incx,float * result) nogil



cdef hipblasStatus_t hipblasDznrm2(void * handle,int n,hipblasDoubleComplex * x,int incx,double * result) nogil


# ! @{
#     \brief BLAS Level 1 API
# 
#     \details
#     rot applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
#         Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
#     - Supported precisions in rocBLAS : s,d,c,z,sc,dz
#     - Supported precisions in cuBLAS  : s,d,c,z,cs,zd
# 
#     @param[in]
#     handle  [hipblasHandle_t]
#             handle to the hipblas library context queue.
#     @param[in]
#     n       [int]
#             number of elements in the x and y vectors.
#     @param[inout]
#     x       device pointer storing vector x.
#     @param[in]
#     incx    [int]
#             specifies the increment between elements of x.
#     @param[inout]
#     y       device pointer storing vector y.
#     @param[in]
#     incy    [int]
#             specifies the increment between elements of y.
#     @param[in]
#     c       device pointer or host pointer storing scalar cosine component of the rotation matrix.
#     @param[in]
#     s       device pointer or host pointer storing scalar sine component of the rotation matrix.
#
cdef hipblasStatus_t hipblasSrot(void * handle,int n,float * x,int incx,float * y,int incy,const float * c,const float * s) nogil



cdef hipblasStatus_t hipblasDrot(void * handle,int n,double * x,int incx,double * y,int incy,const double * c,const double * s) nogil



cdef hipblasStatus_t hipblasCrot(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,const float * c,hipblasComplex * s) nogil



cdef hipblasStatus_t hipblasCsrot(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,const float * c,const float * s) nogil



cdef hipblasStatus_t hipblasZrot(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,const double * c,hipblasDoubleComplex * s) nogil



cdef hipblasStatus_t hipblasZdrot(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,const double * c,const double * s) nogil


# ! @{
#     \brief BLAS Level 1 API
# 
#     \details
#     rotg creates the Givens rotation matrix for the vector (a b).
#          Scalars c and s and arrays a and b may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#          If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#          If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle  [hipblasHandle_t]
#             handle to the hipblas library context queue.
#     @param[inout]
#     a       device pointer or host pointer to input vector element, overwritten with r.
#     @param[inout]
#     b       device pointer or host pointer to input vector element, overwritten with z.
#     @param[inout]
#     c       device pointer or host pointer to cosine element of Givens rotation.
#     @param[inout]
#     s       device pointer or host pointer sine element of Givens rotation.
#
cdef hipblasStatus_t hipblasSrotg(void * handle,float * a,float * b,float * c,float * s) nogil



cdef hipblasStatus_t hipblasDrotg(void * handle,double * a,double * b,double * c,double * s) nogil



cdef hipblasStatus_t hipblasCrotg(void * handle,hipblasComplex * a,hipblasComplex * b,float * c,hipblasComplex * s) nogil



cdef hipblasStatus_t hipblasZrotg(void * handle,hipblasDoubleComplex * a,hipblasDoubleComplex * b,double * c,hipblasDoubleComplex * s) nogil


# ! @{
#     \brief BLAS Level 1 API
# 
#     \details
#     rotm applies the modified Givens rotation matrix defined by param to vectors x and y.
# 
#     - Supported precisions in rocBLAS : s,d
#     - Supported precisions in cuBLAS  : s,d
# 
#     @param[in]
#     handle  [hipblasHandle_t]
#             handle to the hipblas library context queue.
#     @param[in]
#     n       [int]
#             number of elements in the x and y vectors.
#     @param[inout]
#     x       device pointer storing vector x.
#     @param[in]
#     incx    [int]
#             specifies the increment between elements of x.
#     @param[inout]
#     y       device pointer storing vector y.
#     @param[in]
#     incy    [int]
#             specifies the increment between elements of y.
#     @param[in]
#     param   device vector or host vector of 5 elements defining the rotation.
#             param[0] = flag
#             param[1] = H11
#             param[2] = H21
#             param[3] = H12
#             param[4] = H22
#             The flag parameter defines the form of H:
#             flag = -1 => H = ( H11 H12 H21 H22 )
#             flag =  0 => H = ( 1.0 H12 H21 1.0 )
#             flag =  1 => H = ( H11 1.0 -1.0 H22 )
#             flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#             param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#
cdef hipblasStatus_t hipblasSrotm(void * handle,int n,float * x,int incx,float * y,int incy,const float * param) nogil



cdef hipblasStatus_t hipblasDrotm(void * handle,int n,double * x,int incx,double * y,int incy,const double * param) nogil


# ! @{
#     \brief BLAS Level 1 API
# 
#     \details
#     rotmg creates the modified Givens rotation matrix for the vector (d1 * x1, d2 * y1).
#           Parameters may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#           If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#           If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.
# 
#     - Supported precisions in rocBLAS : s,d
#     - Supported precisions in cuBLAS  : s,d
# 
#     @param[in]
#     handle  [hipblasHandle_t]
#             handle to the hipblas library context queue.
#     @param[inout]
#     d1      device pointer or host pointer to input scalar that is overwritten.
#     @param[inout]
#     d2      device pointer or host pointer to input scalar that is overwritten.
#     @param[inout]
#     x1      device pointer or host pointer to input scalar that is overwritten.
#     @param[in]
#     y1      device pointer or host pointer to input scalar.
#     @param[out]
#     param   device vector or host vector of 5 elements defining the rotation.
#             param[0] = flag
#             param[1] = H11
#             param[2] = H21
#             param[3] = H12
#             param[4] = H22
#             The flag parameter defines the form of H:
#             flag = -1 => H = ( H11 H12 H21 H22 )
#             flag =  0 => H = ( 1.0 H12 H21 1.0 )
#             flag =  1 => H = ( H11 1.0 -1.0 H22 )
#             flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#             param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#
cdef hipblasStatus_t hipblasSrotmg(void * handle,float * d1,float * d2,float * x1,const float * y1,float * param) nogil



cdef hipblasStatus_t hipblasDrotmg(void * handle,double * d1,double * d2,double * x1,const double * y1,double * param) nogil


# ! @{
#     \brief BLAS Level 1 API
# 
#     \details
#     scal  scales each element of vector x with scalar alpha.
# 
#         x := alpha * x
# 
#     - Supported precisions in rocBLAS : s,d,c,z,cs,zd
#     - Supported precisions in cuBLAS  : s,d,c,z,cs,zd
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     n         [int]
#               the number of elements in x.
#     @param[in]
#     alpha     device pointer or host pointer for the scalar alpha.
#     @param[inout]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
# 
#
cdef hipblasStatus_t hipblasSscal(void * handle,int n,const float * alpha,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDscal(void * handle,int n,const double * alpha,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCscal(void * handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasCsscal(void * handle,int n,const float * alpha,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZscal(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZdscal(void * handle,int n,const double * alpha,hipblasDoubleComplex * x,int incx) nogil


# ! @{
#     \brief BLAS Level 1 API
# 
#     \details
#     swap  interchanges vectors x and y.
# 
#         y := x; x := y
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     n         [int]
#               the number of elements in x and y.
#     @param[inout]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[inout]
#     y         device pointer storing vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasSswap(void * handle,int n,float * x,int incx,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDswap(void * handle,int n,double * x,int incx,double * y,int incy) nogil



cdef hipblasStatus_t hipblasCswap(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZswap(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     gbmv performs one of the matrix-vector operations
# 
#         y := alpha*A*x    + beta*y,   or
#         y := alpha*A**T*x + beta*y,   or
#         y := alpha*A**H*x + beta*y,
# 
#     where alpha and beta are scalars, x and y are vectors and A is an
#     m by n banded matrix with kl sub-diagonals and ku super-diagonals.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     trans     [hipblasOperation_t]
#               indicates whether matrix A is tranposed (conjugated) or not
#     @param[in]
#     m         [int]
#               number of rows of matrix A
#     @param[in]
#     n         [int]
#               number of columns of matrix A
#     @param[in]
#     kl        [int]
#               number of sub-diagonals of A
#     @param[in]
#     ku        [int]
#               number of super-diagonals of A
#     @param[in]
#     alpha     device pointer or host pointer to scalar alpha.
#     @param[in]
#         AP    device pointer storing banded matrix A.
#               Leading (kl + ku + 1) by n part of the matrix contains the coefficients
#               of the banded matrix. The leading diagonal resides in row (ku + 1) with
#               the first super-diagonal above on the RHS of row ku. The first sub-diagonal
#               resides below on the LHS of row ku + 2. This propogates up and down across
#               sub/super-diagonals.
#                 Ex: (m = n = 7; ku = 2, kl = 2)
#                 1 2 3 0 0 0 0             0 0 3 3 3 3 3
#                 4 1 2 3 0 0 0             0 2 2 2 2 2 2
#                 5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
#                 0 5 4 1 2 3 0             4 4 4 4 4 4 0
#                 0 0 5 4 1 2 0             5 5 5 5 5 0 0
#                 0 0 0 5 4 1 2             0 0 0 0 0 0 0
#                 0 0 0 0 5 4 1             0 0 0 0 0 0 0
#               Note that the empty elements which don't correspond to data will not
#               be referenced.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A. Must be >= (kl + ku + 1)
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[in]
#     beta      device pointer or host pointer to scalar beta.
#     @param[inout]
#     y         device pointer storing vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasSgbmv(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDgbmv(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil



cdef hipblasStatus_t hipblasCgbmv(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZgbmv(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     gemv performs one of the matrix-vector operations
# 
#         y := alpha*A*x    + beta*y,   or
#         y := alpha*A**T*x + beta*y,   or
#         y := alpha*A**H*x + beta*y,
# 
#     where alpha and beta are scalars, x and y are vectors and A is an
#     m by n matrix.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     trans     [hipblasOperation_t]
#               indicates whether matrix A is tranposed (conjugated) or not
#     @param[in]
#     m         [int]
#               number of rows of matrix A
#     @param[in]
#     n         [int]
#               number of columns of matrix A
#     @param[in]
#     alpha     device pointer or host pointer to scalar alpha.
#     @param[in]
#     AP        device pointer storing matrix A.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[in]
#     beta      device pointer or host pointer to scalar beta.
#     @param[inout]
#     y         device pointer storing vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasSgemv(void * handle,hipblasOperation_t trans,int m,int n,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDgemv(void * handle,hipblasOperation_t trans,int m,int n,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil



cdef hipblasStatus_t hipblasCgemv(void * handle,hipblasOperation_t trans,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZgemv(void * handle,hipblasOperation_t trans,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     ger,geru,gerc performs the matrix-vector operations
# 
#         A := A + alpha*x*y**T , OR
#         A := A + alpha*x*y**H for gerc
# 
#     where alpha is a scalar, x and y are vectors, and A is an
#     m by n matrix.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     m         [int]
#               the number of rows of the matrix A.
#     @param[in]
#     n         [int]
#               the number of columns of the matrix A.
#     @param[in]
#     alpha
#               device pointer or host pointer to scalar alpha.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[in]
#     y         device pointer storing vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#     @param[inout]
#     AP         device pointer storing matrix A.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A.
#
cdef hipblasStatus_t hipblasSger(void * handle,int m,int n,const float * alpha,const float * x,int incx,const float * y,int incy,float * AP,int lda) nogil



cdef hipblasStatus_t hipblasDger(void * handle,int m,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP,int lda) nogil



cdef hipblasStatus_t hipblasCgeru(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasCgerc(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasZgeru(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasZgerc(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     hbmv performs the matrix-vector operations
# 
#         y := alpha*A*x + beta*y
# 
#     where alpha and beta are scalars, x and y are n element vectors and A is an
#     n by n Hermitian band matrix, with k super-diagonals.
# 
#     - Supported precisions in rocBLAS : c,z
#     - Supported precisions in cuBLAS  : c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is being supplied.
#               HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is being supplied.
#     @param[in]
#     n         [int]
#               the order of the matrix A.
#     @param[in]
#     k         [int]
#               the number of super-diagonals of the matrix A. Must be >= 0.
#     @param[in]
#     alpha     device pointer or host pointer to scalar alpha.
#     @param[in]
#     AP        device pointer storing matrix A. Of dimension (lda, n).
#               if uplo == HIPBLAS_FILL_MODE_UPPER:
#                 The leading (k + 1) by n part of A must contain the upper
#                 triangular band part of the Hermitian matrix, with the leading
#                 diagonal in row (k + 1), the first super-diagonal on the RHS
#                 of row k, etc.
#                 The top left k by x triangle of A will not be referenced.
#                     Ex (upper, lda = n = 4, k = 1):
#                     A                             Represented matrix
#                     (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
#                     (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
#                     (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
#                     (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)
# 
#               if uplo == HIPBLAS_FILL_MODE_LOWER:
#                 The leading (k + 1) by n part of A must contain the lower
#                 triangular band part of the Hermitian matrix, with the leading
#                 diagonal in row (1), the first sub-diagonal on the LHS of
#                 row 2, etc.
#                 The bottom right k by k triangle of A will not be referenced.
#                     Ex (lower, lda = 2, n = 4, k = 1):
#                     A                               Represented matrix
#                     (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
#                     (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
#                                                     (0, 0) (6, 8) (3, 0) (7,-7)
#                                                     (0, 0) (0, 0) (7, 7) (4, 0)
# 
#               As a Hermitian matrix, the imaginary part of the main diagonal
#               of A will not be referenced and is assumed to be == 0.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A. must be >= k + 1
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[in]
#     beta      device pointer or host pointer to scalar beta.
#     @param[inout]
#     y         device pointer storing vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasChbmv(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZhbmv(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     hemv performs one of the matrix-vector operations
# 
#         y := alpha*A*x + beta*y
# 
#     where alpha and beta are scalars, x and y are n element vectors and A is an
#     n by n Hermitian matrix.
# 
#     - Supported precisions in rocBLAS : c,z
#     - Supported precisions in cuBLAS  : c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied.
#               HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied.
#     @param[in]
#     n         [int]
#               the order of the matrix A.
#     @param[in]
#     alpha     device pointer or host pointer to scalar alpha.
#     @param[in]
#     AP        device pointer storing matrix A. Of dimension (lda, n).
#               if uplo == HIPBLAS_FILL_MODE_UPPER:
#                 The upper triangular part of A must contain
#                 the upper triangular part of a Hermitian matrix. The lower
#                 triangular part of A will not be referenced.
#               if uplo == HIPBLAS_FILL_MODE_LOWER:
#                 The lower triangular part of A must contain
#                 the lower triangular part of a Hermitian matrix. The upper
#                 triangular part of A will not be referenced.
#               As a Hermitian matrix, the imaginary part of the main diagonal
#               of A will not be referenced and is assumed to be == 0.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A. must be >= max(1, n)
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[in]
#     beta      device pointer or host pointer to scalar beta.
#     @param[inout]
#     y         device pointer storing vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasChemv(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZhemv(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     her performs the matrix-vector operations
# 
#         A := A + alpha*x*x**H
# 
#     where alpha is a real scalar, x is a vector, and A is an
#     n by n Hermitian matrix.
# 
#     - Supported precisions in rocBLAS : c,z
#     - Supported precisions in cuBLAS  : c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#               HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in A.
#               HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in A.
#     @param[in]
#     n         [int]
#               the number of rows and columns of matrix A, must be at least 0.
#     @param[in]
#     alpha
#               device pointer or host pointer to scalar alpha.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[inout]
#     AP        device pointer storing the specified triangular portion of
#               the Hermitian matrix A. Of size (lda * n).
#               if uplo == HIPBLAS_FILL_MODE_UPPER:
#                 The upper triangular portion of the Hermitian matrix A is supplied. The lower
#                 triangluar portion will not be touched.
#             if uplo == HIPBLAS_FILL_MODE_LOWER:
#                 The lower triangular portion of the Hermitian matrix A is supplied. The upper
#                 triangular portion will not be touched.
#             Note that the imaginary part of the diagonal elements are not accessed and are assumed
#             to be 0.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A. Must be at least max(1, n).
cdef hipblasStatus_t hipblasCher(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex * x,int incx,hipblasComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasZher(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP,int lda) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     her2 performs the matrix-vector operations
# 
#         A := A + alpha*x*y**H + conj(alpha)*y*x**H
# 
#     where alpha is a complex scalar, x and y are vectors, and A is an
#     n by n Hermitian matrix.
# 
#     - Supported precisions in rocBLAS : c,z
#     - Supported precisions in cuBLAS  : c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#               HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied.
#               HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied.
#     @param[in]
#     n         [int]
#               the number of rows and columns of matrix A, must be at least 0.
#     @param[in]
#     alpha
#               device pointer or host pointer to scalar alpha.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[in]
#     y         device pointer storing vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#     @param[inout]
#     AP         device pointer storing the specified triangular portion of
#               the Hermitian matrix A. Of size (lda, n).
#               if uplo == HIPBLAS_FILL_MODE_UPPER:
#                 The upper triangular portion of the Hermitian matrix A is supplied. The lower triangular
#                 portion of A will not be touched.
#             if uplo == HIPBLAS_FILL_MODE_LOWER:
#                 The lower triangular portion of the Hermitian matrix A is supplied. The upper triangular
#                 portion of A will not be touched.
#             Note that the imaginary part of the diagonal elements are not accessed and are assumed
#             to be 0.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A. Must be at least max(lda, 1).
cdef hipblasStatus_t hipblasCher2(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasZher2(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     hpmv performs the matrix-vector operation
# 
#         y := alpha*A*x + beta*y
# 
#     where alpha and beta are scalars, x and y are n element vectors and A is an
#     n by n Hermitian matrix, supplied in packed form (see description below).
# 
#     - Supported precisions in rocBLAS : c,z
#     - Supported precisions in cuBLAS  : c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied in AP.
#               HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied in AP.
#     @param[in]
#     n         [int]
#               the order of the matrix A, must be >= 0.
#     @param[in]
#     alpha     device pointer or host pointer to scalar alpha.
#     @param[in]
#     AP        device pointer storing the packed version of the specified triangular portion of
#               the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
#               if uplo == HIPBLAS_FILL_MODE_UPPER:
#                 The upper triangular portion of the Hermitian matrix A is supplied.
#                 The matrix is compacted so that AP contains the triangular portion column-by-column
#                 so that:
#                 AP(0) = A(0,0)
#                 AP(1) = A(0,1)
#                 AP(2) = A(1,1), etc.
#                     Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                         (1, 0) (2, 1) (3, 2)
#                         (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
#                         (3,-2) (5, 1) (6, 0)
#             if uplo == HIPBLAS_FILL_MODE_LOWER:
#                 The lower triangular portion of the Hermitian matrix A is supplied.
#                 The matrix is compacted so that AP contains the triangular portion column-by-column
#                 so that:
#                 AP(0) = A(0,0)
#                 AP(1) = A(1,0)
#                 AP(2) = A(2,1), etc.
#                     Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                         (1, 0) (2, 1) (3, 2)
#                         (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
#                         (3,-2) (5, 1) (6, 0)
#             Note that the imaginary part of the diagonal elements are not accessed and are assumed
#             to be 0.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[in]
#     beta      device pointer or host pointer to scalar beta.
#     @param[inout]
#     y         device pointer storing vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasChpmv(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZhpmv(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     hpr performs the matrix-vector operations
# 
#         A := A + alpha*x*x**H
# 
#     where alpha is a real scalar, x is a vector, and A is an
#     n by n Hermitian matrix, supplied in packed form.
# 
#     - Supported precisions in rocBLAS : c,z
#     - Supported precisions in cuBLAS  : c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#               HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
#               HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
#     @param[in]
#     n         [int]
#               the number of rows and columns of matrix A, must be at least 0.
#     @param[in]
#     alpha
#               device pointer or host pointer to scalar alpha.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[inout]
#     AP        device pointer storing the packed version of the specified triangular portion of
#               the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
#               if uplo == HIPBLAS_FILL_MODE_UPPER:
#                 The upper triangular portion of the Hermitian matrix A is supplied.
#                 The matrix is compacted so that AP contains the triangular portion column-by-column
#                 so that:
#                 AP(0) = A(0,0)
#                 AP(1) = A(0,1)
#                 AP(2) = A(1,1), etc.
#                     Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                         (1, 0) (2, 1) (4,9)
#                         (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                         (4,-9) (5,-3) (6,0)
#             if uplo == HIPBLAS_FILL_MODE_LOWER:
#                 The lower triangular portion of the Hermitian matrix A is supplied.
#                 The matrix is compacted so that AP contains the triangular portion column-by-column
#                 so that:
#                 AP(0) = A(0,0)
#                 AP(1) = A(1,0)
#                 AP(2) = A(2,1), etc.
#                     Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                         (1, 0) (2, 1) (4,9)
#                         (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                         (4,-9) (5,-3) (6,0)
#             Note that the imaginary part of the diagonal elements are not accessed and are assumed
#             to be 0.
cdef hipblasStatus_t hipblasChpr(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex * x,int incx,hipblasComplex * AP) nogil



cdef hipblasStatus_t hipblasZhpr(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     hpr2 performs the matrix-vector operations
# 
#         A := A + alpha*x*y**H + conj(alpha)*y*x**H
# 
#     where alpha is a complex scalar, x and y are vectors, and A is an
#     n by n Hermitian matrix, supplied in packed form.
# 
#     - Supported precisions in rocBLAS : c,z
#     - Supported precisions in cuBLAS  : c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#               HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
#               HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
#     @param[in]
#     n         [int]
#               the number of rows and columns of matrix A, must be at least 0.
#     @param[in]
#     alpha
#               device pointer or host pointer to scalar alpha.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[in]
#     y         device pointer storing vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#     @param[inout]
#     AP        device pointer storing the packed version of the specified triangular portion of
#               the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
#               if uplo == HIPBLAS_FILL_MODE_UPPER:
#                 The upper triangular portion of the Hermitian matrix A is supplied.
#                 The matrix is compacted so that AP contains the triangular portion column-by-column
#                 so that:
#                 AP(0) = A(0,0)
#                 AP(1) = A(0,1)
#                 AP(2) = A(1,1), etc.
#                     Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                         (1, 0) (2, 1) (4,9)
#                         (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                         (4,-9) (5,-3) (6,0)
#             if uplo == HIPBLAS_FILL_MODE_LOWER:
#                 The lower triangular portion of the Hermitian matrix A is supplied.
#                 The matrix is compacted so that AP contains the triangular portion column-by-column
#                 so that:
#                 AP(0) = A(0,0)
#                 AP(1) = A(1,0)
#                 AP(2) = A(2,1), etc.
#                     Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                         (1, 0) (2, 1) (4,9)
#                         (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                         (4,-9) (5,-3) (6,0)
#             Note that the imaginary part of the diagonal elements are not accessed and are assumed
#             to be 0.
cdef hipblasStatus_t hipblasChpr2(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP) nogil



cdef hipblasStatus_t hipblasZhpr2(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     sbmv performs the matrix-vector operation:
# 
#         y := alpha*A*x + beta*y,
# 
#     where alpha and beta are scalars, x and y are n element vectors and
#     A should contain an upper or lower triangular n by n symmetric banded matrix.
# 
#     - Supported precisions in rocBLAS : s,d
#     - Supported precisions in cuBLAS  : s,d
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#               if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#               if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
#     @param[in]
#     n         [int]
#     @param[in]
#     k         [int]
#               specifies the number of sub- and super-diagonals
#     @param[in]
#     alpha
#               specifies the scalar alpha
#     @param[in]
#     AP         pointer storing matrix A on the GPU
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of matrix A
#     @param[in]
#     x         pointer storing vector x on the GPU
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x
#     @param[in]
#     beta      specifies the scalar beta
#     @param[out]
#     y         pointer storing vector y on the GPU
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y
#
cdef hipblasStatus_t hipblasSsbmv(void * handle,hipblasFillMode_t uplo,int n,int k,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDsbmv(void * handle,hipblasFillMode_t uplo,int n,int k,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     spmv performs the matrix-vector operation:
# 
#         y := alpha*A*x + beta*y,
# 
#     where alpha and beta are scalars, x and y are n element vectors and
#     A should contain an upper or lower triangular n by n packed symmetric matrix.
# 
#     - Supported precisions in rocBLAS : s,d
#     - Supported precisions in cuBLAS  : s,d
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#               if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#               if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
#     @param[in]
#     n         [int]
#     @param[in]
#     alpha
#               specifies the scalar alpha
#     @param[in]
#     AP         pointer storing matrix A on the GPU
#     @param[in]
#     x         pointer storing vector x on the GPU
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x
#     @param[in]
#     beta      specifies the scalar beta
#     @param[out]
#     y         pointer storing vector y on the GPU
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y
#
cdef hipblasStatus_t hipblasSspmv(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * AP,const float * x,int incx,const float * beta,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDspmv(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,const double * x,int incx,const double * beta,double * y,int incy) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     spr performs the matrix-vector operations
# 
#         A := A + alpha*x*x**T
# 
#     where alpha is a scalar, x is a vector, and A is an
#     n by n symmetric matrix, supplied in packed form.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#               HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
#               HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
#     @param[in]
#     n         [int]
#               the number of rows and columns of matrix A, must be at least 0.
#     @param[in]
#     alpha
#               device pointer or host pointer to scalar alpha.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[inout]
#     AP        device pointer storing the packed version of the specified triangular portion of
#               the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
#               if uplo == HIPBLAS_FILL_MODE_UPPER:
#                 The upper triangular portion of the symmetric matrix A is supplied.
#                 The matrix is compacted so that AP contains the triangular portion column-by-column
#                 so that:
#                 AP(0) = A(0,0)
#                 AP(1) = A(0,1)
#                 AP(2) = A(1,1), etc.
#                     Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                         1 2 4 7
#                         2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                         4 5 6 9
#                         7 8 9 0
#             if uplo == HIPBLAS_FILL_MODE_LOWER:
#                 The lower triangular portion of the symmetric matrix A is supplied.
#                 The matrix is compacted so that AP contains the triangular portion column-by-column
#                 so that:
#                 AP(0) = A(0,0)
#                 AP(1) = A(1,0)
#                 AP(2) = A(2,1), etc.
#                     Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                         1 2 3 4
#                         2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                         3 6 8 9
#                         4 7 9 0
cdef hipblasStatus_t hipblasSspr(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,float * AP) nogil



cdef hipblasStatus_t hipblasDspr(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,double * AP) nogil



cdef hipblasStatus_t hipblasCspr(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * AP) nogil



cdef hipblasStatus_t hipblasZspr(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     spr2 performs the matrix-vector operation
# 
#         A := A + alpha*x*y**T + alpha*y*x**T
# 
#     where alpha is a scalar, x and y are vectors, and A is an
#     n by n symmetric matrix, supplied in packed form.
# 
#     - Supported precisions in rocBLAS : s,d
#     - Supported precisions in cuBLAS  : s,d
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#               HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
#               HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
#     @param[in]
#     n         [int]
#               the number of rows and columns of matrix A, must be at least 0.
#     @param[in]
#     alpha
#               device pointer or host pointer to scalar alpha.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[in]
#     y         device pointer storing vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#     @param[inout]
#     AP        device pointer storing the packed version of the specified triangular portion of
#               the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
#               if uplo == HIPBLAS_FILL_MODE_UPPER:
#                 The upper triangular portion of the symmetric matrix A is supplied.
#                 The matrix is compacted so that AP contains the triangular portion column-by-column
#                 so that:
#                 AP(0) = A(0,0)
#                 AP(1) = A(0,1)
#                 AP(2) = A(1,1), etc.
#                     Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                         1 2 4 7
#                         2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                         4 5 6 9
#                         7 8 9 0
#             if uplo == HIPBLAS_FILL_MODE_LOWER:
#                 The lower triangular portion of the symmetric matrix A is supplied.
#                 The matrix is compacted so that AP contains the triangular portion column-by-column
#                 so that:
#                 AP(0) = A(0,0)
#                 AP(1) = A(1,0)
#                 AP(n) = A(2,1), etc.
#                     Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                         1 2 3 4
#                         2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                         3 6 8 9
#                         4 7 9 0
cdef hipblasStatus_t hipblasSspr2(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,const float * y,int incy,float * AP) nogil



cdef hipblasStatus_t hipblasDspr2(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     symv performs the matrix-vector operation:
# 
#         y := alpha*A*x + beta*y,
# 
#     where alpha and beta are scalars, x and y are n element vectors and
#     A should contain an upper or lower triangular n by n symmetric matrix.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#               if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#               if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
#     @param[in]
#     n         [int]
#     @param[in]
#     alpha
#               specifies the scalar alpha
#     @param[in]
#     AP         pointer storing matrix A on the GPU
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A
#     @param[in]
#     x         pointer storing vector x on the GPU
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x
#     @param[in]
#     beta      specifies the scalar beta
#     @param[out]
#     y         pointer storing vector y on the GPU
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y
#
cdef hipblasStatus_t hipblasSsymv(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDsymv(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil



cdef hipblasStatus_t hipblasCsymv(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZsymv(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     syr performs the matrix-vector operations
# 
#         A := A + alpha*x*x**T
# 
#     where alpha is a scalar, x is a vector, and A is an
#     n by n symmetric matrix.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#               if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#               if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# 
#     @param[in]
#     n         [int]
#               the number of rows and columns of matrix A.
#     @param[in]
#     alpha
#               device pointer or host pointer to scalar alpha.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[inout]
#     AP         device pointer storing matrix A.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A.
#
cdef hipblasStatus_t hipblasSsyr(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,float * AP,int lda) nogil



cdef hipblasStatus_t hipblasDsyr(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,double * AP,int lda) nogil



cdef hipblasStatus_t hipblasCsyr(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasZsyr(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP,int lda) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     syr2 performs the matrix-vector operations
# 
#         A := A + alpha*x*y**T + alpha*y*x**T
# 
#     where alpha is a scalar, x and y are vectors, and A is an
#     n by n symmetric matrix.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : No support
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#               if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#               if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# 
#     @param[in]
#     n         [int]
#               the number of rows and columns of matrix A.
#     @param[in]
#     alpha
#               device pointer or host pointer to scalar alpha.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[in]
#     y         device pointer storing vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#     @param[inout]
#     AP         device pointer storing matrix A.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A.
#
cdef hipblasStatus_t hipblasSsyr2(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,const float * y,int incy,float * AP,int lda) nogil



cdef hipblasStatus_t hipblasDsyr2(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP,int lda) nogil



cdef hipblasStatus_t hipblasCsyr2(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasZsyr2(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     tbmv performs one of the matrix-vector operations
# 
#         x := A*x      or
#         x := A**T*x   or
#         x := A**H*x,
# 
#     x is a vectors and A is a banded m by m matrix (see description below).
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               HIPBLAS_FILL_MODE_UPPER: A is an upper banded triangular matrix.
#               HIPBLAS_FILL_MODE_LOWER: A is a  lower banded triangular matrix.
#     @param[in]
#     transA     [hipblasOperation_t]
#               indicates whether matrix A is tranposed (conjugated) or not.
#     @param[in]
#     diag      [hipblasDiagType_t]
#               HIPBLAS_DIAG_UNIT: The main diagonal of A is assumed to consist of only
#                                      1's and is not referenced.
#               HIPBLAS_DIAG_NON_UNIT: No assumptions are made of A's main diagonal.
#     @param[in]
#     m         [int]
#               the number of rows and columns of the matrix represented by A.
#     @param[in]
#     k         [int]
#               if uplo == HIPBLAS_FILL_MODE_UPPER, k specifies the number of super-diagonals
#               of the matrix A.
#               if uplo == HIPBLAS_FILL_MODE_LOWER, k specifies the number of sub-diagonals
#               of the matrix A.
#               k must satisfy k > 0 && k < lda.
#     @param[in]
#     AP         device pointer storing banded triangular matrix A.
#               if uplo == HIPBLAS_FILL_MODE_UPPER:
#                 The matrix represented is an upper banded triangular matrix
#                 with the main diagonal and k super-diagonals, everything
#                 else can be assumed to be 0.
#                 The matrix is compacted so that the main diagonal resides on the k'th
#                 row, the first super diagonal resides on the RHS of the k-1'th row, etc,
#                 with the k'th diagonal on the RHS of the 0'th row.
#                    Ex: (HIPBLAS_FILL_MODE_UPPER; m = 5; k = 2)
#                       1 6 9 0 0              0 0 9 8 7
#                       0 2 7 8 0              0 6 7 8 9
#                       0 0 3 8 7     ---->    1 2 3 4 5
#                       0 0 0 4 9              0 0 0 0 0
#                       0 0 0 0 5              0 0 0 0 0
#               if uplo == HIPBLAS_FILL_MODE_LOWER:
#                 The matrix represnted is a lower banded triangular matrix
#                 with the main diagonal and k sub-diagonals, everything else can be
#                 assumed to be 0.
#                 The matrix is compacted so that the main diagonal resides on the 0'th row,
#                 working up to the k'th diagonal residing on the LHS of the k'th row.
#                    Ex: (HIPBLAS_FILL_MODE_LOWER; m = 5; k = 2)
#                       1 0 0 0 0              1 2 3 4 5
#                       6 2 0 0 0              6 7 8 9 0
#                       9 7 3 0 0     ---->    9 8 7 0 0
#                       0 8 8 4 0              0 0 0 0 0
#                       0 0 7 9 5              0 0 0 0 0
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A. lda must satisfy lda > k.
#     @param[inout]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#
cdef hipblasStatus_t hipblasStbmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const float * AP,int lda,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDtbmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const double * AP,int lda,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCtbmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZtbmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     tbsv solves
# 
#          A*x = b or A**T*x = b or A**H*x = b,
# 
#     where x and b are vectors and A is a banded triangular matrix.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#             HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
#     @param[in]
#     transA     [hipblasOperation_t]
#                HIPBLAS_OP_N: Solves A*x = b
#                HIPBLAS_OP_T: Solves A**T*x = b
#                HIPBLAS_OP_C: Solves A**H*x = b
# 
#     @param[in]
#     diag    [hipblasDiagType_t]
#             HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular (i.e. the diagonal elements
#                                        of A are not used in computations).
#             HIPBLAS_DIAG_NON_UNIT: A is not assumed to be unit triangular.
# 
#     @param[in]
#     n         [int]
#               n specifies the number of rows of b. n >= 0.
#     @param[in]
#     k         [int]
#               if(uplo == HIPBLAS_FILL_MODE_UPPER)
#                 k specifies the number of super-diagonals of A.
#               if(uplo == HIPBLAS_FILL_MODE_LOWER)
#                 k specifies the number of sub-diagonals of A.
#               k >= 0.
# 
#     @param[in]
#     AP         device pointer storing the matrix A in banded format.
# 
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A.
#               lda >= (k + 1).
# 
#     @param[inout]
#     x         device pointer storing input vector b. Overwritten by the output vector x.
# 
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#
cdef hipblasStatus_t hipblasStbsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const float * AP,int lda,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDtbsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const double * AP,int lda,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCtbsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZtbsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     tpmv performs one of the matrix-vector operations
# 
#          x = A*x or x = A**T*x,
# 
#     where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix, supplied in the pack form.
# 
#     The vector x is overwritten.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#             HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
#     @param[in]
#     transA     [hipblasOperation_t]
# 
#     @param[in]
#     diag    [hipblasDiagType_t]
#             HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#             HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
#     @param[in]
#     m       [int]
#             m specifies the number of rows of A. m >= 0.
# 
#     @param[in]
#     AP       device pointer storing matrix A,
#             of dimension at leat ( m * ( m + 1 ) / 2 ).
#           Before entry with uplo = HIPBLAS_FILL_MODE_UPPER, the array A
#           must contain the upper triangular matrix packed sequentially,
#           column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
#           a_{0,1} and a_{1, 1} respectively, and so on.
#           Before entry with uplo = HIPBLAS_FILL_MODE_LOWER, the array A
#           must contain the lower triangular matrix packed sequentially,
#           column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
#           a_{1,0} and a_{2,0} respectively, and so on.
#           Note that when DIAG = HIPBLAS_DIAG_UNIT, the diagonal elements of A are
#           not referenced, but are assumed to be unity.
# 
#     @param[in]
#     x       device pointer storing vector x.
# 
#     @param[in]
#     incx    [int]
#             specifies the increment for the elements of x. incx must not be zero.
#
cdef hipblasStatus_t hipblasStpmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDtpmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCtpmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZtpmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     tpsv solves
# 
#          A*x = b or A**T*x = b, or A**H*x = b,
# 
#     where x and b are vectors and A is a triangular matrix stored in the packed format.
# 
#     The input vector b is overwritten by the output vector x.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#             HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
#     @param[in]
#     transA  [hipblasOperation_t]
#             HIPBLAS_OP_N: Solves A*x = b
#             HIPBLAS_OP_T: Solves A**T*x = b
#             HIPBLAS_OP_C: Solves A**H*x = b
# 
#     @param[in]
#     diag    [hipblasDiagType_t]
#             HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular (i.e. the diagonal elements
#                                        of A are not used in computations).
#             HIPBLAS_DIAG_NON_UNIT: A is not assumed to be unit triangular.
# 
#     @param[in]
#     m         [int]
#               m specifies the number of rows of b. m >= 0.
# 
#     @param[in]
#     AP        device pointer storing the packed version of matrix A,
#               of dimension >= (n * (n + 1) / 2)
# 
#     @param[inout]
#     x         device pointer storing vector b on input, overwritten by x on output.
# 
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#
cdef hipblasStatus_t hipblasStpsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDtpsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCtpsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZtpsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     trmv performs one of the matrix-vector operations
# 
#          x = A*x or x = A**T*x,
# 
#     where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix.
# 
#     The vector x is overwritten.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#             HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
#     @param[in]
#     transA     [hipblasOperation_t]
# 
#     @param[in]
#     diag    [hipblasDiagType_t]
#             HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#             HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
#     @param[in]
#     m         [int]
#               m specifies the number of rows of A. m >= 0.
# 
#     @param[in]
#     AP        device pointer storing matrix A,
#               of dimension ( lda, m )
# 
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A.
#               lda = max( 1, m ).
# 
#     @param[in]
#     x         device pointer storing vector x.
# 
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#
cdef hipblasStatus_t hipblasStrmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,int lda,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDtrmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCtrmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZtrmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil


# ! @{
#     \brief BLAS Level 2 API
# 
#     \details
#     trsv solves
# 
#          A*x = b or A**T*x = b,
# 
#     where x and b are vectors and A is a triangular matrix.
# 
#     The vector x is overwritten on b.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#             HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
#     @param[in]
#     transA     [hipblasOperation_t]
# 
#     @param[in]
#     diag    [hipblasDiagType_t]
#             HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#             HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
#     @param[in]
#     m         [int]
#               m specifies the number of rows of b. m >= 0.
# 
#     @param[in]
#     AP        device pointer storing matrix A,
#               of dimension ( lda, m )
# 
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A.
#               lda = max( 1, m ).
# 
#     @param[in]
#     x         device pointer storing vector x.
# 
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#
cdef hipblasStatus_t hipblasStrsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,int lda,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDtrsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCtrsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZtrsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
#     gemm performs one of the matrix-matrix operations
# 
#         C = alpha*op( A )*op( B ) + beta*C,
# 
#     where op( X ) is one of
# 
#         op( X ) = X      or
#         op( X ) = X**T   or
#         op( X ) = X**H,
# 
#     alpha and beta are scalars, and A, B and C are matrices, with
#     op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
# 
#     - Supported precisions in rocBLAS : h,s,d,c,z
#     - Supported precisions in cuBLAS  : h,s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
# 
#               .
#     @param[in]
#     transA    [hipblasOperation_t]
#               specifies the form of op( A )
#     @param[in]
#     transB    [hipblasOperation_t]
#               specifies the form of op( B )
#     @param[in]
#     m         [int]
#               number or rows of matrices op( A ) and C
#     @param[in]
#     n         [int]
#               number of columns of matrices op( B ) and C
#     @param[in]
#     k         [int]
#               number of columns of matrix op( A ) and number of rows of matrix op( B )
#     @param[in]
#     alpha     device pointer or host pointer specifying the scalar alpha.
#     @param[in]
#     AP         device pointer storing matrix A.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A.
#     @param[in]
#     BP         device pointer storing matrix B.
#     @param[in]
#     ldb       [int]
#               specifies the leading dimension of B.
#     @param[in]
#     beta      device pointer or host pointer specifying the scalar beta.
#     @param[in, out]
#     CP         device pointer storing matrix C on the GPU.
#     @param[in]
#     ldc       [int]
#               specifies the leading dimension of C.
#
cdef hipblasStatus_t hipblasHgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const unsigned short * alpha,const unsigned short * AP,int lda,const unsigned short * BP,int ldb,const unsigned short * beta,unsigned short * CP,int ldc) nogil



cdef hipblasStatus_t hipblasSgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
# 
#     herk performs one of the matrix-matrix operations for a Hermitian rank-k update
# 
#     C := alpha*op( A )*op( A )^H + beta*C
# 
#     where  alpha and beta are scalars, op(A) is an n by k matrix, and
#     C is a n x n Hermitian matrix stored as either upper or lower.
# 
#         op( A ) = A,  and A is n by k if transA == HIPBLAS_OP_N
#         op( A ) = A^H and A is k by n if transA == HIPBLAS_OP_C
# 
#     - Supported precisions in rocBLAS : c,z
#     - Supported precisions in cuBLAS  : c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#             HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
#     @param[in]
#     transA  [hipblasOperation_t]
#             HIPBLAS_OP_C:  op(A) = A^H
#             HIPBLAS_ON_N:  op(A) = A
# 
#     @param[in]
#     n       [int]
#             n specifies the number of rows and columns of C. n >= 0.
# 
#     @param[in]
#     k       [int]
#             k specifies the number of columns of op(A). k >= 0.
# 
#     @param[in]
#     alpha
#             alpha specifies the scalar alpha. When alpha is
#             zero then A is not referenced and A need not be set before
#             entry.
# 
#     @param[in]
#     AP       pointer storing matrix A on the GPU.
#             Martrix dimension is ( lda, k ) when if transA = HIPBLAS_OP_N, otherwise (lda, n)
#             only the upper/lower triangular part is accessed.
# 
#     @param[in]
#     lda     [int]
#             lda specifies the first dimension of A.
#             if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#             otherwise lda >= max( 1, k ).
# 
#     @param[in]
#     beta
#             beta specifies the scalar beta. When beta is
#             zero then C need not be set before entry.
# 
#     @param[in]
#     CP       pointer storing matrix C on the GPU.
#             The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
#     @param[in]
#     ldc    [int]
#            ldc specifies the first dimension of C. ldc >= max( 1, n ).
#
cdef hipblasStatus_t hipblasCherk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,hipblasComplex * AP,int lda,const float * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZherk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,hipblasDoubleComplex * AP,int lda,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
# 
#     herkx performs one of the matrix-matrix operations for a Hermitian rank-k update
# 
#     C := alpha*op( A )*op( B )^H + beta*C
# 
#     where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
#     C is a n x n Hermitian matrix stored as either upper or lower.
#     This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermitian.
# 
# 
#         op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
#         op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == HIPBLAS_OP_C
# 
#     - Supported precisions in rocBLAS : c,z
#     - Supported precisions in cuBLAS  : c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#             HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
#     @param[in]
#     transA  [hipblasOperation_t]
#             HIPBLAS_OP_C:  op( A ) = A^H, op( B ) = B^H
#             HIPBLAS_OP_N:  op( A ) = A, op( B ) = B
# 
#     @param[in]
#     n       [int]
#             n specifies the number of rows and columns of C. n >= 0.
# 
#     @param[in]
#     k       [int]
#             k specifies the number of columns of op(A). k >= 0.
# 
#     @param[in]
#     alpha
#             alpha specifies the scalar alpha. When alpha is
#             zero then A is not referenced and A need not be set before
#             entry.
# 
#     @param[in]
#     AP      pointer storing matrix A on the GPU.
#             Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
#             only the upper/lower triangular part is accessed.
# 
#     @param[in]
#     lda     [int]
#             lda specifies the first dimension of A.
#             if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#             otherwise lda >= max( 1, k ).
#     @param[in]
#     BP       pointer storing matrix B on the GPU.
#             Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
#             only the upper/lower triangular part is accessed.
# 
#     @param[in]
#     ldb     [int]
#             ldb specifies the first dimension of B.
#             if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#             otherwise ldb >= max( 1, k ).
#     @param[in]
#     beta
#             beta specifies the scalar beta. When beta is
#             zero then C need not be set before entry.
# 
#     @param[in]
#     CP       pointer storing matrix C on the GPU.
#             The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
#     @param[in]
#     ldc    [int]
#            ldc specifies the first dimension of C. ldc >= max( 1, n ).
#
cdef hipblasStatus_t hipblasCherkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,const float * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZherkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
# 
#     her2k performs one of the matrix-matrix operations for a Hermitian rank-2k update
# 
#     C := alpha*op( A )*op( B )^H + conj(alpha)*op( B )*op( A )^H + beta*C
# 
#     where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
#     C is a n x n Hermitian matrix stored as either upper or lower.
# 
#         op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
#         op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == HIPBLAS_OP_C
# 
#     - Supported precisions in rocBLAS : c,z
#     - Supported precisions in cuBLAS  : c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#             HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
#     @param[in]
#     transA  [hipblasOperation_t]
#             HIPBLAS_OP_C:  op( A ) = A^H, op( B ) = B^H
#             HIPBLAS_OP_N:  op( A ) = A, op( B ) = B
# 
#     @param[in]
#     n       [int]
#             n specifies the number of rows and columns of C. n >= 0.
# 
#     @param[in]
#     k       [int]
#             k specifies the number of columns of op(A). k >= 0.
# 
#     @param[in]
#     alpha
#             alpha specifies the scalar alpha. When alpha is
#             zero then A is not referenced and A need not be set before
#             entry.
# 
#     @param[in]
#     AP       pointer storing matrix A on the GPU.
#             Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
#             only the upper/lower triangular part is accessed.
# 
#     @param[in]
#     lda     [int]
#             lda specifies the first dimension of A.
#             if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#             otherwise lda >= max( 1, k ).
#     @param[in]
#     BP       pointer storing matrix B on the GPU.
#             Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
#             only the upper/lower triangular part is accessed.
# 
#     @param[in]
#     ldb     [int]
#             ldb specifies the first dimension of B.
#             if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#             otherwise ldb >= max( 1, k ).
#     @param[in]
#     beta
#             beta specifies the scalar beta. When beta is
#             zero then C need not be set before entry.
# 
#     @param[in]
#     CP       pointer storing matrix C on the GPU.
#             The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
#     @param[in]
#     ldc    [int]
#            ldc specifies the first dimension of C. ldc >= max( 1, n ).
#
cdef hipblasStatus_t hipblasCher2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,const float * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZher2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
# 
#     symm performs one of the matrix-matrix operations:
# 
#     C := alpha*A*B + beta*C if side == HIPBLAS_SIDE_LEFT,
#     C := alpha*B*A + beta*C if side == HIPBLAS_SIDE_RIGHT,
# 
#     where alpha and beta are scalars, B and C are m by n matrices, and
#     A is a symmetric matrix stored as either upper or lower.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     side  [hipblasSideMode_t]
#             HIPBLAS_SIDE_LEFT:      C := alpha*A*B + beta*C
#             HIPBLAS_SIDE_RIGHT:     C := alpha*B*A + beta*C
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix
#             HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix
# 
#     @param[in]
#     m       [int]
#             m specifies the number of rows of B and C. m >= 0.
# 
#     @param[in]
#     n       [int]
#             n specifies the number of columns of B and C. n >= 0.
# 
#     @param[in]
#     alpha
#             alpha specifies the scalar alpha. When alpha is
#             zero then A and B are not referenced.
# 
#     @param[in]
#     AP       pointer storing matrix A on the GPU.
#             A is m by m if side == HIPBLAS_SIDE_LEFT
#             A is n by n if side == HIPBLAS_SIDE_RIGHT
#             only the upper/lower triangular part is accessed.
# 
#     @param[in]
#     lda     [int]
#             lda specifies the first dimension of A.
#             if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#             otherwise lda >= max( 1, n ).
# 
#     @param[in]
#     BP       pointer storing matrix B on the GPU.
#             Matrix dimension is m by n
# 
#     @param[in]
#     ldb     [int]
#             ldb specifies the first dimension of B. ldb >= max( 1, m )
# 
#     @param[in]
#     beta
#             beta specifies the scalar beta. When beta is
#             zero then C need not be set before entry.
# 
#     @param[in]
#     CP       pointer storing matrix C on the GPU.
#             Matrix dimension is m by n
# 
#     @param[in]
#     ldc    [int]
#            ldc specifies the first dimension of C. ldc >= max( 1, m )
#
cdef hipblasStatus_t hipblasSsymm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDsymm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCsymm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZsymm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
# 
#     syrk performs one of the matrix-matrix operations for a symmetric rank-k update
# 
#     C := alpha*op( A )*op( A )^T + beta*C
# 
#     where  alpha and beta are scalars, op(A) is an n by k matrix, and
#     C is a symmetric n x n matrix stored as either upper or lower.
# 
#         op( A ) = A, and A is n by k if transA == HIPBLAS_OP_N
#         op( A ) = A^T and A is k by n if transA == HIPBLAS_OP_T
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#             HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
#     @param[in]
#     transA  [hipblasOperation_t]
#             HIPBLAS_OP_T: op(A) = A^T
#             HIPBLAS_OP_N: op(A) = A
#             HIPBLAS_OP_C: op(A) = A^T
# 
#             HIPBLAS_OP_C is not supported for complex types, see cherk
#             and zherk.
# 
#     @param[in]
#     n       [int]
#             n specifies the number of rows and columns of C. n >= 0.
# 
#     @param[in]
#     k       [int]
#             k specifies the number of columns of op(A). k >= 0.
# 
#     @param[in]
#     alpha
#             alpha specifies the scalar alpha. When alpha is
#             zero then A is not referenced and A need not be set before
#             entry.
# 
#     @param[in]
#     AP       pointer storing matrix A on the GPU.
#             Martrix dimension is ( lda, k ) when if transA = HIPBLAS_OP_N, otherwise (lda, n)
#             only the upper/lower triangular part is accessed.
# 
#     @param[in]
#     lda     [int]
#             lda specifies the first dimension of A.
#             if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#             otherwise lda >= max( 1, k ).
# 
#     @param[in]
#     beta
#             beta specifies the scalar beta. When beta is
#             zero then C need not be set before entry.
# 
#     @param[in]
#     CP       pointer storing matrix C on the GPU.
# 
#     @param[in]
#     ldc    [int]
#            ldc specifies the first dimension of C. ldc >= max( 1, n ).
#
cdef hipblasStatus_t hipblasSsyrk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,const float * beta,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDsyrk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * beta,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCsyrk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZsyrk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
# 
#     syr2k performs one of the matrix-matrix operations for a symmetric rank-2k update
# 
#     C := alpha*(op( A )*op( B )^T + op( B )*op( A )^T) + beta*C
# 
#     where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
#     C is a symmetric n x n matrix stored as either upper or lower.
# 
#         op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
#         op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == HIPBLAS_OP_T
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#             HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
#     @param[in]
#     transA  [hipblasOperation_t]
#             HIPBLAS_OP_T:      op( A ) = A^T, op( B ) = B^T
#             HIPBLAS_OP_N:           op( A ) = A, op( B ) = B
# 
#     @param[in]
#     n       [int]
#             n specifies the number of rows and columns of C. n >= 0.
# 
#     @param[in]
#     k       [int]
#             k specifies the number of columns of op(A) and op(B). k >= 0.
# 
#     @param[in]
#     alpha
#             alpha specifies the scalar alpha. When alpha is
#             zero then A is not referenced and A need not be set before
#             entry.
# 
#     @param[in]
#     AP       pointer storing matrix A on the GPU.
#             Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
#             only the upper/lower triangular part is accessed.
# 
#     @param[in]
#     lda     [int]
#             lda specifies the first dimension of A.
#             if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#             otherwise lda >= max( 1, k ).
#     @param[in]
#     BP       pointer storing matrix B on the GPU.
#             Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
#             only the upper/lower triangular part is accessed.
# 
#     @param[in]
#     ldb     [int]
#             ldb specifies the first dimension of B.
#             if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#             otherwise ldb >= max( 1, k ).
#     @param[in]
#     beta
#             beta specifies the scalar beta. When beta is
#             zero then C need not be set before entry.
# 
#     @param[in]
#     CP       pointer storing matrix C on the GPU.
# 
#     @param[in]
#     ldc    [int]
#            ldc specifies the first dimension of C. ldc >= max( 1, n ).
#
cdef hipblasStatus_t hipblasSsyr2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDsyr2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCsyr2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZsyr2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
# 
#     syrkx performs one of the matrix-matrix operations for a symmetric rank-k update
# 
#     C := alpha*op( A )*op( B )^T + beta*C
# 
#     where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
#     C is a symmetric n x n matrix stored as either upper or lower.
#     This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be symmetric.
# 
#         op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
#         op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == HIPBLAS_OP_T
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#             HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
#     @param[in]
#     transA  [hipblasOperation_t]
#             HIPBLAS_OP_T:      op( A ) = A^T, op( B ) = B^T
#             HIPBLAS_OP_N:           op( A ) = A, op( B ) = B
# 
#     @param[in]
#     n       [int]
#             n specifies the number of rows and columns of C. n >= 0.
# 
#     @param[in]
#     k       [int]
#             k specifies the number of columns of op(A) and op(B). k >= 0.
# 
#     @param[in]
#     alpha
#             alpha specifies the scalar alpha. When alpha is
#             zero then A is not referenced and A need not be set before
#             entry.
# 
#     @param[in]
#     AP       pointer storing matrix A on the GPU.
#             Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
#             only the upper/lower triangular part is accessed.
# 
#     @param[in]
#     lda     [int]
#             lda specifies the first dimension of A.
#             if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#             otherwise lda >= max( 1, k ).
# 
#     @param[in]
#     BP       pointer storing matrix B on the GPU.
#             Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
#             only the upper/lower triangular part is accessed.
# 
#     @param[in]
#     ldb     [int]
#             ldb specifies the first dimension of B.
#             if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#             otherwise ldb >= max( 1, k ).
# 
#     @param[in]
#     beta
#             beta specifies the scalar beta. When beta is
#             zero then C need not be set before entry.
# 
#     @param[in]
#     CP       pointer storing matrix C on the GPU.
# 
#     @param[in]
#     ldc    [int]
#            ldc specifies the first dimension of C. ldc >= max( 1, n ).
#
cdef hipblasStatus_t hipblasSsyrkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDsyrkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCsyrkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZsyrkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
#     geam performs one of the matrix-matrix operations
# 
#         C = alpha*op( A ) + beta*op( B ),
# 
#     where op( X ) is one of
# 
#         op( X ) = X      or
#         op( X ) = X**T   or
#         op( X ) = X**H,
# 
#     alpha and beta are scalars, and A, B and C are matrices, with
#     op( A ) an m by n matrix, op( B ) an m by n matrix, and C an m by n matrix.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     transA    [hipblasOperation_t]
#               specifies the form of op( A )
#     @param[in]
#     transB    [hipblasOperation_t]
#               specifies the form of op( B )
#     @param[in]
#     m         [int]
#               matrix dimension m.
#     @param[in]
#     n         [int]
#               matrix dimension n.
#     @param[in]
#     alpha     device pointer or host pointer specifying the scalar alpha.
#     @param[in]
#     AP         device pointer storing matrix A.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A.
#     @param[in]
#     beta      device pointer or host pointer specifying the scalar beta.
#     @param[in]
#     BP         device pointer storing matrix B.
#     @param[in]
#     ldb       [int]
#               specifies the leading dimension of B.
#     @param[in, out]
#     CP         device pointer storing matrix C.
#     @param[in]
#     ldc       [int]
#               specifies the leading dimension of C.
#
cdef hipblasStatus_t hipblasSgeam(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const float * alpha,const float * AP,int lda,const float * beta,const float * BP,int ldb,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDgeam(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const double * alpha,const double * AP,int lda,const double * beta,const double * BP,int ldb,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCgeam(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * beta,hipblasComplex * BP,int ldb,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZgeam(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * CP,int ldc) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
# 
#     hemm performs one of the matrix-matrix operations:
# 
#     C := alpha*A*B + beta*C if side == HIPBLAS_SIDE_LEFT,
#     C := alpha*B*A + beta*C if side == HIPBLAS_SIDE_RIGHT,
# 
#     where alpha and beta are scalars, B and C are m by n matrices, and
#     A is a Hermitian matrix stored as either upper or lower.
# 
#     - Supported precisions in rocBLAS : c,z
#     - Supported precisions in cuBLAS  : c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     side  [hipblasSideMode_t]
#             HIPBLAS_SIDE_LEFT:      C := alpha*A*B + beta*C
#             HIPBLAS_SIDE_RIGHT:     C := alpha*B*A + beta*C
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix
#             HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix
# 
#     @param[in]
#     n       [int]
#             n specifies the number of rows of B and C. n >= 0.
# 
#     @param[in]
#     k       [int]
#             n specifies the number of columns of B and C. k >= 0.
# 
#     @param[in]
#     alpha
#             alpha specifies the scalar alpha. When alpha is
#             zero then A and B are not referenced.
# 
#     @param[in]
#     AP       pointer storing matrix A on the GPU.
#             A is m by m if side == HIPBLAS_SIDE_LEFT
#             A is n by n if side == HIPBLAS_SIDE_RIGHT
#             Only the upper/lower triangular part is accessed.
#             The imaginary component of the diagonal elements is not used.
# 
#     @param[in]
#     lda     [int]
#             lda specifies the first dimension of A.
#             if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#             otherwise lda >= max( 1, n ).
# 
#     @param[in]
#     BP       pointer storing matrix B on the GPU.
#             Matrix dimension is m by n
# 
#     @param[in]
#     ldb     [int]
#             ldb specifies the first dimension of B. ldb >= max( 1, m )
# 
#     @param[in]
#     beta
#             beta specifies the scalar beta. When beta is
#             zero then C need not be set before entry.
# 
#     @param[in]
#     CP       pointer storing matrix C on the GPU.
#             Matrix dimension is m by n
# 
#     @param[in]
#     ldc    [int]
#            ldc specifies the first dimension of C. ldc >= max( 1, m )
#
cdef hipblasStatus_t hipblasChemm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZhemm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
# 
#     trmm performs one of the matrix-matrix operations
# 
#     B := alpha*op( A )*B,   or   B := alpha*B*op( A )
# 
#     where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
#     non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
# 
#         op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.
# 
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     side    [hipblasSideMode_t]
#             Specifies whether op(A) multiplies B from the left or right as follows:
#             HIPBLAS_SIDE_LEFT:       B := alpha*op( A )*B.
#             HIPBLAS_SIDE_RIGHT:      B := alpha*B*op( A ).
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             Specifies whether the matrix A is an upper or lower triangular matrix as follows:
#             HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#             HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
#     @param[in]
#     transA  [hipblasOperation_t]
#             Specifies the form of op(A) to be used in the matrix multiplication as follows:
#             HIPBLAS_OP_N: op(A) = A.
#             HIPBLAS_OP_T: op(A) = A^T.
#             HIPBLAS_OP_C:  op(A) = A^H.
# 
#     @param[in]
#     diag    [hipblasDiagType_t]
#             Specifies whether or not A is unit triangular as follows:
#             HIPBLAS_DIAG_UNIT:      A is assumed to be unit triangular.
#             HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
#     @param[in]
#     m       [int]
#             m specifies the number of rows of B. m >= 0.
# 
#     @param[in]
#     n       [int]
#             n specifies the number of columns of B. n >= 0.
# 
#     @param[in]
#     alpha
#             alpha specifies the scalar alpha. When alpha is
#             zero then A is not referenced and B need not be set before
#             entry.
# 
#     @param[in]
#     AP       Device pointer to matrix A on the GPU.
#             A has dimension ( lda, k ), where k is m
#             when  side == HIPBLAS_SIDE_LEFT  and
#             is  n  when  side == HIPBLAS_SIDE_RIGHT.
# 
#         When uplo == HIPBLAS_FILL_MODE_UPPER the  leading  k by k
#         upper triangular part of the array  A must contain the upper
#         triangular matrix  and the strictly lower triangular part of
#         A is not referenced.
# 
#         When uplo == HIPBLAS_FILL_MODE_LOWER the  leading  k by k
#         lower triangular part of the array  A must contain the lower
#         triangular matrix  and the strictly upper triangular part of
#         A is not referenced.
# 
#         Note that when  diag == HIPBLAS_DIAG_UNIT  the diagonal elements of
#         A  are not referenced either,  but are assumed to be  unity.
# 
#     @param[in]
#     lda     [int]
#             lda specifies the first dimension of A.
#             if side == HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#             if side == HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
#     @param[inout]
#     BP       Device pointer to the first matrix B_0 on the GPU.
#             On entry,  the leading  m by n part of the array  B must
#            contain the matrix  B,  and  on exit  is overwritten  by the
#            transformed matrix.
# 
#     @param[in]
#     ldb    [int]
#            ldb specifies the first dimension of B. ldb >= max( 1, m ).
#
cdef hipblasStatus_t hipblasStrmm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,const float * AP,int lda,float * BP,int ldb) nogil



cdef hipblasStatus_t hipblasDtrmm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,const double * AP,int lda,double * BP,int ldb) nogil



cdef hipblasStatus_t hipblasCtrmm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb) nogil



cdef hipblasStatus_t hipblasZtrmm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
# 
#     trsm solves
# 
#         op(A)*X = alpha*B or  X*op(A) = alpha*B,
# 
#     where alpha is a scalar, X and B are m by n matrices,
#     A is triangular matrix and op(A) is one of
# 
#         op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.
# 
#     The matrix X is overwritten on B.
# 
#     Note about memory allocation:
#     When trsm is launched with a k evenly divisible by the internal block size of 128,
#     and is no larger than 10 of these blocks, the API takes advantage of utilizing pre-allocated
#     memory found in the handle to increase overall performance. This memory can be managed by using
#     the environment variable WORKBUF_TRSM_B_CHNK. When this variable is not set the device memory
#     used for temporary storage will default to 1 MB and may result in chunking, which in turn may
#     reduce performance. Under these circumstances it is recommended that WORKBUF_TRSM_B_CHNK be set
#     to the desired chunk of right hand sides to be used at a time.
# 
#     (where k is m when HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT)
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
# 
#     @param[in]
#     side    [hipblasSideMode_t]
#             HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#             HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#             HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
#     @param[in]
#     transA  [hipblasOperation_t]
#             HIPBLAS_OP_N: op(A) = A.
#             HIPBLAS_OP_T: op(A) = A^T.
#             HIPBLAS_OP_C: op(A) = A^H.
# 
#     @param[in]
#     diag    [hipblasDiagType_t]
#             HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#             HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
#     @param[in]
#     m       [int]
#             m specifies the number of rows of B. m >= 0.
# 
#     @param[in]
#     n       [int]
#             n specifies the number of columns of B. n >= 0.
# 
#     @param[in]
#     alpha
#             device pointer or host pointer specifying the scalar alpha. When alpha is
#             &zero then A is not referenced and B need not be set before
#             entry.
# 
#     @param[in]
#     AP       device pointer storing matrix A.
#             of dimension ( lda, k ), where k is m
#             when  HIPBLAS_SIDE_LEFT  and
#             is  n  when  HIPBLAS_SIDE_RIGHT
#             only the upper/lower triangular part is accessed.
# 
#     @param[in]
#     lda     [int]
#             lda specifies the first dimension of A.
#             if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#             if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
#     @param[in,out]
#     BP       device pointer storing matrix B.
# 
#     @param[in]
#     ldb    [int]
#            ldb specifies the first dimension of B. ldb >= max( 1, m ).
#
cdef hipblasStatus_t hipblasStrsm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,float * AP,int lda,float * BP,int ldb) nogil



cdef hipblasStatus_t hipblasDtrsm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,double * AP,int lda,double * BP,int ldb) nogil



cdef hipblasStatus_t hipblasCtrsm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb) nogil



cdef hipblasStatus_t hipblasZtrsm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
#     trtri  compute the inverse of a matrix A, namely, invA
# 
#         and write the result into invA;
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : No support
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     uplo      [hipblasFillMode_t]
#               specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#               if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#               if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
#     @param[in]
#     diag      [hipblasDiagType_t]
#               = 'HIPBLAS_DIAG_NON_UNIT', A is non-unit triangular;
#               = 'HIPBLAS_DIAG_UNIT', A is unit triangular;
#     @param[in]
#     n         [int]
#               size of matrix A and invA
#     @param[in]
#     AP         device pointer storing matrix A.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A.
#     @param[out]
#     invA      device pointer storing matrix invA.
#     @param[in]
#     ldinvA    [int]
#               specifies the leading dimension of invA.
#
cdef hipblasStatus_t hipblasStrtri(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const float * AP,int lda,float * invA,int ldinvA) nogil



cdef hipblasStatus_t hipblasDtrtri(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const double * AP,int lda,double * invA,int ldinvA) nogil



cdef hipblasStatus_t hipblasCtrtri(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasComplex * AP,int lda,hipblasComplex * invA,int ldinvA) nogil



cdef hipblasStatus_t hipblasZtrtri(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * invA,int ldinvA) nogil


# ! @{
#     \brief BLAS Level 3 API
# 
#     \details
#     dgmm performs one of the matrix-matrix operations
# 
#         C = A * diag(x) if side == HIPBLAS_SIDE_RIGHT
#         C = diag(x) * A if side == HIPBLAS_SIDE_LEFT
# 
#     where C and A are m by n dimensional matrices. diag( x ) is a diagonal matrix
#     and x is vector of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension m
#     if side == HIPBLAS_SIDE_LEFT.
# 
#     - Supported precisions in rocBLAS : s,d,c,z
#     - Supported precisions in cuBLAS  : s,d,c,z
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     side      [hipblasSideMode_t]
#               specifies the side of diag(x)
#     @param[in]
#     m         [int]
#               matrix dimension m.
#     @param[in]
#     n         [int]
#               matrix dimension n.
#     @param[in]
#     AP         device pointer storing matrix A.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment between values of x
#     @param[in, out]
#     CP         device pointer storing matrix C.
#     @param[in]
#     ldc       [int]
#               specifies the leading dimension of C.
#
cdef hipblasStatus_t hipblasSdgmm(void * handle,hipblasSideMode_t side,int m,int n,const float * AP,int lda,const float * x,int incx,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDdgmm(void * handle,hipblasSideMode_t side,int m,int n,const double * AP,int lda,const double * x,int incx,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCdgmm(void * handle,hipblasSideMode_t side,int m,int n,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZdgmm(void * handle,hipblasSideMode_t side,int m,int n,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * CP,int ldc) nogil


# ! @{
#     \brief SOLVER API
# 
#     \details
#     getrf computes the LU factorization of a general n-by-n matrix A
#     using partial pivoting with row interchanges. The LU factorization can
#     be done without pivoting if ipiv is passed as a nullptr.
# 
#     In the case that ipiv is not null, the factorization has the form:
# 
#     \f[
#         A = PLU
#     \f]
# 
#     where P is a permutation matrix, L is lower triangular with unit
#     diagonal elements, and U is upper triangular.
# 
#     In the case that ipiv is null, the factorization is done without pivoting:
# 
#     \f[
#         A = LU
#     \f]
# 
#     - Supported precisions in rocSOLVER : s,d,c,z
#     - Supported precisions in cuBLAS    : s,d,c,z
# 
#     @param[in]
#     handle    hipblasHandle_t.
#     @param[in]
#     n         int. n >= 0.\n
#               The number of columns and rows of the matrix A.
#     @param[inout]
#     A         pointer to type. Array on the GPU of dimension lda*n.\n
#               On entry, the n-by-n matrix A to be factored.
#               On exit, the factors L and U from the factorization.
#               The unit diagonal elements of L are not stored.
#     @param[in]
#     lda       int. lda >= n.\n
#               Specifies the leading dimension of A.
#     @param[out]
#     ipiv      pointer to int. Array on the GPU of dimension n.\n
#               The vector of pivot indices. Elements of ipiv are 1-based indices.
#               For 1 <= i <= n, the row i of the
#               matrix was interchanged with row ipiv[i].
#               Matrix P of the factorization can be derived from ipiv.
#               The factorization here can be done without pivoting if ipiv is passed
#               in as a nullptr.
#     @param[out]
#     info      pointer to a int on the GPU.\n
#               If info = 0, successful exit.
#               If info = j > 0, U is singular. U[j,j] is the first zero pivot.
cdef hipblasStatus_t hipblasSgetrf(void * handle,const int n,float * A,const int lda,int * ipiv,int * info) nogil



cdef hipblasStatus_t hipblasDgetrf(void * handle,const int n,double * A,const int lda,int * ipiv,int * info) nogil



cdef hipblasStatus_t hipblasCgetrf(void * handle,const int n,hipblasComplex * A,const int lda,int * ipiv,int * info) nogil



cdef hipblasStatus_t hipblasZgetrf(void * handle,const int n,hipblasDoubleComplex * A,const int lda,int * ipiv,int * info) nogil


# ! @{
#     \brief SOLVER API
# 
#     \details
#     getrs solves a system of n linear equations on n variables in its factorized form.
# 
#     It solves one of the following systems, depending on the value of trans:
# 
#     \f[
#         \begin{array}{cl}
#         A X = B & \: \text{not transposed,}\\
#         A^T X = B & \: \text{transposed, or}\\
#         A^H X = B & \: \text{conjugate transposed.}
#         \end{array}
#     \f]
# 
#     Matrix A is defined by its triangular factors as returned by \ref hipblasSgetrf "getrf".
# 
#     - Supported precisions in rocSOLVER : s,d,c,z
#     - Supported precisions in cuBLAS    : s,d,c,z
# 
# 
#     @param[in]
#     handle      hipblasHandle_t.
#     @param[in]
#     trans       hipblasOperation_t.\n
#                 Specifies the form of the system of equations.
#     @param[in]
#     n           int. n >= 0.\n
#                 The order of the system, i.e. the number of columns and rows of A.
#     @param[in]
#     nrhs        int. nrhs >= 0.\n
#                 The number of right hand sides, i.e., the number of columns
#                 of the matrix B.
#     @param[in]
#     A           pointer to type. Array on the GPU of dimension lda*n.\n
#                 The factors L and U of the factorization A = P*L*U returned by \ref hipblasSgetrf "getrf".
#     @param[in]
#     lda         int. lda >= n.\n
#                 The leading dimension of A.
#     @param[in]
#     ipiv        pointer to int. Array on the GPU of dimension n.\n
#                 The pivot indices returned by \ref hipblasSgetrf "getrf".
#     @param[in,out]
#     B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
#                 On entry, the right hand side matrix B.
#                 On exit, the solution matrix X.
#     @param[in]
#     ldb         int. ldb >= n.\n
#                 The leading dimension of B.
#     @param[out]
#     info      pointer to a int on the host.\n
#               If info = 0, successful exit.
#               If info = j < 0, the j-th argument is invalid.
cdef hipblasStatus_t hipblasSgetrs(void * handle,hipblasOperation_t trans,const int n,const int nrhs,float * A,const int lda,const int * ipiv,float * B,const int ldb,int * info) nogil



cdef hipblasStatus_t hipblasDgetrs(void * handle,hipblasOperation_t trans,const int n,const int nrhs,double * A,const int lda,const int * ipiv,double * B,const int ldb,int * info) nogil



cdef hipblasStatus_t hipblasCgetrs(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasComplex * A,const int lda,const int * ipiv,hipblasComplex * B,const int ldb,int * info) nogil



cdef hipblasStatus_t hipblasZgetrs(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,const int * ipiv,hipblasDoubleComplex * B,const int ldb,int * info) nogil


# ! @{
#     \brief GELS solves an overdetermined (or underdetermined) linear system defined by an m-by-n
#     matrix A, and a corresponding matrix B, using the QR factorization computed by \ref hipblasSgeqrf "GEQRF" (or the LQ
#     factorization computed by "GELQF").
# 
#     \details
#     Depending on the value of trans, the problem solved by this function is either of the form
# 
#     \f[
#         \begin{array}{cl}
#         A X = B & \: \text{not transposed, or}\\
#         A' X = B & \: \text{transposed if real, or conjugate transposed if complex}
#         \end{array}
#     \f]
# 
#     If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
#     and a least-squares solution approximating X is found by minimizing
# 
#     \f[
#         || B - A  X || \quad \text{(or} \: || B - A' X ||\text{)}
#     \f]
# 
#     If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
#     and a unique solution for X is chosen such that \f$|| X ||\f$ is minimal.
# 
#     - Supported precisions in rocSOLVER : s,d,c,z
#     - Supported precisions in cuBLAS    : currently unsupported
# 
#     @param[in]
#     handle      hipblasHandle_t.
#     @param[in]
#     trans       hipblasOperation_t.\n
#                 Specifies the form of the system of equations.
#     @param[in]
#     m           int. m >= 0.\n
#                 The number of rows of matrix A.
#     @param[in]
#     n           int. n >= 0.\n
#                 The number of columns of matrix A.
#     @param[in]
#     nrhs        int. nrhs >= 0.\n
#                 The number of columns of matrices B and X;
#                 i.e., the columns on the right hand side.
#     @param[inout]
#     A           pointer to type. Array on the GPU of dimension lda*n.\n
#                 On entry, the matrix A.
#                 On exit, the QR (or LQ) factorization of A as returned by "GEQRF" (or "GELQF").
#     @param[in]
#     lda         int. lda >= m.\n
#                 Specifies the leading dimension of matrix A.
#     @param[inout]
#     B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
#                 On entry, the matrix B.
#                 On exit, when info = 0, B is overwritten by the solution vectors (and the residuals in
#                 the overdetermined cases) stored as columns.
#     @param[in]
#     ldb         int. ldb >= max(m,n).\n
#                 Specifies the leading dimension of matrix B.
#     @param[out]
#     info        pointer to an int on the host.\n
#                 If info = 0, successful exit.
#                 If info = j < 0, the j-th argument is invalid.
#     @param[out]
#     deviceInfo  pointer to int on the GPU.\n
#                 If info = 0, successful exit.
#                 If info = i > 0, the solution could not be computed because input matrix A is
#                 rank deficient; the i-th diagonal element of its triangular factor is zero.
cdef hipblasStatus_t hipblasSgels(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,float * A,const int lda,float * B,const int ldb,int * info,int * deviceInfo) nogil



cdef hipblasStatus_t hipblasDgels(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,double * A,const int lda,double * B,const int ldb,int * info,int * deviceInfo) nogil



cdef hipblasStatus_t hipblasCgels(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasComplex * A,const int lda,hipblasComplex * B,const int ldb,int * info,int * deviceInfo) nogil



cdef hipblasStatus_t hipblasZgels(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,hipblasDoubleComplex * B,const int ldb,int * info,int * deviceInfo) nogil


# ! @{
#     \brief SOLVER API
# 
#     \details
#     geqrf computes a QR factorization of a general m-by-n matrix A.
# 
#     The factorization has the form
# 
#     \f[
#         A = Q\left[\begin{array}{c}
#         R\\
#         0
#         \end{array}\right]
#     \f]
# 
#     where R is upper triangular (upper trapezoidal if m < n), and Q is
#     a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices
# 
#     \f[
#         Q = H_1H_2\cdots H_k, \quad \text{with} \: k = \text{min}(m,n)
#     \f]
# 
#     Each Householder matrix \f$H_i\f$ is given by
# 
#     \f[
#         H_i = I - \text{ipiv}[i] \cdot v_i v_i'
#     \f]
# 
#     where the first i-1 elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.
# 
#     - Supported precisions in rocSOLVER : s,d,c,z
#     - Supported precisions in cuBLAS    : s,d,c,z
# 
#     @param[in]
#     handle    hipblasHandle_t.
#     @param[in]
#     m         int. m >= 0.\n
#               The number of rows of the matrix A.
#     @param[in]
#     n         int. n >= 0.\n
#               The number of columns of the matrix A.
#     @param[inout]
#     A         pointer to type. Array on the GPU of dimension lda*n.\n
#               On entry, the m-by-n matrix to be factored.
#               On exit, the elements on and above the diagonal contain the
#               factor R; the elements below the diagonal are the last m - i elements
#               of Householder vector v_i.
#     @param[in]
#     lda       int. lda >= m.\n
#               Specifies the leading dimension of A.
#     @param[out]
#     ipiv      pointer to type. Array on the GPU of dimension min(m,n).\n
#               The Householder scalars.
#     @param[out]
#     info      pointer to a int on the host.\n
#               If info = 0, successful exit.
#               If info = j < 0, the j-th argument is invalid.
#
cdef hipblasStatus_t hipblasSgeqrf(void * handle,const int m,const int n,float * A,const int lda,float * ipiv,int * info) nogil



cdef hipblasStatus_t hipblasDgeqrf(void * handle,const int m,const int n,double * A,const int lda,double * ipiv,int * info) nogil



cdef hipblasStatus_t hipblasCgeqrf(void * handle,const int m,const int n,hipblasComplex * A,const int lda,hipblasComplex * ipiv,int * info) nogil



cdef hipblasStatus_t hipblasZgeqrf(void * handle,const int m,const int n,hipblasDoubleComplex * A,const int lda,hipblasDoubleComplex * ipiv,int * info) nogil


# ! \brief BLAS EX API
# 
#     \details
#     gemmEx performs one of the matrix-matrix operations
# 
#         C = alpha*op( A )*op( B ) + beta*C,
# 
#     where op( X ) is one of
# 
#         op( X ) = X      or
#         op( X ) = X**T   or
#         op( X ) = X**H,
# 
#     alpha and beta are scalars, and A, B, and C are matrices, with
#     op( A ) an m by k matrix, op( B ) a k by n matrix and C is a m by n matrix.
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
#     Note for int8 users - For rocBLAS backend, please read rocblas_gemm_ex documentation on int8
#     data layout requirements. hipBLAS makes the assumption that the data layout is in the preferred
#     format for a given device as documented in rocBLAS.
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     transA    [hipblasOperation_t]
#               specifies the form of op( A ).
#     @param[in]
#     transB    [hipblasOperation_t]
#               specifies the form of op( B ).
#     @param[in]
#     m         [int]
#               matrix dimension m.
#     @param[in]
#     n         [int]
#               matrix dimension n.
#     @param[in]
#     k         [int]
#               matrix dimension k.
#     @param[in]
#     alpha     [const void *]
#               device pointer or host pointer specifying the scalar alpha. Same datatype as computeType.
#     @param[in]
#     A         [void *]
#               device pointer storing matrix A.
#     @param[in]
#     aType    [hipblasDatatype_t]
#               specifies the datatype of matrix A.
#     @param[in]
#     lda       [int]
#               specifies the leading dimension of A.
#     @param[in]
#     B         [void *]
#               device pointer storing matrix B.
#     @param[in]
#     bType    [hipblasDatatype_t]
#               specifies the datatype of matrix B.
#     @param[in]
#     ldb       [int]
#               specifies the leading dimension of B.
#     @param[in]
#     beta      [const void *]
#               device pointer or host pointer specifying the scalar beta. Same datatype as computeType.
#     @param[in]
#     C         [void *]
#               device pointer storing matrix C.
#     @param[in]
#     cType    [hipblasDatatype_t]
#               specifies the datatype of matrix C.
#     @param[in]
#     ldc       [int]
#               specifies the leading dimension of C.
#     @param[in]
#     computeType
#               [hipblasDatatype_t]
#               specifies the datatype of computation.
#     @param[in]
#     algo      [hipblasGemmAlgo_t]
#               enumerant specifying the algorithm type.
#
cdef hipblasStatus_t hipblasGemmEx(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const void * alpha,const void * A,hipblasDatatype_t aType,int lda,const void * B,hipblasDatatype_t bType,int ldb,const void * beta,void * C,hipblasDatatype_t cType,int ldc,hipblasDatatype_t computeType,hipblasGemmAlgo_t algo) nogil


# ! BLAS EX API
# 
#     \details
#     trsmEx solves
# 
#         op(A)*X = alpha*B or X*op(A) = alpha*B,
# 
#     where alpha is a scalar, X and B are m by n matrices,
#     A is triangular matrix and op(A) is one of
# 
#         op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.
# 
#     The matrix X is overwritten on B.
# 
#     This function gives the user the ability to reuse the invA matrix between runs.
#     If invA == NULL, hipblasTrsmEx will automatically calculate invA on every run.
# 
#     Setting up invA:
#     The accepted invA matrix consists of the packed 128x128 inverses of the diagonal blocks of
#     matrix A, followed by any smaller diagonal block that remains.
#     To set up invA it is recommended that hipblasTrtriBatched be used with matrix A as the input.
# 
#     Device memory of size 128 x k should be allocated for invA ahead of time, where k is m when
#     HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT. The actual number of elements in invA
#     should be passed as invAsize.
# 
#     To begin, hipblasTrtriBatched must be called on the full 128x128 sized diagonal blocks of
#     matrix A. Below are the restricted parameters:
#       - n = 128
#       - ldinvA = 128
#       - stride_invA = 128x128
#       - batchCount = k / 128,
# 
#     Then any remaining block may be added:
#       - n = k % 128
#       - invA = invA + stride_invA * previousBatchCount
#       - ldinvA = 128
#       - batchCount = 1
# 
#     @param[in]
#     handle  [hipblasHandle_t]
#             handle to the hipblas library context queue.
# 
#     @param[in]
#     side    [hipblasSideMode_t]
#             HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#             HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# 
#     @param[in]
#     uplo    [hipblasFillMode_t]
#             HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#             HIPBLAS_FILL_MODE_LOWER:  A is a lower triangular matrix.
# 
#     @param[in]
#     transA  [hipblasOperation_t]
#             HIPBLAS_OP_N: op(A) = A.
#             HIPBLAS_OP_T: op(A) = A^T.
#             HIPBLAS_ON_C: op(A) = A^H.
# 
#     @param[in]
#     diag    [hipblasDiagType_t]
#             HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#             HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
#     @param[in]
#     m       [int]
#             m specifies the number of rows of B. m >= 0.
# 
#     @param[in]
#     n       [int]
#             n specifies the number of columns of B. n >= 0.
# 
#     @param[in]
#     alpha   [void *]
#             device pointer or host pointer specifying the scalar alpha. When alpha is
#             &zero then A is not referenced, and B need not be set before
#             entry.
# 
#     @param[in]
#     A       [void *]
#             device pointer storing matrix A.
#             of dimension ( lda, k ), where k is m
#             when HIPBLAS_SIDE_LEFT and
#             is n when HIPBLAS_SIDE_RIGHT
#             only the upper/lower triangular part is accessed.
# 
#     @param[in]
#     lda     [int]
#             lda specifies the first dimension of A.
#             if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#             if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
#     @param[in, out]
#     B       [void *]
#             device pointer storing matrix B.
#             B is of dimension ( ldb, n ).
#             Before entry, the leading m by n part of the array B must
#             contain the right-hand side matrix B, and on exit is
#             overwritten by the solution matrix X.
# 
#     @param[in]
#     ldb    [int]
#            ldb specifies the first dimension of B. ldb >= max( 1, m ).
# 
#     @param[in]
#     invA    [void *]
#             device pointer storing the inverse diagonal blocks of A.
#             invA is of dimension ( ld_invA, k ), where k is m
#             when HIPBLAS_SIDE_LEFT and
#             is n when HIPBLAS_SIDE_RIGHT.
#             ld_invA must be equal to 128.
# 
#     @param[in]
#     invAsize [int]
#             invAsize specifies the number of elements of device memory in invA.
# 
#     @param[in]
#     computeType [hipblasDatatype_t]
#             specifies the datatype of computation
#
cdef hipblasStatus_t hipblasTrsmEx(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const void * alpha,void * A,int lda,void * B,int ldb,const void * invA,int invAsize,hipblasDatatype_t computeType) nogil


# ! \brief BLAS EX API
# 
#     \details
#     axpyEx computes constant alpha multiplied by vector x, plus vector y
# 
#         y := alpha * x + y
# 
#         - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     n         [int]
#               the number of elements in x and y.
#     @param[in]
#     alpha     device pointer or host pointer to specify the scalar alpha.
#     @param[in]
#     alphaType [hipblasDatatype_t]
#               specifies the datatype of alpha.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     xType [hipblasDatatype_t]
#            specifies the datatype of vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[inout]
#     y         device pointer storing vector y.
#     @param[in]
#     yType [hipblasDatatype_t]
#           specifies the datatype of vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#     @param[in]
#     executionType [hipblasDatatype_t]
#                   specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasAxpyEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,const void * x,hipblasDatatype_t xType,int incx,void * y,hipblasDatatype_t yType,int incy,hipblasDatatype_t executionType) nogil


# ! @{
#     \brief BLAS EX API
# 
#     \details
#     dotEx  performs the dot product of vectors x and y
# 
#         result = x * y;
# 
#     dotcEx  performs the dot product of the conjugate of complex vector x and complex vector y
# 
#         result = conjugate (x) * y;
# 
#         - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     n         [int]
#               the number of elements in x and y.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     xType [hipblasDatatype_t]
#            specifies the datatype of vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of y.
#     @param[in]
#     y         device pointer storing vector y.
#     @param[in]
#     yType [hipblasDatatype_t]
#           specifies the datatype of vector y.
#     @param[in]
#     incy      [int]
#               specifies the increment for the elements of y.
#     @param[inout]
#     result
#               device pointer or host pointer to store the dot product.
#               return is 0.0 if n <= 0.
#     @param[in]
#     resultType [hipblasDatatype_t]
#                 specifies the datatype of the result.
#     @param[in]
#     executionType [hipblasDatatype_t]
#                   specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasDotEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,const void * y,hipblasDatatype_t yType,int incy,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil



cdef hipblasStatus_t hipblasDotcEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,const void * y,hipblasDatatype_t yType,int incy,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil


# ! \brief BLAS_EX API
# 
#     \details
#     nrm2Ex computes the euclidean norm of a real or complex vector
# 
#               result := sqrt( x'*x ) for real vectors
#               result := sqrt( x**H*x ) for complex vectors
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     n         [int]
#               the number of elements in x.
#     @param[in]
#     x         device pointer storing vector x.
#     @param[in]
#     xType [hipblasDatatype_t]
#            specifies the datatype of the vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of y.
#     @param[inout]
#     result
#               device pointer or host pointer to store the nrm2 product.
#               return is 0.0 if n, incx<=0.
#     @param[in]
#     resultType [hipblasDatatype_t]
#                 specifies the datatype of the result.
#     @param[in]
#     executionType [hipblasDatatype_t]
#                   specifies the datatype of computation.
cdef hipblasStatus_t hipblasNrm2Ex(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil


# ! \brief BLAS EX API
# 
#     \details
#     rotEx applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
#         Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
#     In the case where cs_type is real:
#         x := c * x + s * y
#             y := c * y - s * x
# 
#     In the case where cs_type is complex, the imaginary part of c is ignored:
#         x := real(c) * x + s * y
#             y := real(c) * y - conj(s) * x
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
#     @param[in]
#     handle  [hipblasHandle_t]
#             handle to the hipblas library context queue.
#     @param[in]
#     n       [int]
#             number of elements in the x and y vectors.
#     @param[inout]
#     x       device pointer storing vector x.
#     @param[in]
#     xType [hipblasDatatype_t]
#            specifies the datatype of vector x.
#     @param[in]
#     incx    [int]
#             specifies the increment between elements of x.
#     @param[inout]
#     y       device pointer storing vector y.
#     @param[in]
#     yType [hipblasDatatype_t]
#            specifies the datatype of vector y.
#     @param[in]
#     incy    [int]
#             specifies the increment between elements of y.
#     @param[in]
#     c       device pointer or host pointer storing scalar cosine component of the rotation matrix.
#     @param[in]
#     s       device pointer or host pointer storing scalar sine component of the rotation matrix.
#     @param[in]
#     csType [hipblasDatatype_t]
#             specifies the datatype of c and s.
#     @param[in]
#     executionType [hipblasDatatype_t]
#                    specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasRotEx(void * handle,int n,void * x,hipblasDatatype_t xType,int incx,void * y,hipblasDatatype_t yType,int incy,const void * c,const void * s,hipblasDatatype_t csType,hipblasDatatype_t executionType) nogil


# ! \brief BLAS EX API
# 
#     \details
#     scalEx  scales each element of vector x with scalar alpha.
# 
#         x := alpha * x
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
#     @param[in]
#     handle    [hipblasHandle_t]
#               handle to the hipblas library context queue.
#     @param[in]
#     n         [int]
#               the number of elements in x.
#     @param[in]
#     alpha     device pointer or host pointer for the scalar alpha.
#     @param[in]
#     alphaType [hipblasDatatype_t]
#                specifies the datatype of alpha.
#     @param[inout]
#     x         device pointer storing vector x.
#     @param[in]
#     xType [hipblasDatatype_t]
#            specifies the datatype of vector x.
#     @param[in]
#     incx      [int]
#               specifies the increment for the elements of x.
#     @param[in]
#     executionType [hipblasDatatype_t]
#                    specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasScalEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,void * x,hipblasDatatype_t xType,int incx,hipblasDatatype_t executionType) nogil


# ! HIPBLAS Auxiliary API
# 
#     \details
#     hipblasStatusToString
# 
#     Returns string representing hipblasStatus_t value
# 
#     @param[in]
#     status  [hipblasStatus_t]
#             hipBLAS status to convert to string
cdef const char * hipblasStatusToString(hipblasStatus_t status) nogil
