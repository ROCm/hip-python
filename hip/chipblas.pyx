# AMD_COPYRIGHT
from libc.stdint cimport *
from .chip cimport hipStream_t

cimport hip._util.posixloader as loader


cdef void* _lib_handle = loader.open_library("libhipblas.so")


cdef void* hipblasCreate_funptr = NULL
# ! \brief Create hipblas handle. */
cdef hipblasStatus_t hipblasCreate(hipblasHandle_t* handle) nogil:
    global _lib_handle
    global hipblasCreate_funptr
    if hipblasCreate_funptr == NULL:
        with gil:
            hipblasCreate_funptr = loader.load_symbol(_lib_handle, "hipblasCreate")
    return (<hipblasStatus_t (*)(hipblasHandle_t*) nogil> hipblasCreate_funptr)(handle)


cdef void* hipblasDestroy_funptr = NULL
# ! \brief Destroys the library context created using hipblasCreate() */
cdef hipblasStatus_t hipblasDestroy(hipblasHandle_t handle) nogil:
    global _lib_handle
    global hipblasDestroy_funptr
    if hipblasDestroy_funptr == NULL:
        with gil:
            hipblasDestroy_funptr = loader.load_symbol(_lib_handle, "hipblasDestroy")
    return (<hipblasStatus_t (*)(hipblasHandle_t) nogil> hipblasDestroy_funptr)(handle)


cdef void* hipblasSetStream_funptr = NULL
# ! \brief Set stream for handle */
cdef hipblasStatus_t hipblasSetStream(hipblasHandle_t handle,hipStream_t streamId) nogil:
    global _lib_handle
    global hipblasSetStream_funptr
    if hipblasSetStream_funptr == NULL:
        with gil:
            hipblasSetStream_funptr = loader.load_symbol(_lib_handle, "hipblasSetStream")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipStream_t) nogil> hipblasSetStream_funptr)(handle,streamId)


cdef void* hipblasGetStream_funptr = NULL
# ! \brief Get stream[0] for handle */
cdef hipblasStatus_t hipblasGetStream(hipblasHandle_t handle,hipStream_t* streamId) nogil:
    global _lib_handle
    global hipblasGetStream_funptr
    if hipblasGetStream_funptr == NULL:
        with gil:
            hipblasGetStream_funptr = loader.load_symbol(_lib_handle, "hipblasGetStream")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipStream_t*) nogil> hipblasGetStream_funptr)(handle,streamId)


cdef void* hipblasSetPointerMode_funptr = NULL
# ! \brief Set hipblas pointer mode */
cdef hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t handle,hipblasPointerMode_t mode) nogil:
    global _lib_handle
    global hipblasSetPointerMode_funptr
    if hipblasSetPointerMode_funptr == NULL:
        with gil:
            hipblasSetPointerMode_funptr = loader.load_symbol(_lib_handle, "hipblasSetPointerMode")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasPointerMode_t) nogil> hipblasSetPointerMode_funptr)(handle,mode)


cdef void* hipblasGetPointerMode_funptr = NULL
# ! \brief Get hipblas pointer mode */
cdef hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t handle,hipblasPointerMode_t * mode) nogil:
    global _lib_handle
    global hipblasGetPointerMode_funptr
    if hipblasGetPointerMode_funptr == NULL:
        with gil:
            hipblasGetPointerMode_funptr = loader.load_symbol(_lib_handle, "hipblasGetPointerMode")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasPointerMode_t *) nogil> hipblasGetPointerMode_funptr)(handle,mode)


cdef void* hipblasSetInt8Datatype_funptr = NULL
# ! \brief Set hipblas int8 Datatype */
cdef hipblasStatus_t hipblasSetInt8Datatype(hipblasHandle_t handle,hipblasInt8Datatype_t int8Type) nogil:
    global _lib_handle
    global hipblasSetInt8Datatype_funptr
    if hipblasSetInt8Datatype_funptr == NULL:
        with gil:
            hipblasSetInt8Datatype_funptr = loader.load_symbol(_lib_handle, "hipblasSetInt8Datatype")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasInt8Datatype_t) nogil> hipblasSetInt8Datatype_funptr)(handle,int8Type)


cdef void* hipblasGetInt8Datatype_funptr = NULL
# ! \brief Get hipblas int8 Datatype*/
cdef hipblasStatus_t hipblasGetInt8Datatype(hipblasHandle_t handle,hipblasInt8Datatype_t * int8Type) nogil:
    global _lib_handle
    global hipblasGetInt8Datatype_funptr
    if hipblasGetInt8Datatype_funptr == NULL:
        with gil:
            hipblasGetInt8Datatype_funptr = loader.load_symbol(_lib_handle, "hipblasGetInt8Datatype")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasInt8Datatype_t *) nogil> hipblasGetInt8Datatype_funptr)(handle,int8Type)


cdef void* hipblasSetVector_funptr = NULL
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
cdef hipblasStatus_t hipblasSetVector(int n,int elemSize,const void * x,int incx,void * y,int incy) nogil:
    global _lib_handle
    global hipblasSetVector_funptr
    if hipblasSetVector_funptr == NULL:
        with gil:
            hipblasSetVector_funptr = loader.load_symbol(_lib_handle, "hipblasSetVector")
    return (<hipblasStatus_t (*)(int,int,const void *,int,void *,int) nogil> hipblasSetVector_funptr)(n,elemSize,x,incx,y,incy)


cdef void* hipblasGetVector_funptr = NULL
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
cdef hipblasStatus_t hipblasGetVector(int n,int elemSize,const void * x,int incx,void * y,int incy) nogil:
    global _lib_handle
    global hipblasGetVector_funptr
    if hipblasGetVector_funptr == NULL:
        with gil:
            hipblasGetVector_funptr = loader.load_symbol(_lib_handle, "hipblasGetVector")
    return (<hipblasStatus_t (*)(int,int,const void *,int,void *,int) nogil> hipblasGetVector_funptr)(n,elemSize,x,incx,y,incy)


cdef void* hipblasSetMatrix_funptr = NULL
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
cdef hipblasStatus_t hipblasSetMatrix(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb) nogil:
    global _lib_handle
    global hipblasSetMatrix_funptr
    if hipblasSetMatrix_funptr == NULL:
        with gil:
            hipblasSetMatrix_funptr = loader.load_symbol(_lib_handle, "hipblasSetMatrix")
    return (<hipblasStatus_t (*)(int,int,int,const void *,int,void *,int) nogil> hipblasSetMatrix_funptr)(rows,cols,elemSize,AP,lda,BP,ldb)


cdef void* hipblasGetMatrix_funptr = NULL
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
cdef hipblasStatus_t hipblasGetMatrix(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb) nogil:
    global _lib_handle
    global hipblasGetMatrix_funptr
    if hipblasGetMatrix_funptr == NULL:
        with gil:
            hipblasGetMatrix_funptr = loader.load_symbol(_lib_handle, "hipblasGetMatrix")
    return (<hipblasStatus_t (*)(int,int,int,const void *,int,void *,int) nogil> hipblasGetMatrix_funptr)(rows,cols,elemSize,AP,lda,BP,ldb)


cdef void* hipblasSetVectorAsync_funptr = NULL
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
cdef hipblasStatus_t hipblasSetVectorAsync(int n,int elemSize,const void * x,int incx,void * y,int incy,hipStream_t stream) nogil:
    global _lib_handle
    global hipblasSetVectorAsync_funptr
    if hipblasSetVectorAsync_funptr == NULL:
        with gil:
            hipblasSetVectorAsync_funptr = loader.load_symbol(_lib_handle, "hipblasSetVectorAsync")
    return (<hipblasStatus_t (*)(int,int,const void *,int,void *,int,hipStream_t) nogil> hipblasSetVectorAsync_funptr)(n,elemSize,x,incx,y,incy,stream)


cdef void* hipblasGetVectorAsync_funptr = NULL
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
cdef hipblasStatus_t hipblasGetVectorAsync(int n,int elemSize,const void * x,int incx,void * y,int incy,hipStream_t stream) nogil:
    global _lib_handle
    global hipblasGetVectorAsync_funptr
    if hipblasGetVectorAsync_funptr == NULL:
        with gil:
            hipblasGetVectorAsync_funptr = loader.load_symbol(_lib_handle, "hipblasGetVectorAsync")
    return (<hipblasStatus_t (*)(int,int,const void *,int,void *,int,hipStream_t) nogil> hipblasGetVectorAsync_funptr)(n,elemSize,x,incx,y,incy,stream)


cdef void* hipblasSetMatrixAsync_funptr = NULL
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
cdef hipblasStatus_t hipblasSetMatrixAsync(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb,hipStream_t stream) nogil:
    global _lib_handle
    global hipblasSetMatrixAsync_funptr
    if hipblasSetMatrixAsync_funptr == NULL:
        with gil:
            hipblasSetMatrixAsync_funptr = loader.load_symbol(_lib_handle, "hipblasSetMatrixAsync")
    return (<hipblasStatus_t (*)(int,int,int,const void *,int,void *,int,hipStream_t) nogil> hipblasSetMatrixAsync_funptr)(rows,cols,elemSize,AP,lda,BP,ldb,stream)


cdef void* hipblasGetMatrixAsync_funptr = NULL
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
cdef hipblasStatus_t hipblasGetMatrixAsync(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb,hipStream_t stream) nogil:
    global _lib_handle
    global hipblasGetMatrixAsync_funptr
    if hipblasGetMatrixAsync_funptr == NULL:
        with gil:
            hipblasGetMatrixAsync_funptr = loader.load_symbol(_lib_handle, "hipblasGetMatrixAsync")
    return (<hipblasStatus_t (*)(int,int,int,const void *,int,void *,int,hipStream_t) nogil> hipblasGetMatrixAsync_funptr)(rows,cols,elemSize,AP,lda,BP,ldb,stream)


cdef void* hipblasSetAtomicsMode_funptr = NULL
# ! \brief Set hipblasSetAtomicsMode*/
cdef hipblasStatus_t hipblasSetAtomicsMode(hipblasHandle_t handle,hipblasAtomicsMode_t atomics_mode) nogil:
    global _lib_handle
    global hipblasSetAtomicsMode_funptr
    if hipblasSetAtomicsMode_funptr == NULL:
        with gil:
            hipblasSetAtomicsMode_funptr = loader.load_symbol(_lib_handle, "hipblasSetAtomicsMode")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasAtomicsMode_t) nogil> hipblasSetAtomicsMode_funptr)(handle,atomics_mode)


cdef void* hipblasGetAtomicsMode_funptr = NULL
# ! \brief Get hipblasSetAtomicsMode*/
cdef hipblasStatus_t hipblasGetAtomicsMode(hipblasHandle_t handle,hipblasAtomicsMode_t * atomics_mode) nogil:
    global _lib_handle
    global hipblasGetAtomicsMode_funptr
    if hipblasGetAtomicsMode_funptr == NULL:
        with gil:
            hipblasGetAtomicsMode_funptr = loader.load_symbol(_lib_handle, "hipblasGetAtomicsMode")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasAtomicsMode_t *) nogil> hipblasGetAtomicsMode_funptr)(handle,atomics_mode)


cdef void* hipblasIsamax_funptr = NULL
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
cdef hipblasStatus_t hipblasIsamax(hipblasHandle_t handle,int n,const float * x,int incx,int * result) nogil:
    global _lib_handle
    global hipblasIsamax_funptr
    if hipblasIsamax_funptr == NULL:
        with gil:
            hipblasIsamax_funptr = loader.load_symbol(_lib_handle, "hipblasIsamax")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,int,int *) nogil> hipblasIsamax_funptr)(handle,n,x,incx,result)


cdef void* hipblasIdamax_funptr = NULL
cdef hipblasStatus_t hipblasIdamax(hipblasHandle_t handle,int n,const double * x,int incx,int * result) nogil:
    global _lib_handle
    global hipblasIdamax_funptr
    if hipblasIdamax_funptr == NULL:
        with gil:
            hipblasIdamax_funptr = loader.load_symbol(_lib_handle, "hipblasIdamax")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,int,int *) nogil> hipblasIdamax_funptr)(handle,n,x,incx,result)


cdef void* hipblasIcamax_funptr = NULL
cdef hipblasStatus_t hipblasIcamax(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,int * result) nogil:
    global _lib_handle
    global hipblasIcamax_funptr
    if hipblasIcamax_funptr == NULL:
        with gil:
            hipblasIcamax_funptr = loader.load_symbol(_lib_handle, "hipblasIcamax")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,int *) nogil> hipblasIcamax_funptr)(handle,n,x,incx,result)


cdef void* hipblasIzamax_funptr = NULL
cdef hipblasStatus_t hipblasIzamax(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,int * result) nogil:
    global _lib_handle
    global hipblasIzamax_funptr
    if hipblasIzamax_funptr == NULL:
        with gil:
            hipblasIzamax_funptr = loader.load_symbol(_lib_handle, "hipblasIzamax")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,int *) nogil> hipblasIzamax_funptr)(handle,n,x,incx,result)


cdef void* hipblasIsamin_funptr = NULL
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
cdef hipblasStatus_t hipblasIsamin(hipblasHandle_t handle,int n,const float * x,int incx,int * result) nogil:
    global _lib_handle
    global hipblasIsamin_funptr
    if hipblasIsamin_funptr == NULL:
        with gil:
            hipblasIsamin_funptr = loader.load_symbol(_lib_handle, "hipblasIsamin")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,int,int *) nogil> hipblasIsamin_funptr)(handle,n,x,incx,result)


cdef void* hipblasIdamin_funptr = NULL
cdef hipblasStatus_t hipblasIdamin(hipblasHandle_t handle,int n,const double * x,int incx,int * result) nogil:
    global _lib_handle
    global hipblasIdamin_funptr
    if hipblasIdamin_funptr == NULL:
        with gil:
            hipblasIdamin_funptr = loader.load_symbol(_lib_handle, "hipblasIdamin")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,int,int *) nogil> hipblasIdamin_funptr)(handle,n,x,incx,result)


cdef void* hipblasIcamin_funptr = NULL
cdef hipblasStatus_t hipblasIcamin(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,int * result) nogil:
    global _lib_handle
    global hipblasIcamin_funptr
    if hipblasIcamin_funptr == NULL:
        with gil:
            hipblasIcamin_funptr = loader.load_symbol(_lib_handle, "hipblasIcamin")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,int *) nogil> hipblasIcamin_funptr)(handle,n,x,incx,result)


cdef void* hipblasIzamin_funptr = NULL
cdef hipblasStatus_t hipblasIzamin(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,int * result) nogil:
    global _lib_handle
    global hipblasIzamin_funptr
    if hipblasIzamin_funptr == NULL:
        with gil:
            hipblasIzamin_funptr = loader.load_symbol(_lib_handle, "hipblasIzamin")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,int *) nogil> hipblasIzamin_funptr)(handle,n,x,incx,result)


cdef void* hipblasSasum_funptr = NULL
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
cdef hipblasStatus_t hipblasSasum(hipblasHandle_t handle,int n,const float * x,int incx,float * result) nogil:
    global _lib_handle
    global hipblasSasum_funptr
    if hipblasSasum_funptr == NULL:
        with gil:
            hipblasSasum_funptr = loader.load_symbol(_lib_handle, "hipblasSasum")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,int,float *) nogil> hipblasSasum_funptr)(handle,n,x,incx,result)


cdef void* hipblasDasum_funptr = NULL
cdef hipblasStatus_t hipblasDasum(hipblasHandle_t handle,int n,const double * x,int incx,double * result) nogil:
    global _lib_handle
    global hipblasDasum_funptr
    if hipblasDasum_funptr == NULL:
        with gil:
            hipblasDasum_funptr = loader.load_symbol(_lib_handle, "hipblasDasum")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,int,double *) nogil> hipblasDasum_funptr)(handle,n,x,incx,result)


cdef void* hipblasScasum_funptr = NULL
cdef hipblasStatus_t hipblasScasum(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,float * result) nogil:
    global _lib_handle
    global hipblasScasum_funptr
    if hipblasScasum_funptr == NULL:
        with gil:
            hipblasScasum_funptr = loader.load_symbol(_lib_handle, "hipblasScasum")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,float *) nogil> hipblasScasum_funptr)(handle,n,x,incx,result)


cdef void* hipblasDzasum_funptr = NULL
cdef hipblasStatus_t hipblasDzasum(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,double * result) nogil:
    global _lib_handle
    global hipblasDzasum_funptr
    if hipblasDzasum_funptr == NULL:
        with gil:
            hipblasDzasum_funptr = loader.load_symbol(_lib_handle, "hipblasDzasum")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,double *) nogil> hipblasDzasum_funptr)(handle,n,x,incx,result)


cdef void* hipblasHaxpy_funptr = NULL
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
cdef hipblasStatus_t hipblasHaxpy(hipblasHandle_t handle,int n,hipblasHalf * alpha,hipblasHalf * x,int incx,hipblasHalf * y,int incy) nogil:
    global _lib_handle
    global hipblasHaxpy_funptr
    if hipblasHaxpy_funptr == NULL:
        with gil:
            hipblasHaxpy_funptr = loader.load_symbol(_lib_handle, "hipblasHaxpy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasHalf *,hipblasHalf *,int,hipblasHalf *,int) nogil> hipblasHaxpy_funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* hipblasSaxpy_funptr = NULL
cdef hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle,int n,const float * alpha,const float * x,int incx,float * y,int incy) nogil:
    global _lib_handle
    global hipblasSaxpy_funptr
    if hipblasSaxpy_funptr == NULL:
        with gil:
            hipblasSaxpy_funptr = loader.load_symbol(_lib_handle, "hipblasSaxpy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,const float *,int,float *,int) nogil> hipblasSaxpy_funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* hipblasDaxpy_funptr = NULL
cdef hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle,int n,const double * alpha,const double * x,int incx,double * y,int incy) nogil:
    global _lib_handle
    global hipblasDaxpy_funptr
    if hipblasDaxpy_funptr == NULL:
        with gil:
            hipblasDaxpy_funptr = loader.load_symbol(_lib_handle, "hipblasDaxpy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,const double *,int,double *,int) nogil> hipblasDaxpy_funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* hipblasCaxpy_funptr = NULL
cdef hipblasStatus_t hipblasCaxpy(hipblasHandle_t handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasCaxpy_funptr
    if hipblasCaxpy_funptr == NULL:
        with gil:
            hipblasCaxpy_funptr = loader.load_symbol(_lib_handle, "hipblasCaxpy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCaxpy_funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* hipblasZaxpy_funptr = NULL
cdef hipblasStatus_t hipblasZaxpy(hipblasHandle_t handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasZaxpy_funptr
    if hipblasZaxpy_funptr == NULL:
        with gil:
            hipblasZaxpy_funptr = loader.load_symbol(_lib_handle, "hipblasZaxpy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZaxpy_funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* hipblasScopy_funptr = NULL
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
cdef hipblasStatus_t hipblasScopy(hipblasHandle_t handle,int n,const float * x,int incx,float * y,int incy) nogil:
    global _lib_handle
    global hipblasScopy_funptr
    if hipblasScopy_funptr == NULL:
        with gil:
            hipblasScopy_funptr = loader.load_symbol(_lib_handle, "hipblasScopy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,int,float *,int) nogil> hipblasScopy_funptr)(handle,n,x,incx,y,incy)


cdef void* hipblasDcopy_funptr = NULL
cdef hipblasStatus_t hipblasDcopy(hipblasHandle_t handle,int n,const double * x,int incx,double * y,int incy) nogil:
    global _lib_handle
    global hipblasDcopy_funptr
    if hipblasDcopy_funptr == NULL:
        with gil:
            hipblasDcopy_funptr = loader.load_symbol(_lib_handle, "hipblasDcopy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,int,double *,int) nogil> hipblasDcopy_funptr)(handle,n,x,incx,y,incy)


cdef void* hipblasCcopy_funptr = NULL
cdef hipblasStatus_t hipblasCcopy(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasCcopy_funptr
    if hipblasCcopy_funptr == NULL:
        with gil:
            hipblasCcopy_funptr = loader.load_symbol(_lib_handle, "hipblasCcopy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCcopy_funptr)(handle,n,x,incx,y,incy)


cdef void* hipblasZcopy_funptr = NULL
cdef hipblasStatus_t hipblasZcopy(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasZcopy_funptr
    if hipblasZcopy_funptr == NULL:
        with gil:
            hipblasZcopy_funptr = loader.load_symbol(_lib_handle, "hipblasZcopy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZcopy_funptr)(handle,n,x,incx,y,incy)


cdef void* hipblasHdot_funptr = NULL
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
cdef hipblasStatus_t hipblasHdot(hipblasHandle_t handle,int n,hipblasHalf * x,int incx,hipblasHalf * y,int incy,hipblasHalf * result) nogil:
    global _lib_handle
    global hipblasHdot_funptr
    if hipblasHdot_funptr == NULL:
        with gil:
            hipblasHdot_funptr = loader.load_symbol(_lib_handle, "hipblasHdot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasHalf *,int,hipblasHalf *,int,hipblasHalf *) nogil> hipblasHdot_funptr)(handle,n,x,incx,y,incy,result)


cdef void* hipblasBfdot_funptr = NULL
cdef hipblasStatus_t hipblasBfdot(hipblasHandle_t handle,int n,hipblasBfloat16 * x,int incx,hipblasBfloat16 * y,int incy,hipblasBfloat16 * result) nogil:
    global _lib_handle
    global hipblasBfdot_funptr
    if hipblasBfdot_funptr == NULL:
        with gil:
            hipblasBfdot_funptr = loader.load_symbol(_lib_handle, "hipblasBfdot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasBfloat16 *,int,hipblasBfloat16 *,int,hipblasBfloat16 *) nogil> hipblasBfdot_funptr)(handle,n,x,incx,y,incy,result)


cdef void* hipblasSdot_funptr = NULL
cdef hipblasStatus_t hipblasSdot(hipblasHandle_t handle,int n,const float * x,int incx,const float * y,int incy,float * result) nogil:
    global _lib_handle
    global hipblasSdot_funptr
    if hipblasSdot_funptr == NULL:
        with gil:
            hipblasSdot_funptr = loader.load_symbol(_lib_handle, "hipblasSdot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,int,const float *,int,float *) nogil> hipblasSdot_funptr)(handle,n,x,incx,y,incy,result)


cdef void* hipblasDdot_funptr = NULL
cdef hipblasStatus_t hipblasDdot(hipblasHandle_t handle,int n,const double * x,int incx,const double * y,int incy,double * result) nogil:
    global _lib_handle
    global hipblasDdot_funptr
    if hipblasDdot_funptr == NULL:
        with gil:
            hipblasDdot_funptr = loader.load_symbol(_lib_handle, "hipblasDdot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,int,const double *,int,double *) nogil> hipblasDdot_funptr)(handle,n,x,incx,y,incy,result)


cdef void* hipblasCdotc_funptr = NULL
cdef hipblasStatus_t hipblasCdotc(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * result) nogil:
    global _lib_handle
    global hipblasCdotc_funptr
    if hipblasCdotc_funptr == NULL:
        with gil:
            hipblasCdotc_funptr = loader.load_symbol(_lib_handle, "hipblasCdotc")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *) nogil> hipblasCdotc_funptr)(handle,n,x,incx,y,incy,result)


cdef void* hipblasCdotu_funptr = NULL
cdef hipblasStatus_t hipblasCdotu(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * result) nogil:
    global _lib_handle
    global hipblasCdotu_funptr
    if hipblasCdotu_funptr == NULL:
        with gil:
            hipblasCdotu_funptr = loader.load_symbol(_lib_handle, "hipblasCdotu")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *) nogil> hipblasCdotu_funptr)(handle,n,x,incx,y,incy,result)


cdef void* hipblasZdotc_funptr = NULL
cdef hipblasStatus_t hipblasZdotc(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * result) nogil:
    global _lib_handle
    global hipblasZdotc_funptr
    if hipblasZdotc_funptr == NULL:
        with gil:
            hipblasZdotc_funptr = loader.load_symbol(_lib_handle, "hipblasZdotc")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> hipblasZdotc_funptr)(handle,n,x,incx,y,incy,result)


cdef void* hipblasZdotu_funptr = NULL
cdef hipblasStatus_t hipblasZdotu(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * result) nogil:
    global _lib_handle
    global hipblasZdotu_funptr
    if hipblasZdotu_funptr == NULL:
        with gil:
            hipblasZdotu_funptr = loader.load_symbol(_lib_handle, "hipblasZdotu")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> hipblasZdotu_funptr)(handle,n,x,incx,y,incy,result)


cdef void* hipblasSnrm2_funptr = NULL
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
cdef hipblasStatus_t hipblasSnrm2(hipblasHandle_t handle,int n,const float * x,int incx,float * result) nogil:
    global _lib_handle
    global hipblasSnrm2_funptr
    if hipblasSnrm2_funptr == NULL:
        with gil:
            hipblasSnrm2_funptr = loader.load_symbol(_lib_handle, "hipblasSnrm2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,int,float *) nogil> hipblasSnrm2_funptr)(handle,n,x,incx,result)


cdef void* hipblasDnrm2_funptr = NULL
cdef hipblasStatus_t hipblasDnrm2(hipblasHandle_t handle,int n,const double * x,int incx,double * result) nogil:
    global _lib_handle
    global hipblasDnrm2_funptr
    if hipblasDnrm2_funptr == NULL:
        with gil:
            hipblasDnrm2_funptr = loader.load_symbol(_lib_handle, "hipblasDnrm2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,int,double *) nogil> hipblasDnrm2_funptr)(handle,n,x,incx,result)


cdef void* hipblasScnrm2_funptr = NULL
cdef hipblasStatus_t hipblasScnrm2(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,float * result) nogil:
    global _lib_handle
    global hipblasScnrm2_funptr
    if hipblasScnrm2_funptr == NULL:
        with gil:
            hipblasScnrm2_funptr = loader.load_symbol(_lib_handle, "hipblasScnrm2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,float *) nogil> hipblasScnrm2_funptr)(handle,n,x,incx,result)


cdef void* hipblasDznrm2_funptr = NULL
cdef hipblasStatus_t hipblasDznrm2(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,double * result) nogil:
    global _lib_handle
    global hipblasDznrm2_funptr
    if hipblasDznrm2_funptr == NULL:
        with gil:
            hipblasDznrm2_funptr = loader.load_symbol(_lib_handle, "hipblasDznrm2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,double *) nogil> hipblasDznrm2_funptr)(handle,n,x,incx,result)


cdef void* hipblasSrot_funptr = NULL
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
cdef hipblasStatus_t hipblasSrot(hipblasHandle_t handle,int n,float * x,int incx,float * y,int incy,const float * c,const float * s) nogil:
    global _lib_handle
    global hipblasSrot_funptr
    if hipblasSrot_funptr == NULL:
        with gil:
            hipblasSrot_funptr = loader.load_symbol(_lib_handle, "hipblasSrot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,float *,int,float *,int,const float *,const float *) nogil> hipblasSrot_funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* hipblasDrot_funptr = NULL
cdef hipblasStatus_t hipblasDrot(hipblasHandle_t handle,int n,double * x,int incx,double * y,int incy,const double * c,const double * s) nogil:
    global _lib_handle
    global hipblasDrot_funptr
    if hipblasDrot_funptr == NULL:
        with gil:
            hipblasDrot_funptr = loader.load_symbol(_lib_handle, "hipblasDrot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,double *,int,double *,int,const double *,const double *) nogil> hipblasDrot_funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* hipblasCrot_funptr = NULL
cdef hipblasStatus_t hipblasCrot(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,const float * c,hipblasComplex * s) nogil:
    global _lib_handle
    global hipblasCrot_funptr
    if hipblasCrot_funptr == NULL:
        with gil:
            hipblasCrot_funptr = loader.load_symbol(_lib_handle, "hipblasCrot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,hipblasComplex *,int,const float *,hipblasComplex *) nogil> hipblasCrot_funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* hipblasCsrot_funptr = NULL
cdef hipblasStatus_t hipblasCsrot(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,const float * c,const float * s) nogil:
    global _lib_handle
    global hipblasCsrot_funptr
    if hipblasCsrot_funptr == NULL:
        with gil:
            hipblasCsrot_funptr = loader.load_symbol(_lib_handle, "hipblasCsrot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,hipblasComplex *,int,const float *,const float *) nogil> hipblasCsrot_funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* hipblasZrot_funptr = NULL
cdef hipblasStatus_t hipblasZrot(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,const double * c,hipblasDoubleComplex * s) nogil:
    global _lib_handle
    global hipblasZrot_funptr
    if hipblasZrot_funptr == NULL:
        with gil:
            hipblasZrot_funptr = loader.load_symbol(_lib_handle, "hipblasZrot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,const double *,hipblasDoubleComplex *) nogil> hipblasZrot_funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* hipblasZdrot_funptr = NULL
cdef hipblasStatus_t hipblasZdrot(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,const double * c,const double * s) nogil:
    global _lib_handle
    global hipblasZdrot_funptr
    if hipblasZdrot_funptr == NULL:
        with gil:
            hipblasZdrot_funptr = loader.load_symbol(_lib_handle, "hipblasZdrot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,const double *,const double *) nogil> hipblasZdrot_funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* hipblasSrotg_funptr = NULL
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
cdef hipblasStatus_t hipblasSrotg(hipblasHandle_t handle,float * a,float * b,float * c,float * s) nogil:
    global _lib_handle
    global hipblasSrotg_funptr
    if hipblasSrotg_funptr == NULL:
        with gil:
            hipblasSrotg_funptr = loader.load_symbol(_lib_handle, "hipblasSrotg")
    return (<hipblasStatus_t (*)(hipblasHandle_t,float *,float *,float *,float *) nogil> hipblasSrotg_funptr)(handle,a,b,c,s)


cdef void* hipblasDrotg_funptr = NULL
cdef hipblasStatus_t hipblasDrotg(hipblasHandle_t handle,double * a,double * b,double * c,double * s) nogil:
    global _lib_handle
    global hipblasDrotg_funptr
    if hipblasDrotg_funptr == NULL:
        with gil:
            hipblasDrotg_funptr = loader.load_symbol(_lib_handle, "hipblasDrotg")
    return (<hipblasStatus_t (*)(hipblasHandle_t,double *,double *,double *,double *) nogil> hipblasDrotg_funptr)(handle,a,b,c,s)


cdef void* hipblasCrotg_funptr = NULL
cdef hipblasStatus_t hipblasCrotg(hipblasHandle_t handle,hipblasComplex * a,hipblasComplex * b,float * c,hipblasComplex * s) nogil:
    global _lib_handle
    global hipblasCrotg_funptr
    if hipblasCrotg_funptr == NULL:
        with gil:
            hipblasCrotg_funptr = loader.load_symbol(_lib_handle, "hipblasCrotg")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasComplex *,hipblasComplex *,float *,hipblasComplex *) nogil> hipblasCrotg_funptr)(handle,a,b,c,s)


cdef void* hipblasZrotg_funptr = NULL
cdef hipblasStatus_t hipblasZrotg(hipblasHandle_t handle,hipblasDoubleComplex * a,hipblasDoubleComplex * b,double * c,hipblasDoubleComplex * s) nogil:
    global _lib_handle
    global hipblasZrotg_funptr
    if hipblasZrotg_funptr == NULL:
        with gil:
            hipblasZrotg_funptr = loader.load_symbol(_lib_handle, "hipblasZrotg")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasDoubleComplex *,hipblasDoubleComplex *,double *,hipblasDoubleComplex *) nogil> hipblasZrotg_funptr)(handle,a,b,c,s)


cdef void* hipblasSrotm_funptr = NULL
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
cdef hipblasStatus_t hipblasSrotm(hipblasHandle_t handle,int n,float * x,int incx,float * y,int incy,const float * param) nogil:
    global _lib_handle
    global hipblasSrotm_funptr
    if hipblasSrotm_funptr == NULL:
        with gil:
            hipblasSrotm_funptr = loader.load_symbol(_lib_handle, "hipblasSrotm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,float *,int,float *,int,const float *) nogil> hipblasSrotm_funptr)(handle,n,x,incx,y,incy,param)


cdef void* hipblasDrotm_funptr = NULL
cdef hipblasStatus_t hipblasDrotm(hipblasHandle_t handle,int n,double * x,int incx,double * y,int incy,const double * param) nogil:
    global _lib_handle
    global hipblasDrotm_funptr
    if hipblasDrotm_funptr == NULL:
        with gil:
            hipblasDrotm_funptr = loader.load_symbol(_lib_handle, "hipblasDrotm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,double *,int,double *,int,const double *) nogil> hipblasDrotm_funptr)(handle,n,x,incx,y,incy,param)


cdef void* hipblasSrotmg_funptr = NULL
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
cdef hipblasStatus_t hipblasSrotmg(hipblasHandle_t handle,float * d1,float * d2,float * x1,const float * y1,float * param) nogil:
    global _lib_handle
    global hipblasSrotmg_funptr
    if hipblasSrotmg_funptr == NULL:
        with gil:
            hipblasSrotmg_funptr = loader.load_symbol(_lib_handle, "hipblasSrotmg")
    return (<hipblasStatus_t (*)(hipblasHandle_t,float *,float *,float *,const float *,float *) nogil> hipblasSrotmg_funptr)(handle,d1,d2,x1,y1,param)


cdef void* hipblasDrotmg_funptr = NULL
cdef hipblasStatus_t hipblasDrotmg(hipblasHandle_t handle,double * d1,double * d2,double * x1,const double * y1,double * param) nogil:
    global _lib_handle
    global hipblasDrotmg_funptr
    if hipblasDrotmg_funptr == NULL:
        with gil:
            hipblasDrotmg_funptr = loader.load_symbol(_lib_handle, "hipblasDrotmg")
    return (<hipblasStatus_t (*)(hipblasHandle_t,double *,double *,double *,const double *,double *) nogil> hipblasDrotmg_funptr)(handle,d1,d2,x1,y1,param)


cdef void* hipblasSscal_funptr = NULL
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
cdef hipblasStatus_t hipblasSscal(hipblasHandle_t handle,int n,const float * alpha,float * x,int incx) nogil:
    global _lib_handle
    global hipblasSscal_funptr
    if hipblasSscal_funptr == NULL:
        with gil:
            hipblasSscal_funptr = loader.load_symbol(_lib_handle, "hipblasSscal")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,float *,int) nogil> hipblasSscal_funptr)(handle,n,alpha,x,incx)


cdef void* hipblasDscal_funptr = NULL
cdef hipblasStatus_t hipblasDscal(hipblasHandle_t handle,int n,const double * alpha,double * x,int incx) nogil:
    global _lib_handle
    global hipblasDscal_funptr
    if hipblasDscal_funptr == NULL:
        with gil:
            hipblasDscal_funptr = loader.load_symbol(_lib_handle, "hipblasDscal")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,double *,int) nogil> hipblasDscal_funptr)(handle,n,alpha,x,incx)


cdef void* hipblasCscal_funptr = NULL
cdef hipblasStatus_t hipblasCscal(hipblasHandle_t handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasCscal_funptr
    if hipblasCscal_funptr == NULL:
        with gil:
            hipblasCscal_funptr = loader.load_symbol(_lib_handle, "hipblasCscal")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasCscal_funptr)(handle,n,alpha,x,incx)


cdef void* hipblasCsscal_funptr = NULL
cdef hipblasStatus_t hipblasCsscal(hipblasHandle_t handle,int n,const float * alpha,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasCsscal_funptr
    if hipblasCsscal_funptr == NULL:
        with gil:
            hipblasCsscal_funptr = loader.load_symbol(_lib_handle, "hipblasCsscal")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,hipblasComplex *,int) nogil> hipblasCsscal_funptr)(handle,n,alpha,x,incx)


cdef void* hipblasZscal_funptr = NULL
cdef hipblasStatus_t hipblasZscal(hipblasHandle_t handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasZscal_funptr
    if hipblasZscal_funptr == NULL:
        with gil:
            hipblasZscal_funptr = loader.load_symbol(_lib_handle, "hipblasZscal")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZscal_funptr)(handle,n,alpha,x,incx)


cdef void* hipblasZdscal_funptr = NULL
cdef hipblasStatus_t hipblasZdscal(hipblasHandle_t handle,int n,const double * alpha,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasZdscal_funptr
    if hipblasZdscal_funptr == NULL:
        with gil:
            hipblasZdscal_funptr = loader.load_symbol(_lib_handle, "hipblasZdscal")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,hipblasDoubleComplex *,int) nogil> hipblasZdscal_funptr)(handle,n,alpha,x,incx)


cdef void* hipblasSswap_funptr = NULL
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
cdef hipblasStatus_t hipblasSswap(hipblasHandle_t handle,int n,float * x,int incx,float * y,int incy) nogil:
    global _lib_handle
    global hipblasSswap_funptr
    if hipblasSswap_funptr == NULL:
        with gil:
            hipblasSswap_funptr = loader.load_symbol(_lib_handle, "hipblasSswap")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,float *,int,float *,int) nogil> hipblasSswap_funptr)(handle,n,x,incx,y,incy)


cdef void* hipblasDswap_funptr = NULL
cdef hipblasStatus_t hipblasDswap(hipblasHandle_t handle,int n,double * x,int incx,double * y,int incy) nogil:
    global _lib_handle
    global hipblasDswap_funptr
    if hipblasDswap_funptr == NULL:
        with gil:
            hipblasDswap_funptr = loader.load_symbol(_lib_handle, "hipblasDswap")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,double *,int,double *,int) nogil> hipblasDswap_funptr)(handle,n,x,incx,y,incy)


cdef void* hipblasCswap_funptr = NULL
cdef hipblasStatus_t hipblasCswap(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasCswap_funptr
    if hipblasCswap_funptr == NULL:
        with gil:
            hipblasCswap_funptr = loader.load_symbol(_lib_handle, "hipblasCswap")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCswap_funptr)(handle,n,x,incx,y,incy)


cdef void* hipblasZswap_funptr = NULL
cdef hipblasStatus_t hipblasZswap(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasZswap_funptr
    if hipblasZswap_funptr == NULL:
        with gil:
            hipblasZswap_funptr = loader.load_symbol(_lib_handle, "hipblasZswap")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZswap_funptr)(handle,n,x,incx,y,incy)


cdef void* hipblasSgbmv_funptr = NULL
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
cdef hipblasStatus_t hipblasSgbmv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil:
    global _lib_handle
    global hipblasSgbmv_funptr
    if hipblasSgbmv_funptr == NULL:
        with gil:
            hipblasSgbmv_funptr = loader.load_symbol(_lib_handle, "hipblasSgbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> hipblasSgbmv_funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasDgbmv_funptr = NULL
cdef hipblasStatus_t hipblasDgbmv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _lib_handle
    global hipblasDgbmv_funptr
    if hipblasDgbmv_funptr == NULL:
        with gil:
            hipblasDgbmv_funptr = loader.load_symbol(_lib_handle, "hipblasDgbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> hipblasDgbmv_funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasCgbmv_funptr = NULL
cdef hipblasStatus_t hipblasCgbmv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasCgbmv_funptr
    if hipblasCgbmv_funptr == NULL:
        with gil:
            hipblasCgbmv_funptr = loader.load_symbol(_lib_handle, "hipblasCgbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasCgbmv_funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasZgbmv_funptr = NULL
cdef hipblasStatus_t hipblasZgbmv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasZgbmv_funptr
    if hipblasZgbmv_funptr == NULL:
        with gil:
            hipblasZgbmv_funptr = loader.load_symbol(_lib_handle, "hipblasZgbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZgbmv_funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasSgemv_funptr = NULL
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
cdef hipblasStatus_t hipblasSgemv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil:
    global _lib_handle
    global hipblasSgemv_funptr
    if hipblasSgemv_funptr == NULL:
        with gil:
            hipblasSgemv_funptr = loader.load_symbol(_lib_handle, "hipblasSgemv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> hipblasSgemv_funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasDgemv_funptr = NULL
cdef hipblasStatus_t hipblasDgemv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _lib_handle
    global hipblasDgemv_funptr
    if hipblasDgemv_funptr == NULL:
        with gil:
            hipblasDgemv_funptr = loader.load_symbol(_lib_handle, "hipblasDgemv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> hipblasDgemv_funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasCgemv_funptr = NULL
cdef hipblasStatus_t hipblasCgemv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasCgemv_funptr
    if hipblasCgemv_funptr == NULL:
        with gil:
            hipblasCgemv_funptr = loader.load_symbol(_lib_handle, "hipblasCgemv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasCgemv_funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasZgemv_funptr = NULL
cdef hipblasStatus_t hipblasZgemv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasZgemv_funptr
    if hipblasZgemv_funptr == NULL:
        with gil:
            hipblasZgemv_funptr = loader.load_symbol(_lib_handle, "hipblasZgemv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZgemv_funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasSger_funptr = NULL
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
cdef hipblasStatus_t hipblasSger(hipblasHandle_t handle,int m,int n,const float * alpha,const float * x,int incx,const float * y,int incy,float * AP,int lda) nogil:
    global _lib_handle
    global hipblasSger_funptr
    if hipblasSger_funptr == NULL:
        with gil:
            hipblasSger_funptr = loader.load_symbol(_lib_handle, "hipblasSger")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,int,const float *,const float *,int,const float *,int,float *,int) nogil> hipblasSger_funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* hipblasDger_funptr = NULL
cdef hipblasStatus_t hipblasDger(hipblasHandle_t handle,int m,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP,int lda) nogil:
    global _lib_handle
    global hipblasDger_funptr
    if hipblasDger_funptr == NULL:
        with gil:
            hipblasDger_funptr = loader.load_symbol(_lib_handle, "hipblasDger")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,int,const double *,const double *,int,const double *,int,double *,int) nogil> hipblasDger_funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* hipblasCgeru_funptr = NULL
cdef hipblasStatus_t hipblasCgeru(hipblasHandle_t handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil:
    global _lib_handle
    global hipblasCgeru_funptr
    if hipblasCgeru_funptr == NULL:
        with gil:
            hipblasCgeru_funptr = loader.load_symbol(_lib_handle, "hipblasCgeru")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCgeru_funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* hipblasCgerc_funptr = NULL
cdef hipblasStatus_t hipblasCgerc(hipblasHandle_t handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil:
    global _lib_handle
    global hipblasCgerc_funptr
    if hipblasCgerc_funptr == NULL:
        with gil:
            hipblasCgerc_funptr = loader.load_symbol(_lib_handle, "hipblasCgerc")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCgerc_funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* hipblasZgeru_funptr = NULL
cdef hipblasStatus_t hipblasZgeru(hipblasHandle_t handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil:
    global _lib_handle
    global hipblasZgeru_funptr
    if hipblasZgeru_funptr == NULL:
        with gil:
            hipblasZgeru_funptr = loader.load_symbol(_lib_handle, "hipblasZgeru")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZgeru_funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* hipblasZgerc_funptr = NULL
cdef hipblasStatus_t hipblasZgerc(hipblasHandle_t handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil:
    global _lib_handle
    global hipblasZgerc_funptr
    if hipblasZgerc_funptr == NULL:
        with gil:
            hipblasZgerc_funptr = loader.load_symbol(_lib_handle, "hipblasZgerc")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZgerc_funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* hipblasChbmv_funptr = NULL
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
cdef hipblasStatus_t hipblasChbmv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasChbmv_funptr
    if hipblasChbmv_funptr == NULL:
        with gil:
            hipblasChbmv_funptr = loader.load_symbol(_lib_handle, "hipblasChbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasChbmv_funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasZhbmv_funptr = NULL
cdef hipblasStatus_t hipblasZhbmv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasZhbmv_funptr
    if hipblasZhbmv_funptr == NULL:
        with gil:
            hipblasZhbmv_funptr = loader.load_symbol(_lib_handle, "hipblasZhbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZhbmv_funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasChemv_funptr = NULL
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
cdef hipblasStatus_t hipblasChemv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasChemv_funptr
    if hipblasChemv_funptr == NULL:
        with gil:
            hipblasChemv_funptr = loader.load_symbol(_lib_handle, "hipblasChemv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasChemv_funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasZhemv_funptr = NULL
cdef hipblasStatus_t hipblasZhemv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasZhemv_funptr
    if hipblasZhemv_funptr == NULL:
        with gil:
            hipblasZhemv_funptr = loader.load_symbol(_lib_handle, "hipblasZhemv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZhemv_funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasCher_funptr = NULL
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
cdef hipblasStatus_t hipblasCher(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex * x,int incx,hipblasComplex * AP,int lda) nogil:
    global _lib_handle
    global hipblasCher_funptr
    if hipblasCher_funptr == NULL:
        with gil:
            hipblasCher_funptr = loader.load_symbol(_lib_handle, "hipblasCher")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCher_funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* hipblasZher_funptr = NULL
cdef hipblasStatus_t hipblasZher(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP,int lda) nogil:
    global _lib_handle
    global hipblasZher_funptr
    if hipblasZher_funptr == NULL:
        with gil:
            hipblasZher_funptr = loader.load_symbol(_lib_handle, "hipblasZher")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZher_funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* hipblasCher2_funptr = NULL
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
cdef hipblasStatus_t hipblasCher2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil:
    global _lib_handle
    global hipblasCher2_funptr
    if hipblasCher2_funptr == NULL:
        with gil:
            hipblasCher2_funptr = loader.load_symbol(_lib_handle, "hipblasCher2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCher2_funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* hipblasZher2_funptr = NULL
cdef hipblasStatus_t hipblasZher2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil:
    global _lib_handle
    global hipblasZher2_funptr
    if hipblasZher2_funptr == NULL:
        with gil:
            hipblasZher2_funptr = loader.load_symbol(_lib_handle, "hipblasZher2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZher2_funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* hipblasChpmv_funptr = NULL
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
cdef hipblasStatus_t hipblasChpmv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasChpmv_funptr
    if hipblasChpmv_funptr == NULL:
        with gil:
            hipblasChpmv_funptr = loader.load_symbol(_lib_handle, "hipblasChpmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasChpmv_funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)


cdef void* hipblasZhpmv_funptr = NULL
cdef hipblasStatus_t hipblasZhpmv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasZhpmv_funptr
    if hipblasZhpmv_funptr == NULL:
        with gil:
            hipblasZhpmv_funptr = loader.load_symbol(_lib_handle, "hipblasZhpmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZhpmv_funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)


cdef void* hipblasChpr_funptr = NULL
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
cdef hipblasStatus_t hipblasChpr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex * x,int incx,hipblasComplex * AP) nogil:
    global _lib_handle
    global hipblasChpr_funptr
    if hipblasChpr_funptr == NULL:
        with gil:
            hipblasChpr_funptr = loader.load_symbol(_lib_handle, "hipblasChpr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,hipblasComplex *,int,hipblasComplex *) nogil> hipblasChpr_funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* hipblasZhpr_funptr = NULL
cdef hipblasStatus_t hipblasZhpr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP) nogil:
    global _lib_handle
    global hipblasZhpr_funptr
    if hipblasZhpr_funptr == NULL:
        with gil:
            hipblasZhpr_funptr = loader.load_symbol(_lib_handle, "hipblasZhpr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> hipblasZhpr_funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* hipblasChpr2_funptr = NULL
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
cdef hipblasStatus_t hipblasChpr2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP) nogil:
    global _lib_handle
    global hipblasChpr2_funptr
    if hipblasChpr2_funptr == NULL:
        with gil:
            hipblasChpr2_funptr = loader.load_symbol(_lib_handle, "hipblasChpr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *) nogil> hipblasChpr2_funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP)


cdef void* hipblasZhpr2_funptr = NULL
cdef hipblasStatus_t hipblasZhpr2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP) nogil:
    global _lib_handle
    global hipblasZhpr2_funptr
    if hipblasZhpr2_funptr == NULL:
        with gil:
            hipblasZhpr2_funptr = loader.load_symbol(_lib_handle, "hipblasZhpr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> hipblasZhpr2_funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP)


cdef void* hipblasSsbmv_funptr = NULL
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
cdef hipblasStatus_t hipblasSsbmv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,int k,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil:
    global _lib_handle
    global hipblasSsbmv_funptr
    if hipblasSsbmv_funptr == NULL:
        with gil:
            hipblasSsbmv_funptr = loader.load_symbol(_lib_handle, "hipblasSsbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> hipblasSsbmv_funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasDsbmv_funptr = NULL
cdef hipblasStatus_t hipblasDsbmv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,int k,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _lib_handle
    global hipblasDsbmv_funptr
    if hipblasDsbmv_funptr == NULL:
        with gil:
            hipblasDsbmv_funptr = loader.load_symbol(_lib_handle, "hipblasDsbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> hipblasDsbmv_funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasSspmv_funptr = NULL
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
cdef hipblasStatus_t hipblasSspmv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * AP,const float * x,int incx,const float * beta,float * y,int incy) nogil:
    global _lib_handle
    global hipblasSspmv_funptr
    if hipblasSspmv_funptr == NULL:
        with gil:
            hipblasSspmv_funptr = loader.load_symbol(_lib_handle, "hipblasSspmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,const float *,const float *,int,const float *,float *,int) nogil> hipblasSspmv_funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)


cdef void* hipblasDspmv_funptr = NULL
cdef hipblasStatus_t hipblasDspmv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _lib_handle
    global hipblasDspmv_funptr
    if hipblasDspmv_funptr == NULL:
        with gil:
            hipblasDspmv_funptr = loader.load_symbol(_lib_handle, "hipblasDspmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,const double *,const double *,int,const double *,double *,int) nogil> hipblasDspmv_funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)


cdef void* hipblasSspr_funptr = NULL
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
cdef hipblasStatus_t hipblasSspr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,float * AP) nogil:
    global _lib_handle
    global hipblasSspr_funptr
    if hipblasSspr_funptr == NULL:
        with gil:
            hipblasSspr_funptr = loader.load_symbol(_lib_handle, "hipblasSspr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,const float *,int,float *) nogil> hipblasSspr_funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* hipblasDspr_funptr = NULL
cdef hipblasStatus_t hipblasDspr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,double * AP) nogil:
    global _lib_handle
    global hipblasDspr_funptr
    if hipblasDspr_funptr == NULL:
        with gil:
            hipblasDspr_funptr = loader.load_symbol(_lib_handle, "hipblasDspr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,const double *,int,double *) nogil> hipblasDspr_funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* hipblasCspr_funptr = NULL
cdef hipblasStatus_t hipblasCspr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * AP) nogil:
    global _lib_handle
    global hipblasCspr_funptr
    if hipblasCspr_funptr == NULL:
        with gil:
            hipblasCspr_funptr = loader.load_symbol(_lib_handle, "hipblasCspr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *) nogil> hipblasCspr_funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* hipblasZspr_funptr = NULL
cdef hipblasStatus_t hipblasZspr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP) nogil:
    global _lib_handle
    global hipblasZspr_funptr
    if hipblasZspr_funptr == NULL:
        with gil:
            hipblasZspr_funptr = loader.load_symbol(_lib_handle, "hipblasZspr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> hipblasZspr_funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* hipblasSspr2_funptr = NULL
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
cdef hipblasStatus_t hipblasSspr2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,const float * y,int incy,float * AP) nogil:
    global _lib_handle
    global hipblasSspr2_funptr
    if hipblasSspr2_funptr == NULL:
        with gil:
            hipblasSspr2_funptr = loader.load_symbol(_lib_handle, "hipblasSspr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,const float *,int,const float *,int,float *) nogil> hipblasSspr2_funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP)


cdef void* hipblasDspr2_funptr = NULL
cdef hipblasStatus_t hipblasDspr2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP) nogil:
    global _lib_handle
    global hipblasDspr2_funptr
    if hipblasDspr2_funptr == NULL:
        with gil:
            hipblasDspr2_funptr = loader.load_symbol(_lib_handle, "hipblasDspr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,const double *,int,const double *,int,double *) nogil> hipblasDspr2_funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP)


cdef void* hipblasSsymv_funptr = NULL
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
cdef hipblasStatus_t hipblasSsymv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil:
    global _lib_handle
    global hipblasSsymv_funptr
    if hipblasSsymv_funptr == NULL:
        with gil:
            hipblasSsymv_funptr = loader.load_symbol(_lib_handle, "hipblasSsymv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> hipblasSsymv_funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasDsymv_funptr = NULL
cdef hipblasStatus_t hipblasDsymv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _lib_handle
    global hipblasDsymv_funptr
    if hipblasDsymv_funptr == NULL:
        with gil:
            hipblasDsymv_funptr = loader.load_symbol(_lib_handle, "hipblasDsymv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> hipblasDsymv_funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasCsymv_funptr = NULL
cdef hipblasStatus_t hipblasCsymv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasCsymv_funptr
    if hipblasCsymv_funptr == NULL:
        with gil:
            hipblasCsymv_funptr = loader.load_symbol(_lib_handle, "hipblasCsymv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasCsymv_funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasZsymv_funptr = NULL
cdef hipblasStatus_t hipblasZsymv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global hipblasZsymv_funptr
    if hipblasZsymv_funptr == NULL:
        with gil:
            hipblasZsymv_funptr = loader.load_symbol(_lib_handle, "hipblasZsymv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZsymv_funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* hipblasSsyr_funptr = NULL
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
cdef hipblasStatus_t hipblasSsyr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,float * AP,int lda) nogil:
    global _lib_handle
    global hipblasSsyr_funptr
    if hipblasSsyr_funptr == NULL:
        with gil:
            hipblasSsyr_funptr = loader.load_symbol(_lib_handle, "hipblasSsyr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,const float *,int,float *,int) nogil> hipblasSsyr_funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* hipblasDsyr_funptr = NULL
cdef hipblasStatus_t hipblasDsyr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,double * AP,int lda) nogil:
    global _lib_handle
    global hipblasDsyr_funptr
    if hipblasDsyr_funptr == NULL:
        with gil:
            hipblasDsyr_funptr = loader.load_symbol(_lib_handle, "hipblasDsyr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,const double *,int,double *,int) nogil> hipblasDsyr_funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* hipblasCsyr_funptr = NULL
cdef hipblasStatus_t hipblasCsyr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * AP,int lda) nogil:
    global _lib_handle
    global hipblasCsyr_funptr
    if hipblasCsyr_funptr == NULL:
        with gil:
            hipblasCsyr_funptr = loader.load_symbol(_lib_handle, "hipblasCsyr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCsyr_funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* hipblasZsyr_funptr = NULL
cdef hipblasStatus_t hipblasZsyr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP,int lda) nogil:
    global _lib_handle
    global hipblasZsyr_funptr
    if hipblasZsyr_funptr == NULL:
        with gil:
            hipblasZsyr_funptr = loader.load_symbol(_lib_handle, "hipblasZsyr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZsyr_funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* hipblasSsyr2_funptr = NULL
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
cdef hipblasStatus_t hipblasSsyr2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,const float * y,int incy,float * AP,int lda) nogil:
    global _lib_handle
    global hipblasSsyr2_funptr
    if hipblasSsyr2_funptr == NULL:
        with gil:
            hipblasSsyr2_funptr = loader.load_symbol(_lib_handle, "hipblasSsyr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,const float *,int,const float *,int,float *,int) nogil> hipblasSsyr2_funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* hipblasDsyr2_funptr = NULL
cdef hipblasStatus_t hipblasDsyr2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP,int lda) nogil:
    global _lib_handle
    global hipblasDsyr2_funptr
    if hipblasDsyr2_funptr == NULL:
        with gil:
            hipblasDsyr2_funptr = loader.load_symbol(_lib_handle, "hipblasDsyr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,const double *,int,const double *,int,double *,int) nogil> hipblasDsyr2_funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* hipblasCsyr2_funptr = NULL
cdef hipblasStatus_t hipblasCsyr2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil:
    global _lib_handle
    global hipblasCsyr2_funptr
    if hipblasCsyr2_funptr == NULL:
        with gil:
            hipblasCsyr2_funptr = loader.load_symbol(_lib_handle, "hipblasCsyr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCsyr2_funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* hipblasZsyr2_funptr = NULL
cdef hipblasStatus_t hipblasZsyr2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil:
    global _lib_handle
    global hipblasZsyr2_funptr
    if hipblasZsyr2_funptr == NULL:
        with gil:
            hipblasZsyr2_funptr = loader.load_symbol(_lib_handle, "hipblasZsyr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZsyr2_funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* hipblasStbmv_funptr = NULL
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
cdef hipblasStatus_t hipblasStbmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const float * AP,int lda,float * x,int incx) nogil:
    global _lib_handle
    global hipblasStbmv_funptr
    if hipblasStbmv_funptr == NULL:
        with gil:
            hipblasStbmv_funptr = loader.load_symbol(_lib_handle, "hipblasStbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,int,float *,int) nogil> hipblasStbmv_funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx)


cdef void* hipblasDtbmv_funptr = NULL
cdef hipblasStatus_t hipblasDtbmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const double * AP,int lda,double * x,int incx) nogil:
    global _lib_handle
    global hipblasDtbmv_funptr
    if hipblasDtbmv_funptr == NULL:
        with gil:
            hipblasDtbmv_funptr = loader.load_symbol(_lib_handle, "hipblasDtbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,int,double *,int) nogil> hipblasDtbmv_funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx)


cdef void* hipblasCtbmv_funptr = NULL
cdef hipblasStatus_t hipblasCtbmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasCtbmv_funptr
    if hipblasCtbmv_funptr == NULL:
        with gil:
            hipblasCtbmv_funptr = loader.load_symbol(_lib_handle, "hipblasCtbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCtbmv_funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx)


cdef void* hipblasZtbmv_funptr = NULL
cdef hipblasStatus_t hipblasZtbmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasZtbmv_funptr
    if hipblasZtbmv_funptr == NULL:
        with gil:
            hipblasZtbmv_funptr = loader.load_symbol(_lib_handle, "hipblasZtbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZtbmv_funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx)


cdef void* hipblasStbsv_funptr = NULL
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
cdef hipblasStatus_t hipblasStbsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const float * AP,int lda,float * x,int incx) nogil:
    global _lib_handle
    global hipblasStbsv_funptr
    if hipblasStbsv_funptr == NULL:
        with gil:
            hipblasStbsv_funptr = loader.load_symbol(_lib_handle, "hipblasStbsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,int,float *,int) nogil> hipblasStbsv_funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx)


cdef void* hipblasDtbsv_funptr = NULL
cdef hipblasStatus_t hipblasDtbsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const double * AP,int lda,double * x,int incx) nogil:
    global _lib_handle
    global hipblasDtbsv_funptr
    if hipblasDtbsv_funptr == NULL:
        with gil:
            hipblasDtbsv_funptr = loader.load_symbol(_lib_handle, "hipblasDtbsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,int,double *,int) nogil> hipblasDtbsv_funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx)


cdef void* hipblasCtbsv_funptr = NULL
cdef hipblasStatus_t hipblasCtbsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasCtbsv_funptr
    if hipblasCtbsv_funptr == NULL:
        with gil:
            hipblasCtbsv_funptr = loader.load_symbol(_lib_handle, "hipblasCtbsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCtbsv_funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx)


cdef void* hipblasZtbsv_funptr = NULL
cdef hipblasStatus_t hipblasZtbsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasZtbsv_funptr
    if hipblasZtbsv_funptr == NULL:
        with gil:
            hipblasZtbsv_funptr = loader.load_symbol(_lib_handle, "hipblasZtbsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZtbsv_funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx)


cdef void* hipblasStpmv_funptr = NULL
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
cdef hipblasStatus_t hipblasStpmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,float * x,int incx) nogil:
    global _lib_handle
    global hipblasStpmv_funptr
    if hipblasStpmv_funptr == NULL:
        with gil:
            hipblasStpmv_funptr = loader.load_symbol(_lib_handle, "hipblasStpmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,float *,int) nogil> hipblasStpmv_funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* hipblasDtpmv_funptr = NULL
cdef hipblasStatus_t hipblasDtpmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,double * x,int incx) nogil:
    global _lib_handle
    global hipblasDtpmv_funptr
    if hipblasDtpmv_funptr == NULL:
        with gil:
            hipblasDtpmv_funptr = loader.load_symbol(_lib_handle, "hipblasDtpmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,double *,int) nogil> hipblasDtpmv_funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* hipblasCtpmv_funptr = NULL
cdef hipblasStatus_t hipblasCtpmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasCtpmv_funptr
    if hipblasCtpmv_funptr == NULL:
        with gil:
            hipblasCtpmv_funptr = loader.load_symbol(_lib_handle, "hipblasCtpmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasCtpmv_funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* hipblasZtpmv_funptr = NULL
cdef hipblasStatus_t hipblasZtpmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasZtpmv_funptr
    if hipblasZtpmv_funptr == NULL:
        with gil:
            hipblasZtpmv_funptr = loader.load_symbol(_lib_handle, "hipblasZtpmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZtpmv_funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* hipblasStpsv_funptr = NULL
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
cdef hipblasStatus_t hipblasStpsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,float * x,int incx) nogil:
    global _lib_handle
    global hipblasStpsv_funptr
    if hipblasStpsv_funptr == NULL:
        with gil:
            hipblasStpsv_funptr = loader.load_symbol(_lib_handle, "hipblasStpsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,float *,int) nogil> hipblasStpsv_funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* hipblasDtpsv_funptr = NULL
cdef hipblasStatus_t hipblasDtpsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,double * x,int incx) nogil:
    global _lib_handle
    global hipblasDtpsv_funptr
    if hipblasDtpsv_funptr == NULL:
        with gil:
            hipblasDtpsv_funptr = loader.load_symbol(_lib_handle, "hipblasDtpsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,double *,int) nogil> hipblasDtpsv_funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* hipblasCtpsv_funptr = NULL
cdef hipblasStatus_t hipblasCtpsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasCtpsv_funptr
    if hipblasCtpsv_funptr == NULL:
        with gil:
            hipblasCtpsv_funptr = loader.load_symbol(_lib_handle, "hipblasCtpsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasCtpsv_funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* hipblasZtpsv_funptr = NULL
cdef hipblasStatus_t hipblasZtpsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasZtpsv_funptr
    if hipblasZtpsv_funptr == NULL:
        with gil:
            hipblasZtpsv_funptr = loader.load_symbol(_lib_handle, "hipblasZtpsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZtpsv_funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* hipblasStrmv_funptr = NULL
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
cdef hipblasStatus_t hipblasStrmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,int lda,float * x,int incx) nogil:
    global _lib_handle
    global hipblasStrmv_funptr
    if hipblasStrmv_funptr == NULL:
        with gil:
            hipblasStrmv_funptr = loader.load_symbol(_lib_handle, "hipblasStrmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,int,float *,int) nogil> hipblasStrmv_funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* hipblasDtrmv_funptr = NULL
cdef hipblasStatus_t hipblasDtrmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,double * x,int incx) nogil:
    global _lib_handle
    global hipblasDtrmv_funptr
    if hipblasDtrmv_funptr == NULL:
        with gil:
            hipblasDtrmv_funptr = loader.load_symbol(_lib_handle, "hipblasDtrmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,int,double *,int) nogil> hipblasDtrmv_funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* hipblasCtrmv_funptr = NULL
cdef hipblasStatus_t hipblasCtrmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasCtrmv_funptr
    if hipblasCtrmv_funptr == NULL:
        with gil:
            hipblasCtrmv_funptr = loader.load_symbol(_lib_handle, "hipblasCtrmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCtrmv_funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* hipblasZtrmv_funptr = NULL
cdef hipblasStatus_t hipblasZtrmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasZtrmv_funptr
    if hipblasZtrmv_funptr == NULL:
        with gil:
            hipblasZtrmv_funptr = loader.load_symbol(_lib_handle, "hipblasZtrmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZtrmv_funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* hipblasStrsv_funptr = NULL
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
cdef hipblasStatus_t hipblasStrsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,int lda,float * x,int incx) nogil:
    global _lib_handle
    global hipblasStrsv_funptr
    if hipblasStrsv_funptr == NULL:
        with gil:
            hipblasStrsv_funptr = loader.load_symbol(_lib_handle, "hipblasStrsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,int,float *,int) nogil> hipblasStrsv_funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* hipblasDtrsv_funptr = NULL
cdef hipblasStatus_t hipblasDtrsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,double * x,int incx) nogil:
    global _lib_handle
    global hipblasDtrsv_funptr
    if hipblasDtrsv_funptr == NULL:
        with gil:
            hipblasDtrsv_funptr = loader.load_symbol(_lib_handle, "hipblasDtrsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,int,double *,int) nogil> hipblasDtrsv_funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* hipblasCtrsv_funptr = NULL
cdef hipblasStatus_t hipblasCtrsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasCtrsv_funptr
    if hipblasCtrsv_funptr == NULL:
        with gil:
            hipblasCtrsv_funptr = loader.load_symbol(_lib_handle, "hipblasCtrsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCtrsv_funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* hipblasZtrsv_funptr = NULL
cdef hipblasStatus_t hipblasZtrsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global hipblasZtrsv_funptr
    if hipblasZtrsv_funptr == NULL:
        with gil:
            hipblasZtrsv_funptr = loader.load_symbol(_lib_handle, "hipblasZtrsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZtrsv_funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* hipblasHgemm_funptr = NULL
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
cdef hipblasStatus_t hipblasHgemm(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasHalf * alpha,hipblasHalf * AP,int lda,hipblasHalf * BP,int ldb,hipblasHalf * beta,hipblasHalf * CP,int ldc) nogil:
    global _lib_handle
    global hipblasHgemm_funptr
    if hipblasHgemm_funptr == NULL:
        with gil:
            hipblasHgemm_funptr = loader.load_symbol(_lib_handle, "hipblasHgemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,hipblasHalf *,hipblasHalf *,int,hipblasHalf *,int,hipblasHalf *,hipblasHalf *,int) nogil> hipblasHgemm_funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasSgemm_funptr = NULL
cdef hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil:
    global _lib_handle
    global hipblasSgemm_funptr
    if hipblasSgemm_funptr == NULL:
        with gil:
            hipblasSgemm_funptr = loader.load_symbol(_lib_handle, "hipblasSgemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> hipblasSgemm_funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasDgemm_funptr = NULL
cdef hipblasStatus_t hipblasDgemm(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil:
    global _lib_handle
    global hipblasDgemm_funptr
    if hipblasDgemm_funptr == NULL:
        with gil:
            hipblasDgemm_funptr = loader.load_symbol(_lib_handle, "hipblasDgemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> hipblasDgemm_funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasCgemm_funptr = NULL
cdef hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasCgemm_funptr
    if hipblasCgemm_funptr == NULL:
        with gil:
            hipblasCgemm_funptr = loader.load_symbol(_lib_handle, "hipblasCgemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasCgemm_funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasZgemm_funptr = NULL
cdef hipblasStatus_t hipblasZgemm(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasZgemm_funptr
    if hipblasZgemm_funptr == NULL:
        with gil:
            hipblasZgemm_funptr = loader.load_symbol(_lib_handle, "hipblasZgemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZgemm_funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasCherk_funptr = NULL
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
cdef hipblasStatus_t hipblasCherk(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,hipblasComplex * AP,int lda,const float * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasCherk_funptr
    if hipblasCherk_funptr == NULL:
        with gil:
            hipblasCherk_funptr = loader.load_symbol(_lib_handle, "hipblasCherk")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,hipblasComplex *,int,const float *,hipblasComplex *,int) nogil> hipblasCherk_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* hipblasZherk_funptr = NULL
cdef hipblasStatus_t hipblasZherk(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,hipblasDoubleComplex * AP,int lda,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasZherk_funptr
    if hipblasZherk_funptr == NULL:
        with gil:
            hipblasZherk_funptr = loader.load_symbol(_lib_handle, "hipblasZherk")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,hipblasDoubleComplex *,int,const double *,hipblasDoubleComplex *,int) nogil> hipblasZherk_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* hipblasCherkx_funptr = NULL
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
cdef hipblasStatus_t hipblasCherkx(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,const float * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasCherkx_funptr
    if hipblasCherkx_funptr == NULL:
        with gil:
            hipblasCherkx_funptr = loader.load_symbol(_lib_handle, "hipblasCherkx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,const float *,hipblasComplex *,int) nogil> hipblasCherkx_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasZherkx_funptr = NULL
cdef hipblasStatus_t hipblasZherkx(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasZherkx_funptr
    if hipblasZherkx_funptr == NULL:
        with gil:
            hipblasZherkx_funptr = loader.load_symbol(_lib_handle, "hipblasZherkx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,const double *,hipblasDoubleComplex *,int) nogil> hipblasZherkx_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasCher2k_funptr = NULL
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
cdef hipblasStatus_t hipblasCher2k(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,const float * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasCher2k_funptr
    if hipblasCher2k_funptr == NULL:
        with gil:
            hipblasCher2k_funptr = loader.load_symbol(_lib_handle, "hipblasCher2k")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,const float *,hipblasComplex *,int) nogil> hipblasCher2k_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasZher2k_funptr = NULL
cdef hipblasStatus_t hipblasZher2k(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasZher2k_funptr
    if hipblasZher2k_funptr == NULL:
        with gil:
            hipblasZher2k_funptr = loader.load_symbol(_lib_handle, "hipblasZher2k")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,const double *,hipblasDoubleComplex *,int) nogil> hipblasZher2k_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasSsymm_funptr = NULL
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
cdef hipblasStatus_t hipblasSsymm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil:
    global _lib_handle
    global hipblasSsymm_funptr
    if hipblasSsymm_funptr == NULL:
        with gil:
            hipblasSsymm_funptr = loader.load_symbol(_lib_handle, "hipblasSsymm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> hipblasSsymm_funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasDsymm_funptr = NULL
cdef hipblasStatus_t hipblasDsymm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil:
    global _lib_handle
    global hipblasDsymm_funptr
    if hipblasDsymm_funptr == NULL:
        with gil:
            hipblasDsymm_funptr = loader.load_symbol(_lib_handle, "hipblasDsymm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> hipblasDsymm_funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasCsymm_funptr = NULL
cdef hipblasStatus_t hipblasCsymm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasCsymm_funptr
    if hipblasCsymm_funptr == NULL:
        with gil:
            hipblasCsymm_funptr = loader.load_symbol(_lib_handle, "hipblasCsymm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasCsymm_funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasZsymm_funptr = NULL
cdef hipblasStatus_t hipblasZsymm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasZsymm_funptr
    if hipblasZsymm_funptr == NULL:
        with gil:
            hipblasZsymm_funptr = loader.load_symbol(_lib_handle, "hipblasZsymm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZsymm_funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasSsyrk_funptr = NULL
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
cdef hipblasStatus_t hipblasSsyrk(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,const float * beta,float * CP,int ldc) nogil:
    global _lib_handle
    global hipblasSsyrk_funptr
    if hipblasSsyrk_funptr == NULL:
        with gil:
            hipblasSsyrk_funptr = loader.load_symbol(_lib_handle, "hipblasSsyrk")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,float *,int) nogil> hipblasSsyrk_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* hipblasDsyrk_funptr = NULL
cdef hipblasStatus_t hipblasDsyrk(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * beta,double * CP,int ldc) nogil:
    global _lib_handle
    global hipblasDsyrk_funptr
    if hipblasDsyrk_funptr == NULL:
        with gil:
            hipblasDsyrk_funptr = loader.load_symbol(_lib_handle, "hipblasDsyrk")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,double *,int) nogil> hipblasDsyrk_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* hipblasCsyrk_funptr = NULL
cdef hipblasStatus_t hipblasCsyrk(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasCsyrk_funptr
    if hipblasCsyrk_funptr == NULL:
        with gil:
            hipblasCsyrk_funptr = loader.load_symbol(_lib_handle, "hipblasCsyrk")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasCsyrk_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* hipblasZsyrk_funptr = NULL
cdef hipblasStatus_t hipblasZsyrk(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasZsyrk_funptr
    if hipblasZsyrk_funptr == NULL:
        with gil:
            hipblasZsyrk_funptr = loader.load_symbol(_lib_handle, "hipblasZsyrk")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZsyrk_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* hipblasSsyr2k_funptr = NULL
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
cdef hipblasStatus_t hipblasSsyr2k(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil:
    global _lib_handle
    global hipblasSsyr2k_funptr
    if hipblasSsyr2k_funptr == NULL:
        with gil:
            hipblasSsyr2k_funptr = loader.load_symbol(_lib_handle, "hipblasSsyr2k")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> hipblasSsyr2k_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasDsyr2k_funptr = NULL
cdef hipblasStatus_t hipblasDsyr2k(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil:
    global _lib_handle
    global hipblasDsyr2k_funptr
    if hipblasDsyr2k_funptr == NULL:
        with gil:
            hipblasDsyr2k_funptr = loader.load_symbol(_lib_handle, "hipblasDsyr2k")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> hipblasDsyr2k_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasCsyr2k_funptr = NULL
cdef hipblasStatus_t hipblasCsyr2k(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasCsyr2k_funptr
    if hipblasCsyr2k_funptr == NULL:
        with gil:
            hipblasCsyr2k_funptr = loader.load_symbol(_lib_handle, "hipblasCsyr2k")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasCsyr2k_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasZsyr2k_funptr = NULL
cdef hipblasStatus_t hipblasZsyr2k(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasZsyr2k_funptr
    if hipblasZsyr2k_funptr == NULL:
        with gil:
            hipblasZsyr2k_funptr = loader.load_symbol(_lib_handle, "hipblasZsyr2k")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZsyr2k_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasSsyrkx_funptr = NULL
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
cdef hipblasStatus_t hipblasSsyrkx(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil:
    global _lib_handle
    global hipblasSsyrkx_funptr
    if hipblasSsyrkx_funptr == NULL:
        with gil:
            hipblasSsyrkx_funptr = loader.load_symbol(_lib_handle, "hipblasSsyrkx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> hipblasSsyrkx_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasDsyrkx_funptr = NULL
cdef hipblasStatus_t hipblasDsyrkx(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil:
    global _lib_handle
    global hipblasDsyrkx_funptr
    if hipblasDsyrkx_funptr == NULL:
        with gil:
            hipblasDsyrkx_funptr = loader.load_symbol(_lib_handle, "hipblasDsyrkx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> hipblasDsyrkx_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasCsyrkx_funptr = NULL
cdef hipblasStatus_t hipblasCsyrkx(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasCsyrkx_funptr
    if hipblasCsyrkx_funptr == NULL:
        with gil:
            hipblasCsyrkx_funptr = loader.load_symbol(_lib_handle, "hipblasCsyrkx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasCsyrkx_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasZsyrkx_funptr = NULL
cdef hipblasStatus_t hipblasZsyrkx(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasZsyrkx_funptr
    if hipblasZsyrkx_funptr == NULL:
        with gil:
            hipblasZsyrkx_funptr = loader.load_symbol(_lib_handle, "hipblasZsyrkx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZsyrkx_funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasSgeam_funptr = NULL
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
cdef hipblasStatus_t hipblasSgeam(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const float * alpha,const float * AP,int lda,const float * beta,const float * BP,int ldb,float * CP,int ldc) nogil:
    global _lib_handle
    global hipblasSgeam_funptr
    if hipblasSgeam_funptr == NULL:
        with gil:
            hipblasSgeam_funptr = loader.load_symbol(_lib_handle, "hipblasSgeam")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,const float *,int,float *,int) nogil> hipblasSgeam_funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc)


cdef void* hipblasDgeam_funptr = NULL
cdef hipblasStatus_t hipblasDgeam(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const double * alpha,const double * AP,int lda,const double * beta,const double * BP,int ldb,double * CP,int ldc) nogil:
    global _lib_handle
    global hipblasDgeam_funptr
    if hipblasDgeam_funptr == NULL:
        with gil:
            hipblasDgeam_funptr = loader.load_symbol(_lib_handle, "hipblasDgeam")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,const double *,int,double *,int) nogil> hipblasDgeam_funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc)


cdef void* hipblasCgeam_funptr = NULL
cdef hipblasStatus_t hipblasCgeam(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * beta,hipblasComplex * BP,int ldb,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasCgeam_funptr
    if hipblasCgeam_funptr == NULL:
        with gil:
            hipblasCgeam_funptr = loader.load_symbol(_lib_handle, "hipblasCgeam")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCgeam_funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc)


cdef void* hipblasZgeam_funptr = NULL
cdef hipblasStatus_t hipblasZgeam(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasZgeam_funptr
    if hipblasZgeam_funptr == NULL:
        with gil:
            hipblasZgeam_funptr = loader.load_symbol(_lib_handle, "hipblasZgeam")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZgeam_funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc)


cdef void* hipblasChemm_funptr = NULL
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
cdef hipblasStatus_t hipblasChemm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasChemm_funptr
    if hipblasChemm_funptr == NULL:
        with gil:
            hipblasChemm_funptr = loader.load_symbol(_lib_handle, "hipblasChemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> hipblasChemm_funptr)(handle,side,uplo,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasZhemm_funptr = NULL
cdef hipblasStatus_t hipblasZhemm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasZhemm_funptr
    if hipblasZhemm_funptr == NULL:
        with gil:
            hipblasZhemm_funptr = loader.load_symbol(_lib_handle, "hipblasZhemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> hipblasZhemm_funptr)(handle,side,uplo,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* hipblasStrmm_funptr = NULL
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
cdef hipblasStatus_t hipblasStrmm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,const float * AP,int lda,float * BP,int ldb) nogil:
    global _lib_handle
    global hipblasStrmm_funptr
    if hipblasStrmm_funptr == NULL:
        with gil:
            hipblasStrmm_funptr = loader.load_symbol(_lib_handle, "hipblasStrmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,const float *,int,float *,int) nogil> hipblasStrmm_funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* hipblasDtrmm_funptr = NULL
cdef hipblasStatus_t hipblasDtrmm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,const double * AP,int lda,double * BP,int ldb) nogil:
    global _lib_handle
    global hipblasDtrmm_funptr
    if hipblasDtrmm_funptr == NULL:
        with gil:
            hipblasDtrmm_funptr = loader.load_symbol(_lib_handle, "hipblasDtrmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,const double *,int,double *,int) nogil> hipblasDtrmm_funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* hipblasCtrmm_funptr = NULL
cdef hipblasStatus_t hipblasCtrmm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb) nogil:
    global _lib_handle
    global hipblasCtrmm_funptr
    if hipblasCtrmm_funptr == NULL:
        with gil:
            hipblasCtrmm_funptr = loader.load_symbol(_lib_handle, "hipblasCtrmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCtrmm_funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* hipblasZtrmm_funptr = NULL
cdef hipblasStatus_t hipblasZtrmm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb) nogil:
    global _lib_handle
    global hipblasZtrmm_funptr
    if hipblasZtrmm_funptr == NULL:
        with gil:
            hipblasZtrmm_funptr = loader.load_symbol(_lib_handle, "hipblasZtrmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZtrmm_funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* hipblasStrsm_funptr = NULL
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
cdef hipblasStatus_t hipblasStrsm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,float * AP,int lda,float * BP,int ldb) nogil:
    global _lib_handle
    global hipblasStrsm_funptr
    if hipblasStrsm_funptr == NULL:
        with gil:
            hipblasStrsm_funptr = loader.load_symbol(_lib_handle, "hipblasStrsm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,float *,int,float *,int) nogil> hipblasStrsm_funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* hipblasDtrsm_funptr = NULL
cdef hipblasStatus_t hipblasDtrsm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,double * AP,int lda,double * BP,int ldb) nogil:
    global _lib_handle
    global hipblasDtrsm_funptr
    if hipblasDtrsm_funptr == NULL:
        with gil:
            hipblasDtrsm_funptr = loader.load_symbol(_lib_handle, "hipblasDtrsm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,double *,int,double *,int) nogil> hipblasDtrsm_funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* hipblasCtrsm_funptr = NULL
cdef hipblasStatus_t hipblasCtrsm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb) nogil:
    global _lib_handle
    global hipblasCtrsm_funptr
    if hipblasCtrsm_funptr == NULL:
        with gil:
            hipblasCtrsm_funptr = loader.load_symbol(_lib_handle, "hipblasCtrsm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCtrsm_funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* hipblasZtrsm_funptr = NULL
cdef hipblasStatus_t hipblasZtrsm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb) nogil:
    global _lib_handle
    global hipblasZtrsm_funptr
    if hipblasZtrsm_funptr == NULL:
        with gil:
            hipblasZtrsm_funptr = loader.load_symbol(_lib_handle, "hipblasZtrsm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZtrsm_funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* hipblasStrtri_funptr = NULL
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
cdef hipblasStatus_t hipblasStrtri(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const float * AP,int lda,float * invA,int ldinvA) nogil:
    global _lib_handle
    global hipblasStrtri_funptr
    if hipblasStrtri_funptr == NULL:
        with gil:
            hipblasStrtri_funptr = loader.load_symbol(_lib_handle, "hipblasStrtri")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasDiagType_t,int,const float *,int,float *,int) nogil> hipblasStrtri_funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA)


cdef void* hipblasDtrtri_funptr = NULL
cdef hipblasStatus_t hipblasDtrtri(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const double * AP,int lda,double * invA,int ldinvA) nogil:
    global _lib_handle
    global hipblasDtrtri_funptr
    if hipblasDtrtri_funptr == NULL:
        with gil:
            hipblasDtrtri_funptr = loader.load_symbol(_lib_handle, "hipblasDtrtri")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasDiagType_t,int,const double *,int,double *,int) nogil> hipblasDtrtri_funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA)


cdef void* hipblasCtrtri_funptr = NULL
cdef hipblasStatus_t hipblasCtrtri(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasComplex * AP,int lda,hipblasComplex * invA,int ldinvA) nogil:
    global _lib_handle
    global hipblasCtrtri_funptr
    if hipblasCtrtri_funptr == NULL:
        with gil:
            hipblasCtrtri_funptr = loader.load_symbol(_lib_handle, "hipblasCtrtri")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasDiagType_t,int,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCtrtri_funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA)


cdef void* hipblasZtrtri_funptr = NULL
cdef hipblasStatus_t hipblasZtrtri(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * invA,int ldinvA) nogil:
    global _lib_handle
    global hipblasZtrtri_funptr
    if hipblasZtrtri_funptr == NULL:
        with gil:
            hipblasZtrtri_funptr = loader.load_symbol(_lib_handle, "hipblasZtrtri")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasDiagType_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZtrtri_funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA)


cdef void* hipblasSdgmm_funptr = NULL
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
cdef hipblasStatus_t hipblasSdgmm(hipblasHandle_t handle,hipblasSideMode_t side,int m,int n,const float * AP,int lda,const float * x,int incx,float * CP,int ldc) nogil:
    global _lib_handle
    global hipblasSdgmm_funptr
    if hipblasSdgmm_funptr == NULL:
        with gil:
            hipblasSdgmm_funptr = loader.load_symbol(_lib_handle, "hipblasSdgmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,int,int,const float *,int,const float *,int,float *,int) nogil> hipblasSdgmm_funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc)


cdef void* hipblasDdgmm_funptr = NULL
cdef hipblasStatus_t hipblasDdgmm(hipblasHandle_t handle,hipblasSideMode_t side,int m,int n,const double * AP,int lda,const double * x,int incx,double * CP,int ldc) nogil:
    global _lib_handle
    global hipblasDdgmm_funptr
    if hipblasDdgmm_funptr == NULL:
        with gil:
            hipblasDdgmm_funptr = loader.load_symbol(_lib_handle, "hipblasDdgmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,int,int,const double *,int,const double *,int,double *,int) nogil> hipblasDdgmm_funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc)


cdef void* hipblasCdgmm_funptr = NULL
cdef hipblasStatus_t hipblasCdgmm(hipblasHandle_t handle,hipblasSideMode_t side,int m,int n,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasCdgmm_funptr
    if hipblasCdgmm_funptr == NULL:
        with gil:
            hipblasCdgmm_funptr = loader.load_symbol(_lib_handle, "hipblasCdgmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,int,int,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> hipblasCdgmm_funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc)


cdef void* hipblasZdgmm_funptr = NULL
cdef hipblasStatus_t hipblasZdgmm(hipblasHandle_t handle,hipblasSideMode_t side,int m,int n,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global hipblasZdgmm_funptr
    if hipblasZdgmm_funptr == NULL:
        with gil:
            hipblasZdgmm_funptr = loader.load_symbol(_lib_handle, "hipblasZdgmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,int,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> hipblasZdgmm_funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc)


cdef void* hipblasSgetrf_funptr = NULL
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
cdef hipblasStatus_t hipblasSgetrf(hipblasHandle_t handle,const int n,float * A,const int lda,int * ipiv,int * info) nogil:
    global _lib_handle
    global hipblasSgetrf_funptr
    if hipblasSgetrf_funptr == NULL:
        with gil:
            hipblasSgetrf_funptr = loader.load_symbol(_lib_handle, "hipblasSgetrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,float *,const int,int *,int *) nogil> hipblasSgetrf_funptr)(handle,n,A,lda,ipiv,info)


cdef void* hipblasDgetrf_funptr = NULL
cdef hipblasStatus_t hipblasDgetrf(hipblasHandle_t handle,const int n,double * A,const int lda,int * ipiv,int * info) nogil:
    global _lib_handle
    global hipblasDgetrf_funptr
    if hipblasDgetrf_funptr == NULL:
        with gil:
            hipblasDgetrf_funptr = loader.load_symbol(_lib_handle, "hipblasDgetrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,double *,const int,int *,int *) nogil> hipblasDgetrf_funptr)(handle,n,A,lda,ipiv,info)


cdef void* hipblasCgetrf_funptr = NULL
cdef hipblasStatus_t hipblasCgetrf(hipblasHandle_t handle,const int n,hipblasComplex * A,const int lda,int * ipiv,int * info) nogil:
    global _lib_handle
    global hipblasCgetrf_funptr
    if hipblasCgetrf_funptr == NULL:
        with gil:
            hipblasCgetrf_funptr = loader.load_symbol(_lib_handle, "hipblasCgetrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,hipblasComplex *,const int,int *,int *) nogil> hipblasCgetrf_funptr)(handle,n,A,lda,ipiv,info)


cdef void* hipblasZgetrf_funptr = NULL
cdef hipblasStatus_t hipblasZgetrf(hipblasHandle_t handle,const int n,hipblasDoubleComplex * A,const int lda,int * ipiv,int * info) nogil:
    global _lib_handle
    global hipblasZgetrf_funptr
    if hipblasZgetrf_funptr == NULL:
        with gil:
            hipblasZgetrf_funptr = loader.load_symbol(_lib_handle, "hipblasZgetrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,hipblasDoubleComplex *,const int,int *,int *) nogil> hipblasZgetrf_funptr)(handle,n,A,lda,ipiv,info)


cdef void* hipblasSgetrs_funptr = NULL
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
cdef hipblasStatus_t hipblasSgetrs(hipblasHandle_t handle,hipblasOperation_t trans,const int n,const int nrhs,float * A,const int lda,const int * ipiv,float * B,const int ldb,int * info) nogil:
    global _lib_handle
    global hipblasSgetrs_funptr
    if hipblasSgetrs_funptr == NULL:
        with gil:
            hipblasSgetrs_funptr = loader.load_symbol(_lib_handle, "hipblasSgetrs")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,float *,const int,const int *,float *,const int,int *) nogil> hipblasSgetrs_funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info)


cdef void* hipblasDgetrs_funptr = NULL
cdef hipblasStatus_t hipblasDgetrs(hipblasHandle_t handle,hipblasOperation_t trans,const int n,const int nrhs,double * A,const int lda,const int * ipiv,double * B,const int ldb,int * info) nogil:
    global _lib_handle
    global hipblasDgetrs_funptr
    if hipblasDgetrs_funptr == NULL:
        with gil:
            hipblasDgetrs_funptr = loader.load_symbol(_lib_handle, "hipblasDgetrs")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,double *,const int,const int *,double *,const int,int *) nogil> hipblasDgetrs_funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info)


cdef void* hipblasCgetrs_funptr = NULL
cdef hipblasStatus_t hipblasCgetrs(hipblasHandle_t handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasComplex * A,const int lda,const int * ipiv,hipblasComplex * B,const int ldb,int * info) nogil:
    global _lib_handle
    global hipblasCgetrs_funptr
    if hipblasCgetrs_funptr == NULL:
        with gil:
            hipblasCgetrs_funptr = loader.load_symbol(_lib_handle, "hipblasCgetrs")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,hipblasComplex *,const int,const int *,hipblasComplex *,const int,int *) nogil> hipblasCgetrs_funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info)


cdef void* hipblasZgetrs_funptr = NULL
cdef hipblasStatus_t hipblasZgetrs(hipblasHandle_t handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,const int * ipiv,hipblasDoubleComplex * B,const int ldb,int * info) nogil:
    global _lib_handle
    global hipblasZgetrs_funptr
    if hipblasZgetrs_funptr == NULL:
        with gil:
            hipblasZgetrs_funptr = loader.load_symbol(_lib_handle, "hipblasZgetrs")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,hipblasDoubleComplex *,const int,const int *,hipblasDoubleComplex *,const int,int *) nogil> hipblasZgetrs_funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info)


cdef void* hipblasSgels_funptr = NULL
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
cdef hipblasStatus_t hipblasSgels(hipblasHandle_t handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,float * A,const int lda,float * B,const int ldb,int * info,int * deviceInfo) nogil:
    global _lib_handle
    global hipblasSgels_funptr
    if hipblasSgels_funptr == NULL:
        with gil:
            hipblasSgels_funptr = loader.load_symbol(_lib_handle, "hipblasSgels")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,const int,float *,const int,float *,const int,int *,int *) nogil> hipblasSgels_funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo)


cdef void* hipblasDgels_funptr = NULL
cdef hipblasStatus_t hipblasDgels(hipblasHandle_t handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,double * A,const int lda,double * B,const int ldb,int * info,int * deviceInfo) nogil:
    global _lib_handle
    global hipblasDgels_funptr
    if hipblasDgels_funptr == NULL:
        with gil:
            hipblasDgels_funptr = loader.load_symbol(_lib_handle, "hipblasDgels")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,const int,double *,const int,double *,const int,int *,int *) nogil> hipblasDgels_funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo)


cdef void* hipblasCgels_funptr = NULL
cdef hipblasStatus_t hipblasCgels(hipblasHandle_t handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasComplex * A,const int lda,hipblasComplex * B,const int ldb,int * info,int * deviceInfo) nogil:
    global _lib_handle
    global hipblasCgels_funptr
    if hipblasCgels_funptr == NULL:
        with gil:
            hipblasCgels_funptr = loader.load_symbol(_lib_handle, "hipblasCgels")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,const int,hipblasComplex *,const int,hipblasComplex *,const int,int *,int *) nogil> hipblasCgels_funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo)


cdef void* hipblasZgels_funptr = NULL
cdef hipblasStatus_t hipblasZgels(hipblasHandle_t handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,hipblasDoubleComplex * B,const int ldb,int * info,int * deviceInfo) nogil:
    global _lib_handle
    global hipblasZgels_funptr
    if hipblasZgels_funptr == NULL:
        with gil:
            hipblasZgels_funptr = loader.load_symbol(_lib_handle, "hipblasZgels")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,const int,hipblasDoubleComplex *,const int,hipblasDoubleComplex *,const int,int *,int *) nogil> hipblasZgels_funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo)


cdef void* hipblasSgeqrf_funptr = NULL
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
cdef hipblasStatus_t hipblasSgeqrf(hipblasHandle_t handle,const int m,const int n,float * A,const int lda,float * ipiv,int * info) nogil:
    global _lib_handle
    global hipblasSgeqrf_funptr
    if hipblasSgeqrf_funptr == NULL:
        with gil:
            hipblasSgeqrf_funptr = loader.load_symbol(_lib_handle, "hipblasSgeqrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,const int,float *,const int,float *,int *) nogil> hipblasSgeqrf_funptr)(handle,m,n,A,lda,ipiv,info)


cdef void* hipblasDgeqrf_funptr = NULL
cdef hipblasStatus_t hipblasDgeqrf(hipblasHandle_t handle,const int m,const int n,double * A,const int lda,double * ipiv,int * info) nogil:
    global _lib_handle
    global hipblasDgeqrf_funptr
    if hipblasDgeqrf_funptr == NULL:
        with gil:
            hipblasDgeqrf_funptr = loader.load_symbol(_lib_handle, "hipblasDgeqrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,const int,double *,const int,double *,int *) nogil> hipblasDgeqrf_funptr)(handle,m,n,A,lda,ipiv,info)


cdef void* hipblasCgeqrf_funptr = NULL
cdef hipblasStatus_t hipblasCgeqrf(hipblasHandle_t handle,const int m,const int n,hipblasComplex * A,const int lda,hipblasComplex * ipiv,int * info) nogil:
    global _lib_handle
    global hipblasCgeqrf_funptr
    if hipblasCgeqrf_funptr == NULL:
        with gil:
            hipblasCgeqrf_funptr = loader.load_symbol(_lib_handle, "hipblasCgeqrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,const int,hipblasComplex *,const int,hipblasComplex *,int *) nogil> hipblasCgeqrf_funptr)(handle,m,n,A,lda,ipiv,info)


cdef void* hipblasZgeqrf_funptr = NULL
cdef hipblasStatus_t hipblasZgeqrf(hipblasHandle_t handle,const int m,const int n,hipblasDoubleComplex * A,const int lda,hipblasDoubleComplex * ipiv,int * info) nogil:
    global _lib_handle
    global hipblasZgeqrf_funptr
    if hipblasZgeqrf_funptr == NULL:
        with gil:
            hipblasZgeqrf_funptr = loader.load_symbol(_lib_handle, "hipblasZgeqrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,const int,hipblasDoubleComplex *,const int,hipblasDoubleComplex *,int *) nogil> hipblasZgeqrf_funptr)(handle,m,n,A,lda,ipiv,info)


cdef void* hipblasGemmEx_funptr = NULL
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
cdef hipblasStatus_t hipblasGemmEx(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const void * alpha,const void * A,hipblasDatatype_t aType,int lda,const void * B,hipblasDatatype_t bType,int ldb,const void * beta,void * C,hipblasDatatype_t cType,int ldc,hipblasDatatype_t computeType,hipblasGemmAlgo_t algo) nogil:
    global _lib_handle
    global hipblasGemmEx_funptr
    if hipblasGemmEx_funptr == NULL:
        with gil:
            hipblasGemmEx_funptr = loader.load_symbol(_lib_handle, "hipblasGemmEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,const void *,const void *,hipblasDatatype_t,int,const void *,hipblasDatatype_t,int,const void *,void *,hipblasDatatype_t,int,hipblasDatatype_t,hipblasGemmAlgo_t) nogil> hipblasGemmEx_funptr)(handle,transA,transB,m,n,k,alpha,A,aType,lda,B,bType,ldb,beta,C,cType,ldc,computeType,algo)


cdef void* hipblasTrsmEx_funptr = NULL
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
cdef hipblasStatus_t hipblasTrsmEx(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const void * alpha,void * A,int lda,void * B,int ldb,const void * invA,int invAsize,hipblasDatatype_t computeType) nogil:
    global _lib_handle
    global hipblasTrsmEx_funptr
    if hipblasTrsmEx_funptr == NULL:
        with gil:
            hipblasTrsmEx_funptr = loader.load_symbol(_lib_handle, "hipblasTrsmEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const void *,void *,int,void *,int,const void *,int,hipblasDatatype_t) nogil> hipblasTrsmEx_funptr)(handle,side,uplo,transA,diag,m,n,alpha,A,lda,B,ldb,invA,invAsize,computeType)


cdef void* hipblasAxpyEx_funptr = NULL
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
cdef hipblasStatus_t hipblasAxpyEx(hipblasHandle_t handle,int n,const void * alpha,hipblasDatatype_t alphaType,const void * x,hipblasDatatype_t xType,int incx,void * y,hipblasDatatype_t yType,int incy,hipblasDatatype_t executionType) nogil:
    global _lib_handle
    global hipblasAxpyEx_funptr
    if hipblasAxpyEx_funptr == NULL:
        with gil:
            hipblasAxpyEx_funptr = loader.load_symbol(_lib_handle, "hipblasAxpyEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const void *,hipblasDatatype_t,const void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,int,hipblasDatatype_t) nogil> hipblasAxpyEx_funptr)(handle,n,alpha,alphaType,x,xType,incx,y,yType,incy,executionType)


cdef void* hipblasDotEx_funptr = NULL
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
cdef hipblasStatus_t hipblasDotEx(hipblasHandle_t handle,int n,const void * x,hipblasDatatype_t xType,int incx,const void * y,hipblasDatatype_t yType,int incy,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil:
    global _lib_handle
    global hipblasDotEx_funptr
    if hipblasDotEx_funptr == NULL:
        with gil:
            hipblasDotEx_funptr = loader.load_symbol(_lib_handle, "hipblasDotEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const void *,hipblasDatatype_t,int,const void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> hipblasDotEx_funptr)(handle,n,x,xType,incx,y,yType,incy,result,resultType,executionType)


cdef void* hipblasDotcEx_funptr = NULL
cdef hipblasStatus_t hipblasDotcEx(hipblasHandle_t handle,int n,const void * x,hipblasDatatype_t xType,int incx,const void * y,hipblasDatatype_t yType,int incy,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil:
    global _lib_handle
    global hipblasDotcEx_funptr
    if hipblasDotcEx_funptr == NULL:
        with gil:
            hipblasDotcEx_funptr = loader.load_symbol(_lib_handle, "hipblasDotcEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const void *,hipblasDatatype_t,int,const void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> hipblasDotcEx_funptr)(handle,n,x,xType,incx,y,yType,incy,result,resultType,executionType)


cdef void* hipblasNrm2Ex_funptr = NULL
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
cdef hipblasStatus_t hipblasNrm2Ex(hipblasHandle_t handle,int n,const void * x,hipblasDatatype_t xType,int incx,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil:
    global _lib_handle
    global hipblasNrm2Ex_funptr
    if hipblasNrm2Ex_funptr == NULL:
        with gil:
            hipblasNrm2Ex_funptr = loader.load_symbol(_lib_handle, "hipblasNrm2Ex")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> hipblasNrm2Ex_funptr)(handle,n,x,xType,incx,result,resultType,executionType)


cdef void* hipblasRotEx_funptr = NULL
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
cdef hipblasStatus_t hipblasRotEx(hipblasHandle_t handle,int n,void * x,hipblasDatatype_t xType,int incx,void * y,hipblasDatatype_t yType,int incy,const void * c,const void * s,hipblasDatatype_t csType,hipblasDatatype_t executionType) nogil:
    global _lib_handle
    global hipblasRotEx_funptr
    if hipblasRotEx_funptr == NULL:
        with gil:
            hipblasRotEx_funptr = loader.load_symbol(_lib_handle, "hipblasRotEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,int,const void *,const void *,hipblasDatatype_t,hipblasDatatype_t) nogil> hipblasRotEx_funptr)(handle,n,x,xType,incx,y,yType,incy,c,s,csType,executionType)


cdef void* hipblasScalEx_funptr = NULL
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
cdef hipblasStatus_t hipblasScalEx(hipblasHandle_t handle,int n,const void * alpha,hipblasDatatype_t alphaType,void * x,hipblasDatatype_t xType,int incx,hipblasDatatype_t executionType) nogil:
    global _lib_handle
    global hipblasScalEx_funptr
    if hipblasScalEx_funptr == NULL:
        with gil:
            hipblasScalEx_funptr = loader.load_symbol(_lib_handle, "hipblasScalEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const void *,hipblasDatatype_t,void *,hipblasDatatype_t,int,hipblasDatatype_t) nogil> hipblasScalEx_funptr)(handle,n,alpha,alphaType,x,xType,incx,executionType)


cdef void* hipblasStatusToString_funptr = NULL
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
cdef const char * hipblasStatusToString(hipblasStatus_t status) nogil:
    global _lib_handle
    global hipblasStatusToString_funptr
    if hipblasStatusToString_funptr == NULL:
        with gil:
            hipblasStatusToString_funptr = loader.load_symbol(_lib_handle, "hipblasStatusToString")
    return (<const char * (*)(hipblasStatus_t) nogil> hipblasStatusToString_funptr)(status)
