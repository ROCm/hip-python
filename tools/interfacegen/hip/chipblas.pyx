# AMD_COPYRIGHT
from libc.stdint cimport *
from .chip cimport hipStream_t

cimport hip._util.posixloader as loader


cdef void* _lib_handle = loader.open_library("libhipblas.so")


cdef void* _hipblasCreate__funptr = NULL
# ! \brief Create hipblas handle. */
cdef hipblasStatus_t hipblasCreate(hipblasHandle_t* handle) nogil:
    global _lib_handle
    global _hipblasCreate__funptr
    if _hipblasCreate__funptr == NULL:
        with gil:
            _hipblasCreate__funptr = loader.load_symbol(_lib_handle, "hipblasCreate")
    return (<hipblasStatus_t (*)(hipblasHandle_t*) nogil> _hipblasCreate__funptr)(handle)


cdef void* _hipblasDestroy__funptr = NULL
# ! \brief Destroys the library context created using hipblasCreate() */
cdef hipblasStatus_t hipblasDestroy(hipblasHandle_t handle) nogil:
    global _lib_handle
    global _hipblasDestroy__funptr
    if _hipblasDestroy__funptr == NULL:
        with gil:
            _hipblasDestroy__funptr = loader.load_symbol(_lib_handle, "hipblasDestroy")
    return (<hipblasStatus_t (*)(hipblasHandle_t) nogil> _hipblasDestroy__funptr)(handle)


cdef void* _hipblasSetStream__funptr = NULL
# ! \brief Set stream for handle */
cdef hipblasStatus_t hipblasSetStream(hipblasHandle_t handle,hipStream_t streamId) nogil:
    global _lib_handle
    global _hipblasSetStream__funptr
    if _hipblasSetStream__funptr == NULL:
        with gil:
            _hipblasSetStream__funptr = loader.load_symbol(_lib_handle, "hipblasSetStream")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipStream_t) nogil> _hipblasSetStream__funptr)(handle,streamId)


cdef void* _hipblasGetStream__funptr = NULL
# ! \brief Get stream[0] for handle */
cdef hipblasStatus_t hipblasGetStream(hipblasHandle_t handle,hipStream_t* streamId) nogil:
    global _lib_handle
    global _hipblasGetStream__funptr
    if _hipblasGetStream__funptr == NULL:
        with gil:
            _hipblasGetStream__funptr = loader.load_symbol(_lib_handle, "hipblasGetStream")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipStream_t*) nogil> _hipblasGetStream__funptr)(handle,streamId)


cdef void* _hipblasSetPointerMode__funptr = NULL
# ! \brief Set hipblas pointer mode */
cdef hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t handle,hipblasPointerMode_t mode) nogil:
    global _lib_handle
    global _hipblasSetPointerMode__funptr
    if _hipblasSetPointerMode__funptr == NULL:
        with gil:
            _hipblasSetPointerMode__funptr = loader.load_symbol(_lib_handle, "hipblasSetPointerMode")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasPointerMode_t) nogil> _hipblasSetPointerMode__funptr)(handle,mode)


cdef void* _hipblasGetPointerMode__funptr = NULL
# ! \brief Get hipblas pointer mode */
cdef hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t handle,hipblasPointerMode_t * mode) nogil:
    global _lib_handle
    global _hipblasGetPointerMode__funptr
    if _hipblasGetPointerMode__funptr == NULL:
        with gil:
            _hipblasGetPointerMode__funptr = loader.load_symbol(_lib_handle, "hipblasGetPointerMode")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasPointerMode_t *) nogil> _hipblasGetPointerMode__funptr)(handle,mode)


cdef void* _hipblasSetInt8Datatype__funptr = NULL
# ! \brief Set hipblas int8 Datatype */
cdef hipblasStatus_t hipblasSetInt8Datatype(hipblasHandle_t handle,hipblasInt8Datatype_t int8Type) nogil:
    global _lib_handle
    global _hipblasSetInt8Datatype__funptr
    if _hipblasSetInt8Datatype__funptr == NULL:
        with gil:
            _hipblasSetInt8Datatype__funptr = loader.load_symbol(_lib_handle, "hipblasSetInt8Datatype")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasInt8Datatype_t) nogil> _hipblasSetInt8Datatype__funptr)(handle,int8Type)


cdef void* _hipblasGetInt8Datatype__funptr = NULL
# ! \brief Get hipblas int8 Datatype*/
cdef hipblasStatus_t hipblasGetInt8Datatype(hipblasHandle_t handle,hipblasInt8Datatype_t * int8Type) nogil:
    global _lib_handle
    global _hipblasGetInt8Datatype__funptr
    if _hipblasGetInt8Datatype__funptr == NULL:
        with gil:
            _hipblasGetInt8Datatype__funptr = loader.load_symbol(_lib_handle, "hipblasGetInt8Datatype")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasInt8Datatype_t *) nogil> _hipblasGetInt8Datatype__funptr)(handle,int8Type)


cdef void* _hipblasSetVector__funptr = NULL
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
    global _hipblasSetVector__funptr
    if _hipblasSetVector__funptr == NULL:
        with gil:
            _hipblasSetVector__funptr = loader.load_symbol(_lib_handle, "hipblasSetVector")
    return (<hipblasStatus_t (*)(int,int,const void *,int,void *,int) nogil> _hipblasSetVector__funptr)(n,elemSize,x,incx,y,incy)


cdef void* _hipblasGetVector__funptr = NULL
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
    global _hipblasGetVector__funptr
    if _hipblasGetVector__funptr == NULL:
        with gil:
            _hipblasGetVector__funptr = loader.load_symbol(_lib_handle, "hipblasGetVector")
    return (<hipblasStatus_t (*)(int,int,const void *,int,void *,int) nogil> _hipblasGetVector__funptr)(n,elemSize,x,incx,y,incy)


cdef void* _hipblasSetMatrix__funptr = NULL
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
    global _hipblasSetMatrix__funptr
    if _hipblasSetMatrix__funptr == NULL:
        with gil:
            _hipblasSetMatrix__funptr = loader.load_symbol(_lib_handle, "hipblasSetMatrix")
    return (<hipblasStatus_t (*)(int,int,int,const void *,int,void *,int) nogil> _hipblasSetMatrix__funptr)(rows,cols,elemSize,AP,lda,BP,ldb)


cdef void* _hipblasGetMatrix__funptr = NULL
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
    global _hipblasGetMatrix__funptr
    if _hipblasGetMatrix__funptr == NULL:
        with gil:
            _hipblasGetMatrix__funptr = loader.load_symbol(_lib_handle, "hipblasGetMatrix")
    return (<hipblasStatus_t (*)(int,int,int,const void *,int,void *,int) nogil> _hipblasGetMatrix__funptr)(rows,cols,elemSize,AP,lda,BP,ldb)


cdef void* _hipblasSetVectorAsync__funptr = NULL
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
    global _hipblasSetVectorAsync__funptr
    if _hipblasSetVectorAsync__funptr == NULL:
        with gil:
            _hipblasSetVectorAsync__funptr = loader.load_symbol(_lib_handle, "hipblasSetVectorAsync")
    return (<hipblasStatus_t (*)(int,int,const void *,int,void *,int,hipStream_t) nogil> _hipblasSetVectorAsync__funptr)(n,elemSize,x,incx,y,incy,stream)


cdef void* _hipblasGetVectorAsync__funptr = NULL
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
    global _hipblasGetVectorAsync__funptr
    if _hipblasGetVectorAsync__funptr == NULL:
        with gil:
            _hipblasGetVectorAsync__funptr = loader.load_symbol(_lib_handle, "hipblasGetVectorAsync")
    return (<hipblasStatus_t (*)(int,int,const void *,int,void *,int,hipStream_t) nogil> _hipblasGetVectorAsync__funptr)(n,elemSize,x,incx,y,incy,stream)


cdef void* _hipblasSetMatrixAsync__funptr = NULL
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
    global _hipblasSetMatrixAsync__funptr
    if _hipblasSetMatrixAsync__funptr == NULL:
        with gil:
            _hipblasSetMatrixAsync__funptr = loader.load_symbol(_lib_handle, "hipblasSetMatrixAsync")
    return (<hipblasStatus_t (*)(int,int,int,const void *,int,void *,int,hipStream_t) nogil> _hipblasSetMatrixAsync__funptr)(rows,cols,elemSize,AP,lda,BP,ldb,stream)


cdef void* _hipblasGetMatrixAsync__funptr = NULL
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
    global _hipblasGetMatrixAsync__funptr
    if _hipblasGetMatrixAsync__funptr == NULL:
        with gil:
            _hipblasGetMatrixAsync__funptr = loader.load_symbol(_lib_handle, "hipblasGetMatrixAsync")
    return (<hipblasStatus_t (*)(int,int,int,const void *,int,void *,int,hipStream_t) nogil> _hipblasGetMatrixAsync__funptr)(rows,cols,elemSize,AP,lda,BP,ldb,stream)


cdef void* _hipblasSetAtomicsMode__funptr = NULL
# ! \brief Set hipblasSetAtomicsMode*/
cdef hipblasStatus_t hipblasSetAtomicsMode(hipblasHandle_t handle,hipblasAtomicsMode_t atomics_mode) nogil:
    global _lib_handle
    global _hipblasSetAtomicsMode__funptr
    if _hipblasSetAtomicsMode__funptr == NULL:
        with gil:
            _hipblasSetAtomicsMode__funptr = loader.load_symbol(_lib_handle, "hipblasSetAtomicsMode")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasAtomicsMode_t) nogil> _hipblasSetAtomicsMode__funptr)(handle,atomics_mode)


cdef void* _hipblasGetAtomicsMode__funptr = NULL
# ! \brief Get hipblasSetAtomicsMode*/
cdef hipblasStatus_t hipblasGetAtomicsMode(hipblasHandle_t handle,hipblasAtomicsMode_t * atomics_mode) nogil:
    global _lib_handle
    global _hipblasGetAtomicsMode__funptr
    if _hipblasGetAtomicsMode__funptr == NULL:
        with gil:
            _hipblasGetAtomicsMode__funptr = loader.load_symbol(_lib_handle, "hipblasGetAtomicsMode")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasAtomicsMode_t *) nogil> _hipblasGetAtomicsMode__funptr)(handle,atomics_mode)


cdef void* _hipblasIsamax__funptr = NULL
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
    global _hipblasIsamax__funptr
    if _hipblasIsamax__funptr == NULL:
        with gil:
            _hipblasIsamax__funptr = loader.load_symbol(_lib_handle, "hipblasIsamax")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,int,int *) nogil> _hipblasIsamax__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIdamax__funptr = NULL
cdef hipblasStatus_t hipblasIdamax(hipblasHandle_t handle,int n,const double * x,int incx,int * result) nogil:
    global _lib_handle
    global _hipblasIdamax__funptr
    if _hipblasIdamax__funptr == NULL:
        with gil:
            _hipblasIdamax__funptr = loader.load_symbol(_lib_handle, "hipblasIdamax")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,int,int *) nogil> _hipblasIdamax__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIcamax__funptr = NULL
cdef hipblasStatus_t hipblasIcamax(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,int * result) nogil:
    global _lib_handle
    global _hipblasIcamax__funptr
    if _hipblasIcamax__funptr == NULL:
        with gil:
            _hipblasIcamax__funptr = loader.load_symbol(_lib_handle, "hipblasIcamax")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,int *) nogil> _hipblasIcamax__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIzamax__funptr = NULL
cdef hipblasStatus_t hipblasIzamax(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,int * result) nogil:
    global _lib_handle
    global _hipblasIzamax__funptr
    if _hipblasIzamax__funptr == NULL:
        with gil:
            _hipblasIzamax__funptr = loader.load_symbol(_lib_handle, "hipblasIzamax")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,int *) nogil> _hipblasIzamax__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIsamin__funptr = NULL
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
    global _hipblasIsamin__funptr
    if _hipblasIsamin__funptr == NULL:
        with gil:
            _hipblasIsamin__funptr = loader.load_symbol(_lib_handle, "hipblasIsamin")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,int,int *) nogil> _hipblasIsamin__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIdamin__funptr = NULL
cdef hipblasStatus_t hipblasIdamin(hipblasHandle_t handle,int n,const double * x,int incx,int * result) nogil:
    global _lib_handle
    global _hipblasIdamin__funptr
    if _hipblasIdamin__funptr == NULL:
        with gil:
            _hipblasIdamin__funptr = loader.load_symbol(_lib_handle, "hipblasIdamin")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,int,int *) nogil> _hipblasIdamin__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIcamin__funptr = NULL
cdef hipblasStatus_t hipblasIcamin(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,int * result) nogil:
    global _lib_handle
    global _hipblasIcamin__funptr
    if _hipblasIcamin__funptr == NULL:
        with gil:
            _hipblasIcamin__funptr = loader.load_symbol(_lib_handle, "hipblasIcamin")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,int *) nogil> _hipblasIcamin__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIzamin__funptr = NULL
cdef hipblasStatus_t hipblasIzamin(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,int * result) nogil:
    global _lib_handle
    global _hipblasIzamin__funptr
    if _hipblasIzamin__funptr == NULL:
        with gil:
            _hipblasIzamin__funptr = loader.load_symbol(_lib_handle, "hipblasIzamin")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,int *) nogil> _hipblasIzamin__funptr)(handle,n,x,incx,result)


cdef void* _hipblasSasum__funptr = NULL
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
    global _hipblasSasum__funptr
    if _hipblasSasum__funptr == NULL:
        with gil:
            _hipblasSasum__funptr = loader.load_symbol(_lib_handle, "hipblasSasum")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,int,float *) nogil> _hipblasSasum__funptr)(handle,n,x,incx,result)


cdef void* _hipblasDasum__funptr = NULL
cdef hipblasStatus_t hipblasDasum(hipblasHandle_t handle,int n,const double * x,int incx,double * result) nogil:
    global _lib_handle
    global _hipblasDasum__funptr
    if _hipblasDasum__funptr == NULL:
        with gil:
            _hipblasDasum__funptr = loader.load_symbol(_lib_handle, "hipblasDasum")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,int,double *) nogil> _hipblasDasum__funptr)(handle,n,x,incx,result)


cdef void* _hipblasScasum__funptr = NULL
cdef hipblasStatus_t hipblasScasum(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,float * result) nogil:
    global _lib_handle
    global _hipblasScasum__funptr
    if _hipblasScasum__funptr == NULL:
        with gil:
            _hipblasScasum__funptr = loader.load_symbol(_lib_handle, "hipblasScasum")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,float *) nogil> _hipblasScasum__funptr)(handle,n,x,incx,result)


cdef void* _hipblasDzasum__funptr = NULL
cdef hipblasStatus_t hipblasDzasum(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,double * result) nogil:
    global _lib_handle
    global _hipblasDzasum__funptr
    if _hipblasDzasum__funptr == NULL:
        with gil:
            _hipblasDzasum__funptr = loader.load_symbol(_lib_handle, "hipblasDzasum")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,double *) nogil> _hipblasDzasum__funptr)(handle,n,x,incx,result)


cdef void* _hipblasHaxpy__funptr = NULL
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
    global _hipblasHaxpy__funptr
    if _hipblasHaxpy__funptr == NULL:
        with gil:
            _hipblasHaxpy__funptr = loader.load_symbol(_lib_handle, "hipblasHaxpy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasHalf *,hipblasHalf *,int,hipblasHalf *,int) nogil> _hipblasHaxpy__funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* _hipblasSaxpy__funptr = NULL
cdef hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle,int n,const float * alpha,const float * x,int incx,float * y,int incy) nogil:
    global _lib_handle
    global _hipblasSaxpy__funptr
    if _hipblasSaxpy__funptr == NULL:
        with gil:
            _hipblasSaxpy__funptr = loader.load_symbol(_lib_handle, "hipblasSaxpy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,const float *,int,float *,int) nogil> _hipblasSaxpy__funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* _hipblasDaxpy__funptr = NULL
cdef hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle,int n,const double * alpha,const double * x,int incx,double * y,int incy) nogil:
    global _lib_handle
    global _hipblasDaxpy__funptr
    if _hipblasDaxpy__funptr == NULL:
        with gil:
            _hipblasDaxpy__funptr = loader.load_symbol(_lib_handle, "hipblasDaxpy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,const double *,int,double *,int) nogil> _hipblasDaxpy__funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* _hipblasCaxpy__funptr = NULL
cdef hipblasStatus_t hipblasCaxpy(hipblasHandle_t handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasCaxpy__funptr
    if _hipblasCaxpy__funptr == NULL:
        with gil:
            _hipblasCaxpy__funptr = loader.load_symbol(_lib_handle, "hipblasCaxpy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCaxpy__funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* _hipblasZaxpy__funptr = NULL
cdef hipblasStatus_t hipblasZaxpy(hipblasHandle_t handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasZaxpy__funptr
    if _hipblasZaxpy__funptr == NULL:
        with gil:
            _hipblasZaxpy__funptr = loader.load_symbol(_lib_handle, "hipblasZaxpy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZaxpy__funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* _hipblasScopy__funptr = NULL
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
    global _hipblasScopy__funptr
    if _hipblasScopy__funptr == NULL:
        with gil:
            _hipblasScopy__funptr = loader.load_symbol(_lib_handle, "hipblasScopy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,int,float *,int) nogil> _hipblasScopy__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasDcopy__funptr = NULL
cdef hipblasStatus_t hipblasDcopy(hipblasHandle_t handle,int n,const double * x,int incx,double * y,int incy) nogil:
    global _lib_handle
    global _hipblasDcopy__funptr
    if _hipblasDcopy__funptr == NULL:
        with gil:
            _hipblasDcopy__funptr = loader.load_symbol(_lib_handle, "hipblasDcopy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,int,double *,int) nogil> _hipblasDcopy__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasCcopy__funptr = NULL
cdef hipblasStatus_t hipblasCcopy(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasCcopy__funptr
    if _hipblasCcopy__funptr == NULL:
        with gil:
            _hipblasCcopy__funptr = loader.load_symbol(_lib_handle, "hipblasCcopy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCcopy__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasZcopy__funptr = NULL
cdef hipblasStatus_t hipblasZcopy(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasZcopy__funptr
    if _hipblasZcopy__funptr == NULL:
        with gil:
            _hipblasZcopy__funptr = loader.load_symbol(_lib_handle, "hipblasZcopy")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZcopy__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasHdot__funptr = NULL
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
    global _hipblasHdot__funptr
    if _hipblasHdot__funptr == NULL:
        with gil:
            _hipblasHdot__funptr = loader.load_symbol(_lib_handle, "hipblasHdot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasHalf *,int,hipblasHalf *,int,hipblasHalf *) nogil> _hipblasHdot__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasBfdot__funptr = NULL
cdef hipblasStatus_t hipblasBfdot(hipblasHandle_t handle,int n,hipblasBfloat16 * x,int incx,hipblasBfloat16 * y,int incy,hipblasBfloat16 * result) nogil:
    global _lib_handle
    global _hipblasBfdot__funptr
    if _hipblasBfdot__funptr == NULL:
        with gil:
            _hipblasBfdot__funptr = loader.load_symbol(_lib_handle, "hipblasBfdot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasBfloat16 *,int,hipblasBfloat16 *,int,hipblasBfloat16 *) nogil> _hipblasBfdot__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasSdot__funptr = NULL
cdef hipblasStatus_t hipblasSdot(hipblasHandle_t handle,int n,const float * x,int incx,const float * y,int incy,float * result) nogil:
    global _lib_handle
    global _hipblasSdot__funptr
    if _hipblasSdot__funptr == NULL:
        with gil:
            _hipblasSdot__funptr = loader.load_symbol(_lib_handle, "hipblasSdot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,int,const float *,int,float *) nogil> _hipblasSdot__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasDdot__funptr = NULL
cdef hipblasStatus_t hipblasDdot(hipblasHandle_t handle,int n,const double * x,int incx,const double * y,int incy,double * result) nogil:
    global _lib_handle
    global _hipblasDdot__funptr
    if _hipblasDdot__funptr == NULL:
        with gil:
            _hipblasDdot__funptr = loader.load_symbol(_lib_handle, "hipblasDdot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,int,const double *,int,double *) nogil> _hipblasDdot__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasCdotc__funptr = NULL
cdef hipblasStatus_t hipblasCdotc(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * result) nogil:
    global _lib_handle
    global _hipblasCdotc__funptr
    if _hipblasCdotc__funptr == NULL:
        with gil:
            _hipblasCdotc__funptr = loader.load_symbol(_lib_handle, "hipblasCdotc")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *) nogil> _hipblasCdotc__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasCdotu__funptr = NULL
cdef hipblasStatus_t hipblasCdotu(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * result) nogil:
    global _lib_handle
    global _hipblasCdotu__funptr
    if _hipblasCdotu__funptr == NULL:
        with gil:
            _hipblasCdotu__funptr = loader.load_symbol(_lib_handle, "hipblasCdotu")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *) nogil> _hipblasCdotu__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasZdotc__funptr = NULL
cdef hipblasStatus_t hipblasZdotc(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * result) nogil:
    global _lib_handle
    global _hipblasZdotc__funptr
    if _hipblasZdotc__funptr == NULL:
        with gil:
            _hipblasZdotc__funptr = loader.load_symbol(_lib_handle, "hipblasZdotc")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> _hipblasZdotc__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasZdotu__funptr = NULL
cdef hipblasStatus_t hipblasZdotu(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * result) nogil:
    global _lib_handle
    global _hipblasZdotu__funptr
    if _hipblasZdotu__funptr == NULL:
        with gil:
            _hipblasZdotu__funptr = loader.load_symbol(_lib_handle, "hipblasZdotu")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> _hipblasZdotu__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasSnrm2__funptr = NULL
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
    global _hipblasSnrm2__funptr
    if _hipblasSnrm2__funptr == NULL:
        with gil:
            _hipblasSnrm2__funptr = loader.load_symbol(_lib_handle, "hipblasSnrm2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,int,float *) nogil> _hipblasSnrm2__funptr)(handle,n,x,incx,result)


cdef void* _hipblasDnrm2__funptr = NULL
cdef hipblasStatus_t hipblasDnrm2(hipblasHandle_t handle,int n,const double * x,int incx,double * result) nogil:
    global _lib_handle
    global _hipblasDnrm2__funptr
    if _hipblasDnrm2__funptr == NULL:
        with gil:
            _hipblasDnrm2__funptr = loader.load_symbol(_lib_handle, "hipblasDnrm2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,int,double *) nogil> _hipblasDnrm2__funptr)(handle,n,x,incx,result)


cdef void* _hipblasScnrm2__funptr = NULL
cdef hipblasStatus_t hipblasScnrm2(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,float * result) nogil:
    global _lib_handle
    global _hipblasScnrm2__funptr
    if _hipblasScnrm2__funptr == NULL:
        with gil:
            _hipblasScnrm2__funptr = loader.load_symbol(_lib_handle, "hipblasScnrm2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,float *) nogil> _hipblasScnrm2__funptr)(handle,n,x,incx,result)


cdef void* _hipblasDznrm2__funptr = NULL
cdef hipblasStatus_t hipblasDznrm2(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,double * result) nogil:
    global _lib_handle
    global _hipblasDznrm2__funptr
    if _hipblasDznrm2__funptr == NULL:
        with gil:
            _hipblasDznrm2__funptr = loader.load_symbol(_lib_handle, "hipblasDznrm2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,double *) nogil> _hipblasDznrm2__funptr)(handle,n,x,incx,result)


cdef void* _hipblasSrot__funptr = NULL
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
    global _hipblasSrot__funptr
    if _hipblasSrot__funptr == NULL:
        with gil:
            _hipblasSrot__funptr = loader.load_symbol(_lib_handle, "hipblasSrot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,float *,int,float *,int,const float *,const float *) nogil> _hipblasSrot__funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* _hipblasDrot__funptr = NULL
cdef hipblasStatus_t hipblasDrot(hipblasHandle_t handle,int n,double * x,int incx,double * y,int incy,const double * c,const double * s) nogil:
    global _lib_handle
    global _hipblasDrot__funptr
    if _hipblasDrot__funptr == NULL:
        with gil:
            _hipblasDrot__funptr = loader.load_symbol(_lib_handle, "hipblasDrot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,double *,int,double *,int,const double *,const double *) nogil> _hipblasDrot__funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* _hipblasCrot__funptr = NULL
cdef hipblasStatus_t hipblasCrot(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,const float * c,hipblasComplex * s) nogil:
    global _lib_handle
    global _hipblasCrot__funptr
    if _hipblasCrot__funptr == NULL:
        with gil:
            _hipblasCrot__funptr = loader.load_symbol(_lib_handle, "hipblasCrot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,hipblasComplex *,int,const float *,hipblasComplex *) nogil> _hipblasCrot__funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* _hipblasCsrot__funptr = NULL
cdef hipblasStatus_t hipblasCsrot(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,const float * c,const float * s) nogil:
    global _lib_handle
    global _hipblasCsrot__funptr
    if _hipblasCsrot__funptr == NULL:
        with gil:
            _hipblasCsrot__funptr = loader.load_symbol(_lib_handle, "hipblasCsrot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,hipblasComplex *,int,const float *,const float *) nogil> _hipblasCsrot__funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* _hipblasZrot__funptr = NULL
cdef hipblasStatus_t hipblasZrot(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,const double * c,hipblasDoubleComplex * s) nogil:
    global _lib_handle
    global _hipblasZrot__funptr
    if _hipblasZrot__funptr == NULL:
        with gil:
            _hipblasZrot__funptr = loader.load_symbol(_lib_handle, "hipblasZrot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,const double *,hipblasDoubleComplex *) nogil> _hipblasZrot__funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* _hipblasZdrot__funptr = NULL
cdef hipblasStatus_t hipblasZdrot(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,const double * c,const double * s) nogil:
    global _lib_handle
    global _hipblasZdrot__funptr
    if _hipblasZdrot__funptr == NULL:
        with gil:
            _hipblasZdrot__funptr = loader.load_symbol(_lib_handle, "hipblasZdrot")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,const double *,const double *) nogil> _hipblasZdrot__funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* _hipblasSrotg__funptr = NULL
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
    global _hipblasSrotg__funptr
    if _hipblasSrotg__funptr == NULL:
        with gil:
            _hipblasSrotg__funptr = loader.load_symbol(_lib_handle, "hipblasSrotg")
    return (<hipblasStatus_t (*)(hipblasHandle_t,float *,float *,float *,float *) nogil> _hipblasSrotg__funptr)(handle,a,b,c,s)


cdef void* _hipblasDrotg__funptr = NULL
cdef hipblasStatus_t hipblasDrotg(hipblasHandle_t handle,double * a,double * b,double * c,double * s) nogil:
    global _lib_handle
    global _hipblasDrotg__funptr
    if _hipblasDrotg__funptr == NULL:
        with gil:
            _hipblasDrotg__funptr = loader.load_symbol(_lib_handle, "hipblasDrotg")
    return (<hipblasStatus_t (*)(hipblasHandle_t,double *,double *,double *,double *) nogil> _hipblasDrotg__funptr)(handle,a,b,c,s)


cdef void* _hipblasCrotg__funptr = NULL
cdef hipblasStatus_t hipblasCrotg(hipblasHandle_t handle,hipblasComplex * a,hipblasComplex * b,float * c,hipblasComplex * s) nogil:
    global _lib_handle
    global _hipblasCrotg__funptr
    if _hipblasCrotg__funptr == NULL:
        with gil:
            _hipblasCrotg__funptr = loader.load_symbol(_lib_handle, "hipblasCrotg")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasComplex *,hipblasComplex *,float *,hipblasComplex *) nogil> _hipblasCrotg__funptr)(handle,a,b,c,s)


cdef void* _hipblasZrotg__funptr = NULL
cdef hipblasStatus_t hipblasZrotg(hipblasHandle_t handle,hipblasDoubleComplex * a,hipblasDoubleComplex * b,double * c,hipblasDoubleComplex * s) nogil:
    global _lib_handle
    global _hipblasZrotg__funptr
    if _hipblasZrotg__funptr == NULL:
        with gil:
            _hipblasZrotg__funptr = loader.load_symbol(_lib_handle, "hipblasZrotg")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasDoubleComplex *,hipblasDoubleComplex *,double *,hipblasDoubleComplex *) nogil> _hipblasZrotg__funptr)(handle,a,b,c,s)


cdef void* _hipblasSrotm__funptr = NULL
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
    global _hipblasSrotm__funptr
    if _hipblasSrotm__funptr == NULL:
        with gil:
            _hipblasSrotm__funptr = loader.load_symbol(_lib_handle, "hipblasSrotm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,float *,int,float *,int,const float *) nogil> _hipblasSrotm__funptr)(handle,n,x,incx,y,incy,param)


cdef void* _hipblasDrotm__funptr = NULL
cdef hipblasStatus_t hipblasDrotm(hipblasHandle_t handle,int n,double * x,int incx,double * y,int incy,const double * param) nogil:
    global _lib_handle
    global _hipblasDrotm__funptr
    if _hipblasDrotm__funptr == NULL:
        with gil:
            _hipblasDrotm__funptr = loader.load_symbol(_lib_handle, "hipblasDrotm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,double *,int,double *,int,const double *) nogil> _hipblasDrotm__funptr)(handle,n,x,incx,y,incy,param)


cdef void* _hipblasSrotmg__funptr = NULL
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
    global _hipblasSrotmg__funptr
    if _hipblasSrotmg__funptr == NULL:
        with gil:
            _hipblasSrotmg__funptr = loader.load_symbol(_lib_handle, "hipblasSrotmg")
    return (<hipblasStatus_t (*)(hipblasHandle_t,float *,float *,float *,const float *,float *) nogil> _hipblasSrotmg__funptr)(handle,d1,d2,x1,y1,param)


cdef void* _hipblasDrotmg__funptr = NULL
cdef hipblasStatus_t hipblasDrotmg(hipblasHandle_t handle,double * d1,double * d2,double * x1,const double * y1,double * param) nogil:
    global _lib_handle
    global _hipblasDrotmg__funptr
    if _hipblasDrotmg__funptr == NULL:
        with gil:
            _hipblasDrotmg__funptr = loader.load_symbol(_lib_handle, "hipblasDrotmg")
    return (<hipblasStatus_t (*)(hipblasHandle_t,double *,double *,double *,const double *,double *) nogil> _hipblasDrotmg__funptr)(handle,d1,d2,x1,y1,param)


cdef void* _hipblasSscal__funptr = NULL
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
    global _hipblasSscal__funptr
    if _hipblasSscal__funptr == NULL:
        with gil:
            _hipblasSscal__funptr = loader.load_symbol(_lib_handle, "hipblasSscal")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,float *,int) nogil> _hipblasSscal__funptr)(handle,n,alpha,x,incx)


cdef void* _hipblasDscal__funptr = NULL
cdef hipblasStatus_t hipblasDscal(hipblasHandle_t handle,int n,const double * alpha,double * x,int incx) nogil:
    global _lib_handle
    global _hipblasDscal__funptr
    if _hipblasDscal__funptr == NULL:
        with gil:
            _hipblasDscal__funptr = loader.load_symbol(_lib_handle, "hipblasDscal")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,double *,int) nogil> _hipblasDscal__funptr)(handle,n,alpha,x,incx)


cdef void* _hipblasCscal__funptr = NULL
cdef hipblasStatus_t hipblasCscal(hipblasHandle_t handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasCscal__funptr
    if _hipblasCscal__funptr == NULL:
        with gil:
            _hipblasCscal__funptr = loader.load_symbol(_lib_handle, "hipblasCscal")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCscal__funptr)(handle,n,alpha,x,incx)


cdef void* _hipblasCsscal__funptr = NULL
cdef hipblasStatus_t hipblasCsscal(hipblasHandle_t handle,int n,const float * alpha,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasCsscal__funptr
    if _hipblasCsscal__funptr == NULL:
        with gil:
            _hipblasCsscal__funptr = loader.load_symbol(_lib_handle, "hipblasCsscal")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const float *,hipblasComplex *,int) nogil> _hipblasCsscal__funptr)(handle,n,alpha,x,incx)


cdef void* _hipblasZscal__funptr = NULL
cdef hipblasStatus_t hipblasZscal(hipblasHandle_t handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasZscal__funptr
    if _hipblasZscal__funptr == NULL:
        with gil:
            _hipblasZscal__funptr = loader.load_symbol(_lib_handle, "hipblasZscal")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZscal__funptr)(handle,n,alpha,x,incx)


cdef void* _hipblasZdscal__funptr = NULL
cdef hipblasStatus_t hipblasZdscal(hipblasHandle_t handle,int n,const double * alpha,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasZdscal__funptr
    if _hipblasZdscal__funptr == NULL:
        with gil:
            _hipblasZdscal__funptr = loader.load_symbol(_lib_handle, "hipblasZdscal")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const double *,hipblasDoubleComplex *,int) nogil> _hipblasZdscal__funptr)(handle,n,alpha,x,incx)


cdef void* _hipblasSswap__funptr = NULL
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
    global _hipblasSswap__funptr
    if _hipblasSswap__funptr == NULL:
        with gil:
            _hipblasSswap__funptr = loader.load_symbol(_lib_handle, "hipblasSswap")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,float *,int,float *,int) nogil> _hipblasSswap__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasDswap__funptr = NULL
cdef hipblasStatus_t hipblasDswap(hipblasHandle_t handle,int n,double * x,int incx,double * y,int incy) nogil:
    global _lib_handle
    global _hipblasDswap__funptr
    if _hipblasDswap__funptr == NULL:
        with gil:
            _hipblasDswap__funptr = loader.load_symbol(_lib_handle, "hipblasDswap")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,double *,int,double *,int) nogil> _hipblasDswap__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasCswap__funptr = NULL
cdef hipblasStatus_t hipblasCswap(hipblasHandle_t handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasCswap__funptr
    if _hipblasCswap__funptr == NULL:
        with gil:
            _hipblasCswap__funptr = loader.load_symbol(_lib_handle, "hipblasCswap")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCswap__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasZswap__funptr = NULL
cdef hipblasStatus_t hipblasZswap(hipblasHandle_t handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasZswap__funptr
    if _hipblasZswap__funptr == NULL:
        with gil:
            _hipblasZswap__funptr = loader.load_symbol(_lib_handle, "hipblasZswap")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZswap__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasSgbmv__funptr = NULL
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
    global _hipblasSgbmv__funptr
    if _hipblasSgbmv__funptr == NULL:
        with gil:
            _hipblasSgbmv__funptr = loader.load_symbol(_lib_handle, "hipblasSgbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSgbmv__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasDgbmv__funptr = NULL
cdef hipblasStatus_t hipblasDgbmv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _lib_handle
    global _hipblasDgbmv__funptr
    if _hipblasDgbmv__funptr == NULL:
        with gil:
            _hipblasDgbmv__funptr = loader.load_symbol(_lib_handle, "hipblasDgbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDgbmv__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasCgbmv__funptr = NULL
cdef hipblasStatus_t hipblasCgbmv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasCgbmv__funptr
    if _hipblasCgbmv__funptr == NULL:
        with gil:
            _hipblasCgbmv__funptr = loader.load_symbol(_lib_handle, "hipblasCgbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCgbmv__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasZgbmv__funptr = NULL
cdef hipblasStatus_t hipblasZgbmv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasZgbmv__funptr
    if _hipblasZgbmv__funptr == NULL:
        with gil:
            _hipblasZgbmv__funptr = loader.load_symbol(_lib_handle, "hipblasZgbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZgbmv__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasSgemv__funptr = NULL
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
    global _hipblasSgemv__funptr
    if _hipblasSgemv__funptr == NULL:
        with gil:
            _hipblasSgemv__funptr = loader.load_symbol(_lib_handle, "hipblasSgemv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSgemv__funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasDgemv__funptr = NULL
cdef hipblasStatus_t hipblasDgemv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _lib_handle
    global _hipblasDgemv__funptr
    if _hipblasDgemv__funptr == NULL:
        with gil:
            _hipblasDgemv__funptr = loader.load_symbol(_lib_handle, "hipblasDgemv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDgemv__funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasCgemv__funptr = NULL
cdef hipblasStatus_t hipblasCgemv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasCgemv__funptr
    if _hipblasCgemv__funptr == NULL:
        with gil:
            _hipblasCgemv__funptr = loader.load_symbol(_lib_handle, "hipblasCgemv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCgemv__funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasZgemv__funptr = NULL
cdef hipblasStatus_t hipblasZgemv(hipblasHandle_t handle,hipblasOperation_t trans,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasZgemv__funptr
    if _hipblasZgemv__funptr == NULL:
        with gil:
            _hipblasZgemv__funptr = loader.load_symbol(_lib_handle, "hipblasZgemv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZgemv__funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasSger__funptr = NULL
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
    global _hipblasSger__funptr
    if _hipblasSger__funptr == NULL:
        with gil:
            _hipblasSger__funptr = loader.load_symbol(_lib_handle, "hipblasSger")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,int,const float *,const float *,int,const float *,int,float *,int) nogil> _hipblasSger__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasDger__funptr = NULL
cdef hipblasStatus_t hipblasDger(hipblasHandle_t handle,int m,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP,int lda) nogil:
    global _lib_handle
    global _hipblasDger__funptr
    if _hipblasDger__funptr == NULL:
        with gil:
            _hipblasDger__funptr = loader.load_symbol(_lib_handle, "hipblasDger")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,int,const double *,const double *,int,const double *,int,double *,int) nogil> _hipblasDger__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasCgeru__funptr = NULL
cdef hipblasStatus_t hipblasCgeru(hipblasHandle_t handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil:
    global _lib_handle
    global _hipblasCgeru__funptr
    if _hipblasCgeru__funptr == NULL:
        with gil:
            _hipblasCgeru__funptr = loader.load_symbol(_lib_handle, "hipblasCgeru")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCgeru__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasCgerc__funptr = NULL
cdef hipblasStatus_t hipblasCgerc(hipblasHandle_t handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil:
    global _lib_handle
    global _hipblasCgerc__funptr
    if _hipblasCgerc__funptr == NULL:
        with gil:
            _hipblasCgerc__funptr = loader.load_symbol(_lib_handle, "hipblasCgerc")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCgerc__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasZgeru__funptr = NULL
cdef hipblasStatus_t hipblasZgeru(hipblasHandle_t handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil:
    global _lib_handle
    global _hipblasZgeru__funptr
    if _hipblasZgeru__funptr == NULL:
        with gil:
            _hipblasZgeru__funptr = loader.load_symbol(_lib_handle, "hipblasZgeru")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZgeru__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasZgerc__funptr = NULL
cdef hipblasStatus_t hipblasZgerc(hipblasHandle_t handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil:
    global _lib_handle
    global _hipblasZgerc__funptr
    if _hipblasZgerc__funptr == NULL:
        with gil:
            _hipblasZgerc__funptr = loader.load_symbol(_lib_handle, "hipblasZgerc")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZgerc__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasChbmv__funptr = NULL
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
    global _hipblasChbmv__funptr
    if _hipblasChbmv__funptr == NULL:
        with gil:
            _hipblasChbmv__funptr = loader.load_symbol(_lib_handle, "hipblasChbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasChbmv__funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasZhbmv__funptr = NULL
cdef hipblasStatus_t hipblasZhbmv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasZhbmv__funptr
    if _hipblasZhbmv__funptr == NULL:
        with gil:
            _hipblasZhbmv__funptr = loader.load_symbol(_lib_handle, "hipblasZhbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZhbmv__funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasChemv__funptr = NULL
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
    global _hipblasChemv__funptr
    if _hipblasChemv__funptr == NULL:
        with gil:
            _hipblasChemv__funptr = loader.load_symbol(_lib_handle, "hipblasChemv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasChemv__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasZhemv__funptr = NULL
cdef hipblasStatus_t hipblasZhemv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasZhemv__funptr
    if _hipblasZhemv__funptr == NULL:
        with gil:
            _hipblasZhemv__funptr = loader.load_symbol(_lib_handle, "hipblasZhemv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZhemv__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasCher__funptr = NULL
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
    global _hipblasCher__funptr
    if _hipblasCher__funptr == NULL:
        with gil:
            _hipblasCher__funptr = loader.load_symbol(_lib_handle, "hipblasCher")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCher__funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* _hipblasZher__funptr = NULL
cdef hipblasStatus_t hipblasZher(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP,int lda) nogil:
    global _lib_handle
    global _hipblasZher__funptr
    if _hipblasZher__funptr == NULL:
        with gil:
            _hipblasZher__funptr = loader.load_symbol(_lib_handle, "hipblasZher")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZher__funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* _hipblasCher2__funptr = NULL
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
    global _hipblasCher2__funptr
    if _hipblasCher2__funptr == NULL:
        with gil:
            _hipblasCher2__funptr = loader.load_symbol(_lib_handle, "hipblasCher2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCher2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasZher2__funptr = NULL
cdef hipblasStatus_t hipblasZher2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil:
    global _lib_handle
    global _hipblasZher2__funptr
    if _hipblasZher2__funptr == NULL:
        with gil:
            _hipblasZher2__funptr = loader.load_symbol(_lib_handle, "hipblasZher2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZher2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasChpmv__funptr = NULL
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
    global _hipblasChpmv__funptr
    if _hipblasChpmv__funptr == NULL:
        with gil:
            _hipblasChpmv__funptr = loader.load_symbol(_lib_handle, "hipblasChpmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasChpmv__funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)


cdef void* _hipblasZhpmv__funptr = NULL
cdef hipblasStatus_t hipblasZhpmv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasZhpmv__funptr
    if _hipblasZhpmv__funptr == NULL:
        with gil:
            _hipblasZhpmv__funptr = loader.load_symbol(_lib_handle, "hipblasZhpmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZhpmv__funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)


cdef void* _hipblasChpr__funptr = NULL
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
    global _hipblasChpr__funptr
    if _hipblasChpr__funptr == NULL:
        with gil:
            _hipblasChpr__funptr = loader.load_symbol(_lib_handle, "hipblasChpr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,hipblasComplex *,int,hipblasComplex *) nogil> _hipblasChpr__funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* _hipblasZhpr__funptr = NULL
cdef hipblasStatus_t hipblasZhpr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP) nogil:
    global _lib_handle
    global _hipblasZhpr__funptr
    if _hipblasZhpr__funptr == NULL:
        with gil:
            _hipblasZhpr__funptr = loader.load_symbol(_lib_handle, "hipblasZhpr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> _hipblasZhpr__funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* _hipblasChpr2__funptr = NULL
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
    global _hipblasChpr2__funptr
    if _hipblasChpr2__funptr == NULL:
        with gil:
            _hipblasChpr2__funptr = loader.load_symbol(_lib_handle, "hipblasChpr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *) nogil> _hipblasChpr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP)


cdef void* _hipblasZhpr2__funptr = NULL
cdef hipblasStatus_t hipblasZhpr2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP) nogil:
    global _lib_handle
    global _hipblasZhpr2__funptr
    if _hipblasZhpr2__funptr == NULL:
        with gil:
            _hipblasZhpr2__funptr = loader.load_symbol(_lib_handle, "hipblasZhpr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> _hipblasZhpr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP)


cdef void* _hipblasSsbmv__funptr = NULL
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
    global _hipblasSsbmv__funptr
    if _hipblasSsbmv__funptr == NULL:
        with gil:
            _hipblasSsbmv__funptr = loader.load_symbol(_lib_handle, "hipblasSsbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSsbmv__funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasDsbmv__funptr = NULL
cdef hipblasStatus_t hipblasDsbmv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,int k,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _lib_handle
    global _hipblasDsbmv__funptr
    if _hipblasDsbmv__funptr == NULL:
        with gil:
            _hipblasDsbmv__funptr = loader.load_symbol(_lib_handle, "hipblasDsbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDsbmv__funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasSspmv__funptr = NULL
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
    global _hipblasSspmv__funptr
    if _hipblasSspmv__funptr == NULL:
        with gil:
            _hipblasSspmv__funptr = loader.load_symbol(_lib_handle, "hipblasSspmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,const float *,const float *,int,const float *,float *,int) nogil> _hipblasSspmv__funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)


cdef void* _hipblasDspmv__funptr = NULL
cdef hipblasStatus_t hipblasDspmv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _lib_handle
    global _hipblasDspmv__funptr
    if _hipblasDspmv__funptr == NULL:
        with gil:
            _hipblasDspmv__funptr = loader.load_symbol(_lib_handle, "hipblasDspmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,const double *,const double *,int,const double *,double *,int) nogil> _hipblasDspmv__funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)


cdef void* _hipblasSspr__funptr = NULL
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
    global _hipblasSspr__funptr
    if _hipblasSspr__funptr == NULL:
        with gil:
            _hipblasSspr__funptr = loader.load_symbol(_lib_handle, "hipblasSspr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,const float *,int,float *) nogil> _hipblasSspr__funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* _hipblasDspr__funptr = NULL
cdef hipblasStatus_t hipblasDspr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,double * AP) nogil:
    global _lib_handle
    global _hipblasDspr__funptr
    if _hipblasDspr__funptr == NULL:
        with gil:
            _hipblasDspr__funptr = loader.load_symbol(_lib_handle, "hipblasDspr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,const double *,int,double *) nogil> _hipblasDspr__funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* _hipblasCspr__funptr = NULL
cdef hipblasStatus_t hipblasCspr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * AP) nogil:
    global _lib_handle
    global _hipblasCspr__funptr
    if _hipblasCspr__funptr == NULL:
        with gil:
            _hipblasCspr__funptr = loader.load_symbol(_lib_handle, "hipblasCspr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *) nogil> _hipblasCspr__funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* _hipblasZspr__funptr = NULL
cdef hipblasStatus_t hipblasZspr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP) nogil:
    global _lib_handle
    global _hipblasZspr__funptr
    if _hipblasZspr__funptr == NULL:
        with gil:
            _hipblasZspr__funptr = loader.load_symbol(_lib_handle, "hipblasZspr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> _hipblasZspr__funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* _hipblasSspr2__funptr = NULL
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
    global _hipblasSspr2__funptr
    if _hipblasSspr2__funptr == NULL:
        with gil:
            _hipblasSspr2__funptr = loader.load_symbol(_lib_handle, "hipblasSspr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,const float *,int,const float *,int,float *) nogil> _hipblasSspr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP)


cdef void* _hipblasDspr2__funptr = NULL
cdef hipblasStatus_t hipblasDspr2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP) nogil:
    global _lib_handle
    global _hipblasDspr2__funptr
    if _hipblasDspr2__funptr == NULL:
        with gil:
            _hipblasDspr2__funptr = loader.load_symbol(_lib_handle, "hipblasDspr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,const double *,int,const double *,int,double *) nogil> _hipblasDspr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP)


cdef void* _hipblasSsymv__funptr = NULL
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
    global _hipblasSsymv__funptr
    if _hipblasSsymv__funptr == NULL:
        with gil:
            _hipblasSsymv__funptr = loader.load_symbol(_lib_handle, "hipblasSsymv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSsymv__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasDsymv__funptr = NULL
cdef hipblasStatus_t hipblasDsymv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _lib_handle
    global _hipblasDsymv__funptr
    if _hipblasDsymv__funptr == NULL:
        with gil:
            _hipblasDsymv__funptr = loader.load_symbol(_lib_handle, "hipblasDsymv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDsymv__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasCsymv__funptr = NULL
cdef hipblasStatus_t hipblasCsymv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasCsymv__funptr
    if _hipblasCsymv__funptr == NULL:
        with gil:
            _hipblasCsymv__funptr = loader.load_symbol(_lib_handle, "hipblasCsymv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCsymv__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasZsymv__funptr = NULL
cdef hipblasStatus_t hipblasZsymv(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _lib_handle
    global _hipblasZsymv__funptr
    if _hipblasZsymv__funptr == NULL:
        with gil:
            _hipblasZsymv__funptr = loader.load_symbol(_lib_handle, "hipblasZsymv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZsymv__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasSsyr__funptr = NULL
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
    global _hipblasSsyr__funptr
    if _hipblasSsyr__funptr == NULL:
        with gil:
            _hipblasSsyr__funptr = loader.load_symbol(_lib_handle, "hipblasSsyr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,const float *,int,float *,int) nogil> _hipblasSsyr__funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* _hipblasDsyr__funptr = NULL
cdef hipblasStatus_t hipblasDsyr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,double * AP,int lda) nogil:
    global _lib_handle
    global _hipblasDsyr__funptr
    if _hipblasDsyr__funptr == NULL:
        with gil:
            _hipblasDsyr__funptr = loader.load_symbol(_lib_handle, "hipblasDsyr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,const double *,int,double *,int) nogil> _hipblasDsyr__funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* _hipblasCsyr__funptr = NULL
cdef hipblasStatus_t hipblasCsyr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * AP,int lda) nogil:
    global _lib_handle
    global _hipblasCsyr__funptr
    if _hipblasCsyr__funptr == NULL:
        with gil:
            _hipblasCsyr__funptr = loader.load_symbol(_lib_handle, "hipblasCsyr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCsyr__funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* _hipblasZsyr__funptr = NULL
cdef hipblasStatus_t hipblasZsyr(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP,int lda) nogil:
    global _lib_handle
    global _hipblasZsyr__funptr
    if _hipblasZsyr__funptr == NULL:
        with gil:
            _hipblasZsyr__funptr = loader.load_symbol(_lib_handle, "hipblasZsyr")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZsyr__funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* _hipblasSsyr2__funptr = NULL
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
    global _hipblasSsyr2__funptr
    if _hipblasSsyr2__funptr == NULL:
        with gil:
            _hipblasSsyr2__funptr = loader.load_symbol(_lib_handle, "hipblasSsyr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const float *,const float *,int,const float *,int,float *,int) nogil> _hipblasSsyr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasDsyr2__funptr = NULL
cdef hipblasStatus_t hipblasDsyr2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP,int lda) nogil:
    global _lib_handle
    global _hipblasDsyr2__funptr
    if _hipblasDsyr2__funptr == NULL:
        with gil:
            _hipblasDsyr2__funptr = loader.load_symbol(_lib_handle, "hipblasDsyr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,const double *,const double *,int,const double *,int,double *,int) nogil> _hipblasDsyr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasCsyr2__funptr = NULL
cdef hipblasStatus_t hipblasCsyr2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil:
    global _lib_handle
    global _hipblasCsyr2__funptr
    if _hipblasCsyr2__funptr == NULL:
        with gil:
            _hipblasCsyr2__funptr = loader.load_symbol(_lib_handle, "hipblasCsyr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCsyr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasZsyr2__funptr = NULL
cdef hipblasStatus_t hipblasZsyr2(hipblasHandle_t handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil:
    global _lib_handle
    global _hipblasZsyr2__funptr
    if _hipblasZsyr2__funptr == NULL:
        with gil:
            _hipblasZsyr2__funptr = loader.load_symbol(_lib_handle, "hipblasZsyr2")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZsyr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasStbmv__funptr = NULL
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
    global _hipblasStbmv__funptr
    if _hipblasStbmv__funptr == NULL:
        with gil:
            _hipblasStbmv__funptr = loader.load_symbol(_lib_handle, "hipblasStbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,int,float *,int) nogil> _hipblasStbmv__funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx)


cdef void* _hipblasDtbmv__funptr = NULL
cdef hipblasStatus_t hipblasDtbmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const double * AP,int lda,double * x,int incx) nogil:
    global _lib_handle
    global _hipblasDtbmv__funptr
    if _hipblasDtbmv__funptr == NULL:
        with gil:
            _hipblasDtbmv__funptr = loader.load_symbol(_lib_handle, "hipblasDtbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,int,double *,int) nogil> _hipblasDtbmv__funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx)


cdef void* _hipblasCtbmv__funptr = NULL
cdef hipblasStatus_t hipblasCtbmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasCtbmv__funptr
    if _hipblasCtbmv__funptr == NULL:
        with gil:
            _hipblasCtbmv__funptr = loader.load_symbol(_lib_handle, "hipblasCtbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtbmv__funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx)


cdef void* _hipblasZtbmv__funptr = NULL
cdef hipblasStatus_t hipblasZtbmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasZtbmv__funptr
    if _hipblasZtbmv__funptr == NULL:
        with gil:
            _hipblasZtbmv__funptr = loader.load_symbol(_lib_handle, "hipblasZtbmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtbmv__funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx)


cdef void* _hipblasStbsv__funptr = NULL
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
    global _hipblasStbsv__funptr
    if _hipblasStbsv__funptr == NULL:
        with gil:
            _hipblasStbsv__funptr = loader.load_symbol(_lib_handle, "hipblasStbsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,int,float *,int) nogil> _hipblasStbsv__funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx)


cdef void* _hipblasDtbsv__funptr = NULL
cdef hipblasStatus_t hipblasDtbsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const double * AP,int lda,double * x,int incx) nogil:
    global _lib_handle
    global _hipblasDtbsv__funptr
    if _hipblasDtbsv__funptr == NULL:
        with gil:
            _hipblasDtbsv__funptr = loader.load_symbol(_lib_handle, "hipblasDtbsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,int,double *,int) nogil> _hipblasDtbsv__funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx)


cdef void* _hipblasCtbsv__funptr = NULL
cdef hipblasStatus_t hipblasCtbsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasCtbsv__funptr
    if _hipblasCtbsv__funptr == NULL:
        with gil:
            _hipblasCtbsv__funptr = loader.load_symbol(_lib_handle, "hipblasCtbsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtbsv__funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx)


cdef void* _hipblasZtbsv__funptr = NULL
cdef hipblasStatus_t hipblasZtbsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasZtbsv__funptr
    if _hipblasZtbsv__funptr == NULL:
        with gil:
            _hipblasZtbsv__funptr = loader.load_symbol(_lib_handle, "hipblasZtbsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtbsv__funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx)


cdef void* _hipblasStpmv__funptr = NULL
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
    global _hipblasStpmv__funptr
    if _hipblasStpmv__funptr == NULL:
        with gil:
            _hipblasStpmv__funptr = loader.load_symbol(_lib_handle, "hipblasStpmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,float *,int) nogil> _hipblasStpmv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasDtpmv__funptr = NULL
cdef hipblasStatus_t hipblasDtpmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,double * x,int incx) nogil:
    global _lib_handle
    global _hipblasDtpmv__funptr
    if _hipblasDtpmv__funptr == NULL:
        with gil:
            _hipblasDtpmv__funptr = loader.load_symbol(_lib_handle, "hipblasDtpmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,double *,int) nogil> _hipblasDtpmv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasCtpmv__funptr = NULL
cdef hipblasStatus_t hipblasCtpmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasCtpmv__funptr
    if _hipblasCtpmv__funptr == NULL:
        with gil:
            _hipblasCtpmv__funptr = loader.load_symbol(_lib_handle, "hipblasCtpmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCtpmv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasZtpmv__funptr = NULL
cdef hipblasStatus_t hipblasZtpmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasZtpmv__funptr
    if _hipblasZtpmv__funptr == NULL:
        with gil:
            _hipblasZtpmv__funptr = loader.load_symbol(_lib_handle, "hipblasZtpmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZtpmv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasStpsv__funptr = NULL
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
    global _hipblasStpsv__funptr
    if _hipblasStpsv__funptr == NULL:
        with gil:
            _hipblasStpsv__funptr = loader.load_symbol(_lib_handle, "hipblasStpsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,float *,int) nogil> _hipblasStpsv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasDtpsv__funptr = NULL
cdef hipblasStatus_t hipblasDtpsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,double * x,int incx) nogil:
    global _lib_handle
    global _hipblasDtpsv__funptr
    if _hipblasDtpsv__funptr == NULL:
        with gil:
            _hipblasDtpsv__funptr = loader.load_symbol(_lib_handle, "hipblasDtpsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,double *,int) nogil> _hipblasDtpsv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasCtpsv__funptr = NULL
cdef hipblasStatus_t hipblasCtpsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasCtpsv__funptr
    if _hipblasCtpsv__funptr == NULL:
        with gil:
            _hipblasCtpsv__funptr = loader.load_symbol(_lib_handle, "hipblasCtpsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCtpsv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasZtpsv__funptr = NULL
cdef hipblasStatus_t hipblasZtpsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasZtpsv__funptr
    if _hipblasZtpsv__funptr == NULL:
        with gil:
            _hipblasZtpsv__funptr = loader.load_symbol(_lib_handle, "hipblasZtpsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZtpsv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasStrmv__funptr = NULL
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
    global _hipblasStrmv__funptr
    if _hipblasStrmv__funptr == NULL:
        with gil:
            _hipblasStrmv__funptr = loader.load_symbol(_lib_handle, "hipblasStrmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,int,float *,int) nogil> _hipblasStrmv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasDtrmv__funptr = NULL
cdef hipblasStatus_t hipblasDtrmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,double * x,int incx) nogil:
    global _lib_handle
    global _hipblasDtrmv__funptr
    if _hipblasDtrmv__funptr == NULL:
        with gil:
            _hipblasDtrmv__funptr = loader.load_symbol(_lib_handle, "hipblasDtrmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,int,double *,int) nogil> _hipblasDtrmv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasCtrmv__funptr = NULL
cdef hipblasStatus_t hipblasCtrmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasCtrmv__funptr
    if _hipblasCtrmv__funptr == NULL:
        with gil:
            _hipblasCtrmv__funptr = loader.load_symbol(_lib_handle, "hipblasCtrmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtrmv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasZtrmv__funptr = NULL
cdef hipblasStatus_t hipblasZtrmv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasZtrmv__funptr
    if _hipblasZtrmv__funptr == NULL:
        with gil:
            _hipblasZtrmv__funptr = loader.load_symbol(_lib_handle, "hipblasZtrmv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtrmv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasStrsv__funptr = NULL
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
    global _hipblasStrsv__funptr
    if _hipblasStrsv__funptr == NULL:
        with gil:
            _hipblasStrsv__funptr = loader.load_symbol(_lib_handle, "hipblasStrsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,int,float *,int) nogil> _hipblasStrsv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasDtrsv__funptr = NULL
cdef hipblasStatus_t hipblasDtrsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,double * x,int incx) nogil:
    global _lib_handle
    global _hipblasDtrsv__funptr
    if _hipblasDtrsv__funptr == NULL:
        with gil:
            _hipblasDtrsv__funptr = loader.load_symbol(_lib_handle, "hipblasDtrsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,int,double *,int) nogil> _hipblasDtrsv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasCtrsv__funptr = NULL
cdef hipblasStatus_t hipblasCtrsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasCtrsv__funptr
    if _hipblasCtrsv__funptr == NULL:
        with gil:
            _hipblasCtrsv__funptr = loader.load_symbol(_lib_handle, "hipblasCtrsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtrsv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasZtrsv__funptr = NULL
cdef hipblasStatus_t hipblasZtrsv(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil:
    global _lib_handle
    global _hipblasZtrsv__funptr
    if _hipblasZtrsv__funptr == NULL:
        with gil:
            _hipblasZtrsv__funptr = loader.load_symbol(_lib_handle, "hipblasZtrsv")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtrsv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasHgemm__funptr = NULL
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
    global _hipblasHgemm__funptr
    if _hipblasHgemm__funptr == NULL:
        with gil:
            _hipblasHgemm__funptr = loader.load_symbol(_lib_handle, "hipblasHgemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,hipblasHalf *,hipblasHalf *,int,hipblasHalf *,int,hipblasHalf *,hipblasHalf *,int) nogil> _hipblasHgemm__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasSgemm__funptr = NULL
cdef hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasSgemm__funptr
    if _hipblasSgemm__funptr == NULL:
        with gil:
            _hipblasSgemm__funptr = loader.load_symbol(_lib_handle, "hipblasSgemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSgemm__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasDgemm__funptr = NULL
cdef hipblasStatus_t hipblasDgemm(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasDgemm__funptr
    if _hipblasDgemm__funptr == NULL:
        with gil:
            _hipblasDgemm__funptr = loader.load_symbol(_lib_handle, "hipblasDgemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDgemm__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasCgemm__funptr = NULL
cdef hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasCgemm__funptr
    if _hipblasCgemm__funptr == NULL:
        with gil:
            _hipblasCgemm__funptr = loader.load_symbol(_lib_handle, "hipblasCgemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCgemm__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZgemm__funptr = NULL
cdef hipblasStatus_t hipblasZgemm(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasZgemm__funptr
    if _hipblasZgemm__funptr == NULL:
        with gil:
            _hipblasZgemm__funptr = loader.load_symbol(_lib_handle, "hipblasZgemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZgemm__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasCherk__funptr = NULL
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
    global _hipblasCherk__funptr
    if _hipblasCherk__funptr == NULL:
        with gil:
            _hipblasCherk__funptr = loader.load_symbol(_lib_handle, "hipblasCherk")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,hipblasComplex *,int,const float *,hipblasComplex *,int) nogil> _hipblasCherk__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* _hipblasZherk__funptr = NULL
cdef hipblasStatus_t hipblasZherk(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,hipblasDoubleComplex * AP,int lda,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasZherk__funptr
    if _hipblasZherk__funptr == NULL:
        with gil:
            _hipblasZherk__funptr = loader.load_symbol(_lib_handle, "hipblasZherk")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,hipblasDoubleComplex *,int,const double *,hipblasDoubleComplex *,int) nogil> _hipblasZherk__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* _hipblasCherkx__funptr = NULL
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
    global _hipblasCherkx__funptr
    if _hipblasCherkx__funptr == NULL:
        with gil:
            _hipblasCherkx__funptr = loader.load_symbol(_lib_handle, "hipblasCherkx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,const float *,hipblasComplex *,int) nogil> _hipblasCherkx__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZherkx__funptr = NULL
cdef hipblasStatus_t hipblasZherkx(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasZherkx__funptr
    if _hipblasZherkx__funptr == NULL:
        with gil:
            _hipblasZherkx__funptr = loader.load_symbol(_lib_handle, "hipblasZherkx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,const double *,hipblasDoubleComplex *,int) nogil> _hipblasZherkx__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasCher2k__funptr = NULL
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
    global _hipblasCher2k__funptr
    if _hipblasCher2k__funptr == NULL:
        with gil:
            _hipblasCher2k__funptr = loader.load_symbol(_lib_handle, "hipblasCher2k")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,const float *,hipblasComplex *,int) nogil> _hipblasCher2k__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZher2k__funptr = NULL
cdef hipblasStatus_t hipblasZher2k(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasZher2k__funptr
    if _hipblasZher2k__funptr == NULL:
        with gil:
            _hipblasZher2k__funptr = loader.load_symbol(_lib_handle, "hipblasZher2k")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,const double *,hipblasDoubleComplex *,int) nogil> _hipblasZher2k__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasSsymm__funptr = NULL
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
    global _hipblasSsymm__funptr
    if _hipblasSsymm__funptr == NULL:
        with gil:
            _hipblasSsymm__funptr = loader.load_symbol(_lib_handle, "hipblasSsymm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSsymm__funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasDsymm__funptr = NULL
cdef hipblasStatus_t hipblasDsymm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasDsymm__funptr
    if _hipblasDsymm__funptr == NULL:
        with gil:
            _hipblasDsymm__funptr = loader.load_symbol(_lib_handle, "hipblasDsymm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDsymm__funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasCsymm__funptr = NULL
cdef hipblasStatus_t hipblasCsymm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasCsymm__funptr
    if _hipblasCsymm__funptr == NULL:
        with gil:
            _hipblasCsymm__funptr = loader.load_symbol(_lib_handle, "hipblasCsymm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCsymm__funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZsymm__funptr = NULL
cdef hipblasStatus_t hipblasZsymm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasZsymm__funptr
    if _hipblasZsymm__funptr == NULL:
        with gil:
            _hipblasZsymm__funptr = loader.load_symbol(_lib_handle, "hipblasZsymm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZsymm__funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasSsyrk__funptr = NULL
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
    global _hipblasSsyrk__funptr
    if _hipblasSsyrk__funptr == NULL:
        with gil:
            _hipblasSsyrk__funptr = loader.load_symbol(_lib_handle, "hipblasSsyrk")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,float *,int) nogil> _hipblasSsyrk__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* _hipblasDsyrk__funptr = NULL
cdef hipblasStatus_t hipblasDsyrk(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * beta,double * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasDsyrk__funptr
    if _hipblasDsyrk__funptr == NULL:
        with gil:
            _hipblasDsyrk__funptr = loader.load_symbol(_lib_handle, "hipblasDsyrk")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,double *,int) nogil> _hipblasDsyrk__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* _hipblasCsyrk__funptr = NULL
cdef hipblasStatus_t hipblasCsyrk(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasCsyrk__funptr
    if _hipblasCsyrk__funptr == NULL:
        with gil:
            _hipblasCsyrk__funptr = loader.load_symbol(_lib_handle, "hipblasCsyrk")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCsyrk__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* _hipblasZsyrk__funptr = NULL
cdef hipblasStatus_t hipblasZsyrk(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasZsyrk__funptr
    if _hipblasZsyrk__funptr == NULL:
        with gil:
            _hipblasZsyrk__funptr = loader.load_symbol(_lib_handle, "hipblasZsyrk")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZsyrk__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* _hipblasSsyr2k__funptr = NULL
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
    global _hipblasSsyr2k__funptr
    if _hipblasSsyr2k__funptr == NULL:
        with gil:
            _hipblasSsyr2k__funptr = loader.load_symbol(_lib_handle, "hipblasSsyr2k")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSsyr2k__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasDsyr2k__funptr = NULL
cdef hipblasStatus_t hipblasDsyr2k(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasDsyr2k__funptr
    if _hipblasDsyr2k__funptr == NULL:
        with gil:
            _hipblasDsyr2k__funptr = loader.load_symbol(_lib_handle, "hipblasDsyr2k")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDsyr2k__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasCsyr2k__funptr = NULL
cdef hipblasStatus_t hipblasCsyr2k(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasCsyr2k__funptr
    if _hipblasCsyr2k__funptr == NULL:
        with gil:
            _hipblasCsyr2k__funptr = loader.load_symbol(_lib_handle, "hipblasCsyr2k")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCsyr2k__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZsyr2k__funptr = NULL
cdef hipblasStatus_t hipblasZsyr2k(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasZsyr2k__funptr
    if _hipblasZsyr2k__funptr == NULL:
        with gil:
            _hipblasZsyr2k__funptr = loader.load_symbol(_lib_handle, "hipblasZsyr2k")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZsyr2k__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasSsyrkx__funptr = NULL
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
    global _hipblasSsyrkx__funptr
    if _hipblasSsyrkx__funptr == NULL:
        with gil:
            _hipblasSsyrkx__funptr = loader.load_symbol(_lib_handle, "hipblasSsyrkx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSsyrkx__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasDsyrkx__funptr = NULL
cdef hipblasStatus_t hipblasDsyrkx(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasDsyrkx__funptr
    if _hipblasDsyrkx__funptr == NULL:
        with gil:
            _hipblasDsyrkx__funptr = loader.load_symbol(_lib_handle, "hipblasDsyrkx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDsyrkx__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasCsyrkx__funptr = NULL
cdef hipblasStatus_t hipblasCsyrkx(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasCsyrkx__funptr
    if _hipblasCsyrkx__funptr == NULL:
        with gil:
            _hipblasCsyrkx__funptr = loader.load_symbol(_lib_handle, "hipblasCsyrkx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCsyrkx__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZsyrkx__funptr = NULL
cdef hipblasStatus_t hipblasZsyrkx(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasZsyrkx__funptr
    if _hipblasZsyrkx__funptr == NULL:
        with gil:
            _hipblasZsyrkx__funptr = loader.load_symbol(_lib_handle, "hipblasZsyrkx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZsyrkx__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasSgeam__funptr = NULL
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
    global _hipblasSgeam__funptr
    if _hipblasSgeam__funptr == NULL:
        with gil:
            _hipblasSgeam__funptr = loader.load_symbol(_lib_handle, "hipblasSgeam")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,const float *,int,float *,int) nogil> _hipblasSgeam__funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc)


cdef void* _hipblasDgeam__funptr = NULL
cdef hipblasStatus_t hipblasDgeam(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const double * alpha,const double * AP,int lda,const double * beta,const double * BP,int ldb,double * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasDgeam__funptr
    if _hipblasDgeam__funptr == NULL:
        with gil:
            _hipblasDgeam__funptr = loader.load_symbol(_lib_handle, "hipblasDgeam")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,const double *,int,double *,int) nogil> _hipblasDgeam__funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc)


cdef void* _hipblasCgeam__funptr = NULL
cdef hipblasStatus_t hipblasCgeam(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * beta,hipblasComplex * BP,int ldb,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasCgeam__funptr
    if _hipblasCgeam__funptr == NULL:
        with gil:
            _hipblasCgeam__funptr = loader.load_symbol(_lib_handle, "hipblasCgeam")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCgeam__funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc)


cdef void* _hipblasZgeam__funptr = NULL
cdef hipblasStatus_t hipblasZgeam(hipblasHandle_t handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasZgeam__funptr
    if _hipblasZgeam__funptr == NULL:
        with gil:
            _hipblasZgeam__funptr = loader.load_symbol(_lib_handle, "hipblasZgeam")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZgeam__funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc)


cdef void* _hipblasChemm__funptr = NULL
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
    global _hipblasChemm__funptr
    if _hipblasChemm__funptr == NULL:
        with gil:
            _hipblasChemm__funptr = loader.load_symbol(_lib_handle, "hipblasChemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasChemm__funptr)(handle,side,uplo,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZhemm__funptr = NULL
cdef hipblasStatus_t hipblasZhemm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasZhemm__funptr
    if _hipblasZhemm__funptr == NULL:
        with gil:
            _hipblasZhemm__funptr = loader.load_symbol(_lib_handle, "hipblasZhemm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZhemm__funptr)(handle,side,uplo,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasStrmm__funptr = NULL
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
    global _hipblasStrmm__funptr
    if _hipblasStrmm__funptr == NULL:
        with gil:
            _hipblasStrmm__funptr = loader.load_symbol(_lib_handle, "hipblasStrmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,const float *,int,float *,int) nogil> _hipblasStrmm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasDtrmm__funptr = NULL
cdef hipblasStatus_t hipblasDtrmm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,const double * AP,int lda,double * BP,int ldb) nogil:
    global _lib_handle
    global _hipblasDtrmm__funptr
    if _hipblasDtrmm__funptr == NULL:
        with gil:
            _hipblasDtrmm__funptr = loader.load_symbol(_lib_handle, "hipblasDtrmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,const double *,int,double *,int) nogil> _hipblasDtrmm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasCtrmm__funptr = NULL
cdef hipblasStatus_t hipblasCtrmm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb) nogil:
    global _lib_handle
    global _hipblasCtrmm__funptr
    if _hipblasCtrmm__funptr == NULL:
        with gil:
            _hipblasCtrmm__funptr = loader.load_symbol(_lib_handle, "hipblasCtrmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtrmm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasZtrmm__funptr = NULL
cdef hipblasStatus_t hipblasZtrmm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb) nogil:
    global _lib_handle
    global _hipblasZtrmm__funptr
    if _hipblasZtrmm__funptr == NULL:
        with gil:
            _hipblasZtrmm__funptr = loader.load_symbol(_lib_handle, "hipblasZtrmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtrmm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasStrsm__funptr = NULL
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
    global _hipblasStrsm__funptr
    if _hipblasStrsm__funptr == NULL:
        with gil:
            _hipblasStrsm__funptr = loader.load_symbol(_lib_handle, "hipblasStrsm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,float *,int,float *,int) nogil> _hipblasStrsm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasDtrsm__funptr = NULL
cdef hipblasStatus_t hipblasDtrsm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,double * AP,int lda,double * BP,int ldb) nogil:
    global _lib_handle
    global _hipblasDtrsm__funptr
    if _hipblasDtrsm__funptr == NULL:
        with gil:
            _hipblasDtrsm__funptr = loader.load_symbol(_lib_handle, "hipblasDtrsm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,double *,int,double *,int) nogil> _hipblasDtrsm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasCtrsm__funptr = NULL
cdef hipblasStatus_t hipblasCtrsm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb) nogil:
    global _lib_handle
    global _hipblasCtrsm__funptr
    if _hipblasCtrsm__funptr == NULL:
        with gil:
            _hipblasCtrsm__funptr = loader.load_symbol(_lib_handle, "hipblasCtrsm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtrsm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasZtrsm__funptr = NULL
cdef hipblasStatus_t hipblasZtrsm(hipblasHandle_t handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb) nogil:
    global _lib_handle
    global _hipblasZtrsm__funptr
    if _hipblasZtrsm__funptr == NULL:
        with gil:
            _hipblasZtrsm__funptr = loader.load_symbol(_lib_handle, "hipblasZtrsm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtrsm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasStrtri__funptr = NULL
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
    global _hipblasStrtri__funptr
    if _hipblasStrtri__funptr == NULL:
        with gil:
            _hipblasStrtri__funptr = loader.load_symbol(_lib_handle, "hipblasStrtri")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasDiagType_t,int,const float *,int,float *,int) nogil> _hipblasStrtri__funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA)


cdef void* _hipblasDtrtri__funptr = NULL
cdef hipblasStatus_t hipblasDtrtri(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const double * AP,int lda,double * invA,int ldinvA) nogil:
    global _lib_handle
    global _hipblasDtrtri__funptr
    if _hipblasDtrtri__funptr == NULL:
        with gil:
            _hipblasDtrtri__funptr = loader.load_symbol(_lib_handle, "hipblasDtrtri")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasDiagType_t,int,const double *,int,double *,int) nogil> _hipblasDtrtri__funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA)


cdef void* _hipblasCtrtri__funptr = NULL
cdef hipblasStatus_t hipblasCtrtri(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasComplex * AP,int lda,hipblasComplex * invA,int ldinvA) nogil:
    global _lib_handle
    global _hipblasCtrtri__funptr
    if _hipblasCtrtri__funptr == NULL:
        with gil:
            _hipblasCtrtri__funptr = loader.load_symbol(_lib_handle, "hipblasCtrtri")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasDiagType_t,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtrtri__funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA)


cdef void* _hipblasZtrtri__funptr = NULL
cdef hipblasStatus_t hipblasZtrtri(hipblasHandle_t handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * invA,int ldinvA) nogil:
    global _lib_handle
    global _hipblasZtrtri__funptr
    if _hipblasZtrtri__funptr == NULL:
        with gil:
            _hipblasZtrtri__funptr = loader.load_symbol(_lib_handle, "hipblasZtrtri")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasFillMode_t,hipblasDiagType_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtrtri__funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA)


cdef void* _hipblasSdgmm__funptr = NULL
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
    global _hipblasSdgmm__funptr
    if _hipblasSdgmm__funptr == NULL:
        with gil:
            _hipblasSdgmm__funptr = loader.load_symbol(_lib_handle, "hipblasSdgmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,int,int,const float *,int,const float *,int,float *,int) nogil> _hipblasSdgmm__funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc)


cdef void* _hipblasDdgmm__funptr = NULL
cdef hipblasStatus_t hipblasDdgmm(hipblasHandle_t handle,hipblasSideMode_t side,int m,int n,const double * AP,int lda,const double * x,int incx,double * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasDdgmm__funptr
    if _hipblasDdgmm__funptr == NULL:
        with gil:
            _hipblasDdgmm__funptr = loader.load_symbol(_lib_handle, "hipblasDdgmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,int,int,const double *,int,const double *,int,double *,int) nogil> _hipblasDdgmm__funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc)


cdef void* _hipblasCdgmm__funptr = NULL
cdef hipblasStatus_t hipblasCdgmm(hipblasHandle_t handle,hipblasSideMode_t side,int m,int n,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasCdgmm__funptr
    if _hipblasCdgmm__funptr == NULL:
        with gil:
            _hipblasCdgmm__funptr = loader.load_symbol(_lib_handle, "hipblasCdgmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,int,int,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCdgmm__funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc)


cdef void* _hipblasZdgmm__funptr = NULL
cdef hipblasStatus_t hipblasZdgmm(hipblasHandle_t handle,hipblasSideMode_t side,int m,int n,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * CP,int ldc) nogil:
    global _lib_handle
    global _hipblasZdgmm__funptr
    if _hipblasZdgmm__funptr == NULL:
        with gil:
            _hipblasZdgmm__funptr = loader.load_symbol(_lib_handle, "hipblasZdgmm")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,int,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZdgmm__funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc)


cdef void* _hipblasSgetrf__funptr = NULL
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
    global _hipblasSgetrf__funptr
    if _hipblasSgetrf__funptr == NULL:
        with gil:
            _hipblasSgetrf__funptr = loader.load_symbol(_lib_handle, "hipblasSgetrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,float *,const int,int *,int *) nogil> _hipblasSgetrf__funptr)(handle,n,A,lda,ipiv,info)


cdef void* _hipblasDgetrf__funptr = NULL
cdef hipblasStatus_t hipblasDgetrf(hipblasHandle_t handle,const int n,double * A,const int lda,int * ipiv,int * info) nogil:
    global _lib_handle
    global _hipblasDgetrf__funptr
    if _hipblasDgetrf__funptr == NULL:
        with gil:
            _hipblasDgetrf__funptr = loader.load_symbol(_lib_handle, "hipblasDgetrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,double *,const int,int *,int *) nogil> _hipblasDgetrf__funptr)(handle,n,A,lda,ipiv,info)


cdef void* _hipblasCgetrf__funptr = NULL
cdef hipblasStatus_t hipblasCgetrf(hipblasHandle_t handle,const int n,hipblasComplex * A,const int lda,int * ipiv,int * info) nogil:
    global _lib_handle
    global _hipblasCgetrf__funptr
    if _hipblasCgetrf__funptr == NULL:
        with gil:
            _hipblasCgetrf__funptr = loader.load_symbol(_lib_handle, "hipblasCgetrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,hipblasComplex *,const int,int *,int *) nogil> _hipblasCgetrf__funptr)(handle,n,A,lda,ipiv,info)


cdef void* _hipblasZgetrf__funptr = NULL
cdef hipblasStatus_t hipblasZgetrf(hipblasHandle_t handle,const int n,hipblasDoubleComplex * A,const int lda,int * ipiv,int * info) nogil:
    global _lib_handle
    global _hipblasZgetrf__funptr
    if _hipblasZgetrf__funptr == NULL:
        with gil:
            _hipblasZgetrf__funptr = loader.load_symbol(_lib_handle, "hipblasZgetrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,hipblasDoubleComplex *,const int,int *,int *) nogil> _hipblasZgetrf__funptr)(handle,n,A,lda,ipiv,info)


cdef void* _hipblasSgetrs__funptr = NULL
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
    global _hipblasSgetrs__funptr
    if _hipblasSgetrs__funptr == NULL:
        with gil:
            _hipblasSgetrs__funptr = loader.load_symbol(_lib_handle, "hipblasSgetrs")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,float *,const int,const int *,float *,const int,int *) nogil> _hipblasSgetrs__funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info)


cdef void* _hipblasDgetrs__funptr = NULL
cdef hipblasStatus_t hipblasDgetrs(hipblasHandle_t handle,hipblasOperation_t trans,const int n,const int nrhs,double * A,const int lda,const int * ipiv,double * B,const int ldb,int * info) nogil:
    global _lib_handle
    global _hipblasDgetrs__funptr
    if _hipblasDgetrs__funptr == NULL:
        with gil:
            _hipblasDgetrs__funptr = loader.load_symbol(_lib_handle, "hipblasDgetrs")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,double *,const int,const int *,double *,const int,int *) nogil> _hipblasDgetrs__funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info)


cdef void* _hipblasCgetrs__funptr = NULL
cdef hipblasStatus_t hipblasCgetrs(hipblasHandle_t handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasComplex * A,const int lda,const int * ipiv,hipblasComplex * B,const int ldb,int * info) nogil:
    global _lib_handle
    global _hipblasCgetrs__funptr
    if _hipblasCgetrs__funptr == NULL:
        with gil:
            _hipblasCgetrs__funptr = loader.load_symbol(_lib_handle, "hipblasCgetrs")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,hipblasComplex *,const int,const int *,hipblasComplex *,const int,int *) nogil> _hipblasCgetrs__funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info)


cdef void* _hipblasZgetrs__funptr = NULL
cdef hipblasStatus_t hipblasZgetrs(hipblasHandle_t handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,const int * ipiv,hipblasDoubleComplex * B,const int ldb,int * info) nogil:
    global _lib_handle
    global _hipblasZgetrs__funptr
    if _hipblasZgetrs__funptr == NULL:
        with gil:
            _hipblasZgetrs__funptr = loader.load_symbol(_lib_handle, "hipblasZgetrs")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,hipblasDoubleComplex *,const int,const int *,hipblasDoubleComplex *,const int,int *) nogil> _hipblasZgetrs__funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info)


cdef void* _hipblasSgels__funptr = NULL
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
    global _hipblasSgels__funptr
    if _hipblasSgels__funptr == NULL:
        with gil:
            _hipblasSgels__funptr = loader.load_symbol(_lib_handle, "hipblasSgels")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,const int,float *,const int,float *,const int,int *,int *) nogil> _hipblasSgels__funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo)


cdef void* _hipblasDgels__funptr = NULL
cdef hipblasStatus_t hipblasDgels(hipblasHandle_t handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,double * A,const int lda,double * B,const int ldb,int * info,int * deviceInfo) nogil:
    global _lib_handle
    global _hipblasDgels__funptr
    if _hipblasDgels__funptr == NULL:
        with gil:
            _hipblasDgels__funptr = loader.load_symbol(_lib_handle, "hipblasDgels")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,const int,double *,const int,double *,const int,int *,int *) nogil> _hipblasDgels__funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo)


cdef void* _hipblasCgels__funptr = NULL
cdef hipblasStatus_t hipblasCgels(hipblasHandle_t handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasComplex * A,const int lda,hipblasComplex * B,const int ldb,int * info,int * deviceInfo) nogil:
    global _lib_handle
    global _hipblasCgels__funptr
    if _hipblasCgels__funptr == NULL:
        with gil:
            _hipblasCgels__funptr = loader.load_symbol(_lib_handle, "hipblasCgels")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,const int,hipblasComplex *,const int,hipblasComplex *,const int,int *,int *) nogil> _hipblasCgels__funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo)


cdef void* _hipblasZgels__funptr = NULL
cdef hipblasStatus_t hipblasZgels(hipblasHandle_t handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,hipblasDoubleComplex * B,const int ldb,int * info,int * deviceInfo) nogil:
    global _lib_handle
    global _hipblasZgels__funptr
    if _hipblasZgels__funptr == NULL:
        with gil:
            _hipblasZgels__funptr = loader.load_symbol(_lib_handle, "hipblasZgels")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,const int,const int,const int,hipblasDoubleComplex *,const int,hipblasDoubleComplex *,const int,int *,int *) nogil> _hipblasZgels__funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo)


cdef void* _hipblasSgeqrf__funptr = NULL
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
    global _hipblasSgeqrf__funptr
    if _hipblasSgeqrf__funptr == NULL:
        with gil:
            _hipblasSgeqrf__funptr = loader.load_symbol(_lib_handle, "hipblasSgeqrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,const int,float *,const int,float *,int *) nogil> _hipblasSgeqrf__funptr)(handle,m,n,A,lda,ipiv,info)


cdef void* _hipblasDgeqrf__funptr = NULL
cdef hipblasStatus_t hipblasDgeqrf(hipblasHandle_t handle,const int m,const int n,double * A,const int lda,double * ipiv,int * info) nogil:
    global _lib_handle
    global _hipblasDgeqrf__funptr
    if _hipblasDgeqrf__funptr == NULL:
        with gil:
            _hipblasDgeqrf__funptr = loader.load_symbol(_lib_handle, "hipblasDgeqrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,const int,double *,const int,double *,int *) nogil> _hipblasDgeqrf__funptr)(handle,m,n,A,lda,ipiv,info)


cdef void* _hipblasCgeqrf__funptr = NULL
cdef hipblasStatus_t hipblasCgeqrf(hipblasHandle_t handle,const int m,const int n,hipblasComplex * A,const int lda,hipblasComplex * ipiv,int * info) nogil:
    global _lib_handle
    global _hipblasCgeqrf__funptr
    if _hipblasCgeqrf__funptr == NULL:
        with gil:
            _hipblasCgeqrf__funptr = loader.load_symbol(_lib_handle, "hipblasCgeqrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,const int,hipblasComplex *,const int,hipblasComplex *,int *) nogil> _hipblasCgeqrf__funptr)(handle,m,n,A,lda,ipiv,info)


cdef void* _hipblasZgeqrf__funptr = NULL
cdef hipblasStatus_t hipblasZgeqrf(hipblasHandle_t handle,const int m,const int n,hipblasDoubleComplex * A,const int lda,hipblasDoubleComplex * ipiv,int * info) nogil:
    global _lib_handle
    global _hipblasZgeqrf__funptr
    if _hipblasZgeqrf__funptr == NULL:
        with gil:
            _hipblasZgeqrf__funptr = loader.load_symbol(_lib_handle, "hipblasZgeqrf")
    return (<hipblasStatus_t (*)(hipblasHandle_t,const int,const int,hipblasDoubleComplex *,const int,hipblasDoubleComplex *,int *) nogil> _hipblasZgeqrf__funptr)(handle,m,n,A,lda,ipiv,info)


cdef void* _hipblasGemmEx__funptr = NULL
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
    global _hipblasGemmEx__funptr
    if _hipblasGemmEx__funptr == NULL:
        with gil:
            _hipblasGemmEx__funptr = loader.load_symbol(_lib_handle, "hipblasGemmEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,const void *,const void *,hipblasDatatype_t,int,const void *,hipblasDatatype_t,int,const void *,void *,hipblasDatatype_t,int,hipblasDatatype_t,hipblasGemmAlgo_t) nogil> _hipblasGemmEx__funptr)(handle,transA,transB,m,n,k,alpha,A,aType,lda,B,bType,ldb,beta,C,cType,ldc,computeType,algo)


cdef void* _hipblasTrsmEx__funptr = NULL
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
    global _hipblasTrsmEx__funptr
    if _hipblasTrsmEx__funptr == NULL:
        with gil:
            _hipblasTrsmEx__funptr = loader.load_symbol(_lib_handle, "hipblasTrsmEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const void *,void *,int,void *,int,const void *,int,hipblasDatatype_t) nogil> _hipblasTrsmEx__funptr)(handle,side,uplo,transA,diag,m,n,alpha,A,lda,B,ldb,invA,invAsize,computeType)


cdef void* _hipblasAxpyEx__funptr = NULL
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
    global _hipblasAxpyEx__funptr
    if _hipblasAxpyEx__funptr == NULL:
        with gil:
            _hipblasAxpyEx__funptr = loader.load_symbol(_lib_handle, "hipblasAxpyEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const void *,hipblasDatatype_t,const void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,int,hipblasDatatype_t) nogil> _hipblasAxpyEx__funptr)(handle,n,alpha,alphaType,x,xType,incx,y,yType,incy,executionType)


cdef void* _hipblasDotEx__funptr = NULL
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
    global _hipblasDotEx__funptr
    if _hipblasDotEx__funptr == NULL:
        with gil:
            _hipblasDotEx__funptr = loader.load_symbol(_lib_handle, "hipblasDotEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const void *,hipblasDatatype_t,int,const void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasDotEx__funptr)(handle,n,x,xType,incx,y,yType,incy,result,resultType,executionType)


cdef void* _hipblasDotcEx__funptr = NULL
cdef hipblasStatus_t hipblasDotcEx(hipblasHandle_t handle,int n,const void * x,hipblasDatatype_t xType,int incx,const void * y,hipblasDatatype_t yType,int incy,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil:
    global _lib_handle
    global _hipblasDotcEx__funptr
    if _hipblasDotcEx__funptr == NULL:
        with gil:
            _hipblasDotcEx__funptr = loader.load_symbol(_lib_handle, "hipblasDotcEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const void *,hipblasDatatype_t,int,const void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasDotcEx__funptr)(handle,n,x,xType,incx,y,yType,incy,result,resultType,executionType)


cdef void* _hipblasNrm2Ex__funptr = NULL
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
    global _hipblasNrm2Ex__funptr
    if _hipblasNrm2Ex__funptr == NULL:
        with gil:
            _hipblasNrm2Ex__funptr = loader.load_symbol(_lib_handle, "hipblasNrm2Ex")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasNrm2Ex__funptr)(handle,n,x,xType,incx,result,resultType,executionType)


cdef void* _hipblasRotEx__funptr = NULL
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
    global _hipblasRotEx__funptr
    if _hipblasRotEx__funptr == NULL:
        with gil:
            _hipblasRotEx__funptr = loader.load_symbol(_lib_handle, "hipblasRotEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,int,const void *,const void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasRotEx__funptr)(handle,n,x,xType,incx,y,yType,incy,c,s,csType,executionType)


cdef void* _hipblasScalEx__funptr = NULL
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
    global _hipblasScalEx__funptr
    if _hipblasScalEx__funptr == NULL:
        with gil:
            _hipblasScalEx__funptr = loader.load_symbol(_lib_handle, "hipblasScalEx")
    return (<hipblasStatus_t (*)(hipblasHandle_t,int,const void *,hipblasDatatype_t,void *,hipblasDatatype_t,int,hipblasDatatype_t) nogil> _hipblasScalEx__funptr)(handle,n,alpha,alphaType,x,xType,incx,executionType)


cdef void* _hipblasStatusToString__funptr = NULL
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
    global _hipblasStatusToString__funptr
    if _hipblasStatusToString__funptr == NULL:
        with gil:
            _hipblasStatusToString__funptr = loader.load_symbol(_lib_handle, "hipblasStatusToString")
    return (<const char * (*)(hipblasStatus_t) nogil> _hipblasStatusToString__funptr)(status)
