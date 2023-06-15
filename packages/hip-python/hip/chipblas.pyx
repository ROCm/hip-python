# AMD_COPYRIGHT
cimport hip._util.posixloader as loader
cdef void* _lib_handle = NULL

cdef void __init() nogil:
    global _lib_handle
    if _lib_handle == NULL:
        with gil:
            _lib_handle = loader.open_library("libhipblas.so")

cdef void __init_symbol(void** result, const char* name) nogil:
    global _lib_handle
    if _lib_handle == NULL:
        __init()
    if result[0] == NULL:
        with gil:
            result[0] = loader.load_symbol(_lib_handle, name) 


cdef void* _hipblasCreate__funptr = NULL
# \brief Create hipblas handle. */
cdef hipblasStatus_t hipblasCreate(void ** handle) nogil:
    global _hipblasCreate__funptr
    __init_symbol(&_hipblasCreate__funptr,"hipblasCreate")
    return (<hipblasStatus_t (*)(void **) nogil> _hipblasCreate__funptr)(handle)


cdef void* _hipblasDestroy__funptr = NULL
# \brief Destroys the library context created using hipblasCreate() */
cdef hipblasStatus_t hipblasDestroy(void * handle) nogil:
    global _hipblasDestroy__funptr
    __init_symbol(&_hipblasDestroy__funptr,"hipblasDestroy")
    return (<hipblasStatus_t (*)(void *) nogil> _hipblasDestroy__funptr)(handle)


cdef void* _hipblasSetStream__funptr = NULL
# \brief Set stream for handle */
cdef hipblasStatus_t hipblasSetStream(void * handle,hipStream_t streamId) nogil:
    global _hipblasSetStream__funptr
    __init_symbol(&_hipblasSetStream__funptr,"hipblasSetStream")
    return (<hipblasStatus_t (*)(void *,hipStream_t) nogil> _hipblasSetStream__funptr)(handle,streamId)


cdef void* _hipblasGetStream__funptr = NULL
# \brief Get stream[0] for handle */
cdef hipblasStatus_t hipblasGetStream(void * handle,hipStream_t* streamId) nogil:
    global _hipblasGetStream__funptr
    __init_symbol(&_hipblasGetStream__funptr,"hipblasGetStream")
    return (<hipblasStatus_t (*)(void *,hipStream_t*) nogil> _hipblasGetStream__funptr)(handle,streamId)


cdef void* _hipblasSetPointerMode__funptr = NULL
# \brief Set hipblas pointer mode */
cdef hipblasStatus_t hipblasSetPointerMode(void * handle,hipblasPointerMode_t mode) nogil:
    global _hipblasSetPointerMode__funptr
    __init_symbol(&_hipblasSetPointerMode__funptr,"hipblasSetPointerMode")
    return (<hipblasStatus_t (*)(void *,hipblasPointerMode_t) nogil> _hipblasSetPointerMode__funptr)(handle,mode)


cdef void* _hipblasGetPointerMode__funptr = NULL
# \brief Get hipblas pointer mode */
cdef hipblasStatus_t hipblasGetPointerMode(void * handle,hipblasPointerMode_t * mode) nogil:
    global _hipblasGetPointerMode__funptr
    __init_symbol(&_hipblasGetPointerMode__funptr,"hipblasGetPointerMode")
    return (<hipblasStatus_t (*)(void *,hipblasPointerMode_t *) nogil> _hipblasGetPointerMode__funptr)(handle,mode)


cdef void* _hipblasSetInt8Datatype__funptr = NULL
# \brief Set hipblas int8 Datatype */
cdef hipblasStatus_t hipblasSetInt8Datatype(void * handle,hipblasInt8Datatype_t int8Type) nogil:
    global _hipblasSetInt8Datatype__funptr
    __init_symbol(&_hipblasSetInt8Datatype__funptr,"hipblasSetInt8Datatype")
    return (<hipblasStatus_t (*)(void *,hipblasInt8Datatype_t) nogil> _hipblasSetInt8Datatype__funptr)(handle,int8Type)


cdef void* _hipblasGetInt8Datatype__funptr = NULL
# \brief Get hipblas int8 Datatype*/
cdef hipblasStatus_t hipblasGetInt8Datatype(void * handle,hipblasInt8Datatype_t * int8Type) nogil:
    global _hipblasGetInt8Datatype__funptr
    __init_symbol(&_hipblasGetInt8Datatype__funptr,"hipblasGetInt8Datatype")
    return (<hipblasStatus_t (*)(void *,hipblasInt8Datatype_t *) nogil> _hipblasGetInt8Datatype__funptr)(handle,int8Type)


cdef void* _hipblasSetVector__funptr = NULL
# \brief copy vector from host to device
# @param[in]
# n           [int]
#             number of elements in the vector
# @param[in]
# elemSize    [int]
#             Size of both vectors in bytes
# @param[in]
# x           pointer to vector on the host
# @param[in]
# incx        [int]
#             specifies the increment for the elements of the vector
# @param[out]
# y           pointer to vector on the device
# @param[in]
# incy        [int]
#             specifies the increment for the elements of the vector
cdef hipblasStatus_t hipblasSetVector(int n,int elemSize,const void * x,int incx,void * y,int incy) nogil:
    global _hipblasSetVector__funptr
    __init_symbol(&_hipblasSetVector__funptr,"hipblasSetVector")
    return (<hipblasStatus_t (*)(int,int,const void *,int,void *,int) nogil> _hipblasSetVector__funptr)(n,elemSize,x,incx,y,incy)


cdef void* _hipblasGetVector__funptr = NULL
# \brief copy vector from device to host
# @param[in]
# n           [int]
#             number of elements in the vector
# @param[in]
# elemSize    [int]
#             Size of both vectors in bytes
# @param[in]
# x           pointer to vector on the device
# @param[in]
# incx        [int]
#             specifies the increment for the elements of the vector
# @param[out]
# y           pointer to vector on the host
# @param[in]
# incy        [int]
#             specifies the increment for the elements of the vector
cdef hipblasStatus_t hipblasGetVector(int n,int elemSize,const void * x,int incx,void * y,int incy) nogil:
    global _hipblasGetVector__funptr
    __init_symbol(&_hipblasGetVector__funptr,"hipblasGetVector")
    return (<hipblasStatus_t (*)(int,int,const void *,int,void *,int) nogil> _hipblasGetVector__funptr)(n,elemSize,x,incx,y,incy)


cdef void* _hipblasSetMatrix__funptr = NULL
# \brief copy matrix from host to device
# @param[in]
# rows        [int]
#             number of rows in matrices
# @param[in]
# cols        [int]
#             number of columns in matrices
# @param[in]
# elemSize   [int]
#             number of bytes per element in the matrix
# @param[in]
# AP          pointer to matrix on the host
# @param[in]
# lda         [int]
#             specifies the leading dimension of A, lda >= rows
# @param[out]
# BP           pointer to matrix on the GPU
# @param[in]
# ldb         [int]
#             specifies the leading dimension of B, ldb >= rows
cdef hipblasStatus_t hipblasSetMatrix(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb) nogil:
    global _hipblasSetMatrix__funptr
    __init_symbol(&_hipblasSetMatrix__funptr,"hipblasSetMatrix")
    return (<hipblasStatus_t (*)(int,int,int,const void *,int,void *,int) nogil> _hipblasSetMatrix__funptr)(rows,cols,elemSize,AP,lda,BP,ldb)


cdef void* _hipblasGetMatrix__funptr = NULL
# \brief copy matrix from device to host
# @param[in]
# rows        [int]
#             number of rows in matrices
# @param[in]
# cols        [int]
#             number of columns in matrices
# @param[in]
# elemSize   [int]
#             number of bytes per element in the matrix
# @param[in]
# AP          pointer to matrix on the GPU
# @param[in]
# lda         [int]
#             specifies the leading dimension of A, lda >= rows
# @param[out]
# BP          pointer to matrix on the host
# @param[in]
# ldb         [int]
#             specifies the leading dimension of B, ldb >= rows
cdef hipblasStatus_t hipblasGetMatrix(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb) nogil:
    global _hipblasGetMatrix__funptr
    __init_symbol(&_hipblasGetMatrix__funptr,"hipblasGetMatrix")
    return (<hipblasStatus_t (*)(int,int,int,const void *,int,void *,int) nogil> _hipblasGetMatrix__funptr)(rows,cols,elemSize,AP,lda,BP,ldb)


cdef void* _hipblasSetVectorAsync__funptr = NULL
# \brief asynchronously copy vector from host to device
# \details
# hipblasSetVectorAsync copies a vector from pinned host memory to device memory asynchronously.
# Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
# @param[in]
# n           [int]
#             number of elements in the vector
# @param[in]
# elemSize   [int]
#             number of bytes per element in the matrix
# @param[in]
# x           pointer to vector on the host
# @param[in]
# incx        [int]
#             specifies the increment for the elements of the vector
# @param[out]
# y           pointer to vector on the device
# @param[in]
# incy        [int]
#             specifies the increment for the elements of the vector
# @param[in]
# stream      specifies the stream into which this transfer request is queued
cdef hipblasStatus_t hipblasSetVectorAsync(int n,int elemSize,const void * x,int incx,void * y,int incy,hipStream_t stream) nogil:
    global _hipblasSetVectorAsync__funptr
    __init_symbol(&_hipblasSetVectorAsync__funptr,"hipblasSetVectorAsync")
    return (<hipblasStatus_t (*)(int,int,const void *,int,void *,int,hipStream_t) nogil> _hipblasSetVectorAsync__funptr)(n,elemSize,x,incx,y,incy,stream)


cdef void* _hipblasGetVectorAsync__funptr = NULL
# \brief asynchronously copy vector from device to host
# \details
# hipblasGetVectorAsync copies a vector from pinned host memory to device memory asynchronously.
# Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
# @param[in]
# n           [int]
#             number of elements in the vector
# @param[in]
# elemSize   [int]
#             number of bytes per element in the matrix
# @param[in]
# x           pointer to vector on the device
# @param[in]
# incx        [int]
#             specifies the increment for the elements of the vector
# @param[out]
# y           pointer to vector on the host
# @param[in]
# incy        [int]
#             specifies the increment for the elements of the vector
# @param[in]
# stream      specifies the stream into which this transfer request is queued
cdef hipblasStatus_t hipblasGetVectorAsync(int n,int elemSize,const void * x,int incx,void * y,int incy,hipStream_t stream) nogil:
    global _hipblasGetVectorAsync__funptr
    __init_symbol(&_hipblasGetVectorAsync__funptr,"hipblasGetVectorAsync")
    return (<hipblasStatus_t (*)(int,int,const void *,int,void *,int,hipStream_t) nogil> _hipblasGetVectorAsync__funptr)(n,elemSize,x,incx,y,incy,stream)


cdef void* _hipblasSetMatrixAsync__funptr = NULL
# \brief asynchronously copy matrix from host to device
# \details
# hipblasSetMatrixAsync copies a matrix from pinned host memory to device memory asynchronously.
# Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
# @param[in]
# rows        [int]
#             number of rows in matrices
# @param[in]
# cols        [int]
#             number of columns in matrices
# @param[in]
# elemSize   [int]
#             number of bytes per element in the matrix
# @param[in]
# AP           pointer to matrix on the host
# @param[in]
# lda         [int]
#             specifies the leading dimension of A, lda >= rows
# @param[out]
# BP           pointer to matrix on the GPU
# @param[in]
# ldb         [int]
#             specifies the leading dimension of B, ldb >= rows
# @param[in]
# stream      specifies the stream into which this transfer request is queued
cdef hipblasStatus_t hipblasSetMatrixAsync(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb,hipStream_t stream) nogil:
    global _hipblasSetMatrixAsync__funptr
    __init_symbol(&_hipblasSetMatrixAsync__funptr,"hipblasSetMatrixAsync")
    return (<hipblasStatus_t (*)(int,int,int,const void *,int,void *,int,hipStream_t) nogil> _hipblasSetMatrixAsync__funptr)(rows,cols,elemSize,AP,lda,BP,ldb,stream)


cdef void* _hipblasGetMatrixAsync__funptr = NULL
# \brief asynchronously copy matrix from device to host
# \details
# hipblasGetMatrixAsync copies a matrix from device memory to pinned host memory asynchronously.
# Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
# @param[in]
# rows        [int]
#             number of rows in matrices
# @param[in]
# cols        [int]
#             number of columns in matrices
# @param[in]
# elemSize   [int]
#             number of bytes per element in the matrix
# @param[in]
# AP          pointer to matrix on the GPU
# @param[in]
# lda         [int]
#             specifies the leading dimension of A, lda >= rows
# @param[out]
# BP           pointer to matrix on the host
# @param[in]
# ldb         [int]
#             specifies the leading dimension of B, ldb >= rows
# @param[in]
# stream      specifies the stream into which this transfer request is queued
cdef hipblasStatus_t hipblasGetMatrixAsync(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb,hipStream_t stream) nogil:
    global _hipblasGetMatrixAsync__funptr
    __init_symbol(&_hipblasGetMatrixAsync__funptr,"hipblasGetMatrixAsync")
    return (<hipblasStatus_t (*)(int,int,int,const void *,int,void *,int,hipStream_t) nogil> _hipblasGetMatrixAsync__funptr)(rows,cols,elemSize,AP,lda,BP,ldb,stream)


cdef void* _hipblasSetAtomicsMode__funptr = NULL
# \brief Set hipblasSetAtomicsMode*/
cdef hipblasStatus_t hipblasSetAtomicsMode(void * handle,hipblasAtomicsMode_t atomics_mode) nogil:
    global _hipblasSetAtomicsMode__funptr
    __init_symbol(&_hipblasSetAtomicsMode__funptr,"hipblasSetAtomicsMode")
    return (<hipblasStatus_t (*)(void *,hipblasAtomicsMode_t) nogil> _hipblasSetAtomicsMode__funptr)(handle,atomics_mode)


cdef void* _hipblasGetAtomicsMode__funptr = NULL
# \brief Get hipblasSetAtomicsMode*/
cdef hipblasStatus_t hipblasGetAtomicsMode(void * handle,hipblasAtomicsMode_t * atomics_mode) nogil:
    global _hipblasGetAtomicsMode__funptr
    __init_symbol(&_hipblasGetAtomicsMode__funptr,"hipblasGetAtomicsMode")
    return (<hipblasStatus_t (*)(void *,hipblasAtomicsMode_t *) nogil> _hipblasGetAtomicsMode__funptr)(handle,atomics_mode)


cdef void* _hipblasIsamax__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# amax finds the first index of the element of maximum magnitude of a vector x.
# 
# - Supported precisions in rocBLAS : s,d,c,z.
# - Supported precisions in cuBLAS  : s,d,c,z.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# result
#           device pointer or host pointer to store the amax index.
#           return is 0.0 if n, incx<=0.
cdef hipblasStatus_t hipblasIsamax(void * handle,int n,const float * x,int incx,int * result) nogil:
    global _hipblasIsamax__funptr
    __init_symbol(&_hipblasIsamax__funptr,"hipblasIsamax")
    return (<hipblasStatus_t (*)(void *,int,const float *,int,int *) nogil> _hipblasIsamax__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIdamax__funptr = NULL
cdef hipblasStatus_t hipblasIdamax(void * handle,int n,const double * x,int incx,int * result) nogil:
    global _hipblasIdamax__funptr
    __init_symbol(&_hipblasIdamax__funptr,"hipblasIdamax")
    return (<hipblasStatus_t (*)(void *,int,const double *,int,int *) nogil> _hipblasIdamax__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIcamax__funptr = NULL
cdef hipblasStatus_t hipblasIcamax(void * handle,int n,hipblasComplex * x,int incx,int * result) nogil:
    global _hipblasIcamax__funptr
    __init_symbol(&_hipblasIcamax__funptr,"hipblasIcamax")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,int *) nogil> _hipblasIcamax__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIzamax__funptr = NULL
cdef hipblasStatus_t hipblasIzamax(void * handle,int n,hipblasDoubleComplex * x,int incx,int * result) nogil:
    global _hipblasIzamax__funptr
    __init_symbol(&_hipblasIzamax__funptr,"hipblasIzamax")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,int *) nogil> _hipblasIzamax__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIsamaxBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
#  amaxBatched finds the first index of the element of maximum magnitude of each vector x_i in a batch, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z.
# - Supported precisions in cuBLAS  : No support.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each vector x_i
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# batchCount [int]
#           number of instances in the batch, must be > 0.
# @param[out]
# result
#           device or host array of pointers of batchCount size for results.
#           return is 0 if n, incx<=0.
cdef hipblasStatus_t hipblasIsamaxBatched(void * handle,int n,const float *const* x,int incx,int batchCount,int * result) nogil:
    global _hipblasIsamaxBatched__funptr
    __init_symbol(&_hipblasIsamaxBatched__funptr,"hipblasIsamaxBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *const*,int,int,int *) nogil> _hipblasIsamaxBatched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasIdamaxBatched__funptr = NULL
cdef hipblasStatus_t hipblasIdamaxBatched(void * handle,int n,const double *const* x,int incx,int batchCount,int * result) nogil:
    global _hipblasIdamaxBatched__funptr
    __init_symbol(&_hipblasIdamaxBatched__funptr,"hipblasIdamaxBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *const*,int,int,int *) nogil> _hipblasIdamaxBatched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasIcamaxBatched__funptr = NULL
cdef hipblasStatus_t hipblasIcamaxBatched(void * handle,int n,hipblasComplex *const* x,int incx,int batchCount,int * result) nogil:
    global _hipblasIcamaxBatched__funptr
    __init_symbol(&_hipblasIcamaxBatched__funptr,"hipblasIcamaxBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *const*,int,int,int *) nogil> _hipblasIcamaxBatched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasIzamaxBatched__funptr = NULL
cdef hipblasStatus_t hipblasIzamaxBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,int batchCount,int * result) nogil:
    global _hipblasIzamaxBatched__funptr
    __init_symbol(&_hipblasIzamaxBatched__funptr,"hipblasIzamaxBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *const*,int,int,int *) nogil> _hipblasIzamaxBatched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasIsamaxStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
#  amaxStridedBatched finds the first index of the element of maximum magnitude of each vector x_i in a batch, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each vector x_i
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# stridex   [hipblasStride]
#           specifies the pointer increment between one x_i and the next x_(i + 1).
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device or host pointer for storing contiguous batchCount results.
#           return is 0 if n <= 0, incx<=0.
#
cdef hipblasStatus_t hipblasIsamaxStridedBatched(void * handle,int n,const float * x,int incx,long stridex,int batchCount,int * result) nogil:
    global _hipblasIsamaxStridedBatched__funptr
    __init_symbol(&_hipblasIsamaxStridedBatched__funptr,"hipblasIsamaxStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *,int,long,int,int *) nogil> _hipblasIsamaxStridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasIdamaxStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasIdamaxStridedBatched(void * handle,int n,const double * x,int incx,long stridex,int batchCount,int * result) nogil:
    global _hipblasIdamaxStridedBatched__funptr
    __init_symbol(&_hipblasIdamaxStridedBatched__funptr,"hipblasIdamaxStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *,int,long,int,int *) nogil> _hipblasIdamaxStridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasIcamaxStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasIcamaxStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,int batchCount,int * result) nogil:
    global _hipblasIcamaxStridedBatched__funptr
    __init_symbol(&_hipblasIcamaxStridedBatched__funptr,"hipblasIcamaxStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,long,int,int *) nogil> _hipblasIcamaxStridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasIzamaxStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasIzamaxStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,int batchCount,int * result) nogil:
    global _hipblasIzamaxStridedBatched__funptr
    __init_symbol(&_hipblasIzamaxStridedBatched__funptr,"hipblasIzamaxStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,long,int,int *) nogil> _hipblasIzamaxStridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasIsamin__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# amin finds the first index of the element of minimum magnitude of a vector x.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# result
#           device pointer or host pointer to store the amin index.
#           return is 0.0 if n, incx<=0.
cdef hipblasStatus_t hipblasIsamin(void * handle,int n,const float * x,int incx,int * result) nogil:
    global _hipblasIsamin__funptr
    __init_symbol(&_hipblasIsamin__funptr,"hipblasIsamin")
    return (<hipblasStatus_t (*)(void *,int,const float *,int,int *) nogil> _hipblasIsamin__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIdamin__funptr = NULL
cdef hipblasStatus_t hipblasIdamin(void * handle,int n,const double * x,int incx,int * result) nogil:
    global _hipblasIdamin__funptr
    __init_symbol(&_hipblasIdamin__funptr,"hipblasIdamin")
    return (<hipblasStatus_t (*)(void *,int,const double *,int,int *) nogil> _hipblasIdamin__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIcamin__funptr = NULL
cdef hipblasStatus_t hipblasIcamin(void * handle,int n,hipblasComplex * x,int incx,int * result) nogil:
    global _hipblasIcamin__funptr
    __init_symbol(&_hipblasIcamin__funptr,"hipblasIcamin")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,int *) nogil> _hipblasIcamin__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIzamin__funptr = NULL
cdef hipblasStatus_t hipblasIzamin(void * handle,int n,hipblasDoubleComplex * x,int incx,int * result) nogil:
    global _hipblasIzamin__funptr
    __init_symbol(&_hipblasIzamin__funptr,"hipblasIzamin")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,int *) nogil> _hipblasIzamin__funptr)(handle,n,x,incx,result)


cdef void* _hipblasIsaminBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# aminBatched finds the first index of the element of minimum magnitude of each vector x_i in a batch, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each vector x_i
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# batchCount [int]
#           number of instances in the batch, must be > 0.
# @param[out]
# result
#           device or host pointers to array of batchCount size for results.
#           return is 0 if n, incx<=0.
cdef hipblasStatus_t hipblasIsaminBatched(void * handle,int n,const float *const* x,int incx,int batchCount,int * result) nogil:
    global _hipblasIsaminBatched__funptr
    __init_symbol(&_hipblasIsaminBatched__funptr,"hipblasIsaminBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *const*,int,int,int *) nogil> _hipblasIsaminBatched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasIdaminBatched__funptr = NULL
cdef hipblasStatus_t hipblasIdaminBatched(void * handle,int n,const double *const* x,int incx,int batchCount,int * result) nogil:
    global _hipblasIdaminBatched__funptr
    __init_symbol(&_hipblasIdaminBatched__funptr,"hipblasIdaminBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *const*,int,int,int *) nogil> _hipblasIdaminBatched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasIcaminBatched__funptr = NULL
cdef hipblasStatus_t hipblasIcaminBatched(void * handle,int n,hipblasComplex *const* x,int incx,int batchCount,int * result) nogil:
    global _hipblasIcaminBatched__funptr
    __init_symbol(&_hipblasIcaminBatched__funptr,"hipblasIcaminBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *const*,int,int,int *) nogil> _hipblasIcaminBatched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasIzaminBatched__funptr = NULL
cdef hipblasStatus_t hipblasIzaminBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,int batchCount,int * result) nogil:
    global _hipblasIzaminBatched__funptr
    __init_symbol(&_hipblasIzaminBatched__funptr,"hipblasIzaminBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *const*,int,int,int *) nogil> _hipblasIzaminBatched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasIsaminStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
#  aminStridedBatched finds the first index of the element of minimum magnitude of each vector x_i in a batch, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each vector x_i
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# stridex   [hipblasStride]
#           specifies the pointer increment between one x_i and the next x_(i + 1)
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device or host pointer to array for storing contiguous batchCount results.
#           return is 0 if n <= 0, incx<=0.
#
cdef hipblasStatus_t hipblasIsaminStridedBatched(void * handle,int n,const float * x,int incx,long stridex,int batchCount,int * result) nogil:
    global _hipblasIsaminStridedBatched__funptr
    __init_symbol(&_hipblasIsaminStridedBatched__funptr,"hipblasIsaminStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *,int,long,int,int *) nogil> _hipblasIsaminStridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasIdaminStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasIdaminStridedBatched(void * handle,int n,const double * x,int incx,long stridex,int batchCount,int * result) nogil:
    global _hipblasIdaminStridedBatched__funptr
    __init_symbol(&_hipblasIdaminStridedBatched__funptr,"hipblasIdaminStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *,int,long,int,int *) nogil> _hipblasIdaminStridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasIcaminStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasIcaminStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,int batchCount,int * result) nogil:
    global _hipblasIcaminStridedBatched__funptr
    __init_symbol(&_hipblasIcaminStridedBatched__funptr,"hipblasIcaminStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,long,int,int *) nogil> _hipblasIcaminStridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasIzaminStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasIzaminStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,int batchCount,int * result) nogil:
    global _hipblasIzaminStridedBatched__funptr
    __init_symbol(&_hipblasIzaminStridedBatched__funptr,"hipblasIzaminStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,long,int,int *) nogil> _hipblasIzaminStridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasSasum__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# asum computes the sum of the magnitudes of elements of a real vector x,
#      or the sum of magnitudes of the real and imaginary parts of elements if x is a complex vector.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x. incx must be > 0.
# @param[inout]
# result
#           device pointer or host pointer to store the asum product.
#           return is 0.0 if n <= 0.
#
cdef hipblasStatus_t hipblasSasum(void * handle,int n,const float * x,int incx,float * result) nogil:
    global _hipblasSasum__funptr
    __init_symbol(&_hipblasSasum__funptr,"hipblasSasum")
    return (<hipblasStatus_t (*)(void *,int,const float *,int,float *) nogil> _hipblasSasum__funptr)(handle,n,x,incx,result)


cdef void* _hipblasDasum__funptr = NULL
cdef hipblasStatus_t hipblasDasum(void * handle,int n,const double * x,int incx,double * result) nogil:
    global _hipblasDasum__funptr
    __init_symbol(&_hipblasDasum__funptr,"hipblasDasum")
    return (<hipblasStatus_t (*)(void *,int,const double *,int,double *) nogil> _hipblasDasum__funptr)(handle,n,x,incx,result)


cdef void* _hipblasScasum__funptr = NULL
cdef hipblasStatus_t hipblasScasum(void * handle,int n,hipblasComplex * x,int incx,float * result) nogil:
    global _hipblasScasum__funptr
    __init_symbol(&_hipblasScasum__funptr,"hipblasScasum")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,float *) nogil> _hipblasScasum__funptr)(handle,n,x,incx,result)


cdef void* _hipblasDzasum__funptr = NULL
cdef hipblasStatus_t hipblasDzasum(void * handle,int n,hipblasDoubleComplex * x,int incx,double * result) nogil:
    global _hipblasDzasum__funptr
    __init_symbol(&_hipblasDzasum__funptr,"hipblasDzasum")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,double *) nogil> _hipblasDzasum__funptr)(handle,n,x,incx,result)


cdef void* _hipblasSasumBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# asumBatched computes the sum of the magnitudes of the elements in a batch of real vectors x_i,
#     or the sum of magnitudes of the real and imaginary parts of elements if x_i is a complex
#     vector, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each vector x_i
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# batchCount [int]
#           number of instances in the batch.
# @param[out]
# result
#           device array or host array of batchCount size for results.
#           return is 0.0 if n, incx<=0.
cdef hipblasStatus_t hipblasSasumBatched(void * handle,int n,const float *const* x,int incx,int batchCount,float * result) nogil:
    global _hipblasSasumBatched__funptr
    __init_symbol(&_hipblasSasumBatched__funptr,"hipblasSasumBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *const*,int,int,float *) nogil> _hipblasSasumBatched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasDasumBatched__funptr = NULL
cdef hipblasStatus_t hipblasDasumBatched(void * handle,int n,const double *const* x,int incx,int batchCount,double * result) nogil:
    global _hipblasDasumBatched__funptr
    __init_symbol(&_hipblasDasumBatched__funptr,"hipblasDasumBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *const*,int,int,double *) nogil> _hipblasDasumBatched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasScasumBatched__funptr = NULL
cdef hipblasStatus_t hipblasScasumBatched(void * handle,int n,hipblasComplex *const* x,int incx,int batchCount,float * result) nogil:
    global _hipblasScasumBatched__funptr
    __init_symbol(&_hipblasScasumBatched__funptr,"hipblasScasumBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *const*,int,int,float *) nogil> _hipblasScasumBatched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasDzasumBatched__funptr = NULL
cdef hipblasStatus_t hipblasDzasumBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,int batchCount,double * result) nogil:
    global _hipblasDzasumBatched__funptr
    __init_symbol(&_hipblasDzasumBatched__funptr,"hipblasDzasumBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *const*,int,int,double *) nogil> _hipblasDzasumBatched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasSasumStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# asumStridedBatched computes the sum of the magnitudes of elements of a real vectors x_i,
#     or the sum of magnitudes of the real and imaginary parts of elements if x_i is a complex
#     vector, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each vector x_i
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
#           There are no restrictions placed on stride_x, however the user should
#           take care to ensure that stride_x is of appropriate size, for a typical
#           case this means stride_x >= n * incx.
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device pointer or host pointer to array for storing contiguous batchCount results.
#           return is 0.0 if n, incx<=0.
cdef hipblasStatus_t hipblasSasumStridedBatched(void * handle,int n,const float * x,int incx,long stridex,int batchCount,float * result) nogil:
    global _hipblasSasumStridedBatched__funptr
    __init_symbol(&_hipblasSasumStridedBatched__funptr,"hipblasSasumStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *,int,long,int,float *) nogil> _hipblasSasumStridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasDasumStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDasumStridedBatched(void * handle,int n,const double * x,int incx,long stridex,int batchCount,double * result) nogil:
    global _hipblasDasumStridedBatched__funptr
    __init_symbol(&_hipblasDasumStridedBatched__funptr,"hipblasDasumStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *,int,long,int,double *) nogil> _hipblasDasumStridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasScasumStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasScasumStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,int batchCount,float * result) nogil:
    global _hipblasScasumStridedBatched__funptr
    __init_symbol(&_hipblasScasumStridedBatched__funptr,"hipblasScasumStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,long,int,float *) nogil> _hipblasScasumStridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasDzasumStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDzasumStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,int batchCount,double * result) nogil:
    global _hipblasDzasumStridedBatched__funptr
    __init_symbol(&_hipblasDzasumStridedBatched__funptr,"hipblasDzasumStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,long,int,double *) nogil> _hipblasDzasumStridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasHaxpy__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# axpy   computes constant alpha multiplied by vector x, plus vector y
# 
#     y := alpha * x + y
# 
# - Supported precisions in rocBLAS : h,s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[in]
# alpha     device pointer or host pointer to specify the scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[out]
# y         device pointer storing vector y.
# @param[inout]
# incy      [int]
#           specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasHaxpy(void * handle,int n,const unsigned short * alpha,const unsigned short * x,int incx,unsigned short * y,int incy) nogil:
    global _hipblasHaxpy__funptr
    __init_symbol(&_hipblasHaxpy__funptr,"hipblasHaxpy")
    return (<hipblasStatus_t (*)(void *,int,const unsigned short *,const unsigned short *,int,unsigned short *,int) nogil> _hipblasHaxpy__funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* _hipblasSaxpy__funptr = NULL
cdef hipblasStatus_t hipblasSaxpy(void * handle,int n,const float * alpha,const float * x,int incx,float * y,int incy) nogil:
    global _hipblasSaxpy__funptr
    __init_symbol(&_hipblasSaxpy__funptr,"hipblasSaxpy")
    return (<hipblasStatus_t (*)(void *,int,const float *,const float *,int,float *,int) nogil> _hipblasSaxpy__funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* _hipblasDaxpy__funptr = NULL
cdef hipblasStatus_t hipblasDaxpy(void * handle,int n,const double * alpha,const double * x,int incx,double * y,int incy) nogil:
    global _hipblasDaxpy__funptr
    __init_symbol(&_hipblasDaxpy__funptr,"hipblasDaxpy")
    return (<hipblasStatus_t (*)(void *,int,const double *,const double *,int,double *,int) nogil> _hipblasDaxpy__funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* _hipblasCaxpy__funptr = NULL
cdef hipblasStatus_t hipblasCaxpy(void * handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil:
    global _hipblasCaxpy__funptr
    __init_symbol(&_hipblasCaxpy__funptr,"hipblasCaxpy")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCaxpy__funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* _hipblasZaxpy__funptr = NULL
cdef hipblasStatus_t hipblasZaxpy(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil:
    global _hipblasZaxpy__funptr
    __init_symbol(&_hipblasZaxpy__funptr,"hipblasZaxpy")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZaxpy__funptr)(handle,n,alpha,x,incx,y,incy)


cdef void* _hipblasHaxpyBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# axpyBatched   compute y := alpha * x + y over a set of batched vectors.
# 
# - Supported precisions in rocBLAS : h,s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[in]
# alpha     specifies the scalar alpha.
# @param[in]
# x         pointer storing vector x on the GPU.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[out]
# y         pointer storing vector y on the GPU.
# @param[inout]
# incy      [int]
#           specifies the increment for the elements of y.
# 
# @param[in]
# batchCount [int]
#           number of instances in the batch
cdef hipblasStatus_t hipblasHaxpyBatched(void * handle,int n,const unsigned short * alpha,const unsigned short *const* x,int incx,unsigned short *const* y,int incy,int batchCount) nogil:
    global _hipblasHaxpyBatched__funptr
    __init_symbol(&_hipblasHaxpyBatched__funptr,"hipblasHaxpyBatched")
    return (<hipblasStatus_t (*)(void *,int,const unsigned short *,const unsigned short *const*,int,unsigned short *const*,int,int) nogil> _hipblasHaxpyBatched__funptr)(handle,n,alpha,x,incx,y,incy,batchCount)


cdef void* _hipblasSaxpyBatched__funptr = NULL
cdef hipblasStatus_t hipblasSaxpyBatched(void * handle,int n,const float * alpha,const float *const* x,int incx,float *const* y,int incy,int batchCount) nogil:
    global _hipblasSaxpyBatched__funptr
    __init_symbol(&_hipblasSaxpyBatched__funptr,"hipblasSaxpyBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *,const float *const*,int,float *const*,int,int) nogil> _hipblasSaxpyBatched__funptr)(handle,n,alpha,x,incx,y,incy,batchCount)


cdef void* _hipblasDaxpyBatched__funptr = NULL
cdef hipblasStatus_t hipblasDaxpyBatched(void * handle,int n,const double * alpha,const double *const* x,int incx,double *const* y,int incy,int batchCount) nogil:
    global _hipblasDaxpyBatched__funptr
    __init_symbol(&_hipblasDaxpyBatched__funptr,"hipblasDaxpyBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *,const double *const*,int,double *const*,int,int) nogil> _hipblasDaxpyBatched__funptr)(handle,n,alpha,x,incx,y,incy,batchCount)


cdef void* _hipblasCaxpyBatched__funptr = NULL
cdef hipblasStatus_t hipblasCaxpyBatched(void * handle,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasCaxpyBatched__funptr
    __init_symbol(&_hipblasCaxpyBatched__funptr,"hipblasCaxpyBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCaxpyBatched__funptr)(handle,n,alpha,x,incx,y,incy,batchCount)


cdef void* _hipblasZaxpyBatched__funptr = NULL
cdef hipblasStatus_t hipblasZaxpyBatched(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasZaxpyBatched__funptr
    __init_symbol(&_hipblasZaxpyBatched__funptr,"hipblasZaxpyBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZaxpyBatched__funptr)(handle,n,alpha,x,incx,y,incy,batchCount)


cdef void* _hipblasHaxpyStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# axpyStridedBatched   compute y := alpha * x + y over a set of strided batched vectors.
# 
# - Supported precisions in rocBLAS : h,s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
# @param[in]
# alpha     specifies the scalar alpha.
# @param[in]
# x         pointer storing vector x on the GPU.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# stridex   [hipblasStride]
#           specifies the increment between vectors of x.
# @param[out]
# y         pointer storing vector y on the GPU.
# @param[inout]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# stridey   [hipblasStride]
#           specifies the increment between vectors of y.
# 
# @param[in]
# batchCount [int]
#           number of instances in the batch
#
cdef hipblasStatus_t hipblasHaxpyStridedBatched(void * handle,int n,const unsigned short * alpha,const unsigned short * x,int incx,long stridex,unsigned short * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasHaxpyStridedBatched__funptr
    __init_symbol(&_hipblasHaxpyStridedBatched__funptr,"hipblasHaxpyStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const unsigned short *,const unsigned short *,int,long,unsigned short *,int,long,int) nogil> _hipblasHaxpyStridedBatched__funptr)(handle,n,alpha,x,incx,stridex,y,incy,stridey,batchCount)


cdef void* _hipblasSaxpyStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasSaxpyStridedBatched(void * handle,int n,const float * alpha,const float * x,int incx,long stridex,float * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasSaxpyStridedBatched__funptr
    __init_symbol(&_hipblasSaxpyStridedBatched__funptr,"hipblasSaxpyStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *,const float *,int,long,float *,int,long,int) nogil> _hipblasSaxpyStridedBatched__funptr)(handle,n,alpha,x,incx,stridex,y,incy,stridey,batchCount)


cdef void* _hipblasDaxpyStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDaxpyStridedBatched(void * handle,int n,const double * alpha,const double * x,int incx,long stridex,double * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasDaxpyStridedBatched__funptr
    __init_symbol(&_hipblasDaxpyStridedBatched__funptr,"hipblasDaxpyStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *,const double *,int,long,double *,int,long,int) nogil> _hipblasDaxpyStridedBatched__funptr)(handle,n,alpha,x,incx,stridex,y,incy,stridey,batchCount)


cdef void* _hipblasCaxpyStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCaxpyStridedBatched(void * handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasCaxpyStridedBatched__funptr
    __init_symbol(&_hipblasCaxpyStridedBatched__funptr,"hipblasCaxpyStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCaxpyStridedBatched__funptr)(handle,n,alpha,x,incx,stridex,y,incy,stridey,batchCount)


cdef void* _hipblasZaxpyStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZaxpyStridedBatched(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasZaxpyStridedBatched__funptr
    __init_symbol(&_hipblasZaxpyStridedBatched__funptr,"hipblasZaxpyStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZaxpyStridedBatched__funptr)(handle,n,alpha,x,incx,stridex,y,incy,stridey,batchCount)


cdef void* _hipblasScopy__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# copy  copies each element x[i] into y[i], for  i = 1 , ... , n
# 
#     y := x,
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x to be copied to y.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[out]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasScopy(void * handle,int n,const float * x,int incx,float * y,int incy) nogil:
    global _hipblasScopy__funptr
    __init_symbol(&_hipblasScopy__funptr,"hipblasScopy")
    return (<hipblasStatus_t (*)(void *,int,const float *,int,float *,int) nogil> _hipblasScopy__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasDcopy__funptr = NULL
cdef hipblasStatus_t hipblasDcopy(void * handle,int n,const double * x,int incx,double * y,int incy) nogil:
    global _hipblasDcopy__funptr
    __init_symbol(&_hipblasDcopy__funptr,"hipblasDcopy")
    return (<hipblasStatus_t (*)(void *,int,const double *,int,double *,int) nogil> _hipblasDcopy__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasCcopy__funptr = NULL
cdef hipblasStatus_t hipblasCcopy(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil:
    global _hipblasCcopy__funptr
    __init_symbol(&_hipblasCcopy__funptr,"hipblasCcopy")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCcopy__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasZcopy__funptr = NULL
cdef hipblasStatus_t hipblasZcopy(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil:
    global _hipblasZcopy__funptr
    __init_symbol(&_hipblasZcopy__funptr,"hipblasZcopy")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZcopy__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasScopyBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# copyBatched copies each element x_i[j] into y_i[j], for  j = 1 , ... , n; i = 1 , ... , batchCount
# 
#     y_i := x_i,
# 
# where (x_i, y_i) is the i-th instance of the batch.
# x_i and y_i are vectors.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i to be copied to y_i.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i.
# @param[out]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasScopyBatched(void * handle,int n,const float *const* x,int incx,float *const* y,int incy,int batchCount) nogil:
    global _hipblasScopyBatched__funptr
    __init_symbol(&_hipblasScopyBatched__funptr,"hipblasScopyBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *const*,int,float *const*,int,int) nogil> _hipblasScopyBatched__funptr)(handle,n,x,incx,y,incy,batchCount)


cdef void* _hipblasDcopyBatched__funptr = NULL
cdef hipblasStatus_t hipblasDcopyBatched(void * handle,int n,const double *const* x,int incx,double *const* y,int incy,int batchCount) nogil:
    global _hipblasDcopyBatched__funptr
    __init_symbol(&_hipblasDcopyBatched__funptr,"hipblasDcopyBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *const*,int,double *const*,int,int) nogil> _hipblasDcopyBatched__funptr)(handle,n,x,incx,y,incy,batchCount)


cdef void* _hipblasCcopyBatched__funptr = NULL
cdef hipblasStatus_t hipblasCcopyBatched(void * handle,int n,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasCcopyBatched__funptr
    __init_symbol(&_hipblasCcopyBatched__funptr,"hipblasCcopyBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCcopyBatched__funptr)(handle,n,x,incx,y,incy,batchCount)


cdef void* _hipblasZcopyBatched__funptr = NULL
cdef hipblasStatus_t hipblasZcopyBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasZcopyBatched__funptr
    __init_symbol(&_hipblasZcopyBatched__funptr,"hipblasZcopyBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZcopyBatched__funptr)(handle,n,x,incx,y,incy,batchCount)


cdef void* _hipblasScopyStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# copyStridedBatched copies each element x_i[j] into y_i[j], for  j = 1 , ... , n; i = 1 , ... , batchCount
# 
#     y_i := x_i,
# 
# where (x_i, y_i) is the i-th instance of the batch.
# x_i and y_i are vectors.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i to be copied to y_i.
# @param[in]
# x         device pointer to the first vector (x_1) in the batch.
# @param[in]
# incx      [int]
#           specifies the increments for the elements of vectors x_i.
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1).
#             There are no restrictions placed on stride_x, however the user should
#             take care to ensure that stride_x is of appropriate size, for a typical
#             case this means stride_x >= n * incx.
# @param[out]
# y         device pointer to the first vector (y_1) in the batch.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of vectors y_i.
# @param[in]
# stridey     [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1).
#             There are no restrictions placed on stride_y, however the user should
#             take care to ensure that stride_y is of appropriate size, for a typical
#             case this means stride_y >= n * incy. stridey should be non zero.
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasScopyStridedBatched(void * handle,int n,const float * x,int incx,long stridex,float * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasScopyStridedBatched__funptr
    __init_symbol(&_hipblasScopyStridedBatched__funptr,"hipblasScopyStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *,int,long,float *,int,long,int) nogil> _hipblasScopyStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount)


cdef void* _hipblasDcopyStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDcopyStridedBatched(void * handle,int n,const double * x,int incx,long stridex,double * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasDcopyStridedBatched__funptr
    __init_symbol(&_hipblasDcopyStridedBatched__funptr,"hipblasDcopyStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *,int,long,double *,int,long,int) nogil> _hipblasDcopyStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount)


cdef void* _hipblasCcopyStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCcopyStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasCcopyStridedBatched__funptr
    __init_symbol(&_hipblasCcopyStridedBatched__funptr,"hipblasCcopyStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCcopyStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount)


cdef void* _hipblasZcopyStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZcopyStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasZcopyStridedBatched__funptr
    __init_symbol(&_hipblasZcopyStridedBatched__funptr,"hipblasZcopyStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZcopyStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount)


cdef void* _hipblasHdot__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# dot(u)  performs the dot product of vectors x and y
# 
#     result = x * y;
# 
# dotc  performs the dot product of the conjugate of complex vector x and complex vector y
# 
#     result = conjugate (x) * y;
# 
# - Supported precisions in rocBLAS : h,bf,s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of y.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# result
#           device pointer or host pointer to store the dot product.
#           return is 0.0 if n <= 0.
#
cdef hipblasStatus_t hipblasHdot(void * handle,int n,const unsigned short * x,int incx,const unsigned short * y,int incy,unsigned short * result) nogil:
    global _hipblasHdot__funptr
    __init_symbol(&_hipblasHdot__funptr,"hipblasHdot")
    return (<hipblasStatus_t (*)(void *,int,const unsigned short *,int,const unsigned short *,int,unsigned short *) nogil> _hipblasHdot__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasBfdot__funptr = NULL
cdef hipblasStatus_t hipblasBfdot(void * handle,int n,hipblasBfloat16 * x,int incx,hipblasBfloat16 * y,int incy,hipblasBfloat16 * result) nogil:
    global _hipblasBfdot__funptr
    __init_symbol(&_hipblasBfdot__funptr,"hipblasBfdot")
    return (<hipblasStatus_t (*)(void *,int,hipblasBfloat16 *,int,hipblasBfloat16 *,int,hipblasBfloat16 *) nogil> _hipblasBfdot__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasSdot__funptr = NULL
cdef hipblasStatus_t hipblasSdot(void * handle,int n,const float * x,int incx,const float * y,int incy,float * result) nogil:
    global _hipblasSdot__funptr
    __init_symbol(&_hipblasSdot__funptr,"hipblasSdot")
    return (<hipblasStatus_t (*)(void *,int,const float *,int,const float *,int,float *) nogil> _hipblasSdot__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasDdot__funptr = NULL
cdef hipblasStatus_t hipblasDdot(void * handle,int n,const double * x,int incx,const double * y,int incy,double * result) nogil:
    global _hipblasDdot__funptr
    __init_symbol(&_hipblasDdot__funptr,"hipblasDdot")
    return (<hipblasStatus_t (*)(void *,int,const double *,int,const double *,int,double *) nogil> _hipblasDdot__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasCdotc__funptr = NULL
cdef hipblasStatus_t hipblasCdotc(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * result) nogil:
    global _hipblasCdotc__funptr
    __init_symbol(&_hipblasCdotc__funptr,"hipblasCdotc")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *) nogil> _hipblasCdotc__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasCdotu__funptr = NULL
cdef hipblasStatus_t hipblasCdotu(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * result) nogil:
    global _hipblasCdotu__funptr
    __init_symbol(&_hipblasCdotu__funptr,"hipblasCdotu")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *) nogil> _hipblasCdotu__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasZdotc__funptr = NULL
cdef hipblasStatus_t hipblasZdotc(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * result) nogil:
    global _hipblasZdotc__funptr
    __init_symbol(&_hipblasZdotc__funptr,"hipblasZdotc")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> _hipblasZdotc__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasZdotu__funptr = NULL
cdef hipblasStatus_t hipblasZdotu(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * result) nogil:
    global _hipblasZdotu__funptr
    __init_symbol(&_hipblasZdotu__funptr,"hipblasZdotu")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> _hipblasZdotu__funptr)(handle,n,x,incx,y,incy,result)


cdef void* _hipblasHdotBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# dotBatched(u) performs a batch of dot products of vectors x and y
# 
#     result_i = x_i * y_i;
# 
# dotcBatched  performs a batch of dot products of the conjugate of complex vector x and complex vector y
# 
#     result_i = conjugate (x_i) * y_i;
# 
# where (x_i, y_i) is the i-th instance of the batch.
# x_i and y_i are vectors, for i = 1, ..., batchCount
# 
# - Supported precisions in rocBLAS : h,bf,s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# @param[inout]
# result
#           device array or host array of batchCount size to store the dot products of each batch.
#           return 0.0 for each element if n <= 0.
#
cdef hipblasStatus_t hipblasHdotBatched(void * handle,int n,const unsigned short *const* x,int incx,const unsigned short *const* y,int incy,int batchCount,unsigned short * result) nogil:
    global _hipblasHdotBatched__funptr
    __init_symbol(&_hipblasHdotBatched__funptr,"hipblasHdotBatched")
    return (<hipblasStatus_t (*)(void *,int,const unsigned short *const*,int,const unsigned short *const*,int,int,unsigned short *) nogil> _hipblasHdotBatched__funptr)(handle,n,x,incx,y,incy,batchCount,result)


cdef void* _hipblasBfdotBatched__funptr = NULL
cdef hipblasStatus_t hipblasBfdotBatched(void * handle,int n,hipblasBfloat16 *const* x,int incx,hipblasBfloat16 *const* y,int incy,int batchCount,hipblasBfloat16 * result) nogil:
    global _hipblasBfdotBatched__funptr
    __init_symbol(&_hipblasBfdotBatched__funptr,"hipblasBfdotBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasBfloat16 *const*,int,hipblasBfloat16 *const*,int,int,hipblasBfloat16 *) nogil> _hipblasBfdotBatched__funptr)(handle,n,x,incx,y,incy,batchCount,result)


cdef void* _hipblasSdotBatched__funptr = NULL
cdef hipblasStatus_t hipblasSdotBatched(void * handle,int n,const float *const* x,int incx,const float *const* y,int incy,int batchCount,float * result) nogil:
    global _hipblasSdotBatched__funptr
    __init_symbol(&_hipblasSdotBatched__funptr,"hipblasSdotBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *const*,int,const float *const*,int,int,float *) nogil> _hipblasSdotBatched__funptr)(handle,n,x,incx,y,incy,batchCount,result)


cdef void* _hipblasDdotBatched__funptr = NULL
cdef hipblasStatus_t hipblasDdotBatched(void * handle,int n,const double *const* x,int incx,const double *const* y,int incy,int batchCount,double * result) nogil:
    global _hipblasDdotBatched__funptr
    __init_symbol(&_hipblasDdotBatched__funptr,"hipblasDdotBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *const*,int,const double *const*,int,int,double *) nogil> _hipblasDdotBatched__funptr)(handle,n,x,incx,y,incy,batchCount,result)


cdef void* _hipblasCdotcBatched__funptr = NULL
cdef hipblasStatus_t hipblasCdotcBatched(void * handle,int n,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,int batchCount,hipblasComplex * result) nogil:
    global _hipblasCdotcBatched__funptr
    __init_symbol(&_hipblasCdotcBatched__funptr,"hipblasCdotcBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *const*,int,hipblasComplex *const*,int,int,hipblasComplex *) nogil> _hipblasCdotcBatched__funptr)(handle,n,x,incx,y,incy,batchCount,result)


cdef void* _hipblasCdotuBatched__funptr = NULL
cdef hipblasStatus_t hipblasCdotuBatched(void * handle,int n,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,int batchCount,hipblasComplex * result) nogil:
    global _hipblasCdotuBatched__funptr
    __init_symbol(&_hipblasCdotuBatched__funptr,"hipblasCdotuBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *const*,int,hipblasComplex *const*,int,int,hipblasComplex *) nogil> _hipblasCdotuBatched__funptr)(handle,n,x,incx,y,incy,batchCount,result)


cdef void* _hipblasZdotcBatched__funptr = NULL
cdef hipblasStatus_t hipblasZdotcBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,int batchCount,hipblasDoubleComplex * result) nogil:
    global _hipblasZdotcBatched__funptr
    __init_symbol(&_hipblasZdotcBatched__funptr,"hipblasZdotcBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int,hipblasDoubleComplex *) nogil> _hipblasZdotcBatched__funptr)(handle,n,x,incx,y,incy,batchCount,result)


cdef void* _hipblasZdotuBatched__funptr = NULL
cdef hipblasStatus_t hipblasZdotuBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,int batchCount,hipblasDoubleComplex * result) nogil:
    global _hipblasZdotuBatched__funptr
    __init_symbol(&_hipblasZdotuBatched__funptr,"hipblasZdotuBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int,hipblasDoubleComplex *) nogil> _hipblasZdotuBatched__funptr)(handle,n,x,incx,y,incy,batchCount,result)


cdef void* _hipblasHdotStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# dotStridedBatched(u)  performs a batch of dot products of vectors x and y
# 
#     result_i = x_i * y_i;
# 
# dotcStridedBatched  performs a batch of dot products of the conjugate of complex vector x and complex vector y
# 
#     result_i = conjugate (x_i) * y_i;
# 
# where (x_i, y_i) is the i-th instance of the batch.
# x_i and y_i are vectors, for i = 1, ..., batchCount
# 
# - Supported precisions in rocBLAS : h,bf,s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[in]
# x         device pointer to the first vector (x_1) in the batch.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1)
# @param[in]
# y         device pointer to the first vector (y_1) in the batch.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey     [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch
# @param[inout]
# result
#           device array or host array of batchCount size to store the dot products of each batch.
#           return 0.0 for each element if n <= 0.
#
cdef hipblasStatus_t hipblasHdotStridedBatched(void * handle,int n,const unsigned short * x,int incx,long stridex,const unsigned short * y,int incy,long stridey,int batchCount,unsigned short * result) nogil:
    global _hipblasHdotStridedBatched__funptr
    __init_symbol(&_hipblasHdotStridedBatched__funptr,"hipblasHdotStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const unsigned short *,int,long,const unsigned short *,int,long,int,unsigned short *) nogil> _hipblasHdotStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount,result)


cdef void* _hipblasBfdotStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasBfdotStridedBatched(void * handle,int n,hipblasBfloat16 * x,int incx,long stridex,hipblasBfloat16 * y,int incy,long stridey,int batchCount,hipblasBfloat16 * result) nogil:
    global _hipblasBfdotStridedBatched__funptr
    __init_symbol(&_hipblasBfdotStridedBatched__funptr,"hipblasBfdotStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasBfloat16 *,int,long,hipblasBfloat16 *,int,long,int,hipblasBfloat16 *) nogil> _hipblasBfdotStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount,result)


cdef void* _hipblasSdotStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasSdotStridedBatched(void * handle,int n,const float * x,int incx,long stridex,const float * y,int incy,long stridey,int batchCount,float * result) nogil:
    global _hipblasSdotStridedBatched__funptr
    __init_symbol(&_hipblasSdotStridedBatched__funptr,"hipblasSdotStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *,int,long,const float *,int,long,int,float *) nogil> _hipblasSdotStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount,result)


cdef void* _hipblasDdotStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDdotStridedBatched(void * handle,int n,const double * x,int incx,long stridex,const double * y,int incy,long stridey,int batchCount,double * result) nogil:
    global _hipblasDdotStridedBatched__funptr
    __init_symbol(&_hipblasDdotStridedBatched__funptr,"hipblasDdotStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *,int,long,const double *,int,long,int,double *) nogil> _hipblasDdotStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount,result)


cdef void* _hipblasCdotcStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCdotcStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,int batchCount,hipblasComplex * result) nogil:
    global _hipblasCdotcStridedBatched__funptr
    __init_symbol(&_hipblasCdotcStridedBatched__funptr,"hipblasCdotcStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,long,hipblasComplex *,int,long,int,hipblasComplex *) nogil> _hipblasCdotcStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount,result)


cdef void* _hipblasCdotuStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCdotuStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,int batchCount,hipblasComplex * result) nogil:
    global _hipblasCdotuStridedBatched__funptr
    __init_symbol(&_hipblasCdotuStridedBatched__funptr,"hipblasCdotuStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,long,hipblasComplex *,int,long,int,hipblasComplex *) nogil> _hipblasCdotuStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount,result)


cdef void* _hipblasZdotcStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZdotcStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,int batchCount,hipblasDoubleComplex * result) nogil:
    global _hipblasZdotcStridedBatched__funptr
    __init_symbol(&_hipblasZdotcStridedBatched__funptr,"hipblasZdotcStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int,hipblasDoubleComplex *) nogil> _hipblasZdotcStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount,result)


cdef void* _hipblasZdotuStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZdotuStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,int batchCount,hipblasDoubleComplex * result) nogil:
    global _hipblasZdotuStridedBatched__funptr
    __init_symbol(&_hipblasZdotuStridedBatched__funptr,"hipblasZdotuStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int,hipblasDoubleComplex *) nogil> _hipblasZdotuStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount,result)


cdef void* _hipblasSnrm2__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# nrm2 computes the euclidean norm of a real or complex vector
# 
#           result := sqrt( x'*x ) for real vectors
#           result := sqrt( x**H*x ) for complex vectors
# 
# - Supported precisions in rocBLAS : s,d,c,z,sc,dz
# - Supported precisions in cuBLAS  : s,d,sc,dz
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# result
#           device pointer or host pointer to store the nrm2 product.
#           return is 0.0 if n, incx<=0.
cdef hipblasStatus_t hipblasSnrm2(void * handle,int n,const float * x,int incx,float * result) nogil:
    global _hipblasSnrm2__funptr
    __init_symbol(&_hipblasSnrm2__funptr,"hipblasSnrm2")
    return (<hipblasStatus_t (*)(void *,int,const float *,int,float *) nogil> _hipblasSnrm2__funptr)(handle,n,x,incx,result)


cdef void* _hipblasDnrm2__funptr = NULL
cdef hipblasStatus_t hipblasDnrm2(void * handle,int n,const double * x,int incx,double * result) nogil:
    global _hipblasDnrm2__funptr
    __init_symbol(&_hipblasDnrm2__funptr,"hipblasDnrm2")
    return (<hipblasStatus_t (*)(void *,int,const double *,int,double *) nogil> _hipblasDnrm2__funptr)(handle,n,x,incx,result)


cdef void* _hipblasScnrm2__funptr = NULL
cdef hipblasStatus_t hipblasScnrm2(void * handle,int n,hipblasComplex * x,int incx,float * result) nogil:
    global _hipblasScnrm2__funptr
    __init_symbol(&_hipblasScnrm2__funptr,"hipblasScnrm2")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,float *) nogil> _hipblasScnrm2__funptr)(handle,n,x,incx,result)


cdef void* _hipblasDznrm2__funptr = NULL
cdef hipblasStatus_t hipblasDznrm2(void * handle,int n,hipblasDoubleComplex * x,int incx,double * result) nogil:
    global _hipblasDznrm2__funptr
    __init_symbol(&_hipblasDznrm2__funptr,"hipblasDznrm2")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,double *) nogil> _hipblasDznrm2__funptr)(handle,n,x,incx,result)


cdef void* _hipblasSnrm2Batched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# nrm2Batched computes the euclidean norm over a batch of real or complex vectors
# 
#           result := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batchCount
#           result := sqrt( x_i**H*x_i ) for complex vectors x, for i = 1, ..., batchCount
# 
# - Supported precisions in rocBLAS : s,d,c,z,sc,dz
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each x_i.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device pointer or host pointer to array of batchCount size for nrm2 results.
#           return is 0.0 for each element if n <= 0, incx<=0.
#
cdef hipblasStatus_t hipblasSnrm2Batched(void * handle,int n,const float *const* x,int incx,int batchCount,float * result) nogil:
    global _hipblasSnrm2Batched__funptr
    __init_symbol(&_hipblasSnrm2Batched__funptr,"hipblasSnrm2Batched")
    return (<hipblasStatus_t (*)(void *,int,const float *const*,int,int,float *) nogil> _hipblasSnrm2Batched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasDnrm2Batched__funptr = NULL
cdef hipblasStatus_t hipblasDnrm2Batched(void * handle,int n,const double *const* x,int incx,int batchCount,double * result) nogil:
    global _hipblasDnrm2Batched__funptr
    __init_symbol(&_hipblasDnrm2Batched__funptr,"hipblasDnrm2Batched")
    return (<hipblasStatus_t (*)(void *,int,const double *const*,int,int,double *) nogil> _hipblasDnrm2Batched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasScnrm2Batched__funptr = NULL
cdef hipblasStatus_t hipblasScnrm2Batched(void * handle,int n,hipblasComplex *const* x,int incx,int batchCount,float * result) nogil:
    global _hipblasScnrm2Batched__funptr
    __init_symbol(&_hipblasScnrm2Batched__funptr,"hipblasScnrm2Batched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *const*,int,int,float *) nogil> _hipblasScnrm2Batched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasDznrm2Batched__funptr = NULL
cdef hipblasStatus_t hipblasDznrm2Batched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,int batchCount,double * result) nogil:
    global _hipblasDznrm2Batched__funptr
    __init_symbol(&_hipblasDznrm2Batched__funptr,"hipblasDznrm2Batched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *const*,int,int,double *) nogil> _hipblasDznrm2Batched__funptr)(handle,n,x,incx,batchCount,result)


cdef void* _hipblasSnrm2StridedBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# nrm2StridedBatched computes the euclidean norm over a batch of real or complex vectors
# 
#           := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batchCount
#           := sqrt( x_i**H*x_i ) for complex vectors, for i = 1, ..., batchCount
# 
# - Supported precisions in rocBLAS : s,d,c,z,sc,dz
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each x_i.
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
#           There are no restrictions placed on stride_x, however the user should
#           take care to ensure that stride_x is of appropriate size, for a typical
#           case this means stride_x >= n * incx.
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device pointer or host pointer to array for storing contiguous batchCount results.
#           return is 0.0 for each element if n <= 0, incx<=0.
#
cdef hipblasStatus_t hipblasSnrm2StridedBatched(void * handle,int n,const float * x,int incx,long stridex,int batchCount,float * result) nogil:
    global _hipblasSnrm2StridedBatched__funptr
    __init_symbol(&_hipblasSnrm2StridedBatched__funptr,"hipblasSnrm2StridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *,int,long,int,float *) nogil> _hipblasSnrm2StridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasDnrm2StridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDnrm2StridedBatched(void * handle,int n,const double * x,int incx,long stridex,int batchCount,double * result) nogil:
    global _hipblasDnrm2StridedBatched__funptr
    __init_symbol(&_hipblasDnrm2StridedBatched__funptr,"hipblasDnrm2StridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *,int,long,int,double *) nogil> _hipblasDnrm2StridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasScnrm2StridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasScnrm2StridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,int batchCount,float * result) nogil:
    global _hipblasScnrm2StridedBatched__funptr
    __init_symbol(&_hipblasScnrm2StridedBatched__funptr,"hipblasScnrm2StridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,long,int,float *) nogil> _hipblasScnrm2StridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasDznrm2StridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDznrm2StridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,int batchCount,double * result) nogil:
    global _hipblasDznrm2StridedBatched__funptr
    __init_symbol(&_hipblasDznrm2StridedBatched__funptr,"hipblasDznrm2StridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,long,int,double *) nogil> _hipblasDznrm2StridedBatched__funptr)(handle,n,x,incx,stridex,batchCount,result)


cdef void* _hipblasSrot__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# rot applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
#     Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
# - Supported precisions in rocBLAS : s,d,c,z,sc,dz
# - Supported precisions in cuBLAS  : s,d,c,z,cs,zd
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in the x and y vectors.
# @param[inout]
# x       device pointer storing vector x.
# @param[in]
# incx    [int]
#         specifies the increment between elements of x.
# @param[inout]
# y       device pointer storing vector y.
# @param[in]
# incy    [int]
#         specifies the increment between elements of y.
# @param[in]
# c       device pointer or host pointer storing scalar cosine component of the rotation matrix.
# @param[in]
# s       device pointer or host pointer storing scalar sine component of the rotation matrix.
#
cdef hipblasStatus_t hipblasSrot(void * handle,int n,float * x,int incx,float * y,int incy,const float * c,const float * s) nogil:
    global _hipblasSrot__funptr
    __init_symbol(&_hipblasSrot__funptr,"hipblasSrot")
    return (<hipblasStatus_t (*)(void *,int,float *,int,float *,int,const float *,const float *) nogil> _hipblasSrot__funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* _hipblasDrot__funptr = NULL
cdef hipblasStatus_t hipblasDrot(void * handle,int n,double * x,int incx,double * y,int incy,const double * c,const double * s) nogil:
    global _hipblasDrot__funptr
    __init_symbol(&_hipblasDrot__funptr,"hipblasDrot")
    return (<hipblasStatus_t (*)(void *,int,double *,int,double *,int,const double *,const double *) nogil> _hipblasDrot__funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* _hipblasCrot__funptr = NULL
cdef hipblasStatus_t hipblasCrot(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,const float * c,hipblasComplex * s) nogil:
    global _hipblasCrot__funptr
    __init_symbol(&_hipblasCrot__funptr,"hipblasCrot")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,hipblasComplex *,int,const float *,hipblasComplex *) nogil> _hipblasCrot__funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* _hipblasCsrot__funptr = NULL
cdef hipblasStatus_t hipblasCsrot(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,const float * c,const float * s) nogil:
    global _hipblasCsrot__funptr
    __init_symbol(&_hipblasCsrot__funptr,"hipblasCsrot")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,hipblasComplex *,int,const float *,const float *) nogil> _hipblasCsrot__funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* _hipblasZrot__funptr = NULL
cdef hipblasStatus_t hipblasZrot(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,const double * c,hipblasDoubleComplex * s) nogil:
    global _hipblasZrot__funptr
    __init_symbol(&_hipblasZrot__funptr,"hipblasZrot")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,const double *,hipblasDoubleComplex *) nogil> _hipblasZrot__funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* _hipblasZdrot__funptr = NULL
cdef hipblasStatus_t hipblasZdrot(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,const double * c,const double * s) nogil:
    global _hipblasZdrot__funptr
    __init_symbol(&_hipblasZdrot__funptr,"hipblasZdrot")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,const double *,const double *) nogil> _hipblasZdrot__funptr)(handle,n,x,incx,y,incy,c,s)


cdef void* _hipblasSrotBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# rotBatched applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to batched vectors x_i and y_i, for i = 1, ..., batchCount.
#     Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
# - Supported precisions in rocBLAS : s,d,sc,dz
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in each x_i and y_i vectors.
# @param[inout]
# x       device array of deivce pointers storing each vector x_i.
# @param[in]
# incx    [int]
#         specifies the increment between elements of each x_i.
# @param[inout]
# y       device array of device pointers storing each vector y_i.
# @param[in]
# incy    [int]
#         specifies the increment between elements of each y_i.
# @param[in]
# c       device pointer or host pointer to scalar cosine component of the rotation matrix.
# @param[in]
# s       device pointer or host pointer to scalar sine component of the rotation matrix.
# @param[in]
# batchCount [int]
#             the number of x and y arrays, i.e. the number of batches.
#
cdef hipblasStatus_t hipblasSrotBatched(void * handle,int n,float *const* x,int incx,float *const* y,int incy,const float * c,const float * s,int batchCount) nogil:
    global _hipblasSrotBatched__funptr
    __init_symbol(&_hipblasSrotBatched__funptr,"hipblasSrotBatched")
    return (<hipblasStatus_t (*)(void *,int,float *const*,int,float *const*,int,const float *,const float *,int) nogil> _hipblasSrotBatched__funptr)(handle,n,x,incx,y,incy,c,s,batchCount)


cdef void* _hipblasDrotBatched__funptr = NULL
cdef hipblasStatus_t hipblasDrotBatched(void * handle,int n,double *const* x,int incx,double *const* y,int incy,const double * c,const double * s,int batchCount) nogil:
    global _hipblasDrotBatched__funptr
    __init_symbol(&_hipblasDrotBatched__funptr,"hipblasDrotBatched")
    return (<hipblasStatus_t (*)(void *,int,double *const*,int,double *const*,int,const double *,const double *,int) nogil> _hipblasDrotBatched__funptr)(handle,n,x,incx,y,incy,c,s,batchCount)


cdef void* _hipblasCrotBatched__funptr = NULL
cdef hipblasStatus_t hipblasCrotBatched(void * handle,int n,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,const float * c,hipblasComplex * s,int batchCount) nogil:
    global _hipblasCrotBatched__funptr
    __init_symbol(&_hipblasCrotBatched__funptr,"hipblasCrotBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *const*,int,hipblasComplex *const*,int,const float *,hipblasComplex *,int) nogil> _hipblasCrotBatched__funptr)(handle,n,x,incx,y,incy,c,s,batchCount)


cdef void* _hipblasCsrotBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsrotBatched(void * handle,int n,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,const float * c,const float * s,int batchCount) nogil:
    global _hipblasCsrotBatched__funptr
    __init_symbol(&_hipblasCsrotBatched__funptr,"hipblasCsrotBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *const*,int,hipblasComplex *const*,int,const float *,const float *,int) nogil> _hipblasCsrotBatched__funptr)(handle,n,x,incx,y,incy,c,s,batchCount)


cdef void* _hipblasZrotBatched__funptr = NULL
cdef hipblasStatus_t hipblasZrotBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,const double * c,hipblasDoubleComplex * s,int batchCount) nogil:
    global _hipblasZrotBatched__funptr
    __init_symbol(&_hipblasZrotBatched__funptr,"hipblasZrotBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,const double *,hipblasDoubleComplex *,int) nogil> _hipblasZrotBatched__funptr)(handle,n,x,incx,y,incy,c,s,batchCount)


cdef void* _hipblasZdrotBatched__funptr = NULL
cdef hipblasStatus_t hipblasZdrotBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,const double * c,const double * s,int batchCount) nogil:
    global _hipblasZdrotBatched__funptr
    __init_symbol(&_hipblasZdrotBatched__funptr,"hipblasZdrotBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,const double *,const double *,int) nogil> _hipblasZdrotBatched__funptr)(handle,n,x,incx,y,incy,c,s,batchCount)


cdef void* _hipblasSrotStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# rotStridedBatched applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to strided batched vectors x_i and y_i, for i = 1, ..., batchCount.
#     Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
# - Supported precisions in rocBLAS : s,d,sc,dz
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in each x_i and y_i vectors.
# @param[inout]
# x       device pointer to the first vector x_1.
# @param[in]
# incx    [int]
#         specifies the increment between elements of each x_i.
# @param[in]
# stridex [hipblasStride]
#          specifies the increment from the beginning of x_i to the beginning of x_(i+1)
# @param[inout]
# y       device pointer to the first vector y_1.
# @param[in]
# incy    [int]
#         specifies the increment between elements of each y_i.
# @param[in]
# stridey  [hipblasStride]
#          specifies the increment from the beginning of y_i to the beginning of y_(i+1)
# @param[in]
# c       device pointer or host pointer to scalar cosine component of the rotation matrix.
# @param[in]
# s       device pointer or host pointer to scalar sine component of the rotation matrix.
# @param[in]
# batchCount [int]
#         the number of x and y arrays, i.e. the number of batches.
#
cdef hipblasStatus_t hipblasSrotStridedBatched(void * handle,int n,float * x,int incx,long stridex,float * y,int incy,long stridey,const float * c,const float * s,int batchCount) nogil:
    global _hipblasSrotStridedBatched__funptr
    __init_symbol(&_hipblasSrotStridedBatched__funptr,"hipblasSrotStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,float *,int,long,float *,int,long,const float *,const float *,int) nogil> _hipblasSrotStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,c,s,batchCount)


cdef void* _hipblasDrotStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDrotStridedBatched(void * handle,int n,double * x,int incx,long stridex,double * y,int incy,long stridey,const double * c,const double * s,int batchCount) nogil:
    global _hipblasDrotStridedBatched__funptr
    __init_symbol(&_hipblasDrotStridedBatched__funptr,"hipblasDrotStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,double *,int,long,double *,int,long,const double *,const double *,int) nogil> _hipblasDrotStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,c,s,batchCount)


cdef void* _hipblasCrotStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCrotStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,const float * c,hipblasComplex * s,int batchCount) nogil:
    global _hipblasCrotStridedBatched__funptr
    __init_symbol(&_hipblasCrotStridedBatched__funptr,"hipblasCrotStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,long,hipblasComplex *,int,long,const float *,hipblasComplex *,int) nogil> _hipblasCrotStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,c,s,batchCount)


cdef void* _hipblasCsrotStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsrotStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,const float * c,const float * s,int batchCount) nogil:
    global _hipblasCsrotStridedBatched__funptr
    __init_symbol(&_hipblasCsrotStridedBatched__funptr,"hipblasCsrotStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,long,hipblasComplex *,int,long,const float *,const float *,int) nogil> _hipblasCsrotStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,c,s,batchCount)


cdef void* _hipblasZrotStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZrotStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,const double * c,hipblasDoubleComplex * s,int batchCount) nogil:
    global _hipblasZrotStridedBatched__funptr
    __init_symbol(&_hipblasZrotStridedBatched__funptr,"hipblasZrotStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,const double *,hipblasDoubleComplex *,int) nogil> _hipblasZrotStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,c,s,batchCount)


cdef void* _hipblasZdrotStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZdrotStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,const double * c,const double * s,int batchCount) nogil:
    global _hipblasZdrotStridedBatched__funptr
    __init_symbol(&_hipblasZdrotStridedBatched__funptr,"hipblasZdrotStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,const double *,const double *,int) nogil> _hipblasZdrotStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,c,s,batchCount)


cdef void* _hipblasSrotg__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# rotg creates the Givens rotation matrix for the vector (a b).
#      Scalars c and s and arrays a and b may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#      If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#      If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[inout]
# a       device pointer or host pointer to input vector element, overwritten with r.
# @param[inout]
# b       device pointer or host pointer to input vector element, overwritten with z.
# @param[inout]
# c       device pointer or host pointer to cosine element of Givens rotation.
# @param[inout]
# s       device pointer or host pointer sine element of Givens rotation.
#
cdef hipblasStatus_t hipblasSrotg(void * handle,float * a,float * b,float * c,float * s) nogil:
    global _hipblasSrotg__funptr
    __init_symbol(&_hipblasSrotg__funptr,"hipblasSrotg")
    return (<hipblasStatus_t (*)(void *,float *,float *,float *,float *) nogil> _hipblasSrotg__funptr)(handle,a,b,c,s)


cdef void* _hipblasDrotg__funptr = NULL
cdef hipblasStatus_t hipblasDrotg(void * handle,double * a,double * b,double * c,double * s) nogil:
    global _hipblasDrotg__funptr
    __init_symbol(&_hipblasDrotg__funptr,"hipblasDrotg")
    return (<hipblasStatus_t (*)(void *,double *,double *,double *,double *) nogil> _hipblasDrotg__funptr)(handle,a,b,c,s)


cdef void* _hipblasCrotg__funptr = NULL
cdef hipblasStatus_t hipblasCrotg(void * handle,hipblasComplex * a,hipblasComplex * b,float * c,hipblasComplex * s) nogil:
    global _hipblasCrotg__funptr
    __init_symbol(&_hipblasCrotg__funptr,"hipblasCrotg")
    return (<hipblasStatus_t (*)(void *,hipblasComplex *,hipblasComplex *,float *,hipblasComplex *) nogil> _hipblasCrotg__funptr)(handle,a,b,c,s)


cdef void* _hipblasZrotg__funptr = NULL
cdef hipblasStatus_t hipblasZrotg(void * handle,hipblasDoubleComplex * a,hipblasDoubleComplex * b,double * c,hipblasDoubleComplex * s) nogil:
    global _hipblasZrotg__funptr
    __init_symbol(&_hipblasZrotg__funptr,"hipblasZrotg")
    return (<hipblasStatus_t (*)(void *,hipblasDoubleComplex *,hipblasDoubleComplex *,double *,hipblasDoubleComplex *) nogil> _hipblasZrotg__funptr)(handle,a,b,c,s)


cdef void* _hipblasSrotgBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# rotgBatched creates the Givens rotation matrix for the batched vectors (a_i b_i), for i = 1, ..., batchCount.
#      a, b, c, and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#      If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#      If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[inout]
# a       device array of device pointers storing each single input vector element a_i, overwritten with r_i.
# @param[inout]
# b       device array of device pointers storing each single input vector element b_i, overwritten with z_i.
# @param[inout]
# c       device array of device pointers storing each cosine element of Givens rotation for the batch.
# @param[inout]
# s       device array of device pointers storing each sine element of Givens rotation for the batch.
# @param[in]
# batchCount [int]
#             number of batches (length of arrays a, b, c, and s).
#
cdef hipblasStatus_t hipblasSrotgBatched(void * handle,float *const* a,float *const* b,float *const* c,float *const* s,int batchCount) nogil:
    global _hipblasSrotgBatched__funptr
    __init_symbol(&_hipblasSrotgBatched__funptr,"hipblasSrotgBatched")
    return (<hipblasStatus_t (*)(void *,float *const*,float *const*,float *const*,float *const*,int) nogil> _hipblasSrotgBatched__funptr)(handle,a,b,c,s,batchCount)


cdef void* _hipblasDrotgBatched__funptr = NULL
cdef hipblasStatus_t hipblasDrotgBatched(void * handle,double *const* a,double *const* b,double *const* c,double *const* s,int batchCount) nogil:
    global _hipblasDrotgBatched__funptr
    __init_symbol(&_hipblasDrotgBatched__funptr,"hipblasDrotgBatched")
    return (<hipblasStatus_t (*)(void *,double *const*,double *const*,double *const*,double *const*,int) nogil> _hipblasDrotgBatched__funptr)(handle,a,b,c,s,batchCount)


cdef void* _hipblasCrotgBatched__funptr = NULL
cdef hipblasStatus_t hipblasCrotgBatched(void * handle,hipblasComplex *const* a,hipblasComplex *const* b,float *const* c,hipblasComplex *const* s,int batchCount) nogil:
    global _hipblasCrotgBatched__funptr
    __init_symbol(&_hipblasCrotgBatched__funptr,"hipblasCrotgBatched")
    return (<hipblasStatus_t (*)(void *,hipblasComplex *const*,hipblasComplex *const*,float *const*,hipblasComplex *const*,int) nogil> _hipblasCrotgBatched__funptr)(handle,a,b,c,s,batchCount)


cdef void* _hipblasZrotgBatched__funptr = NULL
cdef hipblasStatus_t hipblasZrotgBatched(void * handle,hipblasDoubleComplex *const* a,hipblasDoubleComplex *const* b,double *const* c,hipblasDoubleComplex *const* s,int batchCount) nogil:
    global _hipblasZrotgBatched__funptr
    __init_symbol(&_hipblasZrotgBatched__funptr,"hipblasZrotgBatched")
    return (<hipblasStatus_t (*)(void *,hipblasDoubleComplex *const*,hipblasDoubleComplex *const*,double *const*,hipblasDoubleComplex *const*,int) nogil> _hipblasZrotgBatched__funptr)(handle,a,b,c,s,batchCount)


cdef void* _hipblasSrotgStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# rotgStridedBatched creates the Givens rotation matrix for the strided batched vectors (a_i b_i), for i = 1, ..., batchCount.
#      a, b, c, and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#      If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#      If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function returns immediately and synchronization is required to read the results.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[inout]
# a       device strided_batched pointer or host strided_batched pointer to first single input vector element a_1, overwritten with r.
# @param[in]
# stridea [hipblasStride]
#          distance between elements of a in batch (distance between a_i and a_(i + 1))
# @param[inout]
# b       device strided_batched pointer or host strided_batched pointer to first single input vector element b_1, overwritten with z.
# @param[in]
# strideb [hipblasStride]
#          distance between elements of b in batch (distance between b_i and b_(i + 1))
# @param[inout]
# c       device strided_batched pointer or host strided_batched pointer to first cosine element of Givens rotations c_1.
# @param[in]
# stridec [hipblasStride]
#          distance between elements of c in batch (distance between c_i and c_(i + 1))
# @param[inout]
# s       device strided_batched pointer or host strided_batched pointer to sine element of Givens rotations s_1.
# @param[in]
# strides [hipblasStride]
#          distance between elements of s in batch (distance between s_i and s_(i + 1))
# @param[in]
# batchCount [int]
#             number of batches (length of arrays a, b, c, and s).
#
cdef hipblasStatus_t hipblasSrotgStridedBatched(void * handle,float * a,long stridea,float * b,long strideb,float * c,long stridec,float * s,long strides,int batchCount) nogil:
    global _hipblasSrotgStridedBatched__funptr
    __init_symbol(&_hipblasSrotgStridedBatched__funptr,"hipblasSrotgStridedBatched")
    return (<hipblasStatus_t (*)(void *,float *,long,float *,long,float *,long,float *,long,int) nogil> _hipblasSrotgStridedBatched__funptr)(handle,a,stridea,b,strideb,c,stridec,s,strides,batchCount)


cdef void* _hipblasDrotgStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDrotgStridedBatched(void * handle,double * a,long stridea,double * b,long strideb,double * c,long stridec,double * s,long strides,int batchCount) nogil:
    global _hipblasDrotgStridedBatched__funptr
    __init_symbol(&_hipblasDrotgStridedBatched__funptr,"hipblasDrotgStridedBatched")
    return (<hipblasStatus_t (*)(void *,double *,long,double *,long,double *,long,double *,long,int) nogil> _hipblasDrotgStridedBatched__funptr)(handle,a,stridea,b,strideb,c,stridec,s,strides,batchCount)


cdef void* _hipblasCrotgStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCrotgStridedBatched(void * handle,hipblasComplex * a,long stridea,hipblasComplex * b,long strideb,float * c,long stridec,hipblasComplex * s,long strides,int batchCount) nogil:
    global _hipblasCrotgStridedBatched__funptr
    __init_symbol(&_hipblasCrotgStridedBatched__funptr,"hipblasCrotgStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasComplex *,long,hipblasComplex *,long,float *,long,hipblasComplex *,long,int) nogil> _hipblasCrotgStridedBatched__funptr)(handle,a,stridea,b,strideb,c,stridec,s,strides,batchCount)


cdef void* _hipblasZrotgStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZrotgStridedBatched(void * handle,hipblasDoubleComplex * a,long stridea,hipblasDoubleComplex * b,long strideb,double * c,long stridec,hipblasDoubleComplex * s,long strides,int batchCount) nogil:
    global _hipblasZrotgStridedBatched__funptr
    __init_symbol(&_hipblasZrotgStridedBatched__funptr,"hipblasZrotgStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasDoubleComplex *,long,hipblasDoubleComplex *,long,double *,long,hipblasDoubleComplex *,long,int) nogil> _hipblasZrotgStridedBatched__funptr)(handle,a,stridea,b,strideb,c,stridec,s,strides,batchCount)


cdef void* _hipblasSrotm__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# rotm applies the modified Givens rotation matrix defined by param to vectors x and y.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : s,d
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in the x and y vectors.
# @param[inout]
# x       device pointer storing vector x.
# @param[in]
# incx    [int]
#         specifies the increment between elements of x.
# @param[inout]
# y       device pointer storing vector y.
# @param[in]
# incy    [int]
#         specifies the increment between elements of y.
# @param[in]
# param   device vector or host vector of 5 elements defining the rotation.
#         param[0] = flag
#         param[1] = H11
#         param[2] = H21
#         param[3] = H12
#         param[4] = H22
#         The flag parameter defines the form of H:
#         flag = -1 => H = ( H11 H12 H21 H22 )
#         flag =  0 => H = ( 1.0 H12 H21 1.0 )
#         flag =  1 => H = ( H11 1.0 -1.0 H22 )
#         flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#         param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#
cdef hipblasStatus_t hipblasSrotm(void * handle,int n,float * x,int incx,float * y,int incy,const float * param) nogil:
    global _hipblasSrotm__funptr
    __init_symbol(&_hipblasSrotm__funptr,"hipblasSrotm")
    return (<hipblasStatus_t (*)(void *,int,float *,int,float *,int,const float *) nogil> _hipblasSrotm__funptr)(handle,n,x,incx,y,incy,param)


cdef void* _hipblasDrotm__funptr = NULL
cdef hipblasStatus_t hipblasDrotm(void * handle,int n,double * x,int incx,double * y,int incy,const double * param) nogil:
    global _hipblasDrotm__funptr
    __init_symbol(&_hipblasDrotm__funptr,"hipblasDrotm")
    return (<hipblasStatus_t (*)(void *,int,double *,int,double *,int,const double *) nogil> _hipblasDrotm__funptr)(handle,n,x,incx,y,incy,param)


cdef void* _hipblasSrotmBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# rotmBatched applies the modified Givens rotation matrix defined by param_i to batched vectors x_i and y_i, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in the x and y vectors.
# @param[inout]
# x       device array of device pointers storing each vector x_i.
# @param[in]
# incx    [int]
#         specifies the increment between elements of each x_i.
# @param[inout]
# y       device array of device pointers storing each vector y_1.
# @param[in]
# incy    [int]
#         specifies the increment between elements of each y_i.
# @param[in]
# param   device array of device vectors of 5 elements defining the rotation.
#         param[0] = flag
#         param[1] = H11
#         param[2] = H21
#         param[3] = H12
#         param[4] = H22
#         The flag parameter defines the form of H:
#         flag = -1 => H = ( H11 H12 H21 H22 )
#         flag =  0 => H = ( 1.0 H12 H21 1.0 )
#         flag =  1 => H = ( H11 1.0 -1.0 H22 )
#         flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#         param may ONLY be stored on the device for the batched version of this function.
# @param[in]
# batchCount [int]
#             the number of x and y arrays, i.e. the number of batches.
#
cdef hipblasStatus_t hipblasSrotmBatched(void * handle,int n,float *const* x,int incx,float *const* y,int incy,const float *const* param,int batchCount) nogil:
    global _hipblasSrotmBatched__funptr
    __init_symbol(&_hipblasSrotmBatched__funptr,"hipblasSrotmBatched")
    return (<hipblasStatus_t (*)(void *,int,float *const*,int,float *const*,int,const float *const*,int) nogil> _hipblasSrotmBatched__funptr)(handle,n,x,incx,y,incy,param,batchCount)


cdef void* _hipblasDrotmBatched__funptr = NULL
cdef hipblasStatus_t hipblasDrotmBatched(void * handle,int n,double *const* x,int incx,double *const* y,int incy,const double *const* param,int batchCount) nogil:
    global _hipblasDrotmBatched__funptr
    __init_symbol(&_hipblasDrotmBatched__funptr,"hipblasDrotmBatched")
    return (<hipblasStatus_t (*)(void *,int,double *const*,int,double *const*,int,const double *const*,int) nogil> _hipblasDrotmBatched__funptr)(handle,n,x,incx,y,incy,param,batchCount)


cdef void* _hipblasSrotmStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# rotmStridedBatched applies the modified Givens rotation matrix defined by param_i to strided batched vectors x_i and y_i, for i = 1, ..., batchCount
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in the x and y vectors.
# @param[inout]
# x       device pointer pointing to first strided batched vector x_1.
# @param[in]
# incx    [int]
#         specifies the increment between elements of each x_i.
# @param[in]
# stridex [hipblasStride]
#          specifies the increment between the beginning of x_i and x_(i + 1)
# @param[inout]
# y       device pointer pointing to first strided batched vector y_1.
# @param[in]
# incy    [int]
#         specifies the increment between elements of each y_i.
# @param[in]
# stridey  [hipblasStride]
#          specifies the increment between the beginning of y_i and y_(i + 1)
# @param[in]
# param   device pointer pointing to first array of 5 elements defining the rotation (param_1).
#         param[0] = flag
#         param[1] = H11
#         param[2] = H21
#         param[3] = H12
#         param[4] = H22
#         The flag parameter defines the form of H:
#         flag = -1 => H = ( H11 H12 H21 H22 )
#         flag =  0 => H = ( 1.0 H12 H21 1.0 )
#         flag =  1 => H = ( H11 1.0 -1.0 H22 )
#         flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#         param may ONLY be stored on the device for the strided_batched version of this function.
# @param[in]
# strideParam [hipblasStride]
#              specifies the increment between the beginning of param_i and param_(i + 1)
# @param[in]
# batchCount [int]
#             the number of x and y arrays, i.e. the number of batches.
#
cdef hipblasStatus_t hipblasSrotmStridedBatched(void * handle,int n,float * x,int incx,long stridex,float * y,int incy,long stridey,const float * param,long strideParam,int batchCount) nogil:
    global _hipblasSrotmStridedBatched__funptr
    __init_symbol(&_hipblasSrotmStridedBatched__funptr,"hipblasSrotmStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,float *,int,long,float *,int,long,const float *,long,int) nogil> _hipblasSrotmStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,param,strideParam,batchCount)


cdef void* _hipblasDrotmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDrotmStridedBatched(void * handle,int n,double * x,int incx,long stridex,double * y,int incy,long stridey,const double * param,long strideParam,int batchCount) nogil:
    global _hipblasDrotmStridedBatched__funptr
    __init_symbol(&_hipblasDrotmStridedBatched__funptr,"hipblasDrotmStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,double *,int,long,double *,int,long,const double *,long,int) nogil> _hipblasDrotmStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,param,strideParam,batchCount)


cdef void* _hipblasSrotmg__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# rotmg creates the modified Givens rotation matrix for the vector (d1 * x1, d2 * y1).
#       Parameters may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#       If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#       If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : s,d
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[inout]
# d1      device pointer or host pointer to input scalar that is overwritten.
# @param[inout]
# d2      device pointer or host pointer to input scalar that is overwritten.
# @param[inout]
# x1      device pointer or host pointer to input scalar that is overwritten.
# @param[in]
# y1      device pointer or host pointer to input scalar.
# @param[out]
# param   device vector or host vector of 5 elements defining the rotation.
#         param[0] = flag
#         param[1] = H11
#         param[2] = H21
#         param[3] = H12
#         param[4] = H22
#         The flag parameter defines the form of H:
#         flag = -1 => H = ( H11 H12 H21 H22 )
#         flag =  0 => H = ( 1.0 H12 H21 1.0 )
#         flag =  1 => H = ( H11 1.0 -1.0 H22 )
#         flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#         param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#
cdef hipblasStatus_t hipblasSrotmg(void * handle,float * d1,float * d2,float * x1,const float * y1,float * param) nogil:
    global _hipblasSrotmg__funptr
    __init_symbol(&_hipblasSrotmg__funptr,"hipblasSrotmg")
    return (<hipblasStatus_t (*)(void *,float *,float *,float *,const float *,float *) nogil> _hipblasSrotmg__funptr)(handle,d1,d2,x1,y1,param)


cdef void* _hipblasDrotmg__funptr = NULL
cdef hipblasStatus_t hipblasDrotmg(void * handle,double * d1,double * d2,double * x1,const double * y1,double * param) nogil:
    global _hipblasDrotmg__funptr
    __init_symbol(&_hipblasDrotmg__funptr,"hipblasDrotmg")
    return (<hipblasStatus_t (*)(void *,double *,double *,double *,const double *,double *) nogil> _hipblasDrotmg__funptr)(handle,d1,d2,x1,y1,param)


cdef void* _hipblasSrotmgBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# rotmgBatched creates the modified Givens rotation matrix for the batched vectors (d1_i * x1_i, d2_i * y1_i), for i = 1, ..., batchCount.
#       Parameters may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#       If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#       If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[inout]
# d1      device batched array or host batched array of input scalars that is overwritten.
# @param[inout]
# d2      device batched array or host batched array of input scalars that is overwritten.
# @param[inout]
# x1      device batched array or host batched array of input scalars that is overwritten.
# @param[in]
# y1      device batched array or host batched array of input scalars.
# @param[out]
# param   device batched array or host batched array of vectors of 5 elements defining the rotation.
#         param[0] = flag
#         param[1] = H11
#         param[2] = H21
#         param[3] = H12
#         param[4] = H22
#         The flag parameter defines the form of H:
#         flag = -1 => H = ( H11 H12 H21 H22 )
#         flag =  0 => H = ( 1.0 H12 H21 1.0 )
#         flag =  1 => H = ( H11 1.0 -1.0 H22 )
#         flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#         param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# @param[in]
# batchCount [int]
#             the number of instances in the batch.
#
cdef hipblasStatus_t hipblasSrotmgBatched(void * handle,float *const* d1,float *const* d2,float *const* x1,const float *const* y1,float *const* param,int batchCount) nogil:
    global _hipblasSrotmgBatched__funptr
    __init_symbol(&_hipblasSrotmgBatched__funptr,"hipblasSrotmgBatched")
    return (<hipblasStatus_t (*)(void *,float *const*,float *const*,float *const*,const float *const*,float *const*,int) nogil> _hipblasSrotmgBatched__funptr)(handle,d1,d2,x1,y1,param,batchCount)


cdef void* _hipblasDrotmgBatched__funptr = NULL
cdef hipblasStatus_t hipblasDrotmgBatched(void * handle,double *const* d1,double *const* d2,double *const* x1,const double *const* y1,double *const* param,int batchCount) nogil:
    global _hipblasDrotmgBatched__funptr
    __init_symbol(&_hipblasDrotmgBatched__funptr,"hipblasDrotmgBatched")
    return (<hipblasStatus_t (*)(void *,double *const*,double *const*,double *const*,const double *const*,double *const*,int) nogil> _hipblasDrotmgBatched__funptr)(handle,d1,d2,x1,y1,param,batchCount)


cdef void* _hipblasSrotmgStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# rotmgStridedBatched creates the modified Givens rotation matrix for the strided batched vectors (d1_i * x1_i, d2_i * y1_i), for i = 1, ..., batchCount.
#       Parameters may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#       If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#       If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[inout]
# d1      device strided_batched array or host strided_batched array of input scalars that is overwritten.
# @param[in]
# strided1 [hipblasStride]
#           specifies the increment between the beginning of d1_i and d1_(i+1)
# @param[inout]
# d2      device strided_batched array or host strided_batched array of input scalars that is overwritten.
# @param[in]
# strided2 [hipblasStride]
#           specifies the increment between the beginning of d2_i and d2_(i+1)
# @param[inout]
# x1      device strided_batched array or host strided_batched array of input scalars that is overwritten.
# @param[in]
# stridex1 [hipblasStride]
#           specifies the increment between the beginning of x1_i and x1_(i+1)
# @param[in]
# y1      device strided_batched array or host strided_batched array of input scalars.
# @param[in]
# stridey1 [hipblasStride]
#           specifies the increment between the beginning of y1_i and y1_(i+1)
# @param[out]
# param   device stridedBatched array or host stridedBatched array of vectors of 5 elements defining the rotation.
#         param[0] = flag
#         param[1] = H11
#         param[2] = H21
#         param[3] = H12
#         param[4] = H22
#         The flag parameter defines the form of H:
#         flag = -1 => H = ( H11 H12 H21 H22 )
#         flag =  0 => H = ( 1.0 H12 H21 1.0 )
#         flag =  1 => H = ( H11 1.0 -1.0 H22 )
#         flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#         param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# @param[in]
# strideParam [hipblasStride]
#              specifies the increment between the beginning of param_i and param_(i + 1)
# @param[in]
# batchCount [int]
#             the number of instances in the batch.
#
cdef hipblasStatus_t hipblasSrotmgStridedBatched(void * handle,float * d1,long strided1,float * d2,long strided2,float * x1,long stridex1,const float * y1,long stridey1,float * param,long strideParam,int batchCount) nogil:
    global _hipblasSrotmgStridedBatched__funptr
    __init_symbol(&_hipblasSrotmgStridedBatched__funptr,"hipblasSrotmgStridedBatched")
    return (<hipblasStatus_t (*)(void *,float *,long,float *,long,float *,long,const float *,long,float *,long,int) nogil> _hipblasSrotmgStridedBatched__funptr)(handle,d1,strided1,d2,strided2,x1,stridex1,y1,stridey1,param,strideParam,batchCount)


cdef void* _hipblasDrotmgStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDrotmgStridedBatched(void * handle,double * d1,long strided1,double * d2,long strided2,double * x1,long stridex1,const double * y1,long stridey1,double * param,long strideParam,int batchCount) nogil:
    global _hipblasDrotmgStridedBatched__funptr
    __init_symbol(&_hipblasDrotmgStridedBatched__funptr,"hipblasDrotmgStridedBatched")
    return (<hipblasStatus_t (*)(void *,double *,long,double *,long,double *,long,const double *,long,double *,long,int) nogil> _hipblasDrotmgStridedBatched__funptr)(handle,d1,strided1,d2,strided2,x1,stridex1,y1,stridey1,param,strideParam,batchCount)


cdef void* _hipblasSscal__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# scal  scales each element of vector x with scalar alpha.
# 
#     x := alpha * x
# 
# - Supported precisions in rocBLAS : s,d,c,z,cs,zd
# - Supported precisions in cuBLAS  : s,d,c,z,cs,zd
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# alpha     device pointer or host pointer for the scalar alpha.
# @param[inout]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# 
#
cdef hipblasStatus_t hipblasSscal(void * handle,int n,const float * alpha,float * x,int incx) nogil:
    global _hipblasSscal__funptr
    __init_symbol(&_hipblasSscal__funptr,"hipblasSscal")
    return (<hipblasStatus_t (*)(void *,int,const float *,float *,int) nogil> _hipblasSscal__funptr)(handle,n,alpha,x,incx)


cdef void* _hipblasDscal__funptr = NULL
cdef hipblasStatus_t hipblasDscal(void * handle,int n,const double * alpha,double * x,int incx) nogil:
    global _hipblasDscal__funptr
    __init_symbol(&_hipblasDscal__funptr,"hipblasDscal")
    return (<hipblasStatus_t (*)(void *,int,const double *,double *,int) nogil> _hipblasDscal__funptr)(handle,n,alpha,x,incx)


cdef void* _hipblasCscal__funptr = NULL
cdef hipblasStatus_t hipblasCscal(void * handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx) nogil:
    global _hipblasCscal__funptr
    __init_symbol(&_hipblasCscal__funptr,"hipblasCscal")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCscal__funptr)(handle,n,alpha,x,incx)


cdef void* _hipblasCsscal__funptr = NULL
cdef hipblasStatus_t hipblasCsscal(void * handle,int n,const float * alpha,hipblasComplex * x,int incx) nogil:
    global _hipblasCsscal__funptr
    __init_symbol(&_hipblasCsscal__funptr,"hipblasCsscal")
    return (<hipblasStatus_t (*)(void *,int,const float *,hipblasComplex *,int) nogil> _hipblasCsscal__funptr)(handle,n,alpha,x,incx)


cdef void* _hipblasZscal__funptr = NULL
cdef hipblasStatus_t hipblasZscal(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx) nogil:
    global _hipblasZscal__funptr
    __init_symbol(&_hipblasZscal__funptr,"hipblasZscal")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZscal__funptr)(handle,n,alpha,x,incx)


cdef void* _hipblasZdscal__funptr = NULL
cdef hipblasStatus_t hipblasZdscal(void * handle,int n,const double * alpha,hipblasDoubleComplex * x,int incx) nogil:
    global _hipblasZdscal__funptr
    __init_symbol(&_hipblasZdscal__funptr,"hipblasZdscal")
    return (<hipblasStatus_t (*)(void *,int,const double *,hipblasDoubleComplex *,int) nogil> _hipblasZdscal__funptr)(handle,n,alpha,x,incx)


cdef void* _hipblasSscalBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# \details
# scalBatched  scales each element of vector x_i with scalar alpha, for i = 1, ... , batchCount.
# 
#      x_i := alpha * x_i
# 
#  where (x_i) is the i-th instance of the batch.
# 
# - Supported precisions in rocBLAS : s,d,c,z,cs,zd
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle      [hipblasHandle_t]
#             handle to the hipblas library context queue.
# @param[in]
# n           [int]
#             the number of elements in each x_i.
# @param[in]
# alpha       host pointer or device pointer for the scalar alpha.
# @param[inout]
# x           device array of device pointers storing each vector x_i.
# @param[in]
# incx        [int]
#             specifies the increment for the elements of each x_i.
# @param[in]
# batchCount [int]
#             specifies the number of batches in x.
cdef hipblasStatus_t hipblasSscalBatched(void * handle,int n,const float * alpha,float *const* x,int incx,int batchCount) nogil:
    global _hipblasSscalBatched__funptr
    __init_symbol(&_hipblasSscalBatched__funptr,"hipblasSscalBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *,float *const*,int,int) nogil> _hipblasSscalBatched__funptr)(handle,n,alpha,x,incx,batchCount)


cdef void* _hipblasDscalBatched__funptr = NULL
cdef hipblasStatus_t hipblasDscalBatched(void * handle,int n,const double * alpha,double *const* x,int incx,int batchCount) nogil:
    global _hipblasDscalBatched__funptr
    __init_symbol(&_hipblasDscalBatched__funptr,"hipblasDscalBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *,double *const*,int,int) nogil> _hipblasDscalBatched__funptr)(handle,n,alpha,x,incx,batchCount)


cdef void* _hipblasCscalBatched__funptr = NULL
cdef hipblasStatus_t hipblasCscalBatched(void * handle,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasCscalBatched__funptr
    __init_symbol(&_hipblasCscalBatched__funptr,"hipblasCscalBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,hipblasComplex *const*,int,int) nogil> _hipblasCscalBatched__funptr)(handle,n,alpha,x,incx,batchCount)


cdef void* _hipblasZscalBatched__funptr = NULL
cdef hipblasStatus_t hipblasZscalBatched(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasZscalBatched__funptr
    __init_symbol(&_hipblasZscalBatched__funptr,"hipblasZscalBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZscalBatched__funptr)(handle,n,alpha,x,incx,batchCount)


cdef void* _hipblasCsscalBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsscalBatched(void * handle,int n,const float * alpha,hipblasComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasCsscalBatched__funptr
    __init_symbol(&_hipblasCsscalBatched__funptr,"hipblasCsscalBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *,hipblasComplex *const*,int,int) nogil> _hipblasCsscalBatched__funptr)(handle,n,alpha,x,incx,batchCount)


cdef void* _hipblasZdscalBatched__funptr = NULL
cdef hipblasStatus_t hipblasZdscalBatched(void * handle,int n,const double * alpha,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasZdscalBatched__funptr
    __init_symbol(&_hipblasZdscalBatched__funptr,"hipblasZdscalBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZdscalBatched__funptr)(handle,n,alpha,x,incx,batchCount)


cdef void* _hipblasSscalStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# \details
# scalStridedBatched  scales each element of vector x_i with scalar alpha, for i = 1, ... , batchCount.
# 
#      x_i := alpha * x_i ,
# 
#  where (x_i) is the i-th instance of the batch.
# 
# - Supported precisions in rocBLAS : s,d,c,z,cs,zd
# - Supported precisions in cuBLAS  : No support
# 
#  @param[in]
# handle      [hipblasHandle_t]
#             handle to the hipblas library context queue.
# @param[in]
# n           [int]
#             the number of elements in each x_i.
# @param[in]
# alpha       host pointer or device pointer for the scalar alpha.
# @param[inout]
# x           device pointer to the first vector (x_1) in the batch.
# @param[in]
# incx        [int]
#             specifies the increment for the elements of x.
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1).
#             There are no restrictions placed on stride_x, however the user should
#             take care to ensure that stride_x is of appropriate size, for a typical
#             case this means stride_x >= n * incx.
# @param[in]
# batchCount [int]
#             specifies the number of batches in x.
cdef hipblasStatus_t hipblasSscalStridedBatched(void * handle,int n,const float * alpha,float * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasSscalStridedBatched__funptr
    __init_symbol(&_hipblasSscalStridedBatched__funptr,"hipblasSscalStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *,float *,int,long,int) nogil> _hipblasSscalStridedBatched__funptr)(handle,n,alpha,x,incx,stridex,batchCount)


cdef void* _hipblasDscalStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDscalStridedBatched(void * handle,int n,const double * alpha,double * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasDscalStridedBatched__funptr
    __init_symbol(&_hipblasDscalStridedBatched__funptr,"hipblasDscalStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *,double *,int,long,int) nogil> _hipblasDscalStridedBatched__funptr)(handle,n,alpha,x,incx,stridex,batchCount)


cdef void* _hipblasCscalStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCscalStridedBatched(void * handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasCscalStridedBatched__funptr
    __init_symbol(&_hipblasCscalStridedBatched__funptr,"hipblasCscalStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,hipblasComplex *,int,long,int) nogil> _hipblasCscalStridedBatched__funptr)(handle,n,alpha,x,incx,stridex,batchCount)


cdef void* _hipblasZscalStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZscalStridedBatched(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasZscalStridedBatched__funptr
    __init_symbol(&_hipblasZscalStridedBatched__funptr,"hipblasZscalStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZscalStridedBatched__funptr)(handle,n,alpha,x,incx,stridex,batchCount)


cdef void* _hipblasCsscalStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsscalStridedBatched(void * handle,int n,const float * alpha,hipblasComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasCsscalStridedBatched__funptr
    __init_symbol(&_hipblasCsscalStridedBatched__funptr,"hipblasCsscalStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const float *,hipblasComplex *,int,long,int) nogil> _hipblasCsscalStridedBatched__funptr)(handle,n,alpha,x,incx,stridex,batchCount)


cdef void* _hipblasZdscalStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZdscalStridedBatched(void * handle,int n,const double * alpha,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasZdscalStridedBatched__funptr
    __init_symbol(&_hipblasZdscalStridedBatched__funptr,"hipblasZdscalStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,const double *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZdscalStridedBatched__funptr)(handle,n,alpha,x,incx,stridex,batchCount)


cdef void* _hipblasSswap__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# swap  interchanges vectors x and y.
# 
#     y := x; x := y
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[inout]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasSswap(void * handle,int n,float * x,int incx,float * y,int incy) nogil:
    global _hipblasSswap__funptr
    __init_symbol(&_hipblasSswap__funptr,"hipblasSswap")
    return (<hipblasStatus_t (*)(void *,int,float *,int,float *,int) nogil> _hipblasSswap__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasDswap__funptr = NULL
cdef hipblasStatus_t hipblasDswap(void * handle,int n,double * x,int incx,double * y,int incy) nogil:
    global _hipblasDswap__funptr
    __init_symbol(&_hipblasDswap__funptr,"hipblasDswap")
    return (<hipblasStatus_t (*)(void *,int,double *,int,double *,int) nogil> _hipblasDswap__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasCswap__funptr = NULL
cdef hipblasStatus_t hipblasCswap(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil:
    global _hipblasCswap__funptr
    __init_symbol(&_hipblasCswap__funptr,"hipblasCswap")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCswap__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasZswap__funptr = NULL
cdef hipblasStatus_t hipblasZswap(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil:
    global _hipblasZswap__funptr
    __init_symbol(&_hipblasZswap__funptr,"hipblasZswap")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZswap__funptr)(handle,n,x,incx,y,incy)


cdef void* _hipblasSswapBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# swapBatched interchanges vectors x_i and y_i, for i = 1 , ... , batchCount
# 
#     y_i := x_i; x_i := y_i
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[inout]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasSswapBatched(void * handle,int n,float ** x,int incx,float ** y,int incy,int batchCount) nogil:
    global _hipblasSswapBatched__funptr
    __init_symbol(&_hipblasSswapBatched__funptr,"hipblasSswapBatched")
    return (<hipblasStatus_t (*)(void *,int,float **,int,float **,int,int) nogil> _hipblasSswapBatched__funptr)(handle,n,x,incx,y,incy,batchCount)


cdef void* _hipblasDswapBatched__funptr = NULL
cdef hipblasStatus_t hipblasDswapBatched(void * handle,int n,double ** x,int incx,double ** y,int incy,int batchCount) nogil:
    global _hipblasDswapBatched__funptr
    __init_symbol(&_hipblasDswapBatched__funptr,"hipblasDswapBatched")
    return (<hipblasStatus_t (*)(void *,int,double **,int,double **,int,int) nogil> _hipblasDswapBatched__funptr)(handle,n,x,incx,y,incy,batchCount)


cdef void* _hipblasCswapBatched__funptr = NULL
cdef hipblasStatus_t hipblasCswapBatched(void * handle,int n,hipblasComplex ** x,int incx,hipblasComplex ** y,int incy,int batchCount) nogil:
    global _hipblasCswapBatched__funptr
    __init_symbol(&_hipblasCswapBatched__funptr,"hipblasCswapBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex **,int,hipblasComplex **,int,int) nogil> _hipblasCswapBatched__funptr)(handle,n,x,incx,y,incy,batchCount)


cdef void* _hipblasZswapBatched__funptr = NULL
cdef hipblasStatus_t hipblasZswapBatched(void * handle,int n,hipblasDoubleComplex ** x,int incx,hipblasDoubleComplex ** y,int incy,int batchCount) nogil:
    global _hipblasZswapBatched__funptr
    __init_symbol(&_hipblasZswapBatched__funptr,"hipblasZswapBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex **,int,hipblasDoubleComplex **,int,int) nogil> _hipblasZswapBatched__funptr)(handle,n,x,incx,y,incy,batchCount)


cdef void* _hipblasSswapStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 1 API
# 
# \details
# swapStridedBatched interchanges vectors x_i and y_i, for i = 1 , ... , batchCount
# 
#     y_i := x_i; x_i := y_i
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[inout]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
#           There are no restrictions placed on stride_x, however the user should
#           take care to ensure that stride_x is of appropriate size, for a typical
#           case this means stride_x >= n * incx.
# @param[inout]
# y         device pointer to the first vector y_1.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# stridey   [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1).
#           There are no restrictions placed on stride_x, however the user should
#           take care to ensure that stride_y is of appropriate size, for a typical
#           case this means stride_y >= n * incy. stridey should be non zero.
#  @param[in]
#  batchCount [int]
#              number of instances in the batch.
#
cdef hipblasStatus_t hipblasSswapStridedBatched(void * handle,int n,float * x,int incx,long stridex,float * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasSswapStridedBatched__funptr
    __init_symbol(&_hipblasSswapStridedBatched__funptr,"hipblasSswapStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,float *,int,long,float *,int,long,int) nogil> _hipblasSswapStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount)


cdef void* _hipblasDswapStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDswapStridedBatched(void * handle,int n,double * x,int incx,long stridex,double * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasDswapStridedBatched__funptr
    __init_symbol(&_hipblasDswapStridedBatched__funptr,"hipblasDswapStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,double *,int,long,double *,int,long,int) nogil> _hipblasDswapStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount)


cdef void* _hipblasCswapStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCswapStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasCswapStridedBatched__funptr
    __init_symbol(&_hipblasCswapStridedBatched__funptr,"hipblasCswapStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCswapStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount)


cdef void* _hipblasZswapStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZswapStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasZswapStridedBatched__funptr
    __init_symbol(&_hipblasZswapStridedBatched__funptr,"hipblasZswapStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZswapStridedBatched__funptr)(handle,n,x,incx,stridex,y,incy,stridey,batchCount)


cdef void* _hipblasSgbmv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# gbmv performs one of the matrix-vector operations
# 
#     y := alpha*A*x    + beta*y,   or
#     y := alpha*A**T*x + beta*y,   or
#     y := alpha*A**H*x + beta*y,
# 
# where alpha and beta are scalars, x and y are vectors and A is an
# m by n banded matrix with kl sub-diagonals and ku super-diagonals.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# trans     [hipblasOperation_t]
#           indicates whether matrix A is tranposed (conjugated) or not
# @param[in]
# m         [int]
#           number of rows of matrix A
# @param[in]
# n         [int]
#           number of columns of matrix A
# @param[in]
# kl        [int]
#           number of sub-diagonals of A
# @param[in]
# ku        [int]
#           number of super-diagonals of A
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
#     AP    device pointer storing banded matrix A.
#           Leading (kl + ku + 1) by n part of the matrix contains the coefficients
#           of the banded matrix. The leading diagonal resides in row (ku + 1) with
#           the first super-diagonal above on the RHS of row ku. The first sub-diagonal
#           resides below on the LHS of row ku + 2. This propogates up and down across
#           sub/super-diagonals.
#             Ex: (m = n = 7; ku = 2, kl = 2)
#             1 2 3 0 0 0 0             0 0 3 3 3 3 3
#             4 1 2 3 0 0 0             0 2 2 2 2 2 2
#             5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
#             0 5 4 1 2 3 0             4 4 4 4 4 4 0
#             0 0 5 4 1 2 0             5 5 5 5 5 0 0
#             0 0 0 5 4 1 2             0 0 0 0 0 0 0
#             0 0 0 0 5 4 1             0 0 0 0 0 0 0
#           Note that the empty elements which don't correspond to data will not
#           be referenced.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. Must be >= (kl + ku + 1)
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasSgbmv(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil:
    global _hipblasSgbmv__funptr
    __init_symbol(&_hipblasSgbmv__funptr,"hipblasSgbmv")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSgbmv__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasDgbmv__funptr = NULL
cdef hipblasStatus_t hipblasDgbmv(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _hipblasDgbmv__funptr
    __init_symbol(&_hipblasDgbmv__funptr,"hipblasDgbmv")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDgbmv__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasCgbmv__funptr = NULL
cdef hipblasStatus_t hipblasCgbmv(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _hipblasCgbmv__funptr
    __init_symbol(&_hipblasCgbmv__funptr,"hipblasCgbmv")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCgbmv__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasZgbmv__funptr = NULL
cdef hipblasStatus_t hipblasZgbmv(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _hipblasZgbmv__funptr
    __init_symbol(&_hipblasZgbmv__funptr,"hipblasZgbmv")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZgbmv__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasSgbmvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# gbmvBatched performs one of the matrix-vector operations
# 
#     y_i := alpha*A_i*x_i    + beta*y_i,   or
#     y_i := alpha*A_i**T*x_i + beta*y_i,   or
#     y_i := alpha*A_i**H*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# m by n banded matrix with kl sub-diagonals and ku super-diagonals,
# for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# trans     [hipblasOperation_t]
#           indicates whether matrix A is tranposed (conjugated) or not
# @param[in]
# m         [int]
#           number of rows of each matrix A_i
# @param[in]
# n         [int]
#           number of columns of each matrix A_i
# @param[in]
# kl        [int]
#           number of sub-diagonals of each A_i
# @param[in]
# ku        [int]
#           number of super-diagonals of each A_i
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
#     AP    device array of device pointers storing each banded matrix A_i.
#           Leading (kl + ku + 1) by n part of the matrix contains the coefficients
#           of the banded matrix. The leading diagonal resides in row (ku + 1) with
#           the first super-diagonal above on the RHS of row ku. The first sub-diagonal
#           resides below on the LHS of row ku + 2. This propogates up and down across
#           sub/super-diagonals.
#             Ex: (m = n = 7; ku = 2, kl = 2)
#             1 2 3 0 0 0 0             0 0 3 3 3 3 3
#             4 1 2 3 0 0 0             0 2 2 2 2 2 2
#             5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
#             0 5 4 1 2 3 0             4 4 4 4 4 4 0
#             0 0 5 4 1 2 0             5 5 5 5 5 0 0
#             0 0 0 5 4 1 2             0 0 0 0 0 0 0
#             0 0 0 0 5 4 1             0 0 0 0 0 0 0
#           Note that the empty elements which don't correspond to data will not
#           be referenced.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. Must be >= (kl + ku + 1)
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# batchCount [int]
#             specifies the number of instances in the batch.
#
cdef hipblasStatus_t hipblasSgbmvBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const float * alpha,const float *const* AP,int lda,const float *const* x,int incx,const float * beta,float *const* y,int incy,int batchCount) nogil:
    global _hipblasSgbmvBatched__funptr
    __init_symbol(&_hipblasSgbmvBatched__funptr,"hipblasSgbmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,int,int,const float *,const float *const*,int,const float *const*,int,const float *,float *const*,int,int) nogil> _hipblasSgbmvBatched__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasDgbmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgbmvBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const double * alpha,const double *const* AP,int lda,const double *const* x,int incx,const double * beta,double *const* y,int incy,int batchCount) nogil:
    global _hipblasDgbmvBatched__funptr
    __init_symbol(&_hipblasDgbmvBatched__funptr,"hipblasDgbmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,int,int,const double *,const double *const*,int,const double *const*,int,const double *,double *const*,int,int) nogil> _hipblasDgbmvBatched__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasCgbmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgbmvBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,hipblasComplex * beta,hipblasComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasCgbmvBatched__funptr
    __init_symbol(&_hipblasCgbmvBatched__funptr,"hipblasCgbmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *,hipblasComplex *const*,int,int) nogil> _hipblasCgbmvBatched__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasZgbmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgbmvBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasZgbmvBatched__funptr
    __init_symbol(&_hipblasZgbmvBatched__funptr,"hipblasZgbmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZgbmvBatched__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasSgbmvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# gbmvStridedBatched performs one of the matrix-vector operations
# 
#     y_i := alpha*A_i*x_i    + beta*y_i,   or
#     y_i := alpha*A_i**T*x_i + beta*y_i,   or
#     y_i := alpha*A_i**H*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# m by n banded matrix with kl sub-diagonals and ku super-diagonals,
# for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# trans     [hipblasOperation_t]
#           indicates whether matrix A is tranposed (conjugated) or not
# @param[in]
# m         [int]
#           number of rows of matrix A
# @param[in]
# n         [int]
#           number of columns of matrix A
# @param[in]
# kl        [int]
#           number of sub-diagonals of A
# @param[in]
# ku        [int]
#           number of super-diagonals of A
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
#     AP    device pointer to first banded matrix (A_1).
#           Leading (kl + ku + 1) by n part of the matrix contains the coefficients
#           of the banded matrix. The leading diagonal resides in row (ku + 1) with
#           the first super-diagonal above on the RHS of row ku. The first sub-diagonal
#           resides below on the LHS of row ku + 2. This propogates up and down across
#           sub/super-diagonals.
#             Ex: (m = n = 7; ku = 2, kl = 2)
#             1 2 3 0 0 0 0             0 0 3 3 3 3 3
#             4 1 2 3 0 0 0             0 2 2 2 2 2 2
#             5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
#             0 5 4 1 2 3 0             4 4 4 4 4 4 0
#             0 0 5 4 1 2 0             5 5 5 5 5 0 0
#             0 0 0 5 4 1 2             0 0 0 0 0 0 0
#             0 0 0 0 5 4 1             0 0 0 0 0 0 0
#           Note that the empty elements which don't correspond to data will not
#           be referenced.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. Must be >= (kl + ku + 1)
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x         device pointer to first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1)
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device pointer to first vector (y_1).
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# stridey  [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (x_i+1)
# @param[in]
# batchCount [int]
#             specifies the number of instances in the batch.
#
cdef hipblasStatus_t hipblasSgbmvStridedBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const float * alpha,const float * AP,int lda,long strideA,const float * x,int incx,long stridex,const float * beta,float * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasSgbmvStridedBatched__funptr
    __init_symbol(&_hipblasSgbmvStridedBatched__funptr,"hipblasSgbmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,int,int,const float *,const float *,int,long,const float *,int,long,const float *,float *,int,long,int) nogil> _hipblasSgbmvStridedBatched__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasDgbmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgbmvStridedBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const double * alpha,const double * AP,int lda,long strideA,const double * x,int incx,long stridex,const double * beta,double * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasDgbmvStridedBatched__funptr
    __init_symbol(&_hipblasDgbmvStridedBatched__funptr,"hipblasDgbmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,int,int,const double *,const double *,int,long,const double *,int,long,const double *,double *,int,long,int) nogil> _hipblasDgbmvStridedBatched__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasCgbmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgbmvStridedBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * beta,hipblasComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasCgbmvStridedBatched__funptr
    __init_symbol(&_hipblasCgbmvStridedBatched__funptr,"hipblasCgbmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,hipblasComplex *,int,long,int) nogil> _hipblasCgbmvStridedBatched__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasZgbmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgbmvStridedBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasZgbmvStridedBatched__funptr
    __init_symbol(&_hipblasZgbmvStridedBatched__funptr,"hipblasZgbmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZgbmvStridedBatched__funptr)(handle,trans,m,n,kl,ku,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasSgemv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# gemv performs one of the matrix-vector operations
# 
#     y := alpha*A*x    + beta*y,   or
#     y := alpha*A**T*x + beta*y,   or
#     y := alpha*A**H*x + beta*y,
# 
# where alpha and beta are scalars, x and y are vectors and A is an
# m by n matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# trans     [hipblasOperation_t]
#           indicates whether matrix A is tranposed (conjugated) or not
# @param[in]
# m         [int]
#           number of rows of matrix A
# @param[in]
# n         [int]
#           number of columns of matrix A
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasSgemv(void * handle,hipblasOperation_t trans,int m,int n,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil:
    global _hipblasSgemv__funptr
    __init_symbol(&_hipblasSgemv__funptr,"hipblasSgemv")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSgemv__funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasDgemv__funptr = NULL
cdef hipblasStatus_t hipblasDgemv(void * handle,hipblasOperation_t trans,int m,int n,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _hipblasDgemv__funptr
    __init_symbol(&_hipblasDgemv__funptr,"hipblasDgemv")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDgemv__funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasCgemv__funptr = NULL
cdef hipblasStatus_t hipblasCgemv(void * handle,hipblasOperation_t trans,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _hipblasCgemv__funptr
    __init_symbol(&_hipblasCgemv__funptr,"hipblasCgemv")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCgemv__funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasZgemv__funptr = NULL
cdef hipblasStatus_t hipblasZgemv(void * handle,hipblasOperation_t trans,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _hipblasZgemv__funptr
    __init_symbol(&_hipblasZgemv__funptr,"hipblasZgemv")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZgemv__funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasSgemvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# gemvBatched performs a batch of matrix-vector operations
# 
#     y_i := alpha*A_i*x_i    + beta*y_i,   or
#     y_i := alpha*A_i**T*x_i + beta*y_i,   or
#     y_i := alpha*A_i**H*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# m by n matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle      [hipblasHandle_t]
#             handle to the hipblas library context queue.
# @param[in]
# trans       [hipblasOperation_t]
#             indicates whether matrices A_i are tranposed (conjugated) or not
# @param[in]
# m           [int]
#             number of rows of each matrix A_i
# @param[in]
# n           [int]
#             number of columns of each matrix A_i
# @param[in]
# alpha       device pointer or host pointer to scalar alpha.
# @param[in]
# AP         device array of device pointers storing each matrix A_i.
# @param[in]
# lda         [int]
#             specifies the leading dimension of each matrix A_i.
# @param[in]
# x           device array of device pointers storing each vector x_i.
# @param[in]
# incx        [int]
#             specifies the increment for the elements of each vector x_i.
# @param[in]
# beta        device pointer or host pointer to scalar beta.
# @param[inout]
# y           device array of device pointers storing each vector y_i.
# @param[in]
# incy        [int]
#             specifies the increment for the elements of each vector y_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasSgemvBatched(void * handle,hipblasOperation_t trans,int m,int n,const float * alpha,const float *const* AP,int lda,const float *const* x,int incx,const float * beta,float *const* y,int incy,int batchCount) nogil:
    global _hipblasSgemvBatched__funptr
    __init_symbol(&_hipblasSgemvBatched__funptr,"hipblasSgemvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,const float *,const float *const*,int,const float *const*,int,const float *,float *const*,int,int) nogil> _hipblasSgemvBatched__funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasDgemvBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgemvBatched(void * handle,hipblasOperation_t trans,int m,int n,const double * alpha,const double *const* AP,int lda,const double *const* x,int incx,const double * beta,double *const* y,int incy,int batchCount) nogil:
    global _hipblasDgemvBatched__funptr
    __init_symbol(&_hipblasDgemvBatched__funptr,"hipblasDgemvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,const double *,const double *const*,int,const double *const*,int,const double *,double *const*,int,int) nogil> _hipblasDgemvBatched__funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasCgemvBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgemvBatched(void * handle,hipblasOperation_t trans,int m,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,hipblasComplex * beta,hipblasComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasCgemvBatched__funptr
    __init_symbol(&_hipblasCgemvBatched__funptr,"hipblasCgemvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *,hipblasComplex *const*,int,int) nogil> _hipblasCgemvBatched__funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasZgemvBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgemvBatched(void * handle,hipblasOperation_t trans,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasZgemvBatched__funptr
    __init_symbol(&_hipblasZgemvBatched__funptr,"hipblasZgemvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZgemvBatched__funptr)(handle,trans,m,n,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasSgemvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# gemvStridedBatched performs a batch of matrix-vector operations
# 
#     y_i := alpha*A_i*x_i    + beta*y_i,   or
#     y_i := alpha*A_i**T*x_i + beta*y_i,   or
#     y_i := alpha*A_i**H*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# m by n matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle      [hipblasHandle_t]
#             handle to the hipblas library context queue.
# @param[in]
# transA      [hipblasOperation_t]
#             indicates whether matrices A_i are tranposed (conjugated) or not
# @param[in]
# m           [int]
#             number of rows of matrices A_i
# @param[in]
# n           [int]
#             number of columns of matrices A_i
# @param[in]
# alpha       device pointer or host pointer to scalar alpha.
# @param[in]
# AP          device pointer to the first matrix (A_1) in the batch.
# @param[in]
# lda         [int]
#             specifies the leading dimension of matrices A_i.
# @param[in]
# strideA     [hipblasStride]
#             stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x           device pointer to the first vector (x_1) in the batch.
# @param[in]
# incx        [int]
#             specifies the increment for the elements of vectors x_i.
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1).
#             There are no restrictions placed on stridex, however the user should
#             take care to ensure that stridex is of appropriate size. When trans equals HIPBLAS_OP_N
#             this typically means stridex >= n * incx, otherwise stridex >= m * incx.
# @param[in]
# beta        device pointer or host pointer to scalar beta.
# @param[inout]
# y           device pointer to the first vector (y_1) in the batch.
# @param[in]
# incy        [int]
#             specifies the increment for the elements of vectors y_i.
# @param[in]
# stridey     [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1).
#             There are no restrictions placed on stridey, however the user should
#             take care to ensure that stridey is of appropriate size. When trans equals HIPBLAS_OP_N
#             this typically means stridey >= m * incy, otherwise stridey >= n * incy. stridey should be non zero.
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasSgemvStridedBatched(void * handle,hipblasOperation_t transA,int m,int n,const float * alpha,const float * AP,int lda,long strideA,const float * x,int incx,long stridex,const float * beta,float * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasSgemvStridedBatched__funptr
    __init_symbol(&_hipblasSgemvStridedBatched__funptr,"hipblasSgemvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,const float *,const float *,int,long,const float *,int,long,const float *,float *,int,long,int) nogil> _hipblasSgemvStridedBatched__funptr)(handle,transA,m,n,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasDgemvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgemvStridedBatched(void * handle,hipblasOperation_t transA,int m,int n,const double * alpha,const double * AP,int lda,long strideA,const double * x,int incx,long stridex,const double * beta,double * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasDgemvStridedBatched__funptr
    __init_symbol(&_hipblasDgemvStridedBatched__funptr,"hipblasDgemvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,const double *,const double *,int,long,const double *,int,long,const double *,double *,int,long,int) nogil> _hipblasDgemvStridedBatched__funptr)(handle,transA,m,n,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasCgemvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgemvStridedBatched(void * handle,hipblasOperation_t transA,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * beta,hipblasComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasCgemvStridedBatched__funptr
    __init_symbol(&_hipblasCgemvStridedBatched__funptr,"hipblasCgemvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,hipblasComplex *,int,long,int) nogil> _hipblasCgemvStridedBatched__funptr)(handle,transA,m,n,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasZgemvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgemvStridedBatched(void * handle,hipblasOperation_t transA,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasZgemvStridedBatched__funptr
    __init_symbol(&_hipblasZgemvStridedBatched__funptr,"hipblasZgemvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZgemvStridedBatched__funptr)(handle,transA,m,n,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasSger__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# ger,geru,gerc performs the matrix-vector operations
# 
#     A := A + alpha*x*y**T , OR
#     A := A + alpha*x*y**H for gerc
# 
# where alpha is a scalar, x and y are vectors, and A is an
# m by n matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# m         [int]
#           the number of rows of the matrix A.
# @param[in]
# n         [int]
#           the number of columns of the matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# AP         device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
#
cdef hipblasStatus_t hipblasSger(void * handle,int m,int n,const float * alpha,const float * x,int incx,const float * y,int incy,float * AP,int lda) nogil:
    global _hipblasSger__funptr
    __init_symbol(&_hipblasSger__funptr,"hipblasSger")
    return (<hipblasStatus_t (*)(void *,int,int,const float *,const float *,int,const float *,int,float *,int) nogil> _hipblasSger__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasDger__funptr = NULL
cdef hipblasStatus_t hipblasDger(void * handle,int m,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP,int lda) nogil:
    global _hipblasDger__funptr
    __init_symbol(&_hipblasDger__funptr,"hipblasDger")
    return (<hipblasStatus_t (*)(void *,int,int,const double *,const double *,int,const double *,int,double *,int) nogil> _hipblasDger__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasCgeru__funptr = NULL
cdef hipblasStatus_t hipblasCgeru(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil:
    global _hipblasCgeru__funptr
    __init_symbol(&_hipblasCgeru__funptr,"hipblasCgeru")
    return (<hipblasStatus_t (*)(void *,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCgeru__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasCgerc__funptr = NULL
cdef hipblasStatus_t hipblasCgerc(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil:
    global _hipblasCgerc__funptr
    __init_symbol(&_hipblasCgerc__funptr,"hipblasCgerc")
    return (<hipblasStatus_t (*)(void *,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCgerc__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasZgeru__funptr = NULL
cdef hipblasStatus_t hipblasZgeru(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil:
    global _hipblasZgeru__funptr
    __init_symbol(&_hipblasZgeru__funptr,"hipblasZgeru")
    return (<hipblasStatus_t (*)(void *,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZgeru__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasZgerc__funptr = NULL
cdef hipblasStatus_t hipblasZgerc(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil:
    global _hipblasZgerc__funptr
    __init_symbol(&_hipblasZgerc__funptr,"hipblasZgerc")
    return (<hipblasStatus_t (*)(void *,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZgerc__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasSgerBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# gerBatched,geruBatched,gercBatched performs a batch of the matrix-vector operations
# 
#     A := A + alpha*x*y**T , OR
#     A := A + alpha*x*y**H for gerc
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha is a scalar, x_i and y_i are vectors and A_i is an
# m by n matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# m         [int]
#           the number of rows of each matrix A_i.
# @param[in]
# n         [int]
#           the number of columns of eaceh matrix A_i.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i.
# @param[inout]
# AP        device array of device pointers storing each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasSgerBatched(void * handle,int m,int n,const float * alpha,const float *const* x,int incx,const float *const* y,int incy,float *const* AP,int lda,int batchCount) nogil:
    global _hipblasSgerBatched__funptr
    __init_symbol(&_hipblasSgerBatched__funptr,"hipblasSgerBatched")
    return (<hipblasStatus_t (*)(void *,int,int,const float *,const float *const*,int,const float *const*,int,float *const*,int,int) nogil> _hipblasSgerBatched__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda,batchCount)


cdef void* _hipblasDgerBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgerBatched(void * handle,int m,int n,const double * alpha,const double *const* x,int incx,const double *const* y,int incy,double *const* AP,int lda,int batchCount) nogil:
    global _hipblasDgerBatched__funptr
    __init_symbol(&_hipblasDgerBatched__funptr,"hipblasDgerBatched")
    return (<hipblasStatus_t (*)(void *,int,int,const double *,const double *const*,int,const double *const*,int,double *const*,int,int) nogil> _hipblasDgerBatched__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda,batchCount)


cdef void* _hipblasCgeruBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgeruBatched(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,hipblasComplex *const* AP,int lda,int batchCount) nogil:
    global _hipblasCgeruBatched__funptr
    __init_symbol(&_hipblasCgeruBatched__funptr,"hipblasCgeruBatched")
    return (<hipblasStatus_t (*)(void *,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCgeruBatched__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda,batchCount)


cdef void* _hipblasCgercBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgercBatched(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,hipblasComplex *const* AP,int lda,int batchCount) nogil:
    global _hipblasCgercBatched__funptr
    __init_symbol(&_hipblasCgercBatched__funptr,"hipblasCgercBatched")
    return (<hipblasStatus_t (*)(void *,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCgercBatched__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda,batchCount)


cdef void* _hipblasZgeruBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgeruBatched(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,hipblasDoubleComplex *const* AP,int lda,int batchCount) nogil:
    global _hipblasZgeruBatched__funptr
    __init_symbol(&_hipblasZgeruBatched__funptr,"hipblasZgeruBatched")
    return (<hipblasStatus_t (*)(void *,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZgeruBatched__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda,batchCount)


cdef void* _hipblasZgercBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgercBatched(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,hipblasDoubleComplex *const* AP,int lda,int batchCount) nogil:
    global _hipblasZgercBatched__funptr
    __init_symbol(&_hipblasZgercBatched__funptr,"hipblasZgercBatched")
    return (<hipblasStatus_t (*)(void *,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZgercBatched__funptr)(handle,m,n,alpha,x,incx,y,incy,AP,lda,batchCount)


cdef void* _hipblasSgerStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# gerStridedBatched,geruStridedBatched,gercStridedBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*y_i**T, OR
#     A_i := A_i + alpha*x_i*y_i**H  for gerc
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha is a scalar, x_i and y_i are vectors and A_i is an
# m by n matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# m         [int]
#           the number of rows of each matrix A_i.
# @param[in]
# n         [int]
#           the number of columns of each matrix A_i.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer to the first vector (x_1) in the batch.
# @param[in]
# incx      [int]
#           specifies the increments for the elements of each vector x_i.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
#           There are no restrictions placed on stridex, however the user should
#           take care to ensure that stridex is of appropriate size, for a typical
#           case this means stridex >= m * incx.
# @param[inout]
# y         device pointer to the first vector (y_1) in the batch.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i.
# @param[in]
# stridey   [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1).
#           There are no restrictions placed on stridey, however the user should
#           take care to ensure that stridey is of appropriate size, for a typical
#           case this means stridey >= n * incy.
# @param[inout]
# AP        device pointer to the first matrix (A_1) in the batch.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# strideA     [hipblasStride]
#             stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasSgerStridedBatched(void * handle,int m,int n,const float * alpha,const float * x,int incx,long stridex,const float * y,int incy,long stridey,float * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasSgerStridedBatched__funptr
    __init_symbol(&_hipblasSgerStridedBatched__funptr,"hipblasSgerStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,int,const float *,const float *,int,long,const float *,int,long,float *,int,long,int) nogil> _hipblasSgerStridedBatched__funptr)(handle,m,n,alpha,x,incx,stridex,y,incy,stridey,AP,lda,strideA,batchCount)


cdef void* _hipblasDgerStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgerStridedBatched(void * handle,int m,int n,const double * alpha,const double * x,int incx,long stridex,const double * y,int incy,long stridey,double * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasDgerStridedBatched__funptr
    __init_symbol(&_hipblasDgerStridedBatched__funptr,"hipblasDgerStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,int,const double *,const double *,int,long,const double *,int,long,double *,int,long,int) nogil> _hipblasDgerStridedBatched__funptr)(handle,m,n,alpha,x,incx,stridex,y,incy,stridey,AP,lda,strideA,batchCount)


cdef void* _hipblasCgeruStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgeruStridedBatched(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,hipblasComplex * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasCgeruStridedBatched__funptr
    __init_symbol(&_hipblasCgeruStridedBatched__funptr,"hipblasCgeruStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCgeruStridedBatched__funptr)(handle,m,n,alpha,x,incx,stridex,y,incy,stridey,AP,lda,strideA,batchCount)


cdef void* _hipblasCgercStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgercStridedBatched(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,hipblasComplex * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasCgercStridedBatched__funptr
    __init_symbol(&_hipblasCgercStridedBatched__funptr,"hipblasCgercStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCgercStridedBatched__funptr)(handle,m,n,alpha,x,incx,stridex,y,incy,stridey,AP,lda,strideA,batchCount)


cdef void* _hipblasZgeruStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgeruStridedBatched(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,hipblasDoubleComplex * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasZgeruStridedBatched__funptr
    __init_symbol(&_hipblasZgeruStridedBatched__funptr,"hipblasZgeruStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZgeruStridedBatched__funptr)(handle,m,n,alpha,x,incx,stridex,y,incy,stridey,AP,lda,strideA,batchCount)


cdef void* _hipblasZgercStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgercStridedBatched(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,hipblasDoubleComplex * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasZgercStridedBatched__funptr
    __init_symbol(&_hipblasZgercStridedBatched__funptr,"hipblasZgercStridedBatched")
    return (<hipblasStatus_t (*)(void *,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZgercStridedBatched__funptr)(handle,m,n,alpha,x,incx,stridex,y,incy,stridey,AP,lda,strideA,batchCount)


cdef void* _hipblasChbmv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hbmv performs the matrix-vector operations
# 
#     y := alpha*A*x + beta*y
# 
# where alpha and beta are scalars, x and y are n element vectors and A is an
# n by n Hermitian band matrix, with k super-diagonals.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is being supplied.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is being supplied.
# @param[in]
# n         [int]
#           the order of the matrix A.
# @param[in]
# k         [int]
#           the number of super-diagonals of the matrix A. Must be >= 0.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device pointer storing matrix A. Of dimension (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The leading (k + 1) by n part of A must contain the upper
#             triangular band part of the Hermitian matrix, with the leading
#             diagonal in row (k + 1), the first super-diagonal on the RHS
#             of row k, etc.
#             The top left k by x triangle of A will not be referenced.
#                 Ex (upper, lda = n = 4, k = 1):
#                 A                             Represented matrix
#                 (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
#                 (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
#                 (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
#                 (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)
# 
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The leading (k + 1) by n part of A must contain the lower
#             triangular band part of the Hermitian matrix, with the leading
#             diagonal in row (1), the first sub-diagonal on the LHS of
#             row 2, etc.
#             The bottom right k by k triangle of A will not be referenced.
#                 Ex (lower, lda = 2, n = 4, k = 1):
#                 A                               Represented matrix
#                 (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
#                 (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
#                                                 (0, 0) (6, 8) (3, 0) (7,-7)
#                                                 (0, 0) (0, 0) (7, 7) (4, 0)
# 
#           As a Hermitian matrix, the imaginary part of the main diagonal
#           of A will not be referenced and is assumed to be == 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. must be >= k + 1
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasChbmv(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _hipblasChbmv__funptr
    __init_symbol(&_hipblasChbmv__funptr,"hipblasChbmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasChbmv__funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasZhbmv__funptr = NULL
cdef hipblasStatus_t hipblasZhbmv(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _hipblasZhbmv__funptr
    __init_symbol(&_hipblasZhbmv__funptr,"hipblasZhbmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZhbmv__funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasChbmvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hbmvBatched performs one of the matrix-vector operations
# 
#     y_i := alpha*A_i*x_i + beta*y_i
# 
# where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
# n by n Hermitian band matrix with k super-diagonals, for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is being supplied.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is being supplied.
# @param[in]
# n         [int]
#           the order of each matrix A_i.
# @param[in]
# k         [int]
#           the number of super-diagonals of each matrix A_i. Must be >= 0.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device array of device pointers storing each matrix_i A of dimension (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The leading (k + 1) by n part of each A_i must contain the upper
#             triangular band part of the Hermitian matrix, with the leading
#             diagonal in row (k + 1), the first super-diagonal on the RHS
#             of row k, etc.
#             The top left k by x triangle of each A_i will not be referenced.
#                 Ex (upper, lda = n = 4, k = 1):
#                 A                             Represented matrix
#                 (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
#                 (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
#                 (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
#                 (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)
# 
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The leading (k + 1) by n part of each A_i must contain the lower
#             triangular band part of the Hermitian matrix, with the leading
#             diagonal in row (1), the first sub-diagonal on the LHS of
#             row 2, etc.
#             The bottom right k by k triangle of each A_i will not be referenced.
#                 Ex (lower, lda = 2, n = 4, k = 1):
#                 A                               Represented matrix
#                 (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
#                 (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
#                                                 (0, 0) (6, 8) (3, 0) (7,-7)
#                                                 (0, 0) (0, 0) (7, 7) (4, 0)
# 
#           As a Hermitian matrix, the imaginary part of the main diagonal
#           of each A_i will not be referenced and is assumed to be == 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. must be >= max(1, n)
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasChbmvBatched(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,hipblasComplex * beta,hipblasComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasChbmvBatched__funptr
    __init_symbol(&_hipblasChbmvBatched__funptr,"hipblasChbmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *,hipblasComplex *const*,int,int) nogil> _hipblasChbmvBatched__funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasZhbmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasZhbmvBatched(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasZhbmvBatched__funptr
    __init_symbol(&_hipblasZhbmvBatched__funptr,"hipblasZhbmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZhbmvBatched__funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasChbmvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hbmvStridedBatched performs one of the matrix-vector operations
# 
#     y_i := alpha*A_i*x_i + beta*y_i
# 
# where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
# n by n Hermitian band matrix with k super-diagonals, for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is being supplied.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is being supplied.
# @param[in]
# n         [int]
#           the order of each matrix A_i.
# @param[in]
# k         [int]
#           the number of super-diagonals of each matrix A_i. Must be >= 0.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device array pointing to the first matrix A_1. Each A_i is of dimension (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The leading (k + 1) by n part of each A_i must contain the upper
#             triangular band part of the Hermitian matrix, with the leading
#             diagonal in row (k + 1), the first super-diagonal on the RHS
#             of row k, etc.
#             The top left k by x triangle of each A_i will not be referenced.
#                 Ex (upper, lda = n = 4, k = 1):
#                 A                             Represented matrix
#                 (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
#                 (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
#                 (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
#                 (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)
# 
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The leading (k + 1) by n part of each A_i must contain the lower
#             triangular band part of the Hermitian matrix, with the leading
#             diagonal in row (1), the first sub-diagonal on the LHS of
#             row 2, etc.
#             The bottom right k by k triangle of each A_i will not be referenced.
#                 Ex (lower, lda = 2, n = 4, k = 1):
#                 A                               Represented matrix
#                 (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
#                 (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
#                                                 (0, 0) (6, 8) (3, 0) (7,-7)
#                                                 (0, 0) (0, 0) (7, 7) (4, 0)
# 
#           As a Hermitian matrix, the imaginary part of the main diagonal
#           of each A_i will not be referenced and is assumed to be == 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. must be >= max(1, n)
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x         device array pointing to the first vector y_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1)
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array pointing to the first vector y_1.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# stridey  [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasChbmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * beta,hipblasComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasChbmvStridedBatched__funptr
    __init_symbol(&_hipblasChbmvStridedBatched__funptr,"hipblasChbmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,hipblasComplex *,int,long,int) nogil> _hipblasChbmvStridedBatched__funptr)(handle,uplo,n,k,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasZhbmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZhbmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasZhbmvStridedBatched__funptr
    __init_symbol(&_hipblasZhbmvStridedBatched__funptr,"hipblasZhbmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZhbmvStridedBatched__funptr)(handle,uplo,n,k,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasChemv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hemv performs one of the matrix-vector operations
# 
#     y := alpha*A*x + beta*y
# 
# where alpha and beta are scalars, x and y are n element vectors and A is an
# n by n Hermitian matrix.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied.
#           HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied.
# @param[in]
# n         [int]
#           the order of the matrix A.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device pointer storing matrix A. Of dimension (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular part of A must contain
#             the upper triangular part of a Hermitian matrix. The lower
#             triangular part of A will not be referenced.
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular part of A must contain
#             the lower triangular part of a Hermitian matrix. The upper
#             triangular part of A will not be referenced.
#           As a Hermitian matrix, the imaginary part of the main diagonal
#           of A will not be referenced and is assumed to be == 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. must be >= max(1, n)
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasChemv(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _hipblasChemv__funptr
    __init_symbol(&_hipblasChemv__funptr,"hipblasChemv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasChemv__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasZhemv__funptr = NULL
cdef hipblasStatus_t hipblasZhemv(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _hipblasZhemv__funptr
    __init_symbol(&_hipblasZhemv__funptr,"hipblasZhemv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZhemv__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasChemvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hemvBatched performs one of the matrix-vector operations
# 
#     y_i := alpha*A_i*x_i + beta*y_i
# 
# where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
# n by n Hermitian matrix, for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied.
#           HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied.
# @param[in]
# n         [int]
#           the order of each matrix A_i.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device array of device pointers storing each matrix A_i of dimension (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular part of each A_i must contain
#             the upper triangular part of a Hermitian matrix. The lower
#             triangular part of each A_i will not be referenced.
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular part of each A_i must contain
#             the lower triangular part of a Hermitian matrix. The upper
#             triangular part of each A_i will not be referenced.
#           As a Hermitian matrix, the imaginary part of the main diagonal
#           of each A_i will not be referenced and is assumed to be == 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. must be >= max(1, n)
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasChemvBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,hipblasComplex * beta,hipblasComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasChemvBatched__funptr
    __init_symbol(&_hipblasChemvBatched__funptr,"hipblasChemvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *,hipblasComplex *const*,int,int) nogil> _hipblasChemvBatched__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasZhemvBatched__funptr = NULL
cdef hipblasStatus_t hipblasZhemvBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasZhemvBatched__funptr
    __init_symbol(&_hipblasZhemvBatched__funptr,"hipblasZhemvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZhemvBatched__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasChemvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hemvStridedBatched performs one of the matrix-vector operations
# 
#     y_i := alpha*A_i*x_i + beta*y_i
# 
# where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
# n by n Hermitian matrix, for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied.
#           HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied.
# @param[in]
# n         [int]
#           the order of each matrix A_i.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device array of device pointers storing each matrix A_i of dimension (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular part of each A_i must contain
#             the upper triangular part of a Hermitian matrix. The lower
#             triangular part of each A_i will not be referenced.
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular part of each A_i must contain
#             the lower triangular part of a Hermitian matrix. The upper
#             triangular part of each A_i will not be referenced.
#           As a Hermitian matrix, the imaginary part of the main diagonal
#           of each A_i will not be referenced and is assumed to be == 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. must be >= max(1, n)
# @param[in]
# strideA    [hipblasStride]
#             stride from the start of one (A_i) to the next (A_i+1)
# 
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# stridey  [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasChemvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * beta,hipblasComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasChemvStridedBatched__funptr
    __init_symbol(&_hipblasChemvStridedBatched__funptr,"hipblasChemvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,hipblasComplex *,int,long,int) nogil> _hipblasChemvStridedBatched__funptr)(handle,uplo,n,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasZhemvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZhemvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasZhemvStridedBatched__funptr
    __init_symbol(&_hipblasZhemvStridedBatched__funptr,"hipblasZhemvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZhemvStridedBatched__funptr)(handle,uplo,n,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasCher__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# her performs the matrix-vector operations
# 
#     A := A + alpha*x*x**H
# 
# where alpha is a real scalar, x is a vector, and A is an
# n by n Hermitian matrix.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in A.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in A.
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[inout]
# AP        device pointer storing the specified triangular portion of
#           the Hermitian matrix A. Of size (lda * n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the Hermitian matrix A is supplied. The lower
#             triangluar portion will not be touched.
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the Hermitian matrix A is supplied. The upper
#             triangular portion will not be touched.
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. Must be at least max(1, n).
cdef hipblasStatus_t hipblasCher(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex * x,int incx,hipblasComplex * AP,int lda) nogil:
    global _hipblasCher__funptr
    __init_symbol(&_hipblasCher__funptr,"hipblasCher")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCher__funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* _hipblasZher__funptr = NULL
cdef hipblasStatus_t hipblasZher(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP,int lda) nogil:
    global _hipblasZher__funptr
    __init_symbol(&_hipblasZher__funptr,"hipblasZher")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZher__funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* _hipblasCherBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# herBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*x_i**H
# 
# where alpha is a real scalar, x_i is a vector, and A_i is an
# n by n symmetric matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in A.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in A.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[inout]
# AP       device array of device pointers storing the specified triangular portion of
#           each Hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular portion
#             of each A_i will not be touched.
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular portion
#             of each A_i will not be touched.
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. Must be at least max(1, n).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
cdef hipblasStatus_t hipblasCherBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* AP,int lda,int batchCount) nogil:
    global _hipblasCherBatched__funptr
    __init_symbol(&_hipblasCherBatched__funptr,"hipblasCherBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCherBatched__funptr)(handle,uplo,n,alpha,x,incx,AP,lda,batchCount)


cdef void* _hipblasZherBatched__funptr = NULL
cdef hipblasStatus_t hipblasZherBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* AP,int lda,int batchCount) nogil:
    global _hipblasZherBatched__funptr
    __init_symbol(&_hipblasZherBatched__funptr,"hipblasZherBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZherBatched__funptr)(handle,uplo,n,alpha,x,incx,AP,lda,batchCount)


cdef void* _hipblasCherStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# herStridedBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*x_i**H
# 
# where alpha is a real scalar, x_i is a vector, and A_i is an
# n by n Hermitian matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in A.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in A.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer pointing to the first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[inout]
# AP        device array of device pointers storing the specified triangular portion of
#           each Hermitian matrix A_i. Points to the first matrix (A_1).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular
#             portion of each A_i will not be touched.
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular
#             portion of each A_i will not be touched.
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# strideA    [hipblasStride]
#             stride from the start of one (A_i) and the next (A_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch.
cdef hipblasStatus_t hipblasCherStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasCherStridedBatched__funptr
    __init_symbol(&_hipblasCherStridedBatched__funptr,"hipblasCherStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCherStridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,AP,lda,strideA,batchCount)


cdef void* _hipblasZherStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZherStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasZherStridedBatched__funptr
    __init_symbol(&_hipblasZherStridedBatched__funptr,"hipblasZherStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZherStridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,AP,lda,strideA,batchCount)


cdef void* _hipblasCher2__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# her2 performs the matrix-vector operations
# 
#     A := A + alpha*x*y**H + conj(alpha)*y*x**H
# 
# where alpha is a complex scalar, x and y are vectors, and A is an
# n by n Hermitian matrix.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied.
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# AP         device pointer storing the specified triangular portion of
#           the Hermitian matrix A. Of size (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the Hermitian matrix A is supplied. The lower triangular
#             portion of A will not be touched.
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the Hermitian matrix A is supplied. The upper triangular
#             portion of A will not be touched.
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. Must be at least max(lda, 1).
cdef hipblasStatus_t hipblasCher2(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil:
    global _hipblasCher2__funptr
    __init_symbol(&_hipblasCher2__funptr,"hipblasCher2")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCher2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasZher2__funptr = NULL
cdef hipblasStatus_t hipblasZher2(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil:
    global _hipblasZher2__funptr
    __init_symbol(&_hipblasZher2__funptr,"hipblasZher2")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZher2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasCher2Batched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# her2Batched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H
# 
# where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
# n by n Hermitian matrix for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[inout]
# AP         device array of device pointers storing the specified triangular portion of
#           each Hermitian matrix A_i of size (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular
#             portion of each A_i will not be touched.
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular
#             portion of each A_i will not be touched.
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. Must be at least max(lda, 1).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
cdef hipblasStatus_t hipblasCher2Batched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,hipblasComplex *const* AP,int lda,int batchCount) nogil:
    global _hipblasCher2Batched__funptr
    __init_symbol(&_hipblasCher2Batched__funptr,"hipblasCher2Batched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCher2Batched__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda,batchCount)


cdef void* _hipblasZher2Batched__funptr = NULL
cdef hipblasStatus_t hipblasZher2Batched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,hipblasDoubleComplex *const* AP,int lda,int batchCount) nogil:
    global _hipblasZher2Batched__funptr
    __init_symbol(&_hipblasZher2Batched__funptr,"hipblasZher2Batched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZher2Batched__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda,batchCount)


cdef void* _hipblasCher2StridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# her2StridedBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H
# 
# where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
# n by n Hermitian matrix for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer pointing to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           specifies the stride between the beginning of one vector (x_i) and the next (x_i+1).
# @param[in]
# y         device pointer pointing to the first vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey  [hipblasStride]
#           specifies the stride between the beginning of one vector (y_i) and the next (y_i+1).
# @param[inout]
# AP        device pointer pointing to the first matrix (A_1). Stores the specified triangular portion of
#           each Hermitian matrix A_i.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular
#             portion of each A_i will not be touched.
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular
#             portion of each A_i will not be touched.
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. Must be at least max(lda, 1).
# @param[in]
# strideA  [hipblasStride]
#           specifies the stride between the beginning of one matrix (A_i) and the next (A_i+1).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
cdef hipblasStatus_t hipblasCher2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,hipblasComplex * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasCher2StridedBatched__funptr
    __init_symbol(&_hipblasCher2StridedBatched__funptr,"hipblasCher2StridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCher2StridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,y,incy,stridey,AP,lda,strideA,batchCount)


cdef void* _hipblasZher2StridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZher2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,hipblasDoubleComplex * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasZher2StridedBatched__funptr
    __init_symbol(&_hipblasZher2StridedBatched__funptr,"hipblasZher2StridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZher2StridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,y,incy,stridey,AP,lda,strideA,batchCount)


cdef void* _hipblasChpmv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hpmv performs the matrix-vector operation
# 
#     y := alpha*A*x + beta*y
# 
# where alpha and beta are scalars, x and y are n element vectors and A is an
# n by n Hermitian matrix, supplied in packed form (see description below).
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied in AP.
# @param[in]
# n         [int]
#           the order of the matrix A, must be >= 0.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device pointer storing the packed version of the specified triangular portion of
#           the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the Hermitian matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (3, 2)
#                     (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
#                     (3,-2) (5, 1) (6, 0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the Hermitian matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (3, 2)
#                     (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
#                     (3,-2) (5, 1) (6, 0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
#
cdef hipblasStatus_t hipblasChpmv(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _hipblasChpmv__funptr
    __init_symbol(&_hipblasChpmv__funptr,"hipblasChpmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasChpmv__funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)


cdef void* _hipblasZhpmv__funptr = NULL
cdef hipblasStatus_t hipblasZhpmv(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _hipblasZhpmv__funptr
    __init_symbol(&_hipblasZhpmv__funptr,"hipblasZhpmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZhpmv__funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)


cdef void* _hipblasChpmvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hpmvBatched performs the matrix-vector operation
# 
#     y_i := alpha*A_i*x_i + beta*y_i
# 
# where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
# n by n Hermitian matrix, supplied in packed form (see description below),
# for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: the upper triangular part of each Hermitian matrix A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: the lower triangular part of each Hermitian matrix A_i is supplied in AP.
# @param[in]
# n         [int]
#           the order of each matrix A_i.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP      device pointer of device pointers storing the packed version of the specified triangular
#         portion of each Hermitian matrix A_i. Each A_i is of at least size ((n * (n + 1)) / 2).
#         if uplo == HIPBLAS_FILL_MODE_UPPER:
#         The upper triangular portion of each Hermitian matrix A_i is supplied.
#         The matrix is compacted so that each AP_i contains the triangular portion column-by-column
#         so that:
#         AP(0) = A(0,0)
#         AP(1) = A(0,1)
#         AP(2) = A(1,1), etc.
#             Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                 (1, 0) (2, 1) (3, 2)
#                 (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
#                 (3,-2) (5, 1) (6, 0)
#     if uplo == HIPBLAS_FILL_MODE_LOWER:
#         The lower triangular portion of each Hermitian matrix A_i is supplied.
#         The matrix is compacted so that each AP_i contains the triangular portion column-by-column
#         so that:
#         AP(0) = A(0,0)
#         AP(1) = A(1,0)
#         AP(2) = A(2,1), etc.
#             Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                 (1, 0) (2, 1) (3, 2)
#                 (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
#                 (3,-2) (5, 1) (6, 0)
#     Note that the imaginary part of the diagonal elements are not accessed and are assumed
#     to be 0.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasChpmvBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* AP,hipblasComplex *const* x,int incx,hipblasComplex * beta,hipblasComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasChpmvBatched__funptr
    __init_symbol(&_hipblasChpmvBatched__funptr,"hipblasChpmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *const*,hipblasComplex *const*,int,hipblasComplex *,hipblasComplex *const*,int,int) nogil> _hipblasChpmvBatched__funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasZhpmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasZhpmvBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil:
    global _hipblasZhpmvBatched__funptr
    __init_symbol(&_hipblasZhpmvBatched__funptr,"hipblasZhpmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZhpmvBatched__funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasChpmvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hpmvStridedBatched performs the matrix-vector operation
# 
#     y_i := alpha*A_i*x_i + beta*y_i
# 
# where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
# n by n Hermitian matrix, supplied in packed form (see description below),
# for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: the upper triangular part of each Hermitian matrix A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: the lower triangular part of each Hermitian matrix A_i is supplied in AP.
# @param[in]
# n         [int]
#           the order of each matrix A_i.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device pointer pointing to the beginning of the first matrix (AP_1). Stores the packed
#           version of the specified triangular portion of each Hermitian matrix AP_i of size ((n * (n + 1)) / 2).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that each AP_i contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (3, 2)
#                     (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
#                     (3,-2) (5, 1) (6, 0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that each AP_i contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (3, 2)
#                     (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
#                     (3,-2) (5, 1) (6, 0)
#     Note that the imaginary part of the diagonal elements are not accessed and are assumed
#     to be 0.
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (AP_i) and the next one (AP_i+1).
# @param[in]
# x         device array pointing to the beginning of the first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array pointing to the beginning of the first vector (y_1).
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# stridey  [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasChpmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * beta,hipblasComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasChpmvStridedBatched__funptr
    __init_symbol(&_hipblasChpmvStridedBatched__funptr,"hipblasChpmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,long,hipblasComplex *,int,long,hipblasComplex *,hipblasComplex *,int,long,int) nogil> _hipblasChpmvStridedBatched__funptr)(handle,uplo,n,alpha,AP,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasZhpmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZhpmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasZhpmvStridedBatched__funptr
    __init_symbol(&_hipblasZhpmvStridedBatched__funptr,"hipblasZhpmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZhpmvStridedBatched__funptr)(handle,uplo,n,alpha,AP,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasChpr__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hpr performs the matrix-vector operations
# 
#     A := A + alpha*x*x**H
# 
# where alpha is a real scalar, x is a vector, and A is an
# n by n Hermitian matrix, supplied in packed form.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[inout]
# AP        device pointer storing the packed version of the specified triangular portion of
#           the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the Hermitian matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the Hermitian matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
cdef hipblasStatus_t hipblasChpr(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex * x,int incx,hipblasComplex * AP) nogil:
    global _hipblasChpr__funptr
    __init_symbol(&_hipblasChpr__funptr,"hipblasChpr")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,hipblasComplex *,int,hipblasComplex *) nogil> _hipblasChpr__funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* _hipblasZhpr__funptr = NULL
cdef hipblasStatus_t hipblasZhpr(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP) nogil:
    global _hipblasZhpr__funptr
    __init_symbol(&_hipblasZhpr__funptr,"hipblasZhpr")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> _hipblasZhpr__funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* _hipblasChprBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hprBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*x_i**H
# 
# where alpha is a real scalar, x_i is a vector, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[inout]
# AP        device array of device pointers storing the packed version of the specified triangular portion of
#           each Hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
cdef hipblasStatus_t hipblasChprBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* AP,int batchCount) nogil:
    global _hipblasChprBatched__funptr
    __init_symbol(&_hipblasChprBatched__funptr,"hipblasChprBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,hipblasComplex *const*,int,hipblasComplex *const*,int) nogil> _hipblasChprBatched__funptr)(handle,uplo,n,alpha,x,incx,AP,batchCount)


cdef void* _hipblasZhprBatched__funptr = NULL
cdef hipblasStatus_t hipblasZhprBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* AP,int batchCount) nogil:
    global _hipblasZhprBatched__funptr
    __init_symbol(&_hipblasZhprBatched__funptr,"hipblasZhprBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int) nogil> _hipblasZhprBatched__funptr)(handle,uplo,n,alpha,x,incx,AP,batchCount)


cdef void* _hipblasChprStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hprStridedBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*x_i**H
# 
# where alpha is a real scalar, x_i is a vector, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer pointing to the first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[inout]
# AP        device array of device pointers storing the packed version of the specified triangular portion of
#           each Hermitian matrix A_i. Points to the first matrix (A_1).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# strideA   [hipblasStride]
#             stride from the start of one (A_i) and the next (A_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch.
cdef hipblasStatus_t hipblasChprStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * AP,long strideA,int batchCount) nogil:
    global _hipblasChprStridedBatched__funptr
    __init_symbol(&_hipblasChprStridedBatched__funptr,"hipblasChprStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,hipblasComplex *,int,long,hipblasComplex *,long,int) nogil> _hipblasChprStridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,AP,strideA,batchCount)


cdef void* _hipblasZhprStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZhprStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * AP,long strideA,int batchCount) nogil:
    global _hipblasZhprStridedBatched__funptr
    __init_symbol(&_hipblasZhprStridedBatched__funptr,"hipblasZhprStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,long,int) nogil> _hipblasZhprStridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,AP,strideA,batchCount)


cdef void* _hipblasChpr2__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hpr2 performs the matrix-vector operations
# 
#     A := A + alpha*x*y**H + conj(alpha)*y*x**H
# 
# where alpha is a complex scalar, x and y are vectors, and A is an
# n by n Hermitian matrix, supplied in packed form.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# AP        device pointer storing the packed version of the specified triangular portion of
#           the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the Hermitian matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the Hermitian matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
cdef hipblasStatus_t hipblasChpr2(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP) nogil:
    global _hipblasChpr2__funptr
    __init_symbol(&_hipblasChpr2__funptr,"hipblasChpr2")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *) nogil> _hipblasChpr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP)


cdef void* _hipblasZhpr2__funptr = NULL
cdef hipblasStatus_t hipblasZhpr2(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP) nogil:
    global _hipblasZhpr2__funptr
    __init_symbol(&_hipblasZhpr2__funptr,"hipblasZhpr2")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> _hipblasZhpr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP)


cdef void* _hipblasChpr2Batched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hpr2Batched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H
# 
# where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[inout]
# AP        device array of device pointers storing the packed version of the specified triangular portion of
#           each Hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
cdef hipblasStatus_t hipblasChpr2Batched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,hipblasComplex *const* AP,int batchCount) nogil:
    global _hipblasChpr2Batched__funptr
    __init_symbol(&_hipblasChpr2Batched__funptr,"hipblasChpr2Batched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *const*,int) nogil> _hipblasChpr2Batched__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,batchCount)


cdef void* _hipblasZhpr2Batched__funptr = NULL
cdef hipblasStatus_t hipblasZhpr2Batched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,hipblasDoubleComplex *const* AP,int batchCount) nogil:
    global _hipblasZhpr2Batched__funptr
    __init_symbol(&_hipblasZhpr2Batched__funptr,"hipblasZhpr2Batched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int) nogil> _hipblasZhpr2Batched__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,batchCount)


cdef void* _hipblasChpr2StridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# hpr2StridedBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H
# 
# where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer pointing to the first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[in]
# y         device pointer pointing to the first vector (y_1).
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey  [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1).
# @param[inout]
# AP        device array of device pointers storing the packed version of the specified triangular portion of
#           each Hermitian matrix A_i. Points to the first matrix (A_1).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# strideA    [hipblasStride]
#             stride from the start of one (A_i) and the next (A_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch.
cdef hipblasStatus_t hipblasChpr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,hipblasComplex * AP,long strideA,int batchCount) nogil:
    global _hipblasChpr2StridedBatched__funptr
    __init_symbol(&_hipblasChpr2StridedBatched__funptr,"hipblasChpr2StridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,long,int) nogil> _hipblasChpr2StridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,y,incy,stridey,AP,strideA,batchCount)


cdef void* _hipblasZhpr2StridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZhpr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,hipblasDoubleComplex * AP,long strideA,int batchCount) nogil:
    global _hipblasZhpr2StridedBatched__funptr
    __init_symbol(&_hipblasZhpr2StridedBatched__funptr,"hipblasZhpr2StridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,long,int) nogil> _hipblasZhpr2StridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,y,incy,stridey,AP,strideA,batchCount)


cdef void* _hipblasSsbmv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# sbmv performs the matrix-vector operation:
# 
#     y := alpha*A*x + beta*y,
# 
# where alpha and beta are scalars, x and y are n element vectors and
# A should contain an upper or lower triangular n by n symmetric banded matrix.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : s,d
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
# @param[in]
# k         [int]
#           specifies the number of sub- and super-diagonals
# @param[in]
# alpha
#           specifies the scalar alpha
# @param[in]
# AP         pointer storing matrix A on the GPU
# @param[in]
# lda       [int]
#           specifies the leading dimension of matrix A
# @param[in]
# x         pointer storing vector x on the GPU
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x
# @param[in]
# beta      specifies the scalar beta
# @param[out]
# y         pointer storing vector y on the GPU
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y
#
cdef hipblasStatus_t hipblasSsbmv(void * handle,hipblasFillMode_t uplo,int n,int k,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil:
    global _hipblasSsbmv__funptr
    __init_symbol(&_hipblasSsbmv__funptr,"hipblasSsbmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSsbmv__funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasDsbmv__funptr = NULL
cdef hipblasStatus_t hipblasDsbmv(void * handle,hipblasFillMode_t uplo,int n,int k,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _hipblasDsbmv__funptr
    __init_symbol(&_hipblasDsbmv__funptr,"hipblasDsbmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDsbmv__funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasSsbmvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# sbmvBatched performs the matrix-vector operation:
# 
#     y_i := alpha*A_i*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# n by n symmetric banded matrix, for i = 1, ..., batchCount.
# A should contain an upper or lower triangular n by n symmetric banded matrix.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           number of rows and columns of each matrix A_i
# @param[in]
# k         [int]
#           specifies the number of sub- and super-diagonals
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha
# @param[in]
# AP         device array of device pointers storing each matrix A_i
# @param[in]
# lda       [int]
#           specifies the leading dimension of each matrix A_i
# @param[in]
# x         device array of device pointers storing each vector x_i
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i
# @param[in]
# beta      device pointer or host pointer to scalar beta
# @param[out]
# y         device array of device pointers storing each vector y_i
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasSsbmvBatched(void * handle,hipblasFillMode_t uplo,int n,int k,const float * alpha,const float *const* AP,int lda,const float *const* x,int incx,const float * beta,float ** y,int incy,int batchCount) nogil:
    global _hipblasSsbmvBatched__funptr
    __init_symbol(&_hipblasSsbmvBatched__funptr,"hipblasSsbmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,int,const float *,const float *const*,int,const float *const*,int,const float *,float **,int,int) nogil> _hipblasSsbmvBatched__funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasDsbmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsbmvBatched(void * handle,hipblasFillMode_t uplo,int n,int k,const double * alpha,const double *const* AP,int lda,const double *const* x,int incx,const double * beta,double ** y,int incy,int batchCount) nogil:
    global _hipblasDsbmvBatched__funptr
    __init_symbol(&_hipblasDsbmvBatched__funptr,"hipblasDsbmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,int,const double *,const double *const*,int,const double *const*,int,const double *,double **,int,int) nogil> _hipblasDsbmvBatched__funptr)(handle,uplo,n,k,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasSsbmvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# sbmvStridedBatched performs the matrix-vector operation:
# 
#     y_i := alpha*A_i*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# n by n symmetric banded matrix, for i = 1, ..., batchCount.
# A should contain an upper or lower triangular n by n symmetric banded matrix.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           number of rows and columns of each matrix A_i
# @param[in]
# k         [int]
#           specifies the number of sub- and super-diagonals
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha
# @param[in]
# AP        Device pointer to the first matrix A_1 on the GPU
# @param[in]
# lda       [int]
#           specifies the leading dimension of each matrix A_i
# @param[in]
# strideA     [hipblasStride]
#             stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x         Device pointer to the first vector x_1 on the GPU
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1).
#             There are no restrictions placed on stridex, however the user should
#             take care to ensure that stridex is of appropriate size.
#             This typically means stridex >= n * incx. stridex should be non zero.
# @param[in]
# beta      device pointer or host pointer to scalar beta
# @param[out]
# y         Device pointer to the first vector y_1 on the GPU
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i
# @param[in]
# stridey     [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1).
#             There are no restrictions placed on stridey, however the user should
#             take care to ensure that stridey is of appropriate size.
#             This typically means stridey >= n * incy. stridey should be non zero.
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasSsbmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,int k,const float * alpha,const float * AP,int lda,long strideA,const float * x,int incx,long stridex,const float * beta,float * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasSsbmvStridedBatched__funptr
    __init_symbol(&_hipblasSsbmvStridedBatched__funptr,"hipblasSsbmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,int,const float *,const float *,int,long,const float *,int,long,const float *,float *,int,long,int) nogil> _hipblasSsbmvStridedBatched__funptr)(handle,uplo,n,k,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasDsbmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsbmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,int k,const double * alpha,const double * AP,int lda,long strideA,const double * x,int incx,long stridex,const double * beta,double * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasDsbmvStridedBatched__funptr
    __init_symbol(&_hipblasDsbmvStridedBatched__funptr,"hipblasDsbmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,int,const double *,const double *,int,long,const double *,int,long,const double *,double *,int,long,int) nogil> _hipblasDsbmvStridedBatched__funptr)(handle,uplo,n,k,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasSspmv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# spmv performs the matrix-vector operation:
# 
#     y := alpha*A*x + beta*y,
# 
# where alpha and beta are scalars, x and y are n element vectors and
# A should contain an upper or lower triangular n by n packed symmetric matrix.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : s,d
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
# @param[in]
# alpha
#           specifies the scalar alpha
# @param[in]
# AP         pointer storing matrix A on the GPU
# @param[in]
# x         pointer storing vector x on the GPU
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x
# @param[in]
# beta      specifies the scalar beta
# @param[out]
# y         pointer storing vector y on the GPU
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y
#
cdef hipblasStatus_t hipblasSspmv(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * AP,const float * x,int incx,const float * beta,float * y,int incy) nogil:
    global _hipblasSspmv__funptr
    __init_symbol(&_hipblasSspmv__funptr,"hipblasSspmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *,const float *,int,const float *,float *,int) nogil> _hipblasSspmv__funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)


cdef void* _hipblasDspmv__funptr = NULL
cdef hipblasStatus_t hipblasDspmv(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _hipblasDspmv__funptr
    __init_symbol(&_hipblasDspmv__funptr,"hipblasDspmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *,const double *,int,const double *,double *,int) nogil> _hipblasDspmv__funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)


cdef void* _hipblasSspmvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# spmvBatched performs the matrix-vector operation:
# 
#     y_i := alpha*AP_i*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# n by n symmetric matrix, for i = 1, ..., batchCount.
# A should contain an upper or lower triangular n by n packed symmetric matrix.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           number of rows and columns of each matrix A_i
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha
# @param[in]
# AP         device array of device pointers storing each matrix A_i
# @param[in]
# x         device array of device pointers storing each vector x_i
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i
# @param[in]
# beta      device pointer or host pointer to scalar beta
# @param[out]
# y         device array of device pointers storing each vector y_i
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasSspmvBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float *const* AP,const float *const* x,int incx,const float * beta,float ** y,int incy,int batchCount) nogil:
    global _hipblasSspmvBatched__funptr
    __init_symbol(&_hipblasSspmvBatched__funptr,"hipblasSspmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *const*,const float *const*,int,const float *,float **,int,int) nogil> _hipblasSspmvBatched__funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasDspmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasDspmvBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double *const* AP,const double *const* x,int incx,const double * beta,double ** y,int incy,int batchCount) nogil:
    global _hipblasDspmvBatched__funptr
    __init_symbol(&_hipblasDspmvBatched__funptr,"hipblasDspmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *const*,const double *const*,int,const double *,double **,int,int) nogil> _hipblasDspmvBatched__funptr)(handle,uplo,n,alpha,AP,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasSspmvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# spmvStridedBatched performs the matrix-vector operation:
# 
#     y_i := alpha*A_i*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# n by n symmetric matrix, for i = 1, ..., batchCount.
# A should contain an upper or lower triangular n by n packed symmetric matrix.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           number of rows and columns of each matrix A_i
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha
# @param[in]
# AP        Device pointer to the first matrix A_1 on the GPU
# @param[in]
# strideA    [hipblasStride]
#             stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x         Device pointer to the first vector x_1 on the GPU
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1).
#             There are no restrictions placed on stridex, however the user should
#             take care to ensure that stridex is of appropriate size.
#             This typically means stridex >= n * incx. stridex should be non zero.
# @param[in]
# beta      device pointer or host pointer to scalar beta
# @param[out]
# y         Device pointer to the first vector y_1 on the GPU
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i
# @param[in]
# stridey     [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1).
#             There are no restrictions placed on stridey, however the user should
#             take care to ensure that stridey is of appropriate size.
#             This typically means stridey >= n * incy. stridey should be non zero.
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasSspmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * AP,long strideA,const float * x,int incx,long stridex,const float * beta,float * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasSspmvStridedBatched__funptr
    __init_symbol(&_hipblasSspmvStridedBatched__funptr,"hipblasSspmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *,long,const float *,int,long,const float *,float *,int,long,int) nogil> _hipblasSspmvStridedBatched__funptr)(handle,uplo,n,alpha,AP,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasDspmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDspmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,long strideA,const double * x,int incx,long stridex,const double * beta,double * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasDspmvStridedBatched__funptr
    __init_symbol(&_hipblasDspmvStridedBatched__funptr,"hipblasDspmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *,long,const double *,int,long,const double *,double *,int,long,int) nogil> _hipblasDspmvStridedBatched__funptr)(handle,uplo,n,alpha,AP,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasSspr__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# spr performs the matrix-vector operations
# 
#     A := A + alpha*x*x**T
# 
# where alpha is a scalar, x is a vector, and A is an
# n by n symmetric matrix, supplied in packed form.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[inout]
# AP        device pointer storing the packed version of the specified triangular portion of
#           the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the symmetric matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                     1 2 4 7
#                     2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     4 5 6 9
#                     7 8 9 0
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the symmetric matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                     1 2 3 4
#                     2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     3 6 8 9
#                     4 7 9 0
cdef hipblasStatus_t hipblasSspr(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,float * AP) nogil:
    global _hipblasSspr__funptr
    __init_symbol(&_hipblasSspr__funptr,"hipblasSspr")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *,int,float *) nogil> _hipblasSspr__funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* _hipblasDspr__funptr = NULL
cdef hipblasStatus_t hipblasDspr(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,double * AP) nogil:
    global _hipblasDspr__funptr
    __init_symbol(&_hipblasDspr__funptr,"hipblasDspr")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *,int,double *) nogil> _hipblasDspr__funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* _hipblasCspr__funptr = NULL
cdef hipblasStatus_t hipblasCspr(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * AP) nogil:
    global _hipblasCspr__funptr
    __init_symbol(&_hipblasCspr__funptr,"hipblasCspr")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *) nogil> _hipblasCspr__funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* _hipblasZspr__funptr = NULL
cdef hipblasStatus_t hipblasZspr(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP) nogil:
    global _hipblasZspr__funptr
    __init_symbol(&_hipblasZspr__funptr,"hipblasZspr")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *) nogil> _hipblasZspr__funptr)(handle,uplo,n,alpha,x,incx,AP)


cdef void* _hipblasSsprBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# sprBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*x_i**T
# 
# where alpha is a scalar, x_i is a vector, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[inout]
# AP        device array of device pointers storing the packed version of the specified triangular portion of
#           each symmetric matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                     1 2 4 7
#                     2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     4 5 6 9
#                     7 8 9 0
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                     1 2 3 4
#                     2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     3 6 8 9
#                     4 7 9 0
# @param[in]
# batchCount [int]
#             number of instances in the batch.
cdef hipblasStatus_t hipblasSsprBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float *const* x,int incx,float *const* AP,int batchCount) nogil:
    global _hipblasSsprBatched__funptr
    __init_symbol(&_hipblasSsprBatched__funptr,"hipblasSsprBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *const*,int,float *const*,int) nogil> _hipblasSsprBatched__funptr)(handle,uplo,n,alpha,x,incx,AP,batchCount)


cdef void* _hipblasDsprBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsprBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double *const* x,int incx,double *const* AP,int batchCount) nogil:
    global _hipblasDsprBatched__funptr
    __init_symbol(&_hipblasDsprBatched__funptr,"hipblasDsprBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *const*,int,double *const*,int) nogil> _hipblasDsprBatched__funptr)(handle,uplo,n,alpha,x,incx,AP,batchCount)


cdef void* _hipblasCsprBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsprBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* AP,int batchCount) nogil:
    global _hipblasCsprBatched__funptr
    __init_symbol(&_hipblasCsprBatched__funptr,"hipblasCsprBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int) nogil> _hipblasCsprBatched__funptr)(handle,uplo,n,alpha,x,incx,AP,batchCount)


cdef void* _hipblasZsprBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsprBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* AP,int batchCount) nogil:
    global _hipblasZsprBatched__funptr
    __init_symbol(&_hipblasZsprBatched__funptr,"hipblasZsprBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int) nogil> _hipblasZsprBatched__funptr)(handle,uplo,n,alpha,x,incx,AP,batchCount)


cdef void* _hipblasSsprStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# sprStridedBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*x_i**T
# 
# where alpha is a scalar, x_i is a vector, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer pointing to the first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[inout]
# AP        device pointer storing the packed version of the specified triangular portion of
#           each symmetric matrix A_i. Points to the first A_1.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                     1 2 4 7
#                     2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     4 5 6 9
#                     7 8 9 0
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                     1 2 3 4
#                     2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     3 6 8 9
#                     4 7 9 0
# @param[in]
# strideA    [hipblasStride]
#             stride from the start of one (A_i) and the next (A_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch.
cdef hipblasStatus_t hipblasSsprStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,long stridex,float * AP,long strideA,int batchCount) nogil:
    global _hipblasSsprStridedBatched__funptr
    __init_symbol(&_hipblasSsprStridedBatched__funptr,"hipblasSsprStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *,int,long,float *,long,int) nogil> _hipblasSsprStridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,AP,strideA,batchCount)


cdef void* _hipblasDsprStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsprStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,long stridex,double * AP,long strideA,int batchCount) nogil:
    global _hipblasDsprStridedBatched__funptr
    __init_symbol(&_hipblasDsprStridedBatched__funptr,"hipblasDsprStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *,int,long,double *,long,int) nogil> _hipblasDsprStridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,AP,strideA,batchCount)


cdef void* _hipblasCsprStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsprStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * AP,long strideA,int batchCount) nogil:
    global _hipblasCsprStridedBatched__funptr
    __init_symbol(&_hipblasCsprStridedBatched__funptr,"hipblasCsprStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,long,int) nogil> _hipblasCsprStridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,AP,strideA,batchCount)


cdef void* _hipblasZsprStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsprStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * AP,long strideA,int batchCount) nogil:
    global _hipblasZsprStridedBatched__funptr
    __init_symbol(&_hipblasZsprStridedBatched__funptr,"hipblasZsprStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,long,int) nogil> _hipblasZsprStridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,AP,strideA,batchCount)


cdef void* _hipblasSspr2__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# spr2 performs the matrix-vector operation
# 
#     A := A + alpha*x*y**T + alpha*y*x**T
# 
# where alpha is a scalar, x and y are vectors, and A is an
# n by n symmetric matrix, supplied in packed form.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : s,d
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# AP        device pointer storing the packed version of the specified triangular portion of
#           the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the symmetric matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                     1 2 4 7
#                     2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     4 5 6 9
#                     7 8 9 0
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the symmetric matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(n) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                     1 2 3 4
#                     2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     3 6 8 9
#                     4 7 9 0
cdef hipblasStatus_t hipblasSspr2(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,const float * y,int incy,float * AP) nogil:
    global _hipblasSspr2__funptr
    __init_symbol(&_hipblasSspr2__funptr,"hipblasSspr2")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *,int,const float *,int,float *) nogil> _hipblasSspr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP)


cdef void* _hipblasDspr2__funptr = NULL
cdef hipblasStatus_t hipblasDspr2(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP) nogil:
    global _hipblasDspr2__funptr
    __init_symbol(&_hipblasDspr2__funptr,"hipblasDspr2")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *,int,const double *,int,double *) nogil> _hipblasDspr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP)


cdef void* _hipblasSspr2Batched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# spr2Batched performs the matrix-vector operation
# 
#     A_i := A_i + alpha*x_i*y_i**T + alpha*y_i*x_i**T
# 
# where alpha is a scalar, x_i and y_i are vectors, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[inout]
# AP        device array of device pointers storing the packed version of the specified triangular portion of
#           each symmetric matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                     1 2 4 7
#                     2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     4 5 6 9
#                     7 8 9 0
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(n) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                     1 2 3 4
#                     2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     3 6 8 9
#                     4 7 9 0
# @param[in]
# batchCount [int]
#             number of instances in the batch.
cdef hipblasStatus_t hipblasSspr2Batched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float *const* x,int incx,const float *const* y,int incy,float *const* AP,int batchCount) nogil:
    global _hipblasSspr2Batched__funptr
    __init_symbol(&_hipblasSspr2Batched__funptr,"hipblasSspr2Batched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *const*,int,const float *const*,int,float *const*,int) nogil> _hipblasSspr2Batched__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,batchCount)


cdef void* _hipblasDspr2Batched__funptr = NULL
cdef hipblasStatus_t hipblasDspr2Batched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double *const* x,int incx,const double *const* y,int incy,double *const* AP,int batchCount) nogil:
    global _hipblasDspr2Batched__funptr
    __init_symbol(&_hipblasDspr2Batched__funptr,"hipblasDspr2Batched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *const*,int,const double *const*,int,double *const*,int) nogil> _hipblasDspr2Batched__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,batchCount)


cdef void* _hipblasSspr2StridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# spr2StridedBatched performs the matrix-vector operation
# 
#     A_i := A_i + alpha*x_i*y_i**T + alpha*y_i*x_i**T
# 
# where alpha is a scalar, x_i amd y_i are vectors, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer pointing to the first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[in]
# y         device pointer pointing to the first vector (y_1).
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey  [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1).
# @param[inout]
# AP        device pointer storing the packed version of the specified triangular portion of
#           each symmetric matrix A_i. Points to the first A_1.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                     1 2 4 7
#                     2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     4 5 6 9
#                     7 8 9 0
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(n) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                     1 2 3 4
#                     2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     3 6 8 9
#                     4 7 9 0
# @param[in]
# strideA   [hipblasStride]
#             stride from the start of one (A_i) and the next (A_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch.
cdef hipblasStatus_t hipblasSspr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,long stridex,const float * y,int incy,long stridey,float * AP,long strideA,int batchCount) nogil:
    global _hipblasSspr2StridedBatched__funptr
    __init_symbol(&_hipblasSspr2StridedBatched__funptr,"hipblasSspr2StridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *,int,long,const float *,int,long,float *,long,int) nogil> _hipblasSspr2StridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,y,incy,stridey,AP,strideA,batchCount)


cdef void* _hipblasDspr2StridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDspr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,long stridex,const double * y,int incy,long stridey,double * AP,long strideA,int batchCount) nogil:
    global _hipblasDspr2StridedBatched__funptr
    __init_symbol(&_hipblasDspr2StridedBatched__funptr,"hipblasDspr2StridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *,int,long,const double *,int,long,double *,long,int) nogil> _hipblasDspr2StridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,y,incy,stridey,AP,strideA,batchCount)


cdef void* _hipblasSsymv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# symv performs the matrix-vector operation:
# 
#     y := alpha*A*x + beta*y,
# 
# where alpha and beta are scalars, x and y are n element vectors and
# A should contain an upper or lower triangular n by n symmetric matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
# @param[in]
# alpha
#           specifies the scalar alpha
# @param[in]
# AP         pointer storing matrix A on the GPU
# @param[in]
# lda       [int]
#           specifies the leading dimension of A
# @param[in]
# x         pointer storing vector x on the GPU
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x
# @param[in]
# beta      specifies the scalar beta
# @param[out]
# y         pointer storing vector y on the GPU
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y
#
cdef hipblasStatus_t hipblasSsymv(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil:
    global _hipblasSsymv__funptr
    __init_symbol(&_hipblasSsymv__funptr,"hipblasSsymv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSsymv__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasDsymv__funptr = NULL
cdef hipblasStatus_t hipblasDsymv(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil:
    global _hipblasDsymv__funptr
    __init_symbol(&_hipblasDsymv__funptr,"hipblasDsymv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDsymv__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasCsymv__funptr = NULL
cdef hipblasStatus_t hipblasCsymv(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil:
    global _hipblasCsymv__funptr
    __init_symbol(&_hipblasCsymv__funptr,"hipblasCsymv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCsymv__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasZsymv__funptr = NULL
cdef hipblasStatus_t hipblasZsymv(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil:
    global _hipblasZsymv__funptr
    __init_symbol(&_hipblasZsymv__funptr,"hipblasZsymv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZsymv__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy)


cdef void* _hipblasSsymvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# symvBatched performs the matrix-vector operation:
# 
#     y_i := alpha*A_i*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# n by n symmetric matrix, for i = 1, ..., batchCount.
# A a should contain an upper or lower triangular symmetric matrix
# and the opposing triangular part of A is not referenced
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           number of rows and columns of each matrix A_i
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha
# @param[in]
# AP        device array of device pointers storing each matrix A_i
# @param[in]
# lda       [int]
#           specifies the leading dimension of each matrix A_i
# @param[in]
# x         device array of device pointers storing each vector x_i
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i
# @param[in]
# beta      device pointer or host pointer to scalar beta
# @param[out]
# y         device array of device pointers storing each vector y_i
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasSsymvBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float *const* AP,int lda,const float *const* x,int incx,const float * beta,float ** y,int incy,int batchCount) nogil:
    global _hipblasSsymvBatched__funptr
    __init_symbol(&_hipblasSsymvBatched__funptr,"hipblasSsymvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *const*,int,const float *const*,int,const float *,float **,int,int) nogil> _hipblasSsymvBatched__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasDsymvBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsymvBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double *const* AP,int lda,const double *const* x,int incx,const double * beta,double ** y,int incy,int batchCount) nogil:
    global _hipblasDsymvBatched__funptr
    __init_symbol(&_hipblasDsymvBatched__funptr,"hipblasDsymvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *const*,int,const double *const*,int,const double *,double **,int,int) nogil> _hipblasDsymvBatched__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasCsymvBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsymvBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,hipblasComplex * beta,hipblasComplex ** y,int incy,int batchCount) nogil:
    global _hipblasCsymvBatched__funptr
    __init_symbol(&_hipblasCsymvBatched__funptr,"hipblasCsymvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *,hipblasComplex **,int,int) nogil> _hipblasCsymvBatched__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasZsymvBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsymvBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex ** y,int incy,int batchCount) nogil:
    global _hipblasZsymvBatched__funptr
    __init_symbol(&_hipblasZsymvBatched__funptr,"hipblasZsymvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *,hipblasDoubleComplex **,int,int) nogil> _hipblasZsymvBatched__funptr)(handle,uplo,n,alpha,AP,lda,x,incx,beta,y,incy,batchCount)


cdef void* _hipblasSsymvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# symvStridedBatched performs the matrix-vector operation:
# 
#     y_i := alpha*A_i*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# n by n symmetric matrix, for i = 1, ..., batchCount.
# A a should contain an upper or lower triangular symmetric matrix
# and the opposing triangular part of A is not referenced
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           number of rows and columns of each matrix A_i
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha
# @param[in]
# AP         Device pointer to the first matrix A_1 on the GPU
# @param[in]
# lda       [int]
#           specifies the leading dimension of each matrix A_i
# @param[in]
# strideA     [hipblasStride]
#             stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x         Device pointer to the first vector x_1 on the GPU
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1).
#             There are no restrictions placed on stridex, however the user should
#             take care to ensure that stridex is of appropriate size.
#             This typically means stridex >= n * incx. stridex should be non zero.
# @param[in]
# beta      device pointer or host pointer to scalar beta
# @param[out]
# y         Device pointer to the first vector y_1 on the GPU
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i
# @param[in]
# stridey     [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1).
#             There are no restrictions placed on stridey, however the user should
#             take care to ensure that stridey is of appropriate size.
#             This typically means stridey >= n * incy. stridey should be non zero.
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasSsymvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * AP,int lda,long strideA,const float * x,int incx,long stridex,const float * beta,float * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasSsymvStridedBatched__funptr
    __init_symbol(&_hipblasSsymvStridedBatched__funptr,"hipblasSsymvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *,int,long,const float *,int,long,const float *,float *,int,long,int) nogil> _hipblasSsymvStridedBatched__funptr)(handle,uplo,n,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasDsymvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsymvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,int lda,long strideA,const double * x,int incx,long stridex,const double * beta,double * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasDsymvStridedBatched__funptr
    __init_symbol(&_hipblasDsymvStridedBatched__funptr,"hipblasDsymvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *,int,long,const double *,int,long,const double *,double *,int,long,int) nogil> _hipblasDsymvStridedBatched__funptr)(handle,uplo,n,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasCsymvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsymvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * beta,hipblasComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasCsymvStridedBatched__funptr
    __init_symbol(&_hipblasCsymvStridedBatched__funptr,"hipblasCsymvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,hipblasComplex *,int,long,int) nogil> _hipblasCsymvStridedBatched__funptr)(handle,uplo,n,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasZsymvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsymvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil:
    global _hipblasZsymvStridedBatched__funptr
    __init_symbol(&_hipblasZsymvStridedBatched__funptr,"hipblasZsymvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZsymvStridedBatched__funptr)(handle,uplo,n,alpha,AP,lda,strideA,x,incx,stridex,beta,y,incy,stridey,batchCount)


cdef void* _hipblasSsyr__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# syr performs the matrix-vector operations
# 
#     A := A + alpha*x*x**T
# 
# where alpha is a scalar, x is a vector, and A is an
# n by n symmetric matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# 
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[inout]
# AP         device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
#
cdef hipblasStatus_t hipblasSsyr(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,float * AP,int lda) nogil:
    global _hipblasSsyr__funptr
    __init_symbol(&_hipblasSsyr__funptr,"hipblasSsyr")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *,int,float *,int) nogil> _hipblasSsyr__funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* _hipblasDsyr__funptr = NULL
cdef hipblasStatus_t hipblasDsyr(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,double * AP,int lda) nogil:
    global _hipblasDsyr__funptr
    __init_symbol(&_hipblasDsyr__funptr,"hipblasDsyr")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *,int,double *,int) nogil> _hipblasDsyr__funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* _hipblasCsyr__funptr = NULL
cdef hipblasStatus_t hipblasCsyr(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * AP,int lda) nogil:
    global _hipblasCsyr__funptr
    __init_symbol(&_hipblasCsyr__funptr,"hipblasCsyr")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCsyr__funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* _hipblasZsyr__funptr = NULL
cdef hipblasStatus_t hipblasZsyr(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP,int lda) nogil:
    global _hipblasZsyr__funptr
    __init_symbol(&_hipblasZsyr__funptr,"hipblasZsyr")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZsyr__funptr)(handle,uplo,n,alpha,x,incx,AP,lda)


cdef void* _hipblasSsyrBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# syrBatched performs a batch of matrix-vector operations
# 
#     A[i] := A[i] + alpha*x[i]*x[i]**T
# 
# where alpha is a scalar, x is an array of vectors, and A is an array of
# n by n symmetric matrices, for i = 1 , ... , batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[inout]
# AP         device array of device pointers storing each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasSsyrBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float *const* x,int incx,float *const* AP,int lda,int batchCount) nogil:
    global _hipblasSsyrBatched__funptr
    __init_symbol(&_hipblasSsyrBatched__funptr,"hipblasSsyrBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *const*,int,float *const*,int,int) nogil> _hipblasSsyrBatched__funptr)(handle,uplo,n,alpha,x,incx,AP,lda,batchCount)


cdef void* _hipblasDsyrBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsyrBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double *const* x,int incx,double *const* AP,int lda,int batchCount) nogil:
    global _hipblasDsyrBatched__funptr
    __init_symbol(&_hipblasDsyrBatched__funptr,"hipblasDsyrBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *const*,int,double *const*,int,int) nogil> _hipblasDsyrBatched__funptr)(handle,uplo,n,alpha,x,incx,AP,lda,batchCount)


cdef void* _hipblasCsyrBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsyrBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* AP,int lda,int batchCount) nogil:
    global _hipblasCsyrBatched__funptr
    __init_symbol(&_hipblasCsyrBatched__funptr,"hipblasCsyrBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCsyrBatched__funptr)(handle,uplo,n,alpha,x,incx,AP,lda,batchCount)


cdef void* _hipblasZsyrBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsyrBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* AP,int lda,int batchCount) nogil:
    global _hipblasZsyrBatched__funptr
    __init_symbol(&_hipblasZsyrBatched__funptr,"hipblasZsyrBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZsyrBatched__funptr)(handle,uplo,n,alpha,x,incx,AP,lda,batchCount)


cdef void* _hipblasSsyrStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# syrStridedBatched performs the matrix-vector operations
# 
#     A[i] := A[i] + alpha*x[i]*x[i]**T
# 
# where alpha is a scalar, vectors, and A is an array of
# n by n symmetric matrices, for i = 1 , ... , batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex   [hipblasStride]
#           specifies the pointer increment between vectors (x_i) and (x_i+1).
# @param[inout]
# AP         device pointer to the first matrix A_1.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# strideA   [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# batchCount [int]
#           number of instances in the batch
#
cdef hipblasStatus_t hipblasSsyrStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,long stridex,float * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasSsyrStridedBatched__funptr
    __init_symbol(&_hipblasSsyrStridedBatched__funptr,"hipblasSsyrStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *,int,long,float *,int,long,int) nogil> _hipblasSsyrStridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,AP,lda,strideA,batchCount)


cdef void* _hipblasDsyrStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsyrStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,long stridex,double * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasDsyrStridedBatched__funptr
    __init_symbol(&_hipblasDsyrStridedBatched__funptr,"hipblasDsyrStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *,int,long,double *,int,long,int) nogil> _hipblasDsyrStridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,AP,lda,strideA,batchCount)


cdef void* _hipblasCsyrStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsyrStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasCsyrStridedBatched__funptr
    __init_symbol(&_hipblasCsyrStridedBatched__funptr,"hipblasCsyrStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCsyrStridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,AP,lda,strideA,batchCount)


cdef void* _hipblasZsyrStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsyrStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasZsyrStridedBatched__funptr
    __init_symbol(&_hipblasZsyrStridedBatched__funptr,"hipblasZsyrStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZsyrStridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,AP,lda,strideA,batchCount)


cdef void* _hipblasSsyr2__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# syr2 performs the matrix-vector operations
# 
#     A := A + alpha*x*y**T + alpha*y*x**T
# 
# where alpha is a scalar, x and y are vectors, and A is an
# n by n symmetric matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# 
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# AP         device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
#
cdef hipblasStatus_t hipblasSsyr2(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,const float * y,int incy,float * AP,int lda) nogil:
    global _hipblasSsyr2__funptr
    __init_symbol(&_hipblasSsyr2__funptr,"hipblasSsyr2")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *,int,const float *,int,float *,int) nogil> _hipblasSsyr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasDsyr2__funptr = NULL
cdef hipblasStatus_t hipblasDsyr2(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP,int lda) nogil:
    global _hipblasDsyr2__funptr
    __init_symbol(&_hipblasDsyr2__funptr,"hipblasDsyr2")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *,int,const double *,int,double *,int) nogil> _hipblasDsyr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasCsyr2__funptr = NULL
cdef hipblasStatus_t hipblasCsyr2(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil:
    global _hipblasCsyr2__funptr
    __init_symbol(&_hipblasCsyr2__funptr,"hipblasCsyr2")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCsyr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasZsyr2__funptr = NULL
cdef hipblasStatus_t hipblasZsyr2(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil:
    global _hipblasZsyr2__funptr
    __init_symbol(&_hipblasZsyr2__funptr,"hipblasZsyr2")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZsyr2__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda)


cdef void* _hipblasSsyr2Batched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# syr2Batched performs a batch of matrix-vector operations
# 
#     A[i] := A[i] + alpha*x[i]*y[i]**T + alpha*y[i]*x[i]**T
# 
# where alpha is a scalar, x[i] and y[i] are vectors, and A[i] is a
# n by n symmetric matrix, for i = 1 , ... , batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[inout]
# AP         device array of device pointers storing each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasSsyr2Batched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float *const* x,int incx,const float *const* y,int incy,float *const* AP,int lda,int batchCount) nogil:
    global _hipblasSsyr2Batched__funptr
    __init_symbol(&_hipblasSsyr2Batched__funptr,"hipblasSsyr2Batched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *const*,int,const float *const*,int,float *const*,int,int) nogil> _hipblasSsyr2Batched__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda,batchCount)


cdef void* _hipblasDsyr2Batched__funptr = NULL
cdef hipblasStatus_t hipblasDsyr2Batched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double *const* x,int incx,const double *const* y,int incy,double *const* AP,int lda,int batchCount) nogil:
    global _hipblasDsyr2Batched__funptr
    __init_symbol(&_hipblasDsyr2Batched__funptr,"hipblasDsyr2Batched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *const*,int,const double *const*,int,double *const*,int,int) nogil> _hipblasDsyr2Batched__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda,batchCount)


cdef void* _hipblasCsyr2Batched__funptr = NULL
cdef hipblasStatus_t hipblasCsyr2Batched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,hipblasComplex *const* AP,int lda,int batchCount) nogil:
    global _hipblasCsyr2Batched__funptr
    __init_symbol(&_hipblasCsyr2Batched__funptr,"hipblasCsyr2Batched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCsyr2Batched__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda,batchCount)


cdef void* _hipblasZsyr2Batched__funptr = NULL
cdef hipblasStatus_t hipblasZsyr2Batched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,hipblasDoubleComplex *const* AP,int lda,int batchCount) nogil:
    global _hipblasZsyr2Batched__funptr
    __init_symbol(&_hipblasZsyr2Batched__funptr,"hipblasZsyr2Batched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZsyr2Batched__funptr)(handle,uplo,n,alpha,x,incx,y,incy,AP,lda,batchCount)


cdef void* _hipblasSsyr2StridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# syr2StridedBatched the matrix-vector operations
# 
#     A[i] := A[i] + alpha*x[i]*y[i]**T + alpha*y[i]*x[i]**T
# 
# where alpha is a scalar, x[i] and y[i] are vectors, and A[i] is a
# n by n symmetric matrices, for i = 1 , ... , batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex   [hipblasStride]
#           specifies the pointer increment between vectors (x_i) and (x_i+1).
# @param[in]
# y         device pointer to the first vector y_1.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey   [hipblasStride]
#           specifies the pointer increment between vectors (y_i) and (y_i+1).
# @param[inout]
# AP         device pointer to the first matrix A_1.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# strideA   [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# batchCount [int]
#           number of instances in the batch
#
cdef hipblasStatus_t hipblasSsyr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,long stridex,const float * y,int incy,long stridey,float * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasSsyr2StridedBatched__funptr
    __init_symbol(&_hipblasSsyr2StridedBatched__funptr,"hipblasSsyr2StridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const float *,const float *,int,long,const float *,int,long,float *,int,long,int) nogil> _hipblasSsyr2StridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,y,incy,stridey,AP,lda,strideA,batchCount)


cdef void* _hipblasDsyr2StridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsyr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,long stridex,const double * y,int incy,long stridey,double * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasDsyr2StridedBatched__funptr
    __init_symbol(&_hipblasDsyr2StridedBatched__funptr,"hipblasDsyr2StridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,const double *,const double *,int,long,const double *,int,long,double *,int,long,int) nogil> _hipblasDsyr2StridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,y,incy,stridey,AP,lda,strideA,batchCount)


cdef void* _hipblasCsyr2StridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsyr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,hipblasComplex * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasCsyr2StridedBatched__funptr
    __init_symbol(&_hipblasCsyr2StridedBatched__funptr,"hipblasCsyr2StridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCsyr2StridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,y,incy,stridey,AP,lda,strideA,batchCount)


cdef void* _hipblasZsyr2StridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsyr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,hipblasDoubleComplex * AP,int lda,long strideA,int batchCount) nogil:
    global _hipblasZsyr2StridedBatched__funptr
    __init_symbol(&_hipblasZsyr2StridedBatched__funptr,"hipblasZsyr2StridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZsyr2StridedBatched__funptr)(handle,uplo,n,alpha,x,incx,stridex,y,incy,stridey,AP,lda,strideA,batchCount)


cdef void* _hipblasStbmv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# tbmv performs one of the matrix-vector operations
# 
#     x := A*x      or
#     x := A**T*x   or
#     x := A**H*x,
# 
# x is a vectors and A is a banded m by m matrix (see description below).
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: A is an upper banded triangular matrix.
#           HIPBLAS_FILL_MODE_LOWER: A is a  lower banded triangular matrix.
# @param[in]
# transA     [hipblasOperation_t]
#           indicates whether matrix A is tranposed (conjugated) or not.
# @param[in]
# diag      [hipblasDiagType_t]
#           HIPBLAS_DIAG_UNIT: The main diagonal of A is assumed to consist of only
#                                  1's and is not referenced.
#           HIPBLAS_DIAG_NON_UNIT: No assumptions are made of A's main diagonal.
# @param[in]
# m         [int]
#           the number of rows and columns of the matrix represented by A.
# @param[in]
# k         [int]
#           if uplo == HIPBLAS_FILL_MODE_UPPER, k specifies the number of super-diagonals
#           of the matrix A.
#           if uplo == HIPBLAS_FILL_MODE_LOWER, k specifies the number of sub-diagonals
#           of the matrix A.
#           k must satisfy k > 0 && k < lda.
# @param[in]
# AP         device pointer storing banded triangular matrix A.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The matrix represented is an upper banded triangular matrix
#             with the main diagonal and k super-diagonals, everything
#             else can be assumed to be 0.
#             The matrix is compacted so that the main diagonal resides on the k'th
#             row, the first super diagonal resides on the RHS of the k-1'th row, etc,
#             with the k'th diagonal on the RHS of the 0'th row.
#                Ex: (HIPBLAS_FILL_MODE_UPPER; m = 5; k = 2)
#                   1 6 9 0 0              0 0 9 8 7
#                   0 2 7 8 0              0 6 7 8 9
#                   0 0 3 8 7     ---->    1 2 3 4 5
#                   0 0 0 4 9              0 0 0 0 0
#                   0 0 0 0 5              0 0 0 0 0
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The matrix represnted is a lower banded triangular matrix
#             with the main diagonal and k sub-diagonals, everything else can be
#             assumed to be 0.
#             The matrix is compacted so that the main diagonal resides on the 0'th row,
#             working up to the k'th diagonal residing on the LHS of the k'th row.
#                Ex: (HIPBLAS_FILL_MODE_LOWER; m = 5; k = 2)
#                   1 0 0 0 0              1 2 3 4 5
#                   6 2 0 0 0              6 7 8 9 0
#                   9 7 3 0 0     ---->    9 8 7 0 0
#                   0 8 8 4 0              0 0 0 0 0
#                   0 0 7 9 5              0 0 0 0 0
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. lda must satisfy lda > k.
# @param[inout]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
#
cdef hipblasStatus_t hipblasStbmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const float * AP,int lda,float * x,int incx) nogil:
    global _hipblasStbmv__funptr
    __init_symbol(&_hipblasStbmv__funptr,"hipblasStbmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,int,float *,int) nogil> _hipblasStbmv__funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx)


cdef void* _hipblasDtbmv__funptr = NULL
cdef hipblasStatus_t hipblasDtbmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const double * AP,int lda,double * x,int incx) nogil:
    global _hipblasDtbmv__funptr
    __init_symbol(&_hipblasDtbmv__funptr,"hipblasDtbmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,int,double *,int) nogil> _hipblasDtbmv__funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx)


cdef void* _hipblasCtbmv__funptr = NULL
cdef hipblasStatus_t hipblasCtbmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil:
    global _hipblasCtbmv__funptr
    __init_symbol(&_hipblasCtbmv__funptr,"hipblasCtbmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtbmv__funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx)


cdef void* _hipblasZtbmv__funptr = NULL
cdef hipblasStatus_t hipblasZtbmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil:
    global _hipblasZtbmv__funptr
    __init_symbol(&_hipblasZtbmv__funptr,"hipblasZtbmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtbmv__funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx)


cdef void* _hipblasStbmvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# tbmvBatched performs one of the matrix-vector operations
# 
#     x_i := A_i*x_i      or
#     x_i := A_i**T*x_i   or
#     x_i := A_i**H*x_i,
# 
# where (A_i, x_i) is the i-th instance of the batch.
# x_i is a vector and A_i is an m by m matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: each A_i is an upper banded triangular matrix.
#           HIPBLAS_FILL_MODE_LOWER: each A_i is a  lower banded triangular matrix.
# @param[in]
# transA     [hipblasOperation_t]
#           indicates whether each matrix A_i is tranposed (conjugated) or not.
# @param[in]
# diag      [hipblasDiagType_t]
#           HIPBLAS_DIAG_UNIT: The main diagonal of each A_i is assumed to consist of only
#                                  1's and is not referenced.
#           HIPBLAS_DIAG_NON_UNIT: No assumptions are made of each A_i's main diagonal.
# @param[in]
# m         [int]
#           the number of rows and columns of the matrix represented by each A_i.
# @param[in]
# k         [int]
#           if uplo == HIPBLAS_FILL_MODE_UPPER, k specifies the number of super-diagonals
#           of each matrix A_i.
#           if uplo == HIPBLAS_FILL_MODE_LOWER, k specifies the number of sub-diagonals
#           of each matrix A_i.
#           k must satisfy k > 0 && k < lda.
# @param[in]
# AP         device array of device pointers storing each banded triangular matrix A_i.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The matrix represented is an upper banded triangular matrix
#             with the main diagonal and k super-diagonals, everything
#             else can be assumed to be 0.
#             The matrix is compacted so that the main diagonal resides on the k'th
#             row, the first super diagonal resides on the RHS of the k-1'th row, etc,
#             with the k'th diagonal on the RHS of the 0'th row.
#                Ex: (HIPBLAS_FILL_MODE_UPPER; m = 5; k = 2)
#                   1 6 9 0 0              0 0 9 8 7
#                   0 2 7 8 0              0 6 7 8 9
#                   0 0 3 8 7     ---->    1 2 3 4 5
#                   0 0 0 4 9              0 0 0 0 0
#                   0 0 0 0 5              0 0 0 0 0
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The matrix represnted is a lower banded triangular matrix
#             with the main diagonal and k sub-diagonals, everything else can be
#             assumed to be 0.
#             The matrix is compacted so that the main diagonal resides on the 0'th row,
#             working up to the k'th diagonal residing on the LHS of the k'th row.
#                Ex: (HIPBLAS_FILL_MODE_LOWER; m = 5; k = 2)
#                   1 0 0 0 0              1 2 3 4 5
#                   6 2 0 0 0              6 7 8 9 0
#                   9 7 3 0 0     ---->    9 8 7 0 0
#                   0 8 8 4 0              0 0 0 0 0
#                   0 0 7 9 5              0 0 0 0 0
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. lda must satisfy lda > k.
# @param[inout]
# x         device array of device pointer storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasStbmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const float *const* AP,int lda,float *const* x,int incx,int batchCount) nogil:
    global _hipblasStbmvBatched__funptr
    __init_symbol(&_hipblasStbmvBatched__funptr,"hipblasStbmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *const*,int,float *const*,int,int) nogil> _hipblasStbmvBatched__funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx,batchCount)


cdef void* _hipblasDtbmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtbmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const double *const* AP,int lda,double *const* x,int incx,int batchCount) nogil:
    global _hipblasDtbmvBatched__funptr
    __init_symbol(&_hipblasDtbmvBatched__funptr,"hipblasDtbmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *const*,int,double *const*,int,int) nogil> _hipblasDtbmvBatched__funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx,batchCount)


cdef void* _hipblasCtbmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtbmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasCtbmvBatched__funptr
    __init_symbol(&_hipblasCtbmvBatched__funptr,"hipblasCtbmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCtbmvBatched__funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx,batchCount)


cdef void* _hipblasZtbmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtbmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasZtbmvBatched__funptr
    __init_symbol(&_hipblasZtbmvBatched__funptr,"hipblasZtbmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZtbmvBatched__funptr)(handle,uplo,transA,diag,m,k,AP,lda,x,incx,batchCount)


cdef void* _hipblasStbmvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# tbmvStridedBatched performs one of the matrix-vector operations
# 
#     x_i := A_i*x_i      or
#     x_i := A_i**T*x_i   or
#     x_i := A_i**H*x_i,
# 
# where (A_i, x_i) is the i-th instance of the batch.
# x_i is a vector and A_i is an m by m matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: each A_i is an upper banded triangular matrix.
#           HIPBLAS_FILL_MODE_LOWER: each A_i is a  lower banded triangular matrix.
# @param[in]
# transA     [hipblasOperation_t]
#           indicates whether each matrix A_i is tranposed (conjugated) or not.
# @param[in]
# diag      [hipblasDiagType_t]
#           HIPBLAS_DIAG_UNIT: The main diagonal of each A_i is assumed to consist of only
#                                  1's and is not referenced.
#           HIPBLAS_DIAG_NON_UNIT: No assumptions are made of each A_i's main diagonal.
# @param[in]
# m         [int]
#           the number of rows and columns of the matrix represented by each A_i.
# @param[in]
# k         [int]
#           if uplo == HIPBLAS_FILL_MODE_UPPER, k specifies the number of super-diagonals
#           of each matrix A_i.
#           if uplo == HIPBLAS_FILL_MODE_LOWER, k specifies the number of sub-diagonals
#           of each matrix A_i.
#           k must satisfy k > 0 && k < lda.
# @param[in]
# AP         device array to the first matrix A_i of the batch. Stores each banded triangular matrix A_i.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The matrix represented is an upper banded triangular matrix
#             with the main diagonal and k super-diagonals, everything
#             else can be assumed to be 0.
#             The matrix is compacted so that the main diagonal resides on the k'th
#             row, the first super diagonal resides on the RHS of the k-1'th row, etc,
#             with the k'th diagonal on the RHS of the 0'th row.
#                Ex: (HIPBLAS_FILL_MODE_UPPER; m = 5; k = 2)
#                   1 6 9 0 0              0 0 9 8 7
#                   0 2 7 8 0              0 6 7 8 9
#                   0 0 3 8 7     ---->    1 2 3 4 5
#                   0 0 0 4 9              0 0 0 0 0
#                   0 0 0 0 5              0 0 0 0 0
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The matrix represnted is a lower banded triangular matrix
#             with the main diagonal and k sub-diagonals, everything else can be
#             assumed to be 0.
#             The matrix is compacted so that the main diagonal resides on the 0'th row,
#             working up to the k'th diagonal residing on the LHS of the k'th row.
#                Ex: (HIPBLAS_FILL_MODE_LOWER; m = 5; k = 2)
#                   1 0 0 0 0              1 2 3 4 5
#                   6 2 0 0 0              6 7 8 9 0
#                   9 7 3 0 0     ---->    9 8 7 0 0
#                   0 8 8 4 0              0 0 0 0 0
#                   0 0 7 9 5              0 0 0 0 0
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. lda must satisfy lda > k.
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one A_i matrix to the next A_(i + 1).
# @param[inout]
# x         device array to the first vector x_i of the batch.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one x_i matrix to the next x_(i + 1).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasStbmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const float * AP,int lda,long strideA,float * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasStbmvStridedBatched__funptr
    __init_symbol(&_hipblasStbmvStridedBatched__funptr,"hipblasStbmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,int,long,float *,int,long,int) nogil> _hipblasStbmvStridedBatched__funptr)(handle,uplo,transA,diag,m,k,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasDtbmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtbmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const double * AP,int lda,long strideA,double * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasDtbmvStridedBatched__funptr
    __init_symbol(&_hipblasDtbmvStridedBatched__funptr,"hipblasDtbmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,int,long,double *,int,long,int) nogil> _hipblasDtbmvStridedBatched__funptr)(handle,uplo,transA,diag,m,k,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasCtbmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtbmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasCtbmvStridedBatched__funptr
    __init_symbol(&_hipblasCtbmvStridedBatched__funptr,"hipblasCtbmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCtbmvStridedBatched__funptr)(handle,uplo,transA,diag,m,k,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasZtbmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtbmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasZtbmvStridedBatched__funptr
    __init_symbol(&_hipblasZtbmvStridedBatched__funptr,"hipblasZtbmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZtbmvStridedBatched__funptr)(handle,uplo,transA,diag,m,k,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasStbsv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# tbsv solves
# 
#      A*x = b or A**T*x = b or A**H*x = b,
# 
# where x and b are vectors and A is a banded triangular matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
#            HIPBLAS_OP_N: Solves A*x = b
#            HIPBLAS_OP_T: Solves A**T*x = b
#            HIPBLAS_OP_C: Solves A**H*x = b
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular (i.e. the diagonal elements
#                                    of A are not used in computations).
#         HIPBLAS_DIAG_NON_UNIT: A is not assumed to be unit triangular.
# 
# @param[in]
# n         [int]
#           n specifies the number of rows of b. n >= 0.
# @param[in]
# k         [int]
#           if(uplo == HIPBLAS_FILL_MODE_UPPER)
#             k specifies the number of super-diagonals of A.
#           if(uplo == HIPBLAS_FILL_MODE_LOWER)
#             k specifies the number of sub-diagonals of A.
#           k >= 0.
# 
# @param[in]
# AP         device pointer storing the matrix A in banded format.
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
#           lda >= (k + 1).
# 
# @param[inout]
# x         device pointer storing input vector b. Overwritten by the output vector x.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
#
cdef hipblasStatus_t hipblasStbsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const float * AP,int lda,float * x,int incx) nogil:
    global _hipblasStbsv__funptr
    __init_symbol(&_hipblasStbsv__funptr,"hipblasStbsv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,int,float *,int) nogil> _hipblasStbsv__funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx)


cdef void* _hipblasDtbsv__funptr = NULL
cdef hipblasStatus_t hipblasDtbsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const double * AP,int lda,double * x,int incx) nogil:
    global _hipblasDtbsv__funptr
    __init_symbol(&_hipblasDtbsv__funptr,"hipblasDtbsv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,int,double *,int) nogil> _hipblasDtbsv__funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx)


cdef void* _hipblasCtbsv__funptr = NULL
cdef hipblasStatus_t hipblasCtbsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil:
    global _hipblasCtbsv__funptr
    __init_symbol(&_hipblasCtbsv__funptr,"hipblasCtbsv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtbsv__funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx)


cdef void* _hipblasZtbsv__funptr = NULL
cdef hipblasStatus_t hipblasZtbsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil:
    global _hipblasZtbsv__funptr
    __init_symbol(&_hipblasZtbsv__funptr,"hipblasZtbsv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtbsv__funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx)


cdef void* _hipblasStbsvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# tbsvBatched solves
# 
#      A_i*x_i = b_i or A_i**T*x_i = b_i or A_i**H*x_i = b_i,
# 
# where x_i and b_i are vectors and A_i is a banded triangular matrix,
# for i = [1, batchCount].
# 
# The input vectors b_i are overwritten by the output vectors x_i.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
#            HIPBLAS_OP_N: Solves A_i*x_i = b_i
#            HIPBLAS_OP_T: Solves A_i**T*x_i = b_i
#            HIPBLAS_OP_C: Solves A_i**H*x_i = b_i
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
#                                    of each A_i are not used in computations).
#         HIPBLAS_DIAG_NON_UNIT: each A_i is not assumed to be unit triangular.
# 
# @param[in]
# n         [int]
#           n specifies the number of rows of each b_i. n >= 0.
# @param[in]
# k         [int]
#           if(uplo == HIPBLAS_FILL_MODE_UPPER)
#             k specifies the number of super-diagonals of each A_i.
#           if(uplo == HIPBLAS_FILL_MODE_LOWER)
#             k specifies the number of sub-diagonals of each A_i.
#           k >= 0.
# 
# @param[in]
# AP         device vector of device pointers storing each matrix A_i in banded format.
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
#           lda >= (k + 1).
# 
# @param[inout]
# x         device vector of device pointers storing each input vector b_i. Overwritten by each output
#           vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasStbsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const float *const* AP,int lda,float *const* x,int incx,int batchCount) nogil:
    global _hipblasStbsvBatched__funptr
    __init_symbol(&_hipblasStbsvBatched__funptr,"hipblasStbsvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *const*,int,float *const*,int,int) nogil> _hipblasStbsvBatched__funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx,batchCount)


cdef void* _hipblasDtbsvBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtbsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const double *const* AP,int lda,double *const* x,int incx,int batchCount) nogil:
    global _hipblasDtbsvBatched__funptr
    __init_symbol(&_hipblasDtbsvBatched__funptr,"hipblasDtbsvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *const*,int,double *const*,int,int) nogil> _hipblasDtbsvBatched__funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx,batchCount)


cdef void* _hipblasCtbsvBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtbsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasCtbsvBatched__funptr
    __init_symbol(&_hipblasCtbsvBatched__funptr,"hipblasCtbsvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCtbsvBatched__funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx,batchCount)


cdef void* _hipblasZtbsvBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtbsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasZtbsvBatched__funptr
    __init_symbol(&_hipblasZtbsvBatched__funptr,"hipblasZtbsvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZtbsvBatched__funptr)(handle,uplo,transA,diag,n,k,AP,lda,x,incx,batchCount)


cdef void* _hipblasStbsvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# tbsvStridedBatched solves
# 
#      A_i*x_i = b_i or A_i**T*x_i = b_i or A_i**H*x_i = b_i,
# 
# where x_i and b_i are vectors and A_i is a banded triangular matrix,
# for i = [1, batchCount].
# 
# The input vectors b_i are overwritten by the output vectors x_i.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
#            HIPBLAS_OP_N: Solves A_i*x_i = b_i
#            HIPBLAS_OP_T: Solves A_i**T*x_i = b_i
#            HIPBLAS_OP_C: Solves A_i**H*x_i = b_i
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
#                                    of each A_i are not used in computations).
#         HIPBLAS_DIAG_NON_UNIT: each A_i is not assumed to be unit triangular.
# 
# @param[in]
# n         [int]
#           n specifies the number of rows of each b_i. n >= 0.
# @param[in]
# k         [int]
#           if(uplo == HIPBLAS_FILL_MODE_UPPER)
#             k specifies the number of super-diagonals of each A_i.
#           if(uplo == HIPBLAS_FILL_MODE_LOWER)
#             k specifies the number of sub-diagonals of each A_i.
#           k >= 0.
# 
# @param[in]
# AP         device pointer pointing to the first banded matrix A_1.
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
#           lda >= (k + 1).
# @param[in]
# strideA  [hipblasStride]
#           specifies the distance between the start of one matrix (A_i) and the next (A_i+1).
# 
# @param[inout]
# x         device pointer pointing to the first input vector b_1. Overwritten by output vectors x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           specifies the distance between the start of one vector (x_i) and the next (x_i+1).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasStbsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const float * AP,int lda,long strideA,float * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasStbsvStridedBatched__funptr
    __init_symbol(&_hipblasStbsvStridedBatched__funptr,"hipblasStbsvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,int,long,float *,int,long,int) nogil> _hipblasStbsvStridedBatched__funptr)(handle,uplo,transA,diag,n,k,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasDtbsvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtbsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const double * AP,int lda,long strideA,double * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasDtbsvStridedBatched__funptr
    __init_symbol(&_hipblasDtbsvStridedBatched__funptr,"hipblasDtbsvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,int,long,double *,int,long,int) nogil> _hipblasDtbsvStridedBatched__funptr)(handle,uplo,transA,diag,n,k,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasCtbsvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtbsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasCtbsvStridedBatched__funptr
    __init_symbol(&_hipblasCtbsvStridedBatched__funptr,"hipblasCtbsvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCtbsvStridedBatched__funptr)(handle,uplo,transA,diag,n,k,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasZtbsvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtbsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasZtbsvStridedBatched__funptr
    __init_symbol(&_hipblasZtbsvStridedBatched__funptr,"hipblasZtbsvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZtbsvStridedBatched__funptr)(handle,uplo,transA,diag,n,k,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasStpmv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# tpmv performs one of the matrix-vector operations
# 
#      x = A*x or x = A**T*x,
# 
# where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix, supplied in the pack form.
# 
# The vector x is overwritten.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of A. m >= 0.
# 
# @param[in]
# AP       device pointer storing matrix A,
#         of dimension at leat ( m * ( m + 1 ) / 2 ).
#       Before entry with uplo = HIPBLAS_FILL_MODE_UPPER, the array A
#       must contain the upper triangular matrix packed sequentially,
#       column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
#       a_{0,1} and a_{1, 1} respectively, and so on.
#       Before entry with uplo = HIPBLAS_FILL_MODE_LOWER, the array A
#       must contain the lower triangular matrix packed sequentially,
#       column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
#       a_{1,0} and a_{2,0} respectively, and so on.
#       Note that when DIAG = HIPBLAS_DIAG_UNIT, the diagonal elements of A are
#       not referenced, but are assumed to be unity.
# 
# @param[in]
# x       device pointer storing vector x.
# 
# @param[in]
# incx    [int]
#         specifies the increment for the elements of x. incx must not be zero.
#
cdef hipblasStatus_t hipblasStpmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,float * x,int incx) nogil:
    global _hipblasStpmv__funptr
    __init_symbol(&_hipblasStpmv__funptr,"hipblasStpmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,float *,int) nogil> _hipblasStpmv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasDtpmv__funptr = NULL
cdef hipblasStatus_t hipblasDtpmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,double * x,int incx) nogil:
    global _hipblasDtpmv__funptr
    __init_symbol(&_hipblasDtpmv__funptr,"hipblasDtpmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,double *,int) nogil> _hipblasDtpmv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasCtpmv__funptr = NULL
cdef hipblasStatus_t hipblasCtpmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,hipblasComplex * x,int incx) nogil:
    global _hipblasCtpmv__funptr
    __init_symbol(&_hipblasCtpmv__funptr,"hipblasCtpmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCtpmv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasZtpmv__funptr = NULL
cdef hipblasStatus_t hipblasZtpmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx) nogil:
    global _hipblasZtpmv__funptr
    __init_symbol(&_hipblasZtpmv__funptr,"hipblasZtpmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZtpmv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasStpmvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# tpmvBatched performs one of the matrix-vector operations
# 
#      x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batchCount
# 
# where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
# 
# The vectors x_i are overwritten.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of matrices A_i. m >= 0.
# 
# @param[in]
# AP         device pointer storing pointer of matrices A_i,
#           of dimension ( lda, m )
# 
# @param[in]
# x         device pointer storing vectors x_i.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of vectors x_i.
# 
# @param[in]
# batchCount [int]
#           The number of batched matrices/vectors.
# 
#
cdef hipblasStatus_t hipblasStpmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float *const* AP,float *const* x,int incx,int batchCount) nogil:
    global _hipblasStpmvBatched__funptr
    __init_symbol(&_hipblasStpmvBatched__funptr,"hipblasStpmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *const*,float *const*,int,int) nogil> _hipblasStpmvBatched__funptr)(handle,uplo,transA,diag,m,AP,x,incx,batchCount)


cdef void* _hipblasDtpmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtpmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double *const* AP,double *const* x,int incx,int batchCount) nogil:
    global _hipblasDtpmvBatched__funptr
    __init_symbol(&_hipblasDtpmvBatched__funptr,"hipblasDtpmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *const*,double *const*,int,int) nogil> _hipblasDtpmvBatched__funptr)(handle,uplo,transA,diag,m,AP,x,incx,batchCount)


cdef void* _hipblasCtpmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtpmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex *const* AP,hipblasComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasCtpmvBatched__funptr
    __init_symbol(&_hipblasCtpmvBatched__funptr,"hipblasCtpmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *const*,hipblasComplex *const*,int,int) nogil> _hipblasCtpmvBatched__funptr)(handle,uplo,transA,diag,m,AP,x,incx,batchCount)


cdef void* _hipblasZtpmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtpmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex *const* AP,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasZtpmvBatched__funptr
    __init_symbol(&_hipblasZtpmvBatched__funptr,"hipblasZtpmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *const*,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZtpmvBatched__funptr)(handle,uplo,transA,diag,m,AP,x,incx,batchCount)


cdef void* _hipblasStpmvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# tpmvStridedBatched performs one of the matrix-vector operations
# 
#      x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batchCount
# 
# where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
# with strides specifying how to retrieve $x_i$ (resp. $A_i$) from $x_{i-1}$ (resp. $A_i$).
# 
# The vectors x_i are overwritten.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of matrices A_i. m >= 0.
# 
# @param[in]
# AP         device pointer of the matrix A_0,
#           of dimension ( lda, m )
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one A_i matrix to the next A_{i + 1}
# 
# @param[in]
# x         device pointer storing the vector x_0.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of one vector x.
# 
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one x_i vector to the next x_{i + 1}
# 
# @param[in]
# batchCount [int]
#           The number of batched matrices/vectors.
# 
#
cdef hipblasStatus_t hipblasStpmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,long strideA,float * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasStpmvStridedBatched__funptr
    __init_symbol(&_hipblasStpmvStridedBatched__funptr,"hipblasStpmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,long,float *,int,long,int) nogil> _hipblasStpmvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasDtpmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtpmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,long strideA,double * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasDtpmvStridedBatched__funptr
    __init_symbol(&_hipblasDtpmvStridedBatched__funptr,"hipblasDtpmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,long,double *,int,long,int) nogil> _hipblasDtpmvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasCtpmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtpmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,long strideA,hipblasComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasCtpmvStridedBatched__funptr
    __init_symbol(&_hipblasCtpmvStridedBatched__funptr,"hipblasCtpmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,long,hipblasComplex *,int,long,int) nogil> _hipblasCtpmvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasZtpmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtpmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,long strideA,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasZtpmvStridedBatched__funptr
    __init_symbol(&_hipblasZtpmvStridedBatched__funptr,"hipblasZtpmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZtpmvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasStpsv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# tpsv solves
# 
#      A*x = b or A**T*x = b, or A**H*x = b,
# 
# where x and b are vectors and A is a triangular matrix stored in the packed format.
# 
# The input vector b is overwritten by the output vector x.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: Solves A*x = b
#         HIPBLAS_OP_T: Solves A**T*x = b
#         HIPBLAS_OP_C: Solves A**H*x = b
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular (i.e. the diagonal elements
#                                    of A are not used in computations).
#         HIPBLAS_DIAG_NON_UNIT: A is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of b. m >= 0.
# 
# @param[in]
# AP        device pointer storing the packed version of matrix A,
#           of dimension >= (n * (n + 1) / 2)
# 
# @param[inout]
# x         device pointer storing vector b on input, overwritten by x on output.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
#
cdef hipblasStatus_t hipblasStpsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,float * x,int incx) nogil:
    global _hipblasStpsv__funptr
    __init_symbol(&_hipblasStpsv__funptr,"hipblasStpsv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,float *,int) nogil> _hipblasStpsv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasDtpsv__funptr = NULL
cdef hipblasStatus_t hipblasDtpsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,double * x,int incx) nogil:
    global _hipblasDtpsv__funptr
    __init_symbol(&_hipblasDtpsv__funptr,"hipblasDtpsv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,double *,int) nogil> _hipblasDtpsv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasCtpsv__funptr = NULL
cdef hipblasStatus_t hipblasCtpsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,hipblasComplex * x,int incx) nogil:
    global _hipblasCtpsv__funptr
    __init_symbol(&_hipblasCtpsv__funptr,"hipblasCtpsv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCtpsv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasZtpsv__funptr = NULL
cdef hipblasStatus_t hipblasZtpsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx) nogil:
    global _hipblasZtpsv__funptr
    __init_symbol(&_hipblasZtpsv__funptr,"hipblasZtpsv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZtpsv__funptr)(handle,uplo,transA,diag,m,AP,x,incx)


cdef void* _hipblasStpsvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# tpsvBatched solves
# 
#      A_i*x_i = b_i or A_i**T*x_i = b_i, or A_i**H*x_i = b_i,
# 
# where x_i and b_i are vectors and A_i is a triangular matrix stored in the packed format,
# for i in [1, batchCount].
# 
# The input vectors b_i are overwritten by the output vectors x_i.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  each A_i is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: Solves A*x = b
#         HIPBLAS_OP_T: Solves A**T*x = b
#         HIPBLAS_OP_C: Solves A**H*x = b
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
#                                    of each A_i are not used in computations).
#         HIPBLAS_DIAG_NON_UNIT: each A_i is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of each b_i. m >= 0.
# 
# @param[in]
# AP        device array of device pointers storing the packed versions of each matrix A_i,
#           of dimension >= (n * (n + 1) / 2)
# 
# @param[inout]
# x         device array of device pointers storing each input vector b_i, overwritten by x_i on output.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# batchCount [int]
#             specifies the number of instances in the batch.
#
cdef hipblasStatus_t hipblasStpsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float *const* AP,float *const* x,int incx,int batchCount) nogil:
    global _hipblasStpsvBatched__funptr
    __init_symbol(&_hipblasStpsvBatched__funptr,"hipblasStpsvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *const*,float *const*,int,int) nogil> _hipblasStpsvBatched__funptr)(handle,uplo,transA,diag,m,AP,x,incx,batchCount)


cdef void* _hipblasDtpsvBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtpsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double *const* AP,double *const* x,int incx,int batchCount) nogil:
    global _hipblasDtpsvBatched__funptr
    __init_symbol(&_hipblasDtpsvBatched__funptr,"hipblasDtpsvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *const*,double *const*,int,int) nogil> _hipblasDtpsvBatched__funptr)(handle,uplo,transA,diag,m,AP,x,incx,batchCount)


cdef void* _hipblasCtpsvBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtpsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex *const* AP,hipblasComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasCtpsvBatched__funptr
    __init_symbol(&_hipblasCtpsvBatched__funptr,"hipblasCtpsvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *const*,hipblasComplex *const*,int,int) nogil> _hipblasCtpsvBatched__funptr)(handle,uplo,transA,diag,m,AP,x,incx,batchCount)


cdef void* _hipblasZtpsvBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtpsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex *const* AP,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasZtpsvBatched__funptr
    __init_symbol(&_hipblasZtpsvBatched__funptr,"hipblasZtpsvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *const*,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZtpsvBatched__funptr)(handle,uplo,transA,diag,m,AP,x,incx,batchCount)


cdef void* _hipblasStpsvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# tpsvStridedBatched solves
# 
#      A_i*x_i = b_i or A_i**T*x_i = b_i, or A_i**H*x_i = b_i,
# 
# where x_i and b_i are vectors and A_i is a triangular matrix stored in the packed format,
# for i in [1, batchCount].
# 
# The input vectors b_i are overwritten by the output vectors x_i.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  each A_i is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: Solves A*x = b
#         HIPBLAS_OP_T: Solves A**T*x = b
#         HIPBLAS_OP_C: Solves A**H*x = b
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
#                                    of each A_i are not used in computations).
#         HIPBLAS_DIAG_NON_UNIT: each A_i is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of each b_i. m >= 0.
# 
# @param[in]
# AP        device pointer pointing to the first packed matrix A_1,
#           of dimension >= (n * (n + 1) / 2)
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the beginning of one packed matrix (AP_i) and the next (AP_i+1).
# 
# @param[inout]
# x         device pointer pointing to the first input vector b_1. Overwritten by each x_i on output.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the beginning of one vector (x_i) and the next (x_i+1).
# @param[in]
# batchCount [int]
#             specifies the number of instances in the batch.
#
cdef hipblasStatus_t hipblasStpsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,long strideA,float * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasStpsvStridedBatched__funptr
    __init_symbol(&_hipblasStpsvStridedBatched__funptr,"hipblasStpsvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,long,float *,int,long,int) nogil> _hipblasStpsvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasDtpsvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtpsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,long strideA,double * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasDtpsvStridedBatched__funptr
    __init_symbol(&_hipblasDtpsvStridedBatched__funptr,"hipblasDtpsvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,long,double *,int,long,int) nogil> _hipblasDtpsvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasCtpsvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtpsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,long strideA,hipblasComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasCtpsvStridedBatched__funptr
    __init_symbol(&_hipblasCtpsvStridedBatched__funptr,"hipblasCtpsvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,long,hipblasComplex *,int,long,int) nogil> _hipblasCtpsvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasZtpsvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtpsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,long strideA,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasZtpsvStridedBatched__funptr
    __init_symbol(&_hipblasZtpsvStridedBatched__funptr,"hipblasZtpsvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZtpsvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasStrmv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# trmv performs one of the matrix-vector operations
# 
#      x = A*x or x = A**T*x,
# 
# where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix.
# 
# The vector x is overwritten.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of A. m >= 0.
# 
# @param[in]
# AP        device pointer storing matrix A,
#           of dimension ( lda, m )
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
#           lda = max( 1, m ).
# 
# @param[in]
# x         device pointer storing vector x.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
#
cdef hipblasStatus_t hipblasStrmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,int lda,float * x,int incx) nogil:
    global _hipblasStrmv__funptr
    __init_symbol(&_hipblasStrmv__funptr,"hipblasStrmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,int,float *,int) nogil> _hipblasStrmv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasDtrmv__funptr = NULL
cdef hipblasStatus_t hipblasDtrmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,double * x,int incx) nogil:
    global _hipblasDtrmv__funptr
    __init_symbol(&_hipblasDtrmv__funptr,"hipblasDtrmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,int,double *,int) nogil> _hipblasDtrmv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasCtrmv__funptr = NULL
cdef hipblasStatus_t hipblasCtrmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil:
    global _hipblasCtrmv__funptr
    __init_symbol(&_hipblasCtrmv__funptr,"hipblasCtrmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtrmv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasZtrmv__funptr = NULL
cdef hipblasStatus_t hipblasZtrmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil:
    global _hipblasZtrmv__funptr
    __init_symbol(&_hipblasZtrmv__funptr,"hipblasZtrmv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtrmv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasStrmvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# trmvBatched performs one of the matrix-vector operations
# 
#      x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batchCount
# 
# where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
# 
# The vectors x_i are overwritten.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of matrices A_i. m >= 0.
# 
# @param[in]
# AP        device pointer storing pointer of matrices A_i,
#           of dimension ( lda, m )
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of A_i.
#           lda >= max( 1, m ).
# 
# @param[in]
# x         device pointer storing vectors x_i.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of vectors x_i.
# 
# @param[in]
# batchCount [int]
#           The number of batched matrices/vectors.
# 
#
cdef hipblasStatus_t hipblasStrmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float *const* AP,int lda,float *const* x,int incx,int batchCount) nogil:
    global _hipblasStrmvBatched__funptr
    __init_symbol(&_hipblasStrmvBatched__funptr,"hipblasStrmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *const*,int,float *const*,int,int) nogil> _hipblasStrmvBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx,batchCount)


cdef void* _hipblasDtrmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtrmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double *const* AP,int lda,double *const* x,int incx,int batchCount) nogil:
    global _hipblasDtrmvBatched__funptr
    __init_symbol(&_hipblasDtrmvBatched__funptr,"hipblasDtrmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *const*,int,double *const*,int,int) nogil> _hipblasDtrmvBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx,batchCount)


cdef void* _hipblasCtrmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtrmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasCtrmvBatched__funptr
    __init_symbol(&_hipblasCtrmvBatched__funptr,"hipblasCtrmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCtrmvBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx,batchCount)


cdef void* _hipblasZtrmvBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtrmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasZtrmvBatched__funptr
    __init_symbol(&_hipblasZtrmvBatched__funptr,"hipblasZtrmvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZtrmvBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx,batchCount)


cdef void* _hipblasStrmvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# trmvStridedBatched performs one of the matrix-vector operations
# 
#      x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batchCount
# 
# where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
# with strides specifying how to retrieve $x_i$ (resp. $A_i$) from $x_{i-1}$ (resp. $A_i$).
# 
# The vectors x_i are overwritten.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of matrices A_i. m >= 0.
# 
# @param[in]
# AP        device pointer of the matrix A_0,
#           of dimension ( lda, m )
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of A_i.
#           lda >= max( 1, m ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one A_i matrix to the next A_{i + 1}
# 
# @param[in]
# x         device pointer storing the vector x_0.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of one vector x.
# 
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one x_i vector to the next x_{i + 1}
# 
# @param[in]
# batchCount [int]
#           The number of batched matrices/vectors.
# 
#
cdef hipblasStatus_t hipblasStrmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,int lda,long strideA,float * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasStrmvStridedBatched__funptr
    __init_symbol(&_hipblasStrmvStridedBatched__funptr,"hipblasStrmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,int,long,float *,int,long,int) nogil> _hipblasStrmvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasDtrmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtrmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,long strideA,double * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasDtrmvStridedBatched__funptr
    __init_symbol(&_hipblasDtrmvStridedBatched__funptr,"hipblasDtrmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,int,long,double *,int,long,int) nogil> _hipblasDtrmvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasCtrmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtrmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasCtrmvStridedBatched__funptr
    __init_symbol(&_hipblasCtrmvStridedBatched__funptr,"hipblasCtrmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCtrmvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasZtrmvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtrmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasZtrmvStridedBatched__funptr
    __init_symbol(&_hipblasZtrmvStridedBatched__funptr,"hipblasZtrmvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZtrmvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasStrsv__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# trsv solves
# 
#      A*x = b or A**T*x = b,
# 
# where x and b are vectors and A is a triangular matrix.
# 
# The vector x is overwritten on b.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of b. m >= 0.
# 
# @param[in]
# AP        device pointer storing matrix A,
#           of dimension ( lda, m )
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
#           lda = max( 1, m ).
# 
# @param[in]
# x         device pointer storing vector x.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
#
cdef hipblasStatus_t hipblasStrsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,int lda,float * x,int incx) nogil:
    global _hipblasStrsv__funptr
    __init_symbol(&_hipblasStrsv__funptr,"hipblasStrsv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,int,float *,int) nogil> _hipblasStrsv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasDtrsv__funptr = NULL
cdef hipblasStatus_t hipblasDtrsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,double * x,int incx) nogil:
    global _hipblasDtrsv__funptr
    __init_symbol(&_hipblasDtrsv__funptr,"hipblasDtrsv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,int,double *,int) nogil> _hipblasDtrsv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasCtrsv__funptr = NULL
cdef hipblasStatus_t hipblasCtrsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil:
    global _hipblasCtrsv__funptr
    __init_symbol(&_hipblasCtrsv__funptr,"hipblasCtrsv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtrsv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasZtrsv__funptr = NULL
cdef hipblasStatus_t hipblasZtrsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil:
    global _hipblasZtrsv__funptr
    __init_symbol(&_hipblasZtrsv__funptr,"hipblasZtrsv")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtrsv__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx)


cdef void* _hipblasStrsvBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# trsvBatched solves
# 
#      A_i*x_i = b_i or A_i**T*x_i = b_i,
# 
# where (A_i, x_i, b_i) is the i-th instance of the batch.
# x_i and b_i are vectors and A_i is an
# m by m triangular matrix.
# 
# The vector x is overwritten on b.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of b. m >= 0.
# 
# @param[in]
# AP         device array of device pointers storing each matrix A_i.
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
#           lda = max(1, m)
# 
# @param[in]
# x         device array of device pointers storing each vector x_i.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasStrsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float *const* AP,int lda,float *const* x,int incx,int batchCount) nogil:
    global _hipblasStrsvBatched__funptr
    __init_symbol(&_hipblasStrsvBatched__funptr,"hipblasStrsvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *const*,int,float *const*,int,int) nogil> _hipblasStrsvBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx,batchCount)


cdef void* _hipblasDtrsvBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtrsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double *const* AP,int lda,double *const* x,int incx,int batchCount) nogil:
    global _hipblasDtrsvBatched__funptr
    __init_symbol(&_hipblasDtrsvBatched__funptr,"hipblasDtrsvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *const*,int,double *const*,int,int) nogil> _hipblasDtrsvBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx,batchCount)


cdef void* _hipblasCtrsvBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtrsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasCtrsvBatched__funptr
    __init_symbol(&_hipblasCtrsvBatched__funptr,"hipblasCtrsvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCtrsvBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx,batchCount)


cdef void* _hipblasZtrsvBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtrsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil:
    global _hipblasZtrsvBatched__funptr
    __init_symbol(&_hipblasZtrsvBatched__funptr,"hipblasZtrsvBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZtrsvBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,x,incx,batchCount)


cdef void* _hipblasStrsvStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 2 API
# 
# \details
# trsvStridedBatched solves
# 
#      A_i*x_i = b_i or A_i**T*x_i = b_i,
# 
# where (A_i, x_i, b_i) is the i-th instance of the batch.
# x_i and b_i are vectors and A_i is an m by m triangular matrix, for i = 1, ..., batchCount.
# 
# The vector x is overwritten on b.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of each b_i. m >= 0.
# 
# @param[in]
# AP         device pointer to the first matrix (A_1) in the batch, of dimension ( lda, m )
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one A_i matrix to the next A_(i + 1)
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
#           lda = max( 1, m ).
# 
# @param[in, out]
# x         device pointer to the first vector (x_1) in the batch.
# 
# @param[in]
# stridex [hipblasStride]
#          stride from the start of one x_i vector to the next x_(i + 1)
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasStrsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,int lda,long strideA,float * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasStrsvStridedBatched__funptr
    __init_symbol(&_hipblasStrsvStridedBatched__funptr,"hipblasStrsvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const float *,int,long,float *,int,long,int) nogil> _hipblasStrsvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasDtrsvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtrsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,long strideA,double * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasDtrsvStridedBatched__funptr
    __init_symbol(&_hipblasDtrsvStridedBatched__funptr,"hipblasDtrsvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,const double *,int,long,double *,int,long,int) nogil> _hipblasDtrsvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasCtrsvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtrsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasCtrsvStridedBatched__funptr
    __init_symbol(&_hipblasCtrsvStridedBatched__funptr,"hipblasCtrsvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCtrsvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasZtrsvStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtrsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil:
    global _hipblasZtrsvStridedBatched__funptr
    __init_symbol(&_hipblasZtrsvStridedBatched__funptr,"hipblasZtrsvStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZtrsvStridedBatched__funptr)(handle,uplo,transA,diag,m,AP,lda,strideA,x,incx,stridex,batchCount)


cdef void* _hipblasHgemm__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# gemm performs one of the matrix-matrix operations
# 
#     C = alpha*op( A )*op( B ) + beta*C,
# 
# where op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
# 
# alpha and beta are scalars, and A, B and C are matrices, with
# op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
# 
# - Supported precisions in rocBLAS : h,s,d,c,z
# - Supported precisions in cuBLAS  : h,s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
# 
#           .
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A )
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B )
# @param[in]
# m         [int]
#           number or rows of matrices op( A ) and C
# @param[in]
# n         [int]
#           number of columns of matrices op( B ) and C
# @param[in]
# k         [int]
#           number of columns of matrix op( A ) and number of rows of matrix op( B )
# @param[in]
# alpha     device pointer or host pointer specifying the scalar alpha.
# @param[in]
# AP         device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# BP         device pointer storing matrix B.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of B.
# @param[in]
# beta      device pointer or host pointer specifying the scalar beta.
# @param[in, out]
# CP         device pointer storing matrix C on the GPU.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
#
cdef hipblasStatus_t hipblasHgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const unsigned short * alpha,const unsigned short * AP,int lda,const unsigned short * BP,int ldb,const unsigned short * beta,unsigned short * CP,int ldc) nogil:
    global _hipblasHgemm__funptr
    __init_symbol(&_hipblasHgemm__funptr,"hipblasHgemm")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,const unsigned short *,const unsigned short *,int,const unsigned short *,int,const unsigned short *,unsigned short *,int) nogil> _hipblasHgemm__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasSgemm__funptr = NULL
cdef hipblasStatus_t hipblasSgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil:
    global _hipblasSgemm__funptr
    __init_symbol(&_hipblasSgemm__funptr,"hipblasSgemm")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSgemm__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasDgemm__funptr = NULL
cdef hipblasStatus_t hipblasDgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil:
    global _hipblasDgemm__funptr
    __init_symbol(&_hipblasDgemm__funptr,"hipblasDgemm")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDgemm__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasCgemm__funptr = NULL
cdef hipblasStatus_t hipblasCgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _hipblasCgemm__funptr
    __init_symbol(&_hipblasCgemm__funptr,"hipblasCgemm")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCgemm__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZgemm__funptr = NULL
cdef hipblasStatus_t hipblasZgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _hipblasZgemm__funptr
    __init_symbol(&_hipblasZgemm__funptr,"hipblasZgemm")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZgemm__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasHgemmBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
#  \details
# gemmBatched performs one of the batched matrix-matrix operations
#      C_i = alpha*op( A_i )*op( B_i ) + beta*C_i, for i = 1, ..., batchCount.
#  where op( X ) is one of
#      op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
#  alpha and beta are scalars, and A, B and C are strided batched matrices, with
# op( A ) an m by k by batchCount strided_batched matrix,
# op( B ) an k by n by batchCount strided_batched matrix and
# C an m by n by batchCount strided_batched matrix.
# 
# - Supported precisions in rocBLAS : h,s,d,c,z
# - Supported precisions in cuBLAS  : h,s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A )
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B )
# @param[in]
# m         [int]
#           matrix dimention m.
# @param[in]
# n         [int]
#           matrix dimention n.
# @param[in]
# k         [int]
#           matrix dimention k.
# @param[in]
# alpha     device pointer or host pointer specifying the scalar alpha.
# @param[in]
# AP         device array of device pointers storing each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# BP         device array of device pointers storing each matrix B_i.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of each B_i.
# @param[in]
# beta      device pointer or host pointer specifying the scalar beta.
# @param[in, out]
# CP         device array of device pointers storing each matrix C_i.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of each C_i.
# @param[in]
# batchCount
#           [int]
#           number of gemm operations in the batch
cdef hipblasStatus_t hipblasHgemmBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const unsigned short * alpha,const unsigned short *const* AP,int lda,const unsigned short *const* BP,int ldb,const unsigned short * beta,unsigned short *const* CP,int ldc,int batchCount) nogil:
    global _hipblasHgemmBatched__funptr
    __init_symbol(&_hipblasHgemmBatched__funptr,"hipblasHgemmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,const unsigned short *,const unsigned short *const*,int,const unsigned short *const*,int,const unsigned short *,unsigned short *const*,int,int) nogil> _hipblasHgemmBatched__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasSgemmBatched__funptr = NULL
cdef hipblasStatus_t hipblasSgemmBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const float * alpha,const float *const* AP,int lda,const float *const* BP,int ldb,const float * beta,float *const* CP,int ldc,int batchCount) nogil:
    global _hipblasSgemmBatched__funptr
    __init_symbol(&_hipblasSgemmBatched__funptr,"hipblasSgemmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,const float *,const float *const*,int,const float *const*,int,const float *,float *const*,int,int) nogil> _hipblasSgemmBatched__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasDgemmBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgemmBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const double * alpha,const double *const* AP,int lda,const double *const* BP,int ldb,const double * beta,double *const* CP,int ldc,int batchCount) nogil:
    global _hipblasDgemmBatched__funptr
    __init_symbol(&_hipblasDgemmBatched__funptr,"hipblasDgemmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,const double *,const double *const*,int,const double *const*,int,const double *,double *const*,int,int) nogil> _hipblasDgemmBatched__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasCgemmBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgemmBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,hipblasComplex * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasCgemmBatched__funptr
    __init_symbol(&_hipblasCgemmBatched__funptr,"hipblasCgemmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *,hipblasComplex *const*,int,int) nogil> _hipblasCgemmBatched__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasZgemmBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgemmBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasZgemmBatched__funptr
    __init_symbol(&_hipblasZgemmBatched__funptr,"hipblasZgemmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZgemmBatched__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasHgemmStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# gemmStridedBatched performs one of the strided batched matrix-matrix operations
# 
#     C_i = alpha*op( A_i )*op( B_i ) + beta*C_i, for i = 1, ..., batchCount.
# 
# where op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
# 
# alpha and beta are scalars, and A, B and C are strided batched matrices, with
# op( A ) an m by k by batchCount strided_batched matrix,
# op( B ) an k by n by batchCount strided_batched matrix and
# C an m by n by batchCount strided_batched matrix.
# 
# - Supported precisions in rocBLAS : h,s,d,c,z
# - Supported precisions in cuBLAS  : h,s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A )
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B )
# @param[in]
# m         [int]
#           matrix dimention m.
# @param[in]
# n         [int]
#           matrix dimention n.
# @param[in]
# k         [int]
#           matrix dimention k.
# @param[in]
# alpha     device pointer or host pointer specifying the scalar alpha.
# @param[in]
# AP         device pointer pointing to the first matrix A_1.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one A_i matrix to the next A_(i + 1).
# @param[in]
# BP         device pointer pointing to the first matrix B_1.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of each B_i.
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one B_i matrix to the next B_(i + 1).
# @param[in]
# beta      device pointer or host pointer specifying the scalar beta.
# @param[in, out]
# CP         device pointer pointing to the first matrix C_1.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of each C_i.
# @param[in]
# strideC  [hipblasStride]
#           stride from the start of one C_i matrix to the next C_(i + 1).
# @param[in]
# batchCount
#           [int]
#           number of gemm operatons in the batch
#
cdef hipblasStatus_t hipblasHgemmStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const unsigned short * alpha,const unsigned short * AP,int lda,long long strideA,const unsigned short * BP,int ldb,long long strideB,const unsigned short * beta,unsigned short * CP,int ldc,long long strideC,int batchCount) nogil:
    global _hipblasHgemmStridedBatched__funptr
    __init_symbol(&_hipblasHgemmStridedBatched__funptr,"hipblasHgemmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,const unsigned short *,const unsigned short *,int,long long,const unsigned short *,int,long long,const unsigned short *,unsigned short *,int,long long,int) nogil> _hipblasHgemmStridedBatched__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasSgemmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasSgemmStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const float * alpha,const float * AP,int lda,long long strideA,const float * BP,int ldb,long long strideB,const float * beta,float * CP,int ldc,long long strideC,int batchCount) nogil:
    global _hipblasSgemmStridedBatched__funptr
    __init_symbol(&_hipblasSgemmStridedBatched__funptr,"hipblasSgemmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,const float *,const float *,int,long long,const float *,int,long long,const float *,float *,int,long long,int) nogil> _hipblasSgemmStridedBatched__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasDgemmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgemmStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const double * alpha,const double * AP,int lda,long long strideA,const double * BP,int ldb,long long strideB,const double * beta,double * CP,int ldc,long long strideC,int batchCount) nogil:
    global _hipblasDgemmStridedBatched__funptr
    __init_symbol(&_hipblasDgemmStridedBatched__funptr,"hipblasDgemmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,const double *,const double *,int,long long,const double *,int,long long,const double *,double *,int,long long,int) nogil> _hipblasDgemmStridedBatched__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasCgemmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgemmStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long long strideA,hipblasComplex * BP,int ldb,long long strideB,hipblasComplex * beta,hipblasComplex * CP,int ldc,long long strideC,int batchCount) nogil:
    global _hipblasCgemmStridedBatched__funptr
    __init_symbol(&_hipblasCgemmStridedBatched__funptr,"hipblasCgemmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,hipblasComplex *,hipblasComplex *,int,long long,hipblasComplex *,int,long long,hipblasComplex *,hipblasComplex *,int,long long,int) nogil> _hipblasCgemmStridedBatched__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasZgemmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgemmStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long long strideA,hipblasDoubleComplex * BP,int ldb,long long strideB,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc,long long strideC,int batchCount) nogil:
    global _hipblasZgemmStridedBatched__funptr
    __init_symbol(&_hipblasZgemmStridedBatched__funptr,"hipblasZgemmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long long,hipblasDoubleComplex *,int,long long,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long long,int) nogil> _hipblasZgemmStridedBatched__funptr)(handle,transA,transB,m,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasCherk__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# herk performs one of the matrix-matrix operations for a Hermitian rank-k update
# 
# C := alpha*op( A )*op( A )^H + beta*C
# 
# where  alpha and beta are scalars, op(A) is an n by k matrix, and
# C is a n x n Hermitian matrix stored as either upper or lower.
# 
#     op( A ) = A,  and A is n by k if transA == HIPBLAS_OP_N
#     op( A ) = A^H and A is k by n if transA == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C:  op(A) = A^H
#         HIPBLAS_ON_N:  op(A) = A
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         Martrix dimension is ( lda, k ) when if transA = HIPBLAS_OP_N, otherwise (lda, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
#
cdef hipblasStatus_t hipblasCherk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,hipblasComplex * AP,int lda,const float * beta,hipblasComplex * CP,int ldc) nogil:
    global _hipblasCherk__funptr
    __init_symbol(&_hipblasCherk__funptr,"hipblasCherk")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,hipblasComplex *,int,const float *,hipblasComplex *,int) nogil> _hipblasCherk__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* _hipblasZherk__funptr = NULL
cdef hipblasStatus_t hipblasZherk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,hipblasDoubleComplex * AP,int lda,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _hipblasZherk__funptr
    __init_symbol(&_hipblasZherk__funptr,"hipblasZherk")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,hipblasDoubleComplex *,int,const double *,hipblasDoubleComplex *,int) nogil> _hipblasZherk__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* _hipblasCherkBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# herkBatched performs a batch of the matrix-matrix operations for a Hermitian rank-k update
# 
# C_i := alpha*op( A_i )*op( A_i )^H + beta*C_i
# 
# where  alpha and beta are scalars, op(A) is an n by k matrix, and
# C_i is a n x n Hermitian matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, and A_i is n by k if transA == HIPBLAS_OP_N
#     op( A_i ) = A_i^H and A_i is k by n if transA == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C: op(A) = A^H
#         HIPBLAS_OP_N: op(A) = A
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       device array of device pointers storing each matrix_i A of dimension (lda, k)
#         when transA is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasCherkBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,hipblasComplex *const* AP,int lda,const float * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasCherkBatched__funptr
    __init_symbol(&_hipblasCherkBatched__funptr,"hipblasCherkBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,hipblasComplex *const*,int,const float *,hipblasComplex *const*,int,int) nogil> _hipblasCherkBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc,batchCount)


cdef void* _hipblasZherkBatched__funptr = NULL
cdef hipblasStatus_t hipblasZherkBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,hipblasDoubleComplex *const* AP,int lda,const double * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasZherkBatched__funptr
    __init_symbol(&_hipblasZherkBatched__funptr,"hipblasZherkBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,hipblasDoubleComplex *const*,int,const double *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZherkBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc,batchCount)


cdef void* _hipblasCherkStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# herkStridedBatched performs a batch of the matrix-matrix operations for a Hermitian rank-k update
# 
# C_i := alpha*op( A_i )*op( A_i )^H + beta*C_i
# 
# where  alpha and beta are scalars, op(A) is an n by k matrix, and
# C_i is a n x n Hermitian matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, and A_i is n by k if transA == HIPBLAS_OP_N
#     op( A_i ) = A_i^H and A_i is k by n if transA == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C: op(A) = A^H
#         HIPBLAS_OP_N: op(A) = A
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
#         when transA is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       Device pointer to the first matrix C_1 on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasCherkStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,hipblasComplex * AP,int lda,long strideA,const float * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasCherkStridedBatched__funptr
    __init_symbol(&_hipblasCherkStridedBatched__funptr,"hipblasCherkStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,hipblasComplex *,int,long,const float *,hipblasComplex *,int,long,int) nogil> _hipblasCherkStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasZherkStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZherkStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,hipblasDoubleComplex * AP,int lda,long strideA,const double * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasZherkStridedBatched__funptr
    __init_symbol(&_hipblasZherkStridedBatched__funptr,"hipblasZherkStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,hipblasDoubleComplex *,int,long,const double *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZherkStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasCherkx__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# herkx performs one of the matrix-matrix operations for a Hermitian rank-k update
# 
# C := alpha*op( A )*op( B )^H + beta*C
# 
# where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
# C is a n x n Hermitian matrix stored as either upper or lower.
# This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermitian.
# 
# 
#     op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
#     op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C:  op( A ) = A^H, op( B ) = B^H
#         HIPBLAS_OP_N:  op( A ) = A, op( B ) = B
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP      pointer storing matrix A on the GPU.
#         Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# @param[in]
# BP       pointer storing matrix B on the GPU.
#         Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
#
cdef hipblasStatus_t hipblasCherkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,const float * beta,hipblasComplex * CP,int ldc) nogil:
    global _hipblasCherkx__funptr
    __init_symbol(&_hipblasCherkx__funptr,"hipblasCherkx")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,const float *,hipblasComplex *,int) nogil> _hipblasCherkx__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZherkx__funptr = NULL
cdef hipblasStatus_t hipblasZherkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _hipblasZherkx__funptr
    __init_symbol(&_hipblasZherkx__funptr,"hipblasZherkx")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,const double *,hipblasDoubleComplex *,int) nogil> _hipblasZherkx__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasCherkxBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# herkxBatched performs a batch of the matrix-matrix operations for a Hermitian rank-k update
# 
# C_i := alpha*op( A_i )*op( B_i )^H + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
# C_i is a n x n Hermitian matrix stored as either upper or lower.
# This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermitian.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C: op(A) = A^H
#         HIPBLAS_OP_N: op(A) = A
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       device array of device pointers storing each matrix_i A of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# BP       device array of device pointers storing each matrix_i B of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasCherkxBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,const float * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasCherkxBatched__funptr
    __init_symbol(&_hipblasCherkxBatched__funptr,"hipblasCherkxBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,const float *,hipblasComplex *const*,int,int) nogil> _hipblasCherkxBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasZherkxBatched__funptr = NULL
cdef hipblasStatus_t hipblasZherkxBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,const double * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasZherkxBatched__funptr
    __init_symbol(&_hipblasZherkxBatched__funptr,"hipblasZherkxBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,const double *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZherkxBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasCherkxStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# herkxStridedBatched performs a batch of the matrix-matrix operations for a Hermitian rank-k update
# 
# C_i := alpha*op( A_i )*op( B_i )^H + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
# C_i is a n x n Hermitian matrix stored as either upper or lower.
# This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermitian.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C: op( A_i ) = A_i^H, op( B_i ) = B_i^H
#         HIPBLAS_OP_N: op( A_i ) = A_i, op( B_i ) = B_i
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# BP       Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       Device pointer to the first matrix C_1 on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasCherkxStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,const float * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasCherkxStridedBatched__funptr
    __init_symbol(&_hipblasCherkxStridedBatched__funptr,"hipblasCherkxStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,const float *,hipblasComplex *,int,long,int) nogil> _hipblasCherkxStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasZherkxStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZherkxStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,const double * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasZherkxStridedBatched__funptr
    __init_symbol(&_hipblasZherkxStridedBatched__funptr,"hipblasZherkxStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,const double *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZherkxStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasCher2k__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# her2k performs one of the matrix-matrix operations for a Hermitian rank-2k update
# 
# C := alpha*op( A )*op( B )^H + conj(alpha)*op( B )*op( A )^H + beta*C
# 
# where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
# C is a n x n Hermitian matrix stored as either upper or lower.
# 
#     op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
#     op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C:  op( A ) = A^H, op( B ) = B^H
#         HIPBLAS_OP_N:  op( A ) = A, op( B ) = B
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# @param[in]
# BP       pointer storing matrix B on the GPU.
#         Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
#
cdef hipblasStatus_t hipblasCher2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,const float * beta,hipblasComplex * CP,int ldc) nogil:
    global _hipblasCher2k__funptr
    __init_symbol(&_hipblasCher2k__funptr,"hipblasCher2k")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,const float *,hipblasComplex *,int) nogil> _hipblasCher2k__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZher2k__funptr = NULL
cdef hipblasStatus_t hipblasZher2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _hipblasZher2k__funptr
    __init_symbol(&_hipblasZher2k__funptr,"hipblasZher2k")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,const double *,hipblasDoubleComplex *,int) nogil> _hipblasZher2k__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasCher2kBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# her2kBatched performs a batch of the matrix-matrix operations for a Hermitian rank-2k update
# 
# C_i := alpha*op( A_i )*op( B_i )^H + conj(alpha)*op( B_i )*op( A_i )^H + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
# C_i is a n x n Hermitian matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C: op(A) = A^H
#         HIPBLAS_OP_N: op(A) = A
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       device array of device pointers storing each matrix_i A of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# @param[in]
# BP       device array of device pointers storing each matrix_i B of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasCher2kBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,const float * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasCher2kBatched__funptr
    __init_symbol(&_hipblasCher2kBatched__funptr,"hipblasCher2kBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,const float *,hipblasComplex *const*,int,int) nogil> _hipblasCher2kBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasZher2kBatched__funptr = NULL
cdef hipblasStatus_t hipblasZher2kBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,const double * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasZher2kBatched__funptr
    __init_symbol(&_hipblasZher2kBatched__funptr,"hipblasZher2kBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,const double *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZher2kBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasCher2kStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# her2kStridedBatched performs a batch of the matrix-matrix operations for a Hermitian rank-2k update
# 
# C_i := alpha*op( A_i )*op( B_i )^H + conj(alpha)*op( B_i )*op( A_i )^H + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
# C_i is a n x n Hermitian matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C: op( A_i ) = A_i^H, op( B_i ) = B_i^H
#         HIPBLAS_OP_N: op( A_i ) = A_i, op( B_i ) = B_i
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# BP       Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       Device pointer to the first matrix C_1 on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasCher2kStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,const float * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasCher2kStridedBatched__funptr
    __init_symbol(&_hipblasCher2kStridedBatched__funptr,"hipblasCher2kStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,const float *,hipblasComplex *,int,long,int) nogil> _hipblasCher2kStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasZher2kStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZher2kStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,const double * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasZher2kStridedBatched__funptr
    __init_symbol(&_hipblasZher2kStridedBatched__funptr,"hipblasZher2kStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,const double *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZher2kStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasSsymm__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# symm performs one of the matrix-matrix operations:
# 
# C := alpha*A*B + beta*C if side == HIPBLAS_SIDE_LEFT,
# C := alpha*B*A + beta*C if side == HIPBLAS_SIDE_RIGHT,
# 
# where alpha and beta are scalars, B and C are m by n matrices, and
# A is a symmetric matrix stored as either upper or lower.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side  [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:      C := alpha*A*B + beta*C
#         HIPBLAS_SIDE_RIGHT:     C := alpha*B*A + beta*C
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B and C. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B and C. n >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A and B are not referenced.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         A is m by m if side == HIPBLAS_SIDE_LEFT
#         A is n by n if side == HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         otherwise lda >= max( 1, n ).
# 
# @param[in]
# BP       pointer storing matrix B on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B. ldb >= max( 1, m )
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, m )
#
cdef hipblasStatus_t hipblasSsymm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil:
    global _hipblasSsymm__funptr
    __init_symbol(&_hipblasSsymm__funptr,"hipblasSsymm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSsymm__funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasDsymm__funptr = NULL
cdef hipblasStatus_t hipblasDsymm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil:
    global _hipblasDsymm__funptr
    __init_symbol(&_hipblasDsymm__funptr,"hipblasDsymm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDsymm__funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasCsymm__funptr = NULL
cdef hipblasStatus_t hipblasCsymm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _hipblasCsymm__funptr
    __init_symbol(&_hipblasCsymm__funptr,"hipblasCsymm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCsymm__funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZsymm__funptr = NULL
cdef hipblasStatus_t hipblasZsymm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _hipblasZsymm__funptr
    __init_symbol(&_hipblasZsymm__funptr,"hipblasZsymm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZsymm__funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasSsymmBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# symmBatched performs a batch of the matrix-matrix operations:
# 
# C_i := alpha*A_i*B_i + beta*C_i if side == HIPBLAS_SIDE_LEFT,
# C_i := alpha*B_i*A_i + beta*C_i if side == HIPBLAS_SIDE_RIGHT,
# 
# where alpha and beta are scalars, B_i and C_i are m by n matrices, and
# A_i is a symmetric matrix stored as either upper or lower.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side  [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:      C_i := alpha*A_i*B_i + beta*C_i
#         HIPBLAS_SIDE_RIGHT:     C_i := alpha*B_i*A_i + beta*C_i
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B_i and C_i. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B_i and C_i. n >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A_i and B_i are not referenced.
# 
# @param[in]
# AP      device array of device pointers storing each matrix A_i on the GPU.
#         A_i is m by m if side == HIPBLAS_SIDE_LEFT
#         A_i is n by n if side == HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         otherwise lda >= max( 1, n ).
# 
# @param[in]
# BP       device array of device pointers storing each matrix B_i on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i. ldb >= max( 1, m )
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C_i need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C_i. ldc >= max( 1, m )
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasSsymmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const float * alpha,const float *const* AP,int lda,const float *const* BP,int ldb,const float * beta,float *const* CP,int ldc,int batchCount) nogil:
    global _hipblasSsymmBatched__funptr
    __init_symbol(&_hipblasSsymmBatched__funptr,"hipblasSsymmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,const float *,const float *const*,int,const float *const*,int,const float *,float *const*,int,int) nogil> _hipblasSsymmBatched__funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasDsymmBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsymmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const double * alpha,const double *const* AP,int lda,const double *const* BP,int ldb,const double * beta,double *const* CP,int ldc,int batchCount) nogil:
    global _hipblasDsymmBatched__funptr
    __init_symbol(&_hipblasDsymmBatched__funptr,"hipblasDsymmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,const double *,const double *const*,int,const double *const*,int,const double *,double *const*,int,int) nogil> _hipblasDsymmBatched__funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasCsymmBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsymmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,hipblasComplex * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasCsymmBatched__funptr
    __init_symbol(&_hipblasCsymmBatched__funptr,"hipblasCsymmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *,hipblasComplex *const*,int,int) nogil> _hipblasCsymmBatched__funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasZsymmBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsymmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasZsymmBatched__funptr
    __init_symbol(&_hipblasZsymmBatched__funptr,"hipblasZsymmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZsymmBatched__funptr)(handle,side,uplo,m,n,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasSsymmStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# symmStridedBatched performs a batch of the matrix-matrix operations:
# 
# C_i := alpha*A_i*B_i + beta*C_i if side == HIPBLAS_SIDE_LEFT,
# C_i := alpha*B_i*A_i + beta*C_i if side == HIPBLAS_SIDE_RIGHT,
# 
# where alpha and beta are scalars, B_i and C_i are m by n matrices, and
# A_i is a symmetric matrix stored as either upper or lower.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side  [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:      C_i := alpha*A_i*B_i + beta*C_i
#         HIPBLAS_SIDE_RIGHT:     C_i := alpha*B_i*A_i + beta*C_i
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B_i and C_i. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B_i and C_i. n >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A_i and B_i are not referenced.
# 
# @param[in]
# AP       device pointer to first matrix A_1
#         A_i is m by m if side == HIPBLAS_SIDE_LEFT
#         A_i is n by n if side == HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         otherwise lda >= max( 1, n ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# BP       device pointer to first matrix B_1 of dimension (ldb, n) on the GPU.
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i. ldb >= max( 1, m )
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP        device pointer to first matrix C_1 of dimension (ldc, n) on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, m ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasSsymmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const float * alpha,const float * AP,int lda,long strideA,const float * BP,int ldb,long strideB,const float * beta,float * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasSsymmStridedBatched__funptr
    __init_symbol(&_hipblasSsymmStridedBatched__funptr,"hipblasSsymmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,const float *,const float *,int,long,const float *,int,long,const float *,float *,int,long,int) nogil> _hipblasSsymmStridedBatched__funptr)(handle,side,uplo,m,n,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasDsymmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsymmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const double * alpha,const double * AP,int lda,long strideA,const double * BP,int ldb,long strideB,const double * beta,double * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasDsymmStridedBatched__funptr
    __init_symbol(&_hipblasDsymmStridedBatched__funptr,"hipblasDsymmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,const double *,const double *,int,long,const double *,int,long,const double *,double *,int,long,int) nogil> _hipblasDsymmStridedBatched__funptr)(handle,side,uplo,m,n,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasCsymmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsymmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,hipblasComplex * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasCsymmStridedBatched__funptr
    __init_symbol(&_hipblasCsymmStridedBatched__funptr,"hipblasCsymmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,hipblasComplex *,int,long,int) nogil> _hipblasCsymmStridedBatched__funptr)(handle,side,uplo,m,n,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasZsymmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsymmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasZsymmStridedBatched__funptr
    __init_symbol(&_hipblasZsymmStridedBatched__funptr,"hipblasZsymmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZsymmStridedBatched__funptr)(handle,side,uplo,m,n,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasSsyrk__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syrk performs one of the matrix-matrix operations for a symmetric rank-k update
# 
# C := alpha*op( A )*op( A )^T + beta*C
# 
# where  alpha and beta are scalars, op(A) is an n by k matrix, and
# C is a symmetric n x n matrix stored as either upper or lower.
# 
#     op( A ) = A, and A is n by k if transA == HIPBLAS_OP_N
#     op( A ) = A^T and A is k by n if transA == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T: op(A) = A^T
#         HIPBLAS_OP_N: op(A) = A
#         HIPBLAS_OP_C: op(A) = A^T
# 
#         HIPBLAS_OP_C is not supported for complex types, see cherk
#         and zherk.
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         Martrix dimension is ( lda, k ) when if transA = HIPBLAS_OP_N, otherwise (lda, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
#
cdef hipblasStatus_t hipblasSsyrk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,const float * beta,float * CP,int ldc) nogil:
    global _hipblasSsyrk__funptr
    __init_symbol(&_hipblasSsyrk__funptr,"hipblasSsyrk")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,float *,int) nogil> _hipblasSsyrk__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* _hipblasDsyrk__funptr = NULL
cdef hipblasStatus_t hipblasDsyrk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * beta,double * CP,int ldc) nogil:
    global _hipblasDsyrk__funptr
    __init_symbol(&_hipblasDsyrk__funptr,"hipblasDsyrk")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,double *,int) nogil> _hipblasDsyrk__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* _hipblasCsyrk__funptr = NULL
cdef hipblasStatus_t hipblasCsyrk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _hipblasCsyrk__funptr
    __init_symbol(&_hipblasCsyrk__funptr,"hipblasCsyrk")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCsyrk__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* _hipblasZsyrk__funptr = NULL
cdef hipblasStatus_t hipblasZsyrk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _hipblasZsyrk__funptr
    __init_symbol(&_hipblasZsyrk__funptr,"hipblasZsyrk")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZsyrk__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc)


cdef void* _hipblasSsyrkBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syrkBatched performs a batch of the matrix-matrix operations for a symmetric rank-k update
# 
# C_i := alpha*op( A_i )*op( A_i )^T + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) is an n by k matrix, and
# C_i is a symmetric n x n matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, and A_i is n by k if transA == HIPBLAS_OP_N
#     op( A_i ) = A_i^T and A_i is k by n if transA == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T: op(A) = A^T
#         HIPBLAS_OP_N: op(A) = A
#         HIPBLAS_OP_C: op(A) = A^T
# 
#         HIPBLAS_OP_C is not supported for complex types, see cherk
#         and zherk.
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       device array of device pointers storing each matrix_i A of dimension (lda, k)
#         when transA is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasSsyrkBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float *const* AP,int lda,const float * beta,float *const* CP,int ldc,int batchCount) nogil:
    global _hipblasSsyrkBatched__funptr
    __init_symbol(&_hipblasSsyrkBatched__funptr,"hipblasSsyrkBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *const*,int,const float *,float *const*,int,int) nogil> _hipblasSsyrkBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc,batchCount)


cdef void* _hipblasDsyrkBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsyrkBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double *const* AP,int lda,const double * beta,double *const* CP,int ldc,int batchCount) nogil:
    global _hipblasDsyrkBatched__funptr
    __init_symbol(&_hipblasDsyrkBatched__funptr,"hipblasDsyrkBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *const*,int,const double *,double *const*,int,int) nogil> _hipblasDsyrkBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc,batchCount)


cdef void* _hipblasCsyrkBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsyrkBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasCsyrkBatched__funptr
    __init_symbol(&_hipblasCsyrkBatched__funptr,"hipblasCsyrkBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *,hipblasComplex *const*,int,int) nogil> _hipblasCsyrkBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc,batchCount)


cdef void* _hipblasZsyrkBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsyrkBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasZsyrkBatched__funptr
    __init_symbol(&_hipblasZsyrkBatched__funptr,"hipblasZsyrkBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZsyrkBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,beta,CP,ldc,batchCount)


cdef void* _hipblasSsyrkStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syrkStridedBatched performs a batch of the matrix-matrix operations for a symmetric rank-k update
# 
# C_i := alpha*op( A_i )*op( A_i )^T + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) is an n by k matrix, and
# C_i is a symmetric n x n matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, and A_i is n by k if transA == HIPBLAS_OP_N
#     op( A_i ) = A_i^T and A_i is k by n if transA == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T: op(A) = A^T
#         HIPBLAS_OP_N: op(A) = A
#         HIPBLAS_OP_C: op(A) = A^T
# 
#         HIPBLAS_OP_C is not supported for complex types, see cherk
#         and zherk.
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
#         when transA is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       Device pointer to the first matrix C_1 on the GPU. on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasSsyrkStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,long strideA,const float * beta,float * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasSsyrkStridedBatched__funptr
    __init_symbol(&_hipblasSsyrkStridedBatched__funptr,"hipblasSsyrkStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *,int,long,const float *,float *,int,long,int) nogil> _hipblasSsyrkStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasDsyrkStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsyrkStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,long strideA,const double * beta,double * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasDsyrkStridedBatched__funptr
    __init_symbol(&_hipblasDsyrkStridedBatched__funptr,"hipblasDsyrkStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *,int,long,const double *,double *,int,long,int) nogil> _hipblasDsyrkStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasCsyrkStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsyrkStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasCsyrkStridedBatched__funptr
    __init_symbol(&_hipblasCsyrkStridedBatched__funptr,"hipblasCsyrkStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,hipblasComplex *,int,long,int) nogil> _hipblasCsyrkStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasZsyrkStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsyrkStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasZsyrkStridedBatched__funptr
    __init_symbol(&_hipblasZsyrkStridedBatched__funptr,"hipblasZsyrkStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZsyrkStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasSsyr2k__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syr2k performs one of the matrix-matrix operations for a symmetric rank-2k update
# 
# C := alpha*(op( A )*op( B )^T + op( B )*op( A )^T) + beta*C
# 
# where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
# C is a symmetric n x n matrix stored as either upper or lower.
# 
#     op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
#     op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T:      op( A ) = A^T, op( B ) = B^T
#         HIPBLAS_OP_N:           op( A ) = A, op( B ) = B
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A) and op(B). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# @param[in]
# BP       pointer storing matrix B on the GPU.
#         Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
#
cdef hipblasStatus_t hipblasSsyr2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil:
    global _hipblasSsyr2k__funptr
    __init_symbol(&_hipblasSsyr2k__funptr,"hipblasSsyr2k")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSsyr2k__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasDsyr2k__funptr = NULL
cdef hipblasStatus_t hipblasDsyr2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil:
    global _hipblasDsyr2k__funptr
    __init_symbol(&_hipblasDsyr2k__funptr,"hipblasDsyr2k")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDsyr2k__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasCsyr2k__funptr = NULL
cdef hipblasStatus_t hipblasCsyr2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _hipblasCsyr2k__funptr
    __init_symbol(&_hipblasCsyr2k__funptr,"hipblasCsyr2k")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCsyr2k__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZsyr2k__funptr = NULL
cdef hipblasStatus_t hipblasZsyr2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _hipblasZsyr2k__funptr
    __init_symbol(&_hipblasZsyr2k__funptr,"hipblasZsyr2k")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZsyr2k__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasSsyr2kBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syr2kBatched performs a batch of the matrix-matrix operations for a symmetric rank-2k update
# 
# C_i := alpha*(op( A_i )*op( B_i )^T + op( B_i )*op( A_i )^T) + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
# C_i is a symmetric n x n matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
#         HIPBLAS_OP_N:           op( A_i ) = A_i, op( B_i ) = B_i
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       device array of device pointers storing each matrix_i A of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# @param[in]
# BP      device array of device pointers storing each matrix_i B of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP      device array of device pointers storing each matrix C_i on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasSsyr2kBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float *const* AP,int lda,const float *const* BP,int ldb,const float * beta,float *const* CP,int ldc,int batchCount) nogil:
    global _hipblasSsyr2kBatched__funptr
    __init_symbol(&_hipblasSsyr2kBatched__funptr,"hipblasSsyr2kBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *const*,int,const float *const*,int,const float *,float *const*,int,int) nogil> _hipblasSsyr2kBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasDsyr2kBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsyr2kBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double *const* AP,int lda,const double *const* BP,int ldb,const double * beta,double *const* CP,int ldc,int batchCount) nogil:
    global _hipblasDsyr2kBatched__funptr
    __init_symbol(&_hipblasDsyr2kBatched__funptr,"hipblasDsyr2kBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *const*,int,const double *const*,int,const double *,double *const*,int,int) nogil> _hipblasDsyr2kBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasCsyr2kBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsyr2kBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,hipblasComplex * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasCsyr2kBatched__funptr
    __init_symbol(&_hipblasCsyr2kBatched__funptr,"hipblasCsyr2kBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *,hipblasComplex *const*,int,int) nogil> _hipblasCsyr2kBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasZsyr2kBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsyr2kBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasZsyr2kBatched__funptr
    __init_symbol(&_hipblasZsyr2kBatched__funptr,"hipblasZsyr2kBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZsyr2kBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasSsyr2kStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syr2kStridedBatched performs a batch of the matrix-matrix operations for a symmetric rank-2k update
# 
# C_i := alpha*(op( A_i )*op( B_i )^T + op( B_i )*op( A_i )^T) + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
# C_i is a symmetric n x n matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
#         HIPBLAS_OP_N:           op( A_i ) = A_i, op( B_i ) = B_i
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# BP       Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       Device pointer to the first matrix C_1 on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasSsyr2kStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,long strideA,const float * BP,int ldb,long strideB,const float * beta,float * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasSsyr2kStridedBatched__funptr
    __init_symbol(&_hipblasSsyr2kStridedBatched__funptr,"hipblasSsyr2kStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *,int,long,const float *,int,long,const float *,float *,int,long,int) nogil> _hipblasSsyr2kStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasDsyr2kStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsyr2kStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,long strideA,const double * BP,int ldb,long strideB,const double * beta,double * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasDsyr2kStridedBatched__funptr
    __init_symbol(&_hipblasDsyr2kStridedBatched__funptr,"hipblasDsyr2kStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *,int,long,const double *,int,long,const double *,double *,int,long,int) nogil> _hipblasDsyr2kStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasCsyr2kStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsyr2kStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,hipblasComplex * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasCsyr2kStridedBatched__funptr
    __init_symbol(&_hipblasCsyr2kStridedBatched__funptr,"hipblasCsyr2kStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,hipblasComplex *,int,long,int) nogil> _hipblasCsyr2kStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasZsyr2kStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsyr2kStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasZsyr2kStridedBatched__funptr
    __init_symbol(&_hipblasZsyr2kStridedBatched__funptr,"hipblasZsyr2kStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZsyr2kStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasSsyrkx__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syrkx performs one of the matrix-matrix operations for a symmetric rank-k update
# 
# C := alpha*op( A )*op( B )^T + beta*C
# 
# where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
# C is a symmetric n x n matrix stored as either upper or lower.
# This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be symmetric.
# 
#     op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
#     op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T:      op( A ) = A^T, op( B ) = B^T
#         HIPBLAS_OP_N:           op( A ) = A, op( B ) = B
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A) and op(B). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# BP       pointer storing matrix B on the GPU.
#         Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
#
cdef hipblasStatus_t hipblasSsyrkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil:
    global _hipblasSsyrkx__funptr
    __init_symbol(&_hipblasSsyrkx__funptr,"hipblasSsyrkx")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,int,const float *,float *,int) nogil> _hipblasSsyrkx__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasDsyrkx__funptr = NULL
cdef hipblasStatus_t hipblasDsyrkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil:
    global _hipblasDsyrkx__funptr
    __init_symbol(&_hipblasDsyrkx__funptr,"hipblasDsyrkx")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,int,const double *,double *,int) nogil> _hipblasDsyrkx__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasCsyrkx__funptr = NULL
cdef hipblasStatus_t hipblasCsyrkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _hipblasCsyrkx__funptr
    __init_symbol(&_hipblasCsyrkx__funptr,"hipblasCsyrkx")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasCsyrkx__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZsyrkx__funptr = NULL
cdef hipblasStatus_t hipblasZsyrkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _hipblasZsyrkx__funptr
    __init_symbol(&_hipblasZsyrkx__funptr,"hipblasZsyrkx")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZsyrkx__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasSsyrkxBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syrkxBatched performs a batch of the matrix-matrix operations for a symmetric rank-k update
# 
# C_i := alpha*op( A_i )*op( B_i )^T + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
# C_i is a symmetric n x n matrix stored as either upper or lower.
# This routine should only be used when the caller can guarantee that the result of op( A_i )*op( B_i )^T will be symmetric.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
#         HIPBLAS_OP_N:           op( A_i ) = A_i, op( B_i ) = B_i
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       device array of device pointers storing each matrix_i A of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# BP       device array of device pointers storing each matrix_i B of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[in]
# batchCount [int]
#         number of instances in the batch.
#
cdef hipblasStatus_t hipblasSsyrkxBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float *const* AP,int lda,const float *const* BP,int ldb,const float * beta,float *const* CP,int ldc,int batchCount) nogil:
    global _hipblasSsyrkxBatched__funptr
    __init_symbol(&_hipblasSsyrkxBatched__funptr,"hipblasSsyrkxBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *const*,int,const float *const*,int,const float *,float *const*,int,int) nogil> _hipblasSsyrkxBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasDsyrkxBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsyrkxBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double *const* AP,int lda,const double *const* BP,int ldb,const double * beta,double *const* CP,int ldc,int batchCount) nogil:
    global _hipblasDsyrkxBatched__funptr
    __init_symbol(&_hipblasDsyrkxBatched__funptr,"hipblasDsyrkxBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *const*,int,const double *const*,int,const double *,double *const*,int,int) nogil> _hipblasDsyrkxBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasCsyrkxBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsyrkxBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,hipblasComplex * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasCsyrkxBatched__funptr
    __init_symbol(&_hipblasCsyrkxBatched__funptr,"hipblasCsyrkxBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *,hipblasComplex *const*,int,int) nogil> _hipblasCsyrkxBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasZsyrkxBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsyrkxBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasZsyrkxBatched__funptr
    __init_symbol(&_hipblasZsyrkxBatched__funptr,"hipblasZsyrkxBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZsyrkxBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasSsyrkxStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syrkxStridedBatched performs a batch of the matrix-matrix operations for a symmetric rank-k update
# 
# C_i := alpha*op( A_i )*op( B_i )^T + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
# C_i is a symmetric n x n matrix stored as either upper or lower.
# This routine should only be used when the caller can guarantee that the result of op( A_i )*op( B_i )^T will be symmetric.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
#         HIPBLAS_OP_N:           op( A_i ) = A_i, op( B_i ) = B_i
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# BP       Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       Device pointer to the first matrix C_1 on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasSsyrkxStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,long strideA,const float * BP,int ldb,long strideB,const float * beta,float * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasSsyrkxStridedBatched__funptr
    __init_symbol(&_hipblasSsyrkxStridedBatched__funptr,"hipblasSsyrkxStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const float *,const float *,int,long,const float *,int,long,const float *,float *,int,long,int) nogil> _hipblasSsyrkxStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasDsyrkxStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDsyrkxStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,long strideA,const double * BP,int ldb,long strideB,const double * beta,double * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasDsyrkxStridedBatched__funptr
    __init_symbol(&_hipblasDsyrkxStridedBatched__funptr,"hipblasDsyrkxStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,const double *,const double *,int,long,const double *,int,long,const double *,double *,int,long,int) nogil> _hipblasDsyrkxStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasCsyrkxStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCsyrkxStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,hipblasComplex * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasCsyrkxStridedBatched__funptr
    __init_symbol(&_hipblasCsyrkxStridedBatched__funptr,"hipblasCsyrkxStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,hipblasComplex *,int,long,int) nogil> _hipblasCsyrkxStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasZsyrkxStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZsyrkxStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasZsyrkxStridedBatched__funptr
    __init_symbol(&_hipblasZsyrkxStridedBatched__funptr,"hipblasZsyrkxStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZsyrkxStridedBatched__funptr)(handle,uplo,transA,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasSgeam__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# geam performs one of the matrix-matrix operations
# 
#     C = alpha*op( A ) + beta*op( B ),
# 
# where op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
# 
# alpha and beta are scalars, and A, B and C are matrices, with
# op( A ) an m by n matrix, op( B ) an m by n matrix, and C an m by n matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A )
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B )
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# alpha     device pointer or host pointer specifying the scalar alpha.
# @param[in]
# AP         device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# beta      device pointer or host pointer specifying the scalar beta.
# @param[in]
# BP         device pointer storing matrix B.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of B.
# @param[in, out]
# CP         device pointer storing matrix C.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
#
cdef hipblasStatus_t hipblasSgeam(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const float * alpha,const float * AP,int lda,const float * beta,const float * BP,int ldb,float * CP,int ldc) nogil:
    global _hipblasSgeam__funptr
    __init_symbol(&_hipblasSgeam__funptr,"hipblasSgeam")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,const float *,const float *,int,const float *,const float *,int,float *,int) nogil> _hipblasSgeam__funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc)


cdef void* _hipblasDgeam__funptr = NULL
cdef hipblasStatus_t hipblasDgeam(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const double * alpha,const double * AP,int lda,const double * beta,const double * BP,int ldb,double * CP,int ldc) nogil:
    global _hipblasDgeam__funptr
    __init_symbol(&_hipblasDgeam__funptr,"hipblasDgeam")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,const double *,const double *,int,const double *,const double *,int,double *,int) nogil> _hipblasDgeam__funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc)


cdef void* _hipblasCgeam__funptr = NULL
cdef hipblasStatus_t hipblasCgeam(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * beta,hipblasComplex * BP,int ldb,hipblasComplex * CP,int ldc) nogil:
    global _hipblasCgeam__funptr
    __init_symbol(&_hipblasCgeam__funptr,"hipblasCgeam")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCgeam__funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc)


cdef void* _hipblasZgeam__funptr = NULL
cdef hipblasStatus_t hipblasZgeam(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * CP,int ldc) nogil:
    global _hipblasZgeam__funptr
    __init_symbol(&_hipblasZgeam__funptr,"hipblasZgeam")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZgeam__funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc)


cdef void* _hipblasSgeamBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# geamBatched performs one of the batched matrix-matrix operations
# 
#     C_i = alpha*op( A_i ) + beta*op( B_i )  for i = 0, 1, ... batchCount - 1
# 
# where alpha and beta are scalars, and op(A_i), op(B_i) and C_i are m by n matrices
# and op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A )
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B )
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# alpha     device pointer or host pointer specifying the scalar alpha.
# @param[in]
# AP         device array of device pointers storing each matrix A_i on the GPU.
#           Each A_i is of dimension ( lda, k ), where k is m
#           when  transA == HIPBLAS_OP_N and
#           is  n  when  transA == HIPBLAS_OP_T.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# beta      device pointer or host pointer specifying the scalar beta.
# @param[in]
# BP         device array of device pointers storing each matrix B_i on the GPU.
#           Each B_i is of dimension ( ldb, k ), where k is m
#           when  transB == HIPBLAS_OP_N and
#           is  n  when  transB == HIPBLAS_OP_T.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of B.
# @param[in, out]
# CP         device array of device pointers storing each matrix C_i on the GPU.
#           Each C_i is of dimension ( ldc, n ).
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
# 
# @param[in]
# batchCount [int]
#             number of instances i in the batch.
#
cdef hipblasStatus_t hipblasSgeamBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const float * alpha,const float *const* AP,int lda,const float * beta,const float *const* BP,int ldb,float *const* CP,int ldc,int batchCount) nogil:
    global _hipblasSgeamBatched__funptr
    __init_symbol(&_hipblasSgeamBatched__funptr,"hipblasSgeamBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,const float *,const float *const*,int,const float *,const float *const*,int,float *const*,int,int) nogil> _hipblasSgeamBatched__funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc,batchCount)


cdef void* _hipblasDgeamBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgeamBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const double * alpha,const double *const* AP,int lda,const double * beta,const double *const* BP,int ldb,double *const* CP,int ldc,int batchCount) nogil:
    global _hipblasDgeamBatched__funptr
    __init_symbol(&_hipblasDgeamBatched__funptr,"hipblasDgeamBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,const double *,const double *const*,int,const double *,const double *const*,int,double *const*,int,int) nogil> _hipblasDgeamBatched__funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc,batchCount)


cdef void* _hipblasCgeamBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgeamBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex * beta,hipblasComplex *const* BP,int ldb,hipblasComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasCgeamBatched__funptr
    __init_symbol(&_hipblasCgeamBatched__funptr,"hipblasCgeamBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCgeamBatched__funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc,batchCount)


cdef void* _hipblasZgeamBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgeamBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* BP,int ldb,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasZgeamBatched__funptr
    __init_symbol(&_hipblasZgeamBatched__funptr,"hipblasZgeamBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZgeamBatched__funptr)(handle,transA,transB,m,n,alpha,AP,lda,beta,BP,ldb,CP,ldc,batchCount)


cdef void* _hipblasSgeamStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# geamStridedBatched performs one of the batched matrix-matrix operations
# 
#     C_i = alpha*op( A_i ) + beta*op( B_i )  for i = 0, 1, ... batchCount - 1
# 
# where alpha and beta are scalars, and op(A_i), op(B_i) and C_i are m by n matrices
# and op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A )
# 
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B )
# 
# @param[in]
# m         [int]
#           matrix dimension m.
# 
# @param[in]
# n         [int]
#           matrix dimension n.
# 
# @param[in]
# alpha     device pointer or host pointer specifying the scalar alpha.
# 
# @param[in]
# AP         device pointer to the first matrix A_0 on the GPU.
#           Each A_i is of dimension ( lda, k ), where k is m
#           when  transA == HIPBLAS_OP_N and
#           is  n  when  transA == HIPBLAS_OP_T.
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# beta      device pointer or host pointer specifying the scalar beta.
# 
# @param[in]
# BP         pointer to the first matrix B_0 on the GPU.
#           Each B_i is of dimension ( ldb, k ), where k is m
#           when  transB == HIPBLAS_OP_N and
#           is  n  when  transB == HIPBLAS_OP_T.
# 
# @param[in]
# ldb       [int]
#           specifies the leading dimension of B.
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# 
# @param[in, out]
# CP         pointer to the first matrix C_0 on the GPU.
#           Each C_i is of dimension ( ldc, n ).
# 
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
# 
# @param[in]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances i in the batch.
#
cdef hipblasStatus_t hipblasSgeamStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const float * alpha,const float * AP,int lda,long strideA,const float * beta,const float * BP,int ldb,long strideB,float * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasSgeamStridedBatched__funptr
    __init_symbol(&_hipblasSgeamStridedBatched__funptr,"hipblasSgeamStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,const float *,const float *,int,long,const float *,const float *,int,long,float *,int,long,int) nogil> _hipblasSgeamStridedBatched__funptr)(handle,transA,transB,m,n,alpha,AP,lda,strideA,beta,BP,ldb,strideB,CP,ldc,strideC,batchCount)


cdef void* _hipblasDgeamStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgeamStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const double * alpha,const double * AP,int lda,long strideA,const double * beta,const double * BP,int ldb,long strideB,double * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasDgeamStridedBatched__funptr
    __init_symbol(&_hipblasDgeamStridedBatched__funptr,"hipblasDgeamStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,const double *,const double *,int,long,const double *,const double *,int,long,double *,int,long,int) nogil> _hipblasDgeamStridedBatched__funptr)(handle,transA,transB,m,n,alpha,AP,lda,strideA,beta,BP,ldb,strideB,CP,ldc,strideC,batchCount)


cdef void* _hipblasCgeamStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgeamStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * beta,hipblasComplex * BP,int ldb,long strideB,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasCgeamStridedBatched__funptr
    __init_symbol(&_hipblasCgeamStridedBatched__funptr,"hipblasCgeamStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCgeamStridedBatched__funptr)(handle,transA,transB,m,n,alpha,AP,lda,strideA,beta,BP,ldb,strideB,CP,ldc,strideC,batchCount)


cdef void* _hipblasZgeamStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgeamStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * beta,hipblasDoubleComplex * BP,int ldb,long strideB,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasZgeamStridedBatched__funptr
    __init_symbol(&_hipblasZgeamStridedBatched__funptr,"hipblasZgeamStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZgeamStridedBatched__funptr)(handle,transA,transB,m,n,alpha,AP,lda,strideA,beta,BP,ldb,strideB,CP,ldc,strideC,batchCount)


cdef void* _hipblasChemm__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# hemm performs one of the matrix-matrix operations:
# 
# C := alpha*A*B + beta*C if side == HIPBLAS_SIDE_LEFT,
# C := alpha*B*A + beta*C if side == HIPBLAS_SIDE_RIGHT,
# 
# where alpha and beta are scalars, B and C are m by n matrices, and
# A is a Hermitian matrix stored as either upper or lower.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side  [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:      C := alpha*A*B + beta*C
#         HIPBLAS_SIDE_RIGHT:     C := alpha*B*A + beta*C
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix
# 
# @param[in]
# n       [int]
#         n specifies the number of rows of B and C. n >= 0.
# 
# @param[in]
# k       [int]
#         n specifies the number of columns of B and C. k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A and B are not referenced.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         A is m by m if side == HIPBLAS_SIDE_LEFT
#         A is n by n if side == HIPBLAS_SIDE_RIGHT
#         Only the upper/lower triangular part is accessed.
#         The imaginary component of the diagonal elements is not used.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         otherwise lda >= max( 1, n ).
# 
# @param[in]
# BP       pointer storing matrix B on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B. ldb >= max( 1, m )
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, m )
#
cdef hipblasStatus_t hipblasChemm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil:
    global _hipblasChemm__funptr
    __init_symbol(&_hipblasChemm__funptr,"hipblasChemm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,hipblasComplex *,int) nogil> _hipblasChemm__funptr)(handle,side,uplo,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasZhemm__funptr = NULL
cdef hipblasStatus_t hipblasZhemm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil:
    global _hipblasZhemm__funptr
    __init_symbol(&_hipblasZhemm__funptr,"hipblasZhemm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int) nogil> _hipblasZhemm__funptr)(handle,side,uplo,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc)


cdef void* _hipblasChemmBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# hemmBatched performs a batch of the matrix-matrix operations:
# 
# C_i := alpha*A_i*B_i + beta*C_i if side == HIPBLAS_SIDE_LEFT,
# C_i := alpha*B_i*A_i + beta*C_i if side == HIPBLAS_SIDE_RIGHT,
# 
# where alpha and beta are scalars, B_i and C_i are m by n matrices, and
# A_i is a Hermitian matrix stored as either upper or lower.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side  [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:      C_i := alpha*A_i*B_i + beta*C_i
#         HIPBLAS_SIDE_RIGHT:     C_i := alpha*B_i*A_i + beta*C_i
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix
# 
# @param[in]
# n       [int]
#         n specifies the number of rows of B_i and C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of B_i and C_i. k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A_i and B_i are not referenced.
# 
# @param[in]
# AP       device array of device pointers storing each matrix A_i on the GPU.
#         A_i is m by m if side == HIPBLAS_SIDE_LEFT
#         A_i is n by n if side == HIPBLAS_SIDE_RIGHT
#         Only the upper/lower triangular part is accessed.
#         The imaginary component of the diagonal elements is not used.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         otherwise lda >= max( 1, n ).
# 
# @param[in]
# BP       device array of device pointers storing each matrix B_i on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i. ldb >= max( 1, m )
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C_i need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C_i. ldc >= max( 1, m )
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasChemmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,hipblasComplex * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasChemmBatched__funptr
    __init_symbol(&_hipblasChemmBatched__funptr,"hipblasChemmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *,hipblasComplex *const*,int,int) nogil> _hipblasChemmBatched__funptr)(handle,side,uplo,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasZhemmBatched__funptr = NULL
cdef hipblasStatus_t hipblasZhemmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasZhemmBatched__funptr
    __init_symbol(&_hipblasZhemmBatched__funptr,"hipblasZhemmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZhemmBatched__funptr)(handle,side,uplo,n,k,alpha,AP,lda,BP,ldb,beta,CP,ldc,batchCount)


cdef void* _hipblasChemmStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# hemmStridedBatched performs a batch of the matrix-matrix operations:
# 
# C_i := alpha*A_i*B_i + beta*C_i if side == HIPBLAS_SIDE_LEFT,
# C_i := alpha*B_i*A_i + beta*C_i if side == HIPBLAS_SIDE_RIGHT,
# 
# where alpha and beta are scalars, B_i and C_i are m by n matrices, and
# A_i is a Hermitian matrix stored as either upper or lower.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side  [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:      C_i := alpha*A_i*B_i + beta*C_i
#         HIPBLAS_SIDE_RIGHT:     C_i := alpha*B_i*A_i + beta*C_i
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix
# 
# @param[in]
# n       [int]
#         n specifies the number of rows of B_i and C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of B_i and C_i. k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A_i and B_i are not referenced.
# 
# @param[in]
# AP       device pointer to first matrix A_1
#         A_i is m by m if side == HIPBLAS_SIDE_LEFT
#         A_i is n by n if side == HIPBLAS_SIDE_RIGHT
#         Only the upper/lower triangular part is accessed.
#         The imaginary component of the diagonal elements is not used.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         otherwise lda >= max( 1, n ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# BP       device pointer to first matrix B_1 of dimension (ldb, n) on the GPU
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if side = HIPBLAS_OP_N,  ldb >= max( 1, m ),
#         otherwise ldb >= max( 1, n ).
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP        device pointer to first matrix C_1 of dimension (ldc, n) on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, m )
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch
#
cdef hipblasStatus_t hipblasChemmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,hipblasComplex * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasChemmStridedBatched__funptr
    __init_symbol(&_hipblasChemmStridedBatched__funptr,"hipblasChemmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,hipblasComplex *,int,long,int) nogil> _hipblasChemmStridedBatched__funptr)(handle,side,uplo,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasZhemmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZhemmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasZhemmStridedBatched__funptr
    __init_symbol(&_hipblasZhemmStridedBatched__funptr,"hipblasZhemmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZhemmStridedBatched__funptr)(handle,side,uplo,n,k,alpha,AP,lda,strideA,BP,ldb,strideB,beta,CP,ldc,strideC,batchCount)


cdef void* _hipblasStrmm__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# trmm performs one of the matrix-matrix operations
# 
# B := alpha*op( A )*B,   or   B := alpha*B*op( A )
# 
# where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
# non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
# 
#     op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.
# 
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         Specifies whether op(A) multiplies B from the left or right as follows:
#         HIPBLAS_SIDE_LEFT:       B := alpha*op( A )*B.
#         HIPBLAS_SIDE_RIGHT:      B := alpha*B*op( A ).
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         Specifies whether the matrix A is an upper or lower triangular matrix as follows:
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         Specifies the form of op(A) to be used in the matrix multiplication as follows:
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_OP_C:  op(A) = A^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         Specifies whether or not A is unit triangular as follows:
#         HIPBLAS_DIAG_UNIT:      A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B. n >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and B need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to matrix A on the GPU.
#         A has dimension ( lda, k ), where k is m
#         when  side == HIPBLAS_SIDE_LEFT  and
#         is  n  when  side == HIPBLAS_SIDE_RIGHT.
# 
#     When uplo == HIPBLAS_FILL_MODE_UPPER the  leading  k by k
#     upper triangular part of the array  A must contain the upper
#     triangular matrix  and the strictly lower triangular part of
#     A is not referenced.
# 
#     When uplo == HIPBLAS_FILL_MODE_LOWER the  leading  k by k
#     lower triangular part of the array  A must contain the lower
#     triangular matrix  and the strictly upper triangular part of
#     A is not referenced.
# 
#     Note that when  diag == HIPBLAS_DIAG_UNIT  the diagonal elements of
#     A  are not referenced either,  but are assumed to be  unity.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side == HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side == HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[inout]
# BP       Device pointer to the first matrix B_0 on the GPU.
#         On entry,  the leading  m by n part of the array  B must
#        contain the matrix  B,  and  on exit  is overwritten  by the
#        transformed matrix.
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of B. ldb >= max( 1, m ).
#
cdef hipblasStatus_t hipblasStrmm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,const float * AP,int lda,float * BP,int ldb) nogil:
    global _hipblasStrmm__funptr
    __init_symbol(&_hipblasStrmm__funptr,"hipblasStrmm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,const float *,int,float *,int) nogil> _hipblasStrmm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasDtrmm__funptr = NULL
cdef hipblasStatus_t hipblasDtrmm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,const double * AP,int lda,double * BP,int ldb) nogil:
    global _hipblasDtrmm__funptr
    __init_symbol(&_hipblasDtrmm__funptr,"hipblasDtrmm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,const double *,int,double *,int) nogil> _hipblasDtrmm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasCtrmm__funptr = NULL
cdef hipblasStatus_t hipblasCtrmm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb) nogil:
    global _hipblasCtrmm__funptr
    __init_symbol(&_hipblasCtrmm__funptr,"hipblasCtrmm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtrmm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasZtrmm__funptr = NULL
cdef hipblasStatus_t hipblasZtrmm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb) nogil:
    global _hipblasZtrmm__funptr
    __init_symbol(&_hipblasZtrmm__funptr,"hipblasZtrmm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtrmm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasStrmmBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# trmmBatched performs one of the batched matrix-matrix operations
# 
# B_i := alpha*op( A_i )*B_i,   or   B_i := alpha*B_i*op( A_i )  for i = 0, 1, ... batchCount -1
# 
# where  alpha  is a scalar,  B_i  is an m by n matrix,  A_i  is a unit, or
# non-unit,  upper or lower triangular matrix  and  op( A_i )  is one  of
# 
#     op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         Specifies whether op(A_i) multiplies B_i from the left or right as follows:
#         HIPBLAS_SIDE_LEFT:       B_i := alpha*op( A_i )*B_i.
#         HIPBLAS_SIDE_RIGHT:      B_i := alpha*B_i*op( A_i ).
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         Specifies whether the matrix A is an upper or lower triangular matrix as follows:
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         Specifies the form of op(A_i) to be used in the matrix multiplication as follows:
#         HIPBLAS_OP_N:    op(A_i) = A_i.
#         HIPBLAS_OP_T:      op(A_i) = A_i^T.
#         HIPBLAS_OP_C:  op(A_i) = A_i^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         Specifies whether or not A_i is unit triangular as follows:
#         HIPBLAS_DIAG_UNIT:      A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B_i. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B_i. n >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A_i is not referenced and B_i need not be set before
#         entry.
# 
# @param[in]
# AP       Device array of device pointers storing each matrix A_i on the GPU.
#         Each A_i is of dimension ( lda, k ), where k is m
#         when  side == HIPBLAS_SIDE_LEFT  and
#         is  n  when  side == HIPBLAS_SIDE_RIGHT.
# 
#     When uplo == HIPBLAS_FILL_MODE_UPPER the  leading  k by k
#     upper triangular part of the array  A must contain the upper
#     triangular matrix  and the strictly lower triangular part of
#     A is not referenced.
# 
#     When uplo == HIPBLAS_FILL_MODE_LOWER the  leading  k by k
#     lower triangular part of the array  A must contain the lower
#     triangular matrix  and the strictly upper triangular part of
#     A is not referenced.
# 
#     Note that when  diag == HIPBLAS_DIAG_UNIT  the diagonal elements of
#     A_i  are not referenced either,  but are assumed to be  unity.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side == HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side == HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[inout]
# BP       device array of device pointers storing each matrix B_i on the GPU.
#         On entry,  the leading  m by n part of the array  B_i must
#        contain the matrix  B_i,  and  on exit  is overwritten  by the
#        transformed matrix.
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of B_i. ldb >= max( 1, m ).
# 
# @param[in]
# batchCount [int]
#             number of instances i in the batch.
cdef hipblasStatus_t hipblasStrmmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,const float *const* AP,int lda,float *const* BP,int ldb,int batchCount) nogil:
    global _hipblasStrmmBatched__funptr
    __init_symbol(&_hipblasStrmmBatched__funptr,"hipblasStrmmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,const float *const*,int,float *const*,int,int) nogil> _hipblasStrmmBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb,batchCount)


cdef void* _hipblasDtrmmBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtrmmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,const double *const* AP,int lda,double *const* BP,int ldb,int batchCount) nogil:
    global _hipblasDtrmmBatched__funptr
    __init_symbol(&_hipblasDtrmmBatched__funptr,"hipblasDtrmmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,const double *const*,int,double *const*,int,int) nogil> _hipblasDtrmmBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb,batchCount)


cdef void* _hipblasCtrmmBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtrmmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,int batchCount) nogil:
    global _hipblasCtrmmBatched__funptr
    __init_symbol(&_hipblasCtrmmBatched__funptr,"hipblasCtrmmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCtrmmBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb,batchCount)


cdef void* _hipblasZtrmmBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtrmmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,int batchCount) nogil:
    global _hipblasZtrmmBatched__funptr
    __init_symbol(&_hipblasZtrmmBatched__funptr,"hipblasZtrmmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZtrmmBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb,batchCount)


cdef void* _hipblasStrmmStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# trmmStridedBatched performs one of the strided_batched matrix-matrix operations
# 
# B_i := alpha*op( A_i )*B_i,   or   B_i := alpha*B_i*op( A_i )  for i = 0, 1, ... batchCount -1
# 
# where  alpha  is a scalar,  B_i  is an m by n matrix,  A_i  is a unit, or
# non-unit,  upper or lower triangular matrix  and  op( A_i )  is one  of
# 
#     op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         Specifies whether op(A_i) multiplies B_i from the left or right as follows:
#         HIPBLAS_SIDE_LEFT:       B_i := alpha*op( A_i )*B_i.
#         HIPBLAS_SIDE_RIGHT:      B_i := alpha*B_i*op( A_i ).
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         Specifies whether the matrix A is an upper or lower triangular matrix as follows:
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         Specifies the form of op(A_i) to be used in the matrix multiplication as follows:
#         HIPBLAS_OP_N:    op(A_i) = A_i.
#         HIPBLAS_OP_T:      op(A_i) = A_i^T.
#         HIPBLAS_OP_C:  op(A_i) = A_i^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         Specifies whether or not A_i is unit triangular as follows:
#         HIPBLAS_DIAG_UNIT:      A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B_i. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B_i. n >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A_i is not referenced and B_i need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_0 on the GPU.
#         Each A_i is of dimension ( lda, k ), where k is m
#         when  side == HIPBLAS_SIDE_LEFT  and
#         is  n  when  side == HIPBLAS_SIDE_RIGHT.
# 
#     When uplo == HIPBLAS_FILL_MODE_UPPER the  leading  k by k
#     upper triangular part of the array  A must contain the upper
#     triangular matrix  and the strictly lower triangular part of
#     A is not referenced.
# 
#     When uplo == HIPBLAS_FILL_MODE_LOWER the  leading  k by k
#     lower triangular part of the array  A must contain the lower
#     triangular matrix  and the strictly upper triangular part of
#     A is not referenced.
# 
#     Note that when  diag == HIPBLAS_DIAG_UNIT  the diagonal elements of
#     A_i  are not referenced either,  but are assumed to be  unity.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side == HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side == HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[inout]
# BP       Device pointer to the first matrix B_0 on the GPU.
#         On entry,  the leading  m by n part of the array  B_i must
#        contain the matrix  B_i,  and  on exit  is overwritten  by the
#        transformed matrix.
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of B_i. ldb >= max( 1, m ).
# 
#        @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# @param[in]
# batchCount [int]
#             number of instances i in the batch.
cdef hipblasStatus_t hipblasStrmmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,const float * AP,int lda,long strideA,float * BP,int ldb,long strideB,int batchCount) nogil:
    global _hipblasStrmmStridedBatched__funptr
    __init_symbol(&_hipblasStrmmStridedBatched__funptr,"hipblasStrmmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,const float *,int,long,float *,int,long,int) nogil> _hipblasStrmmStridedBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,strideA,BP,ldb,strideB,batchCount)


cdef void* _hipblasDtrmmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtrmmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,const double * AP,int lda,long strideA,double * BP,int ldb,long strideB,int batchCount) nogil:
    global _hipblasDtrmmStridedBatched__funptr
    __init_symbol(&_hipblasDtrmmStridedBatched__funptr,"hipblasDtrmmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,const double *,int,long,double *,int,long,int) nogil> _hipblasDtrmmStridedBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,strideA,BP,ldb,strideB,batchCount)


cdef void* _hipblasCtrmmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtrmmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,int batchCount) nogil:
    global _hipblasCtrmmStridedBatched__funptr
    __init_symbol(&_hipblasCtrmmStridedBatched__funptr,"hipblasCtrmmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCtrmmStridedBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,strideA,BP,ldb,strideB,batchCount)


cdef void* _hipblasZtrmmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtrmmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,int batchCount) nogil:
    global _hipblasZtrmmStridedBatched__funptr
    __init_symbol(&_hipblasZtrmmStridedBatched__funptr,"hipblasZtrmmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZtrmmStridedBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,strideA,BP,ldb,strideB,batchCount)


cdef void* _hipblasStrsm__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# trsm solves
# 
#     op(A)*X = alpha*B or  X*op(A) = alpha*B,
# 
# where alpha is a scalar, X and B are m by n matrices,
# A is triangular matrix and op(A) is one of
# 
#     op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.
# 
# The matrix X is overwritten on B.
# 
# Note about memory allocation:
# When trsm is launched with a k evenly divisible by the internal block size of 128,
# and is no larger than 10 of these blocks, the API takes advantage of utilizing pre-allocated
# memory found in the handle to increase overall performance. This memory can be managed by using
# the environment variable WORKBUF_TRSM_B_CHNK. When this variable is not set the device memory
# used for temporary storage will default to 1 MB and may result in chunking, which in turn may
# reduce performance. Under these circumstances it is recommended that WORKBUF_TRSM_B_CHNK be set
# to the desired chunk of right hand sides to be used at a time.
# 
# (where k is m when HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT)
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#         HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_OP_C: op(A) = A^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B. n >= 0.
# 
# @param[in]
# alpha
#         device pointer or host pointer specifying the scalar alpha. When alpha is
#         &zero then A is not referenced and B need not be set before
#         entry.
# 
# @param[in]
# AP       device pointer storing matrix A.
#         of dimension ( lda, k ), where k is m
#         when  HIPBLAS_SIDE_LEFT  and
#         is  n  when  HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[in,out]
# BP       device pointer storing matrix B.
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of B. ldb >= max( 1, m ).
#
cdef hipblasStatus_t hipblasStrsm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,float * AP,int lda,float * BP,int ldb) nogil:
    global _hipblasStrsm__funptr
    __init_symbol(&_hipblasStrsm__funptr,"hipblasStrsm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,float *,int,float *,int) nogil> _hipblasStrsm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasDtrsm__funptr = NULL
cdef hipblasStatus_t hipblasDtrsm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,double * AP,int lda,double * BP,int ldb) nogil:
    global _hipblasDtrsm__funptr
    __init_symbol(&_hipblasDtrsm__funptr,"hipblasDtrsm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,double *,int,double *,int) nogil> _hipblasDtrsm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasCtrsm__funptr = NULL
cdef hipblasStatus_t hipblasCtrsm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb) nogil:
    global _hipblasCtrsm__funptr
    __init_symbol(&_hipblasCtrsm__funptr,"hipblasCtrsm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtrsm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasZtrsm__funptr = NULL
cdef hipblasStatus_t hipblasZtrsm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb) nogil:
    global _hipblasZtrsm__funptr
    __init_symbol(&_hipblasZtrsm__funptr,"hipblasZtrsm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtrsm__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb)


cdef void* _hipblasStrsmBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# \details
# trsmBatched performs the following batched operation:
# 
#     op(A_i)*X_i = alpha*B_i or  X_i*op(A_i) = alpha*B_i, for i = 1, ..., batchCount.
# 
# where alpha is a scalar, X and B are batched m by n matrices,
# A is triangular batched matrix and op(A) is one of
# 
#     op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.
# 
# Each matrix X_i is overwritten on B_i for i = 1, ..., batchCount.
# 
# Note about memory allocation:
# When trsm is launched with a k evenly divisible by the internal block size of 128,
# and is no larger than 10 of these blocks, the API takes advantage of utilizing pre-allocated
# memory found in the handle to increase overall performance. This memory can be managed by using
# the environment variable WORKBUF_TRSM_B_CHNK. When this variable is not set the device memory
# used for temporary storage will default to 1 MB and may result in chunking, which in turn may
# reduce performance. Under these circumstances it is recommended that WORKBUF_TRSM_B_CHNK be set
# to the desired chunk of right hand sides to be used at a time.
# (where k is m when HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT)
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# side    [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#         HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  each A_i is a  lower triangular matrix.
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_OP_C: op(A) = A^H.
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  each A_i is not assumed to be unit triangular.
# @param[in]
# m       [int]
#         m specifies the number of rows of each B_i. m >= 0.
# @param[in]
# n       [int]
#         n specifies the number of columns of each B_i. n >= 0.
# @param[in]
# alpha
#         device pointer or host pointer specifying the scalar alpha. When alpha is
#         &zero then A is not referenced and B need not be set before
#         entry.
# @param[in]
# AP       device array of device pointers storing each matrix A_i on the GPU.
#         Matricies are of dimension ( lda, k ), where k is m
#         when  HIPBLAS_SIDE_LEFT  and is  n  when  HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# @param[in]
# lda     [int]
#         lda specifies the first dimension of each A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# @param[in,out]
# BP       device array of device pointers storing each matrix B_i on the GPU.
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
# @param[in]
# batchCount [int]
#             number of trsm operatons in the batch.
cdef hipblasStatus_t hipblasStrsmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,float *const* AP,int lda,float ** BP,int ldb,int batchCount) nogil:
    global _hipblasStrsmBatched__funptr
    __init_symbol(&_hipblasStrsmBatched__funptr,"hipblasStrsmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,float *const*,int,float **,int,int) nogil> _hipblasStrsmBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb,batchCount)


cdef void* _hipblasDtrsmBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtrsmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,double *const* AP,int lda,double ** BP,int ldb,int batchCount) nogil:
    global _hipblasDtrsmBatched__funptr
    __init_symbol(&_hipblasDtrsmBatched__funptr,"hipblasDtrsmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,double *const*,int,double **,int,int) nogil> _hipblasDtrsmBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb,batchCount)


cdef void* _hipblasCtrsmBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtrsmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex ** BP,int ldb,int batchCount) nogil:
    global _hipblasCtrsmBatched__funptr
    __init_symbol(&_hipblasCtrsmBatched__funptr,"hipblasCtrsmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,hipblasComplex *const*,int,hipblasComplex **,int,int) nogil> _hipblasCtrsmBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb,batchCount)


cdef void* _hipblasZtrsmBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtrsmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex ** BP,int ldb,int batchCount) nogil:
    global _hipblasZtrsmBatched__funptr
    __init_symbol(&_hipblasZtrsmBatched__funptr,"hipblasZtrsmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *const*,int,hipblasDoubleComplex **,int,int) nogil> _hipblasZtrsmBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,BP,ldb,batchCount)


cdef void* _hipblasStrsmStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# \details
# trsmSridedBatched performs the following strided batched operation:
# 
#     op(A_i)*X_i = alpha*B_i or  X_i*op(A_i) = alpha*B_i, for i = 1, ..., batchCount.
# 
# where alpha is a scalar, X and B are strided batched m by n matrices,
# A is triangular strided batched matrix and op(A) is one of
# 
#     op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.
# 
# Each matrix X_i is overwritten on B_i for i = 1, ..., batchCount.
# 
# Note about memory allocation:
# When trsm is launched with a k evenly divisible by the internal block size of 128,
# and is no larger than 10 of these blocks, the API takes advantage of utilizing pre-allocated
# memory found in the handle to increase overall performance. This memory can be managed by using
# the environment variable WORKBUF_TRSM_B_CHNK. When this variable is not set the device memory
# used for temporary storage will default to 1 MB and may result in chunking, which in turn may
# reduce performance. Under these circumstances it is recommended that WORKBUF_TRSM_B_CHNK be set
# to the desired chunk of right hand sides to be used at a time.
# (where k is m when HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT)
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# side    [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#         HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  each A_i is a  lower triangular matrix.
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_OP_C: op(A) = A^H.
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  each A_i is not assumed to be unit triangular.
# @param[in]
# m       [int]
#         m specifies the number of rows of each B_i. m >= 0.
# @param[in]
# n       [int]
#         n specifies the number of columns of each B_i. n >= 0.
# @param[in]
# alpha
#         device pointer or host pointer specifying the scalar alpha. When alpha is
#         &zero then A is not referenced and B need not be set before
#         entry.
# @param[in]
# AP       device pointer pointing to the first matrix A_1.
#         of dimension ( lda, k ), where k is m
#         when  HIPBLAS_SIDE_LEFT  and
#         is  n  when  HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# @param[in]
# lda     [int]
#         lda specifies the first dimension of each A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# @param[in]
# strideA [hipblasStride]
#          stride from the start of one A_i matrix to the next A_(i + 1).
# @param[in,out]
# BP       device pointer pointing to the first matrix B_1.
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
# @param[in]
# strideB [hipblasStride]
#          stride from the start of one B_i matrix to the next B_(i + 1).
# @param[in]
# batchCount [int]
#             number of trsm operatons in the batch.
cdef hipblasStatus_t hipblasStrsmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,float * AP,int lda,long strideA,float * BP,int ldb,long strideB,int batchCount) nogil:
    global _hipblasStrsmStridedBatched__funptr
    __init_symbol(&_hipblasStrsmStridedBatched__funptr,"hipblasStrsmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const float *,float *,int,long,float *,int,long,int) nogil> _hipblasStrsmStridedBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,strideA,BP,ldb,strideB,batchCount)


cdef void* _hipblasDtrsmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtrsmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,double * AP,int lda,long strideA,double * BP,int ldb,long strideB,int batchCount) nogil:
    global _hipblasDtrsmStridedBatched__funptr
    __init_symbol(&_hipblasDtrsmStridedBatched__funptr,"hipblasDtrsmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double *,double *,int,long,double *,int,long,int) nogil> _hipblasDtrsmStridedBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,strideA,BP,ldb,strideB,batchCount)


cdef void* _hipblasCtrsmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtrsmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,int batchCount) nogil:
    global _hipblasCtrsmStridedBatched__funptr
    __init_symbol(&_hipblasCtrsmStridedBatched__funptr,"hipblasCtrsmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasComplex *,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCtrsmStridedBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,strideA,BP,ldb,strideB,batchCount)


cdef void* _hipblasZtrsmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtrsmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,int batchCount) nogil:
    global _hipblasZtrsmStridedBatched__funptr
    __init_symbol(&_hipblasZtrsmStridedBatched__funptr,"hipblasZtrsmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,hipblasDoubleComplex *,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZtrsmStridedBatched__funptr)(handle,side,uplo,transA,diag,m,n,alpha,AP,lda,strideA,BP,ldb,strideB,batchCount)


cdef void* _hipblasStrtri__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# trtri  compute the inverse of a matrix A, namely, invA
# 
#     and write the result into invA;
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# diag      [hipblasDiagType_t]
#           = 'HIPBLAS_DIAG_NON_UNIT', A is non-unit triangular;
#           = 'HIPBLAS_DIAG_UNIT', A is unit triangular;
# @param[in]
# n         [int]
#           size of matrix A and invA
# @param[in]
# AP         device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[out]
# invA      device pointer storing matrix invA.
# @param[in]
# ldinvA    [int]
#           specifies the leading dimension of invA.
#
cdef hipblasStatus_t hipblasStrtri(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const float * AP,int lda,float * invA,int ldinvA) nogil:
    global _hipblasStrtri__funptr
    __init_symbol(&_hipblasStrtri__funptr,"hipblasStrtri")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasDiagType_t,int,const float *,int,float *,int) nogil> _hipblasStrtri__funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA)


cdef void* _hipblasDtrtri__funptr = NULL
cdef hipblasStatus_t hipblasDtrtri(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const double * AP,int lda,double * invA,int ldinvA) nogil:
    global _hipblasDtrtri__funptr
    __init_symbol(&_hipblasDtrtri__funptr,"hipblasDtrtri")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasDiagType_t,int,const double *,int,double *,int) nogil> _hipblasDtrtri__funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA)


cdef void* _hipblasCtrtri__funptr = NULL
cdef hipblasStatus_t hipblasCtrtri(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasComplex * AP,int lda,hipblasComplex * invA,int ldinvA) nogil:
    global _hipblasCtrtri__funptr
    __init_symbol(&_hipblasCtrtri__funptr,"hipblasCtrtri")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasDiagType_t,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCtrtri__funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA)


cdef void* _hipblasZtrtri__funptr = NULL
cdef hipblasStatus_t hipblasZtrtri(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * invA,int ldinvA) nogil:
    global _hipblasZtrtri__funptr
    __init_symbol(&_hipblasZtrtri__funptr,"hipblasZtrtri")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasDiagType_t,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZtrtri__funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA)


cdef void* _hipblasStrtriBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# trtriBatched  compute the inverse of A_i and write into invA_i where
#                A_i and invA_i are the i-th matrices in the batch,
#                for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
# @param[in]
# diag      [hipblasDiagType_t]
#           = 'HIPBLAS_DIAG_NON_UNIT', A is non-unit triangular;
#           = 'HIPBLAS_DIAG_UNIT', A is unit triangular;
# @param[in]
# n         [int]
# @param[in]
# AP         device array of device pointers storing each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[out]
# invA      device array of device pointers storing the inverse of each matrix A_i.
#           Partial inplace operation is supported, see below.
#           If UPLO = 'U', the leading N-by-N upper triangular part of the invA will store
#           the inverse of the upper triangular matrix, and the strictly lower
#           triangular part of invA is cleared.
#           If UPLO = 'L', the leading N-by-N lower triangular part of the invA will store
#           the inverse of the lower triangular matrix, and the strictly upper
#           triangular part of invA is cleared.
# @param[in]
# ldinvA    [int]
#           specifies the leading dimension of each invA_i.
# @param[in]
# batchCount [int]
#           numbers of matrices in the batch
cdef hipblasStatus_t hipblasStrtriBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const float *const* AP,int lda,float ** invA,int ldinvA,int batchCount) nogil:
    global _hipblasStrtriBatched__funptr
    __init_symbol(&_hipblasStrtriBatched__funptr,"hipblasStrtriBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasDiagType_t,int,const float *const*,int,float **,int,int) nogil> _hipblasStrtriBatched__funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA,batchCount)


cdef void* _hipblasDtrtriBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtrtriBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const double *const* AP,int lda,double ** invA,int ldinvA,int batchCount) nogil:
    global _hipblasDtrtriBatched__funptr
    __init_symbol(&_hipblasDtrtriBatched__funptr,"hipblasDtrtriBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasDiagType_t,int,const double *const*,int,double **,int,int) nogil> _hipblasDtrtriBatched__funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA,batchCount)


cdef void* _hipblasCtrtriBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtrtriBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasComplex *const* AP,int lda,hipblasComplex ** invA,int ldinvA,int batchCount) nogil:
    global _hipblasCtrtriBatched__funptr
    __init_symbol(&_hipblasCtrtriBatched__funptr,"hipblasCtrtriBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasDiagType_t,int,hipblasComplex *const*,int,hipblasComplex **,int,int) nogil> _hipblasCtrtriBatched__funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA,batchCount)


cdef void* _hipblasZtrtriBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtrtriBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex ** invA,int ldinvA,int batchCount) nogil:
    global _hipblasZtrtriBatched__funptr
    __init_symbol(&_hipblasZtrtriBatched__funptr,"hipblasZtrtriBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasDiagType_t,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex **,int,int) nogil> _hipblasZtrtriBatched__funptr)(handle,uplo,diag,n,AP,lda,invA,ldinvA,batchCount)


cdef void* _hipblasStrtriStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# trtriStridedBatched compute the inverse of A_i and write into invA_i where
#                A_i and invA_i are the i-th matrices in the batch,
#                for i = 1, ..., batchCount
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
# @param[in]
# diag      [hipblasDiagType_t]
#           = 'HIPBLAS_DIAG_NON_UNIT', A is non-unit triangular;
#           = 'HIPBLAS_DIAG_UNIT', A is unit triangular;
# @param[in]
# n         [int]
# @param[in]
# AP         device pointer pointing to address of first matrix A_1.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A.
# @param[in]
# strideA  [hipblasStride]
#          "batch stride a": stride from the start of one A_i matrix to the next A_(i + 1).
# @param[out]
# invA      device pointer storing the inverses of each matrix A_i.
#           Partial inplace operation is supported, see below.
#           If UPLO = 'U', the leading N-by-N upper triangular part of the invA will store
#           the inverse of the upper triangular matrix, and the strictly lower
#           triangular part of invA is cleared.
#           If UPLO = 'L', the leading N-by-N lower triangular part of the invA will store
#           the inverse of the lower triangular matrix, and the strictly upper
#           triangular part of invA is cleared.
# @param[in]
# ldinvA    [int]
#           specifies the leading dimension of each invA_i.
# @param[in]
# stride_invA  [hipblasStride]
#              "batch stride invA": stride from the start of one invA_i matrix to the next invA_(i + 1).
# @param[in]
# batchCount  [int]
#              numbers of matrices in the batch
cdef hipblasStatus_t hipblasStrtriStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const float * AP,int lda,long strideA,float * invA,int ldinvA,long stride_invA,int batchCount) nogil:
    global _hipblasStrtriStridedBatched__funptr
    __init_symbol(&_hipblasStrtriStridedBatched__funptr,"hipblasStrtriStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasDiagType_t,int,const float *,int,long,float *,int,long,int) nogil> _hipblasStrtriStridedBatched__funptr)(handle,uplo,diag,n,AP,lda,strideA,invA,ldinvA,stride_invA,batchCount)


cdef void* _hipblasDtrtriStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDtrtriStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const double * AP,int lda,long strideA,double * invA,int ldinvA,long stride_invA,int batchCount) nogil:
    global _hipblasDtrtriStridedBatched__funptr
    __init_symbol(&_hipblasDtrtriStridedBatched__funptr,"hipblasDtrtriStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasDiagType_t,int,const double *,int,long,double *,int,long,int) nogil> _hipblasDtrtriStridedBatched__funptr)(handle,uplo,diag,n,AP,lda,strideA,invA,ldinvA,stride_invA,batchCount)


cdef void* _hipblasCtrtriStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCtrtriStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasComplex * AP,int lda,long strideA,hipblasComplex * invA,int ldinvA,long stride_invA,int batchCount) nogil:
    global _hipblasCtrtriStridedBatched__funptr
    __init_symbol(&_hipblasCtrtriStridedBatched__funptr,"hipblasCtrtriStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasDiagType_t,int,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCtrtriStridedBatched__funptr)(handle,uplo,diag,n,AP,lda,strideA,invA,ldinvA,stride_invA,batchCount)


cdef void* _hipblasZtrtriStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZtrtriStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * invA,int ldinvA,long stride_invA,int batchCount) nogil:
    global _hipblasZtrtriStridedBatched__funptr
    __init_symbol(&_hipblasZtrtriStridedBatched__funptr,"hipblasZtrtriStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasFillMode_t,hipblasDiagType_t,int,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZtrtriStridedBatched__funptr)(handle,uplo,diag,n,AP,lda,strideA,invA,ldinvA,stride_invA,batchCount)


cdef void* _hipblasSdgmm__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# dgmm performs one of the matrix-matrix operations
# 
#     C = A * diag(x) if side == HIPBLAS_SIDE_RIGHT
#     C = diag(x) * A if side == HIPBLAS_SIDE_LEFT
# 
# where C and A are m by n dimensional matrices. diag( x ) is a diagonal matrix
# and x is vector of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension m
# if side == HIPBLAS_SIDE_LEFT.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# side      [hipblasSideMode_t]
#           specifies the side of diag(x)
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# AP         device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment between values of x
# @param[in, out]
# CP         device pointer storing matrix C.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
#
cdef hipblasStatus_t hipblasSdgmm(void * handle,hipblasSideMode_t side,int m,int n,const float * AP,int lda,const float * x,int incx,float * CP,int ldc) nogil:
    global _hipblasSdgmm__funptr
    __init_symbol(&_hipblasSdgmm__funptr,"hipblasSdgmm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,int,int,const float *,int,const float *,int,float *,int) nogil> _hipblasSdgmm__funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc)


cdef void* _hipblasDdgmm__funptr = NULL
cdef hipblasStatus_t hipblasDdgmm(void * handle,hipblasSideMode_t side,int m,int n,const double * AP,int lda,const double * x,int incx,double * CP,int ldc) nogil:
    global _hipblasDdgmm__funptr
    __init_symbol(&_hipblasDdgmm__funptr,"hipblasDdgmm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,int,int,const double *,int,const double *,int,double *,int) nogil> _hipblasDdgmm__funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc)


cdef void* _hipblasCdgmm__funptr = NULL
cdef hipblasStatus_t hipblasCdgmm(void * handle,hipblasSideMode_t side,int m,int n,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * CP,int ldc) nogil:
    global _hipblasCdgmm__funptr
    __init_symbol(&_hipblasCdgmm__funptr,"hipblasCdgmm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,int,int,hipblasComplex *,int,hipblasComplex *,int,hipblasComplex *,int) nogil> _hipblasCdgmm__funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc)


cdef void* _hipblasZdgmm__funptr = NULL
cdef hipblasStatus_t hipblasZdgmm(void * handle,hipblasSideMode_t side,int m,int n,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * CP,int ldc) nogil:
    global _hipblasZdgmm__funptr
    __init_symbol(&_hipblasZdgmm__funptr,"hipblasZdgmm")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,int,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int,hipblasDoubleComplex *,int) nogil> _hipblasZdgmm__funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc)


cdef void* _hipblasSdgmmBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# dgmmBatched performs one of the batched matrix-matrix operations
# 
#     C_i = A_i * diag(x_i) for i = 0, 1, ... batchCount-1 if side == HIPBLAS_SIDE_RIGHT
#     C_i = diag(x_i) * A_i for i = 0, 1, ... batchCount-1 if side == HIPBLAS_SIDE_LEFT
# 
# where C_i and A_i are m by n dimensional matrices. diag(x_i) is a diagonal matrix
# and x_i is vector of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension m
# if side == HIPBLAS_SIDE_LEFT.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# side      [hipblasSideMode_t]
#           specifies the side of diag(x)
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# AP         device array of device pointers storing each matrix A_i on the GPU.
#           Each A_i is of dimension ( lda, n )
# @param[in]
# lda       [int]
#           specifies the leading dimension of A_i.
# @param[in]
# x         device array of device pointers storing each vector x_i on the GPU.
#           Each x_i is of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension
#           m if side == HIPBLAS_SIDE_LEFT
# @param[in]
# incx      [int]
#           specifies the increment between values of x_i
# @param[in, out]
# CP         device array of device pointers storing each matrix C_i on the GPU.
#           Each C_i is of dimension ( ldc, n ).
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#
cdef hipblasStatus_t hipblasSdgmmBatched(void * handle,hipblasSideMode_t side,int m,int n,const float *const* AP,int lda,const float *const* x,int incx,float *const* CP,int ldc,int batchCount) nogil:
    global _hipblasSdgmmBatched__funptr
    __init_symbol(&_hipblasSdgmmBatched__funptr,"hipblasSdgmmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,int,int,const float *const*,int,const float *const*,int,float *const*,int,int) nogil> _hipblasSdgmmBatched__funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc,batchCount)


cdef void* _hipblasDdgmmBatched__funptr = NULL
cdef hipblasStatus_t hipblasDdgmmBatched(void * handle,hipblasSideMode_t side,int m,int n,const double *const* AP,int lda,const double *const* x,int incx,double *const* CP,int ldc,int batchCount) nogil:
    global _hipblasDdgmmBatched__funptr
    __init_symbol(&_hipblasDdgmmBatched__funptr,"hipblasDdgmmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,int,int,const double *const*,int,const double *const*,int,double *const*,int,int) nogil> _hipblasDdgmmBatched__funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc,batchCount)


cdef void* _hipblasCdgmmBatched__funptr = NULL
cdef hipblasStatus_t hipblasCdgmmBatched(void * handle,hipblasSideMode_t side,int m,int n,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,hipblasComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasCdgmmBatched__funptr
    __init_symbol(&_hipblasCdgmmBatched__funptr,"hipblasCdgmmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,int,int,hipblasComplex *const*,int,hipblasComplex *const*,int,hipblasComplex *const*,int,int) nogil> _hipblasCdgmmBatched__funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc,batchCount)


cdef void* _hipblasZdgmmBatched__funptr = NULL
cdef hipblasStatus_t hipblasZdgmmBatched(void * handle,hipblasSideMode_t side,int m,int n,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil:
    global _hipblasZdgmmBatched__funptr
    __init_symbol(&_hipblasZdgmmBatched__funptr,"hipblasZdgmmBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,int,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,hipblasDoubleComplex *const*,int,int) nogil> _hipblasZdgmmBatched__funptr)(handle,side,m,n,AP,lda,x,incx,CP,ldc,batchCount)


cdef void* _hipblasSdgmmStridedBatched__funptr = NULL
# @{
# \brief BLAS Level 3 API
# 
# \details
# dgmmStridedBatched performs one of the batched matrix-matrix operations
# 
#     C_i = A_i * diag(x_i)   if side == HIPBLAS_SIDE_RIGHT   for i = 0, 1, ... batchCount-1
#     C_i = diag(x_i) * A_i   if side == HIPBLAS_SIDE_LEFT    for i = 0, 1, ... batchCount-1
# 
# where C_i and A_i are m by n dimensional matrices. diag(x_i) is a diagonal matrix
# and x_i is vector of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension m
# if side == HIPBLAS_SIDE_LEFT.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# side      [hipblasSideMode_t]
#           specifies the side of diag(x)
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# AP         device pointer to the first matrix A_0 on the GPU.
#           Each A_i is of dimension ( lda, n )
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x         pointer to the first vector x_0 on the GPU.
#           Each x_i is of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension
#           m if side == HIPBLAS_SIDE_LEFT
# @param[in]
# incx      [int]
#           specifies the increment between values of x
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector(x_i) and the next one (x_i+1)
# @param[in, out]
# CP         device pointer to the first matrix C_0 on the GPU.
#           Each C_i is of dimension ( ldc, n ).
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
# @param[in]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# @param[in]
# batchCount [int]
#             number of instances i in the batch.
#
cdef hipblasStatus_t hipblasSdgmmStridedBatched(void * handle,hipblasSideMode_t side,int m,int n,const float * AP,int lda,long strideA,const float * x,int incx,long stridex,float * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasSdgmmStridedBatched__funptr
    __init_symbol(&_hipblasSdgmmStridedBatched__funptr,"hipblasSdgmmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,int,int,const float *,int,long,const float *,int,long,float *,int,long,int) nogil> _hipblasSdgmmStridedBatched__funptr)(handle,side,m,n,AP,lda,strideA,x,incx,stridex,CP,ldc,strideC,batchCount)


cdef void* _hipblasDdgmmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDdgmmStridedBatched(void * handle,hipblasSideMode_t side,int m,int n,const double * AP,int lda,long strideA,const double * x,int incx,long stridex,double * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasDdgmmStridedBatched__funptr
    __init_symbol(&_hipblasDdgmmStridedBatched__funptr,"hipblasDdgmmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,int,int,const double *,int,long,const double *,int,long,double *,int,long,int) nogil> _hipblasDdgmmStridedBatched__funptr)(handle,side,m,n,AP,lda,strideA,x,incx,stridex,CP,ldc,strideC,batchCount)


cdef void* _hipblasCdgmmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCdgmmStridedBatched(void * handle,hipblasSideMode_t side,int m,int n,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasCdgmmStridedBatched__funptr
    __init_symbol(&_hipblasCdgmmStridedBatched__funptr,"hipblasCdgmmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,int,int,hipblasComplex *,int,long,hipblasComplex *,int,long,hipblasComplex *,int,long,int) nogil> _hipblasCdgmmStridedBatched__funptr)(handle,side,m,n,AP,lda,strideA,x,incx,stridex,CP,ldc,strideC,batchCount)


cdef void* _hipblasZdgmmStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZdgmmStridedBatched(void * handle,hipblasSideMode_t side,int m,int n,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil:
    global _hipblasZdgmmStridedBatched__funptr
    __init_symbol(&_hipblasZdgmmStridedBatched__funptr,"hipblasZdgmmStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,int,int,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,hipblasDoubleComplex *,int,long,int) nogil> _hipblasZdgmmStridedBatched__funptr)(handle,side,m,n,AP,lda,strideA,x,incx,stridex,CP,ldc,strideC,batchCount)


cdef void* _hipblasSgetrf__funptr = NULL
# @{
# \brief SOLVER API
# 
# \details
# getrf computes the LU factorization of a general n-by-n matrix A
# using partial pivoting with row interchanges. The LU factorization can
# be done without pivoting if ipiv is passed as a nullptr.
# 
# In the case that ipiv is not null, the factorization has the form:
# 
# \f[
#     A = PLU
# \f]
# 
# where P is a permutation matrix, L is lower triangular with unit
# diagonal elements, and U is upper triangular.
# 
# In the case that ipiv is null, the factorization is done without pivoting:
# 
# \f[
#     A = LU
# \f]
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# n         int. n >= 0.\n
#           The number of columns and rows of the matrix A.
# @param[inout]
# A         pointer to type. Array on the GPU of dimension lda*n.\n
#           On entry, the n-by-n matrix A to be factored.
#           On exit, the factors L and U from the factorization.
#           The unit diagonal elements of L are not stored.
# @param[in]
# lda       int. lda >= n.\n
#           Specifies the leading dimension of A.
# @param[out]
# ipiv      pointer to int. Array on the GPU of dimension n.\n
#           The vector of pivot indices. Elements of ipiv are 1-based indices.
#           For 1 <= i <= n, the row i of the
#           matrix was interchanged with row ipiv[i].
#           Matrix P of the factorization can be derived from ipiv.
#           The factorization here can be done without pivoting if ipiv is passed
#           in as a nullptr.
# @param[out]
# info      pointer to a int on the GPU.\n
#           If info = 0, successful exit.
#           If info = j > 0, U is singular. U[j,j] is the first zero pivot.
cdef hipblasStatus_t hipblasSgetrf(void * handle,const int n,float * A,const int lda,int * ipiv,int * info) nogil:
    global _hipblasSgetrf__funptr
    __init_symbol(&_hipblasSgetrf__funptr,"hipblasSgetrf")
    return (<hipblasStatus_t (*)(void *,const int,float *,const int,int *,int *) nogil> _hipblasSgetrf__funptr)(handle,n,A,lda,ipiv,info)


cdef void* _hipblasDgetrf__funptr = NULL
cdef hipblasStatus_t hipblasDgetrf(void * handle,const int n,double * A,const int lda,int * ipiv,int * info) nogil:
    global _hipblasDgetrf__funptr
    __init_symbol(&_hipblasDgetrf__funptr,"hipblasDgetrf")
    return (<hipblasStatus_t (*)(void *,const int,double *,const int,int *,int *) nogil> _hipblasDgetrf__funptr)(handle,n,A,lda,ipiv,info)


cdef void* _hipblasCgetrf__funptr = NULL
cdef hipblasStatus_t hipblasCgetrf(void * handle,const int n,hipblasComplex * A,const int lda,int * ipiv,int * info) nogil:
    global _hipblasCgetrf__funptr
    __init_symbol(&_hipblasCgetrf__funptr,"hipblasCgetrf")
    return (<hipblasStatus_t (*)(void *,const int,hipblasComplex *,const int,int *,int *) nogil> _hipblasCgetrf__funptr)(handle,n,A,lda,ipiv,info)


cdef void* _hipblasZgetrf__funptr = NULL
cdef hipblasStatus_t hipblasZgetrf(void * handle,const int n,hipblasDoubleComplex * A,const int lda,int * ipiv,int * info) nogil:
    global _hipblasZgetrf__funptr
    __init_symbol(&_hipblasZgetrf__funptr,"hipblasZgetrf")
    return (<hipblasStatus_t (*)(void *,const int,hipblasDoubleComplex *,const int,int *,int *) nogil> _hipblasZgetrf__funptr)(handle,n,A,lda,ipiv,info)


cdef void* _hipblasSgetrfBatched__funptr = NULL
# @{
# \brief SOLVER API
# 
# \details
# getrfBatched computes the LU factorization of a batch of general
# n-by-n matrices using partial pivoting with row interchanges. The LU factorization can
# be done without pivoting if ipiv is passed as a nullptr.
# 
# In the case that ipiv is not null, the factorization of matrix \f$A_i\f$ in the batch has the form:
# 
# \f[
#     A_i = P_iL_iU_i
# \f]
# 
# where \f$P_i\f$ is a permutation matrix, \f$L_i\f$ is lower triangular with unit
# diagonal elements, and \f$U_i\f$ is upper triangular.
# 
# In the case that ipiv is null, the factorization is done without pivoting:
# 
# \f[
#     A_i = L_iU_i
# \f]
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# n         int. n >= 0.\n
#           The number of columns and rows of all matrices A_i in the batch.
# @param[inout]
# A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
#           On entry, the n-by-n matrices A_i to be factored.
#           On exit, the factors L_i and U_i from the factorizations.
#           The unit diagonal elements of L_i are not stored.
# @param[in]
# lda       int. lda >= n.\n
#           Specifies the leading dimension of matrices A_i.
# @param[out]
# ipiv      pointer to int. Array on the GPU.\n
#           Contains the vectors of pivot indices ipiv_i (corresponding to A_i).
#           Dimension of ipiv_i is n.
#           Elements of ipiv_i are 1-based indices.
#           For each instance A_i in the batch and for 1 <= j <= n, the row j of the
#           matrix A_i was interchanged with row ipiv_i[j].
#           Matrix P_i of the factorization can be derived from ipiv_i.
#           The factorization here can be done without pivoting if ipiv is passed
#           in as a nullptr.
# @param[out]
# info      pointer to int. Array of batchCount integers on the GPU.\n
#           If info[i] = 0, successful exit for factorization of A_i.
#           If info[i] = j > 0, U_i is singular. U_i[j,j] is the first zero pivot.
# @param[in]
# batchCount int. batchCount >= 0.\n
#             Number of matrices in the batch.
cdef hipblasStatus_t hipblasSgetrfBatched(void * handle,const int n,float *const* A,const int lda,int * ipiv,int * info,const int batchCount) nogil:
    global _hipblasSgetrfBatched__funptr
    __init_symbol(&_hipblasSgetrfBatched__funptr,"hipblasSgetrfBatched")
    return (<hipblasStatus_t (*)(void *,const int,float *const*,const int,int *,int *,const int) nogil> _hipblasSgetrfBatched__funptr)(handle,n,A,lda,ipiv,info,batchCount)


cdef void* _hipblasDgetrfBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgetrfBatched(void * handle,const int n,double *const* A,const int lda,int * ipiv,int * info,const int batchCount) nogil:
    global _hipblasDgetrfBatched__funptr
    __init_symbol(&_hipblasDgetrfBatched__funptr,"hipblasDgetrfBatched")
    return (<hipblasStatus_t (*)(void *,const int,double *const*,const int,int *,int *,const int) nogil> _hipblasDgetrfBatched__funptr)(handle,n,A,lda,ipiv,info,batchCount)


cdef void* _hipblasCgetrfBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgetrfBatched(void * handle,const int n,hipblasComplex *const* A,const int lda,int * ipiv,int * info,const int batchCount) nogil:
    global _hipblasCgetrfBatched__funptr
    __init_symbol(&_hipblasCgetrfBatched__funptr,"hipblasCgetrfBatched")
    return (<hipblasStatus_t (*)(void *,const int,hipblasComplex *const*,const int,int *,int *,const int) nogil> _hipblasCgetrfBatched__funptr)(handle,n,A,lda,ipiv,info,batchCount)


cdef void* _hipblasZgetrfBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgetrfBatched(void * handle,const int n,hipblasDoubleComplex *const* A,const int lda,int * ipiv,int * info,const int batchCount) nogil:
    global _hipblasZgetrfBatched__funptr
    __init_symbol(&_hipblasZgetrfBatched__funptr,"hipblasZgetrfBatched")
    return (<hipblasStatus_t (*)(void *,const int,hipblasDoubleComplex *const*,const int,int *,int *,const int) nogil> _hipblasZgetrfBatched__funptr)(handle,n,A,lda,ipiv,info,batchCount)


cdef void* _hipblasSgetrfStridedBatched__funptr = NULL
# @{
# \brief SOLVER API
# 
# \details
# getrfStridedBatched computes the LU factorization of a batch of
# general n-by-n matrices using partial pivoting with row interchanges. The LU factorization can
# be done without pivoting if ipiv is passed as a nullptr.
# 
# In the case that ipiv is not null, the factorization of matrix \f$A_i\f$ in the batch has the form:
# 
# \f[
#     A_i = P_iL_iU_i
# \f]
# 
# where \f$P_i\f$ is a permutation matrix, \f$L_i\f$ is lower triangular with unit
# diagonal elements, and \f$U_i\f$ is upper triangular.
# 
# In the case that ipiv is null, the factorization is done without pivoting:
# 
# \f[
#     A_i = L_iU_i
# \f]
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# n         int. n >= 0.\n
#           The number of columns and rows of all matrices A_i in the batch.
# @param[inout]
# A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
#           On entry, the n-by-n matrices A_i to be factored.
#           On exit, the factors L_i and U_i from the factorization.
#           The unit diagonal elements of L_i are not stored.
# @param[in]
# lda       int. lda >= n.\n
#           Specifies the leading dimension of matrices A_i.
# @param[in]
# strideA   hipblasStride.\n
#           Stride from the start of one matrix A_i to the next one A_(i+1).
#           There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
# @param[out]
# ipiv      pointer to int. Array on the GPU (the size depends on the value of strideP).\n
#           Contains the vectors of pivots indices ipiv_i (corresponding to A_i).
#           Dimension of ipiv_i is n.
#           Elements of ipiv_i are 1-based indices.
#           For each instance A_i in the batch and for 1 <= j <= n, the row j of the
#           matrix A_i was interchanged with row ipiv_i[j].
#           Matrix P_i of the factorization can be derived from ipiv_i.
#           The factorization here can be done without pivoting if ipiv is passed
#           in as a nullptr.
# @param[in]
# strideP   hipblasStride.\n
#           Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
#           There is no restriction for the value of strideP. Normal use case is strideP >= n.
# @param[out]
# info      pointer to int. Array of batchCount integers on the GPU.\n
#           If info[i] = 0, successful exit for factorization of A_i.
#           If info[i] = j > 0, U_i is singular. U_i[j,j] is the first zero pivot.
# @param[in]
# batchCount int. batchCount >= 0.\n
#             Number of matrices in the batch.
cdef hipblasStatus_t hipblasSgetrfStridedBatched(void * handle,const int n,float * A,const int lda,const long strideA,int * ipiv,const long strideP,int * info,const int batchCount) nogil:
    global _hipblasSgetrfStridedBatched__funptr
    __init_symbol(&_hipblasSgetrfStridedBatched__funptr,"hipblasSgetrfStridedBatched")
    return (<hipblasStatus_t (*)(void *,const int,float *,const int,const long,int *,const long,int *,const int) nogil> _hipblasSgetrfStridedBatched__funptr)(handle,n,A,lda,strideA,ipiv,strideP,info,batchCount)


cdef void* _hipblasDgetrfStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgetrfStridedBatched(void * handle,const int n,double * A,const int lda,const long strideA,int * ipiv,const long strideP,int * info,const int batchCount) nogil:
    global _hipblasDgetrfStridedBatched__funptr
    __init_symbol(&_hipblasDgetrfStridedBatched__funptr,"hipblasDgetrfStridedBatched")
    return (<hipblasStatus_t (*)(void *,const int,double *,const int,const long,int *,const long,int *,const int) nogil> _hipblasDgetrfStridedBatched__funptr)(handle,n,A,lda,strideA,ipiv,strideP,info,batchCount)


cdef void* _hipblasCgetrfStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgetrfStridedBatched(void * handle,const int n,hipblasComplex * A,const int lda,const long strideA,int * ipiv,const long strideP,int * info,const int batchCount) nogil:
    global _hipblasCgetrfStridedBatched__funptr
    __init_symbol(&_hipblasCgetrfStridedBatched__funptr,"hipblasCgetrfStridedBatched")
    return (<hipblasStatus_t (*)(void *,const int,hipblasComplex *,const int,const long,int *,const long,int *,const int) nogil> _hipblasCgetrfStridedBatched__funptr)(handle,n,A,lda,strideA,ipiv,strideP,info,batchCount)


cdef void* _hipblasZgetrfStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgetrfStridedBatched(void * handle,const int n,hipblasDoubleComplex * A,const int lda,const long strideA,int * ipiv,const long strideP,int * info,const int batchCount) nogil:
    global _hipblasZgetrfStridedBatched__funptr
    __init_symbol(&_hipblasZgetrfStridedBatched__funptr,"hipblasZgetrfStridedBatched")
    return (<hipblasStatus_t (*)(void *,const int,hipblasDoubleComplex *,const int,const long,int *,const long,int *,const int) nogil> _hipblasZgetrfStridedBatched__funptr)(handle,n,A,lda,strideA,ipiv,strideP,info,batchCount)


cdef void* _hipblasSgetrs__funptr = NULL
# @{
# \brief SOLVER API
# 
# \details
# getrs solves a system of n linear equations on n variables in its factorized form.
# 
# It solves one of the following systems, depending on the value of trans:
# 
# \f[
#     \begin{array}{cl}
#     A X = B & \: \text{not transposed,}\\
#     A^T X = B & \: \text{transposed, or}\\
#     A^H X = B & \: \text{conjugate transposed.}
#     \end{array}
# \f]
# 
# Matrix A is defined by its triangular factors as returned by \ref hipblasSgetrf "getrf".
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# 
# @param[in]
# handle      hipblasHandle_t.
# @param[in]
# trans       hipblasOperation_t.\n
#             Specifies the form of the system of equations.
# @param[in]
# n           int. n >= 0.\n
#             The order of the system, i.e. the number of columns and rows of A.
# @param[in]
# nrhs        int. nrhs >= 0.\n
#             The number of right hand sides, i.e., the number of columns
#             of the matrix B.
# @param[in]
# A           pointer to type. Array on the GPU of dimension lda*n.\n
#             The factors L and U of the factorization A = P*L*U returned by \ref hipblasSgetrf "getrf".
# @param[in]
# lda         int. lda >= n.\n
#             The leading dimension of A.
# @param[in]
# ipiv        pointer to int. Array on the GPU of dimension n.\n
#             The pivot indices returned by \ref hipblasSgetrf "getrf".
# @param[in,out]
# B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
#             On entry, the right hand side matrix B.
#             On exit, the solution matrix X.
# @param[in]
# ldb         int. ldb >= n.\n
#             The leading dimension of B.
# @param[out]
# info      pointer to a int on the host.\n
#           If info = 0, successful exit.
#           If info = j < 0, the j-th argument is invalid.
cdef hipblasStatus_t hipblasSgetrs(void * handle,hipblasOperation_t trans,const int n,const int nrhs,float * A,const int lda,const int * ipiv,float * B,const int ldb,int * info) nogil:
    global _hipblasSgetrs__funptr
    __init_symbol(&_hipblasSgetrs__funptr,"hipblasSgetrs")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,float *,const int,const int *,float *,const int,int *) nogil> _hipblasSgetrs__funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info)


cdef void* _hipblasDgetrs__funptr = NULL
cdef hipblasStatus_t hipblasDgetrs(void * handle,hipblasOperation_t trans,const int n,const int nrhs,double * A,const int lda,const int * ipiv,double * B,const int ldb,int * info) nogil:
    global _hipblasDgetrs__funptr
    __init_symbol(&_hipblasDgetrs__funptr,"hipblasDgetrs")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,double *,const int,const int *,double *,const int,int *) nogil> _hipblasDgetrs__funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info)


cdef void* _hipblasCgetrs__funptr = NULL
cdef hipblasStatus_t hipblasCgetrs(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasComplex * A,const int lda,const int * ipiv,hipblasComplex * B,const int ldb,int * info) nogil:
    global _hipblasCgetrs__funptr
    __init_symbol(&_hipblasCgetrs__funptr,"hipblasCgetrs")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,hipblasComplex *,const int,const int *,hipblasComplex *,const int,int *) nogil> _hipblasCgetrs__funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info)


cdef void* _hipblasZgetrs__funptr = NULL
cdef hipblasStatus_t hipblasZgetrs(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,const int * ipiv,hipblasDoubleComplex * B,const int ldb,int * info) nogil:
    global _hipblasZgetrs__funptr
    __init_symbol(&_hipblasZgetrs__funptr,"hipblasZgetrs")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,hipblasDoubleComplex *,const int,const int *,hipblasDoubleComplex *,const int,int *) nogil> _hipblasZgetrs__funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info)


cdef void* _hipblasSgetrsBatched__funptr = NULL
# @{
# \brief SOLVER API
# 
# \details getrsBatched solves a batch of systems of n linear equations on n
# variables in its factorized forms.
# 
# For each instance i in the batch, it solves one of the following systems, depending on the value of trans:
# 
# \f[
#     \begin{array}{cl}
#     A_i X_i = B_i & \: \text{not transposed,}\\
#     A_i^T X_i = B_i & \: \text{transposed, or}\\
#     A_i^H X_i = B_i & \: \text{conjugate transposed.}
#     \end{array}
# \f]
# 
# Matrix \f$A_i\f$ is defined by its triangular factors as returned by \ref hipblasSgetrfBatched "getrfBatched".
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle      hipblasHandle_t.
# @param[in]
# trans       hipblasOperation_t.\n
#             Specifies the form of the system of equations of each instance in the batch.
# @param[in]
# n           int. n >= 0.\n
#             The order of the system, i.e. the number of columns and rows of all A_i matrices.
# @param[in]
# nrhs        int. nrhs >= 0.\n
#             The number of right hand sides, i.e., the number of columns
#             of all the matrices B_i.
# @param[in]
# A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
#             The factors L_i and U_i of the factorization A_i = P_i*L_i*U_i returned by \ref hipblasSgetrfBatched "getrfBatched".
# @param[in]
# lda         int. lda >= n.\n
#             The leading dimension of matrices A_i.
# @param[in]
# ipiv        pointer to int. Array on the GPU.\n
#             Contains the vectors ipiv_i of pivot indices returned by \ref hipblasSgetrfBatched "getrfBatched".
# @param[in,out]
# B           Array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*nrhs.\n
#             On entry, the right hand side matrices B_i.
#             On exit, the solution matrix X_i of each system in the batch.
# @param[in]
# ldb         int. ldb >= n.\n
#             The leading dimension of matrices B_i.
# @param[out]
# info      pointer to a int on the host.\n
#           If info = 0, successful exit.
#           If info = j < 0, the j-th argument is invalid.
# @param[in]
# batchCount int. batchCount >= 0.\n
#             Number of instances (systems) in the batch.
#
cdef hipblasStatus_t hipblasSgetrsBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,float *const* A,const int lda,const int * ipiv,float *const* B,const int ldb,int * info,const int batchCount) nogil:
    global _hipblasSgetrsBatched__funptr
    __init_symbol(&_hipblasSgetrsBatched__funptr,"hipblasSgetrsBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,float *const*,const int,const int *,float *const*,const int,int *,const int) nogil> _hipblasSgetrsBatched__funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info,batchCount)


cdef void* _hipblasDgetrsBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgetrsBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,double *const* A,const int lda,const int * ipiv,double *const* B,const int ldb,int * info,const int batchCount) nogil:
    global _hipblasDgetrsBatched__funptr
    __init_symbol(&_hipblasDgetrsBatched__funptr,"hipblasDgetrsBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,double *const*,const int,const int *,double *const*,const int,int *,const int) nogil> _hipblasDgetrsBatched__funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info,batchCount)


cdef void* _hipblasCgetrsBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgetrsBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasComplex *const* A,const int lda,const int * ipiv,hipblasComplex *const* B,const int ldb,int * info,const int batchCount) nogil:
    global _hipblasCgetrsBatched__funptr
    __init_symbol(&_hipblasCgetrsBatched__funptr,"hipblasCgetrsBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,hipblasComplex *const*,const int,const int *,hipblasComplex *const*,const int,int *,const int) nogil> _hipblasCgetrsBatched__funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info,batchCount)


cdef void* _hipblasZgetrsBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgetrsBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasDoubleComplex *const* A,const int lda,const int * ipiv,hipblasDoubleComplex *const* B,const int ldb,int * info,const int batchCount) nogil:
    global _hipblasZgetrsBatched__funptr
    __init_symbol(&_hipblasZgetrsBatched__funptr,"hipblasZgetrsBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,hipblasDoubleComplex *const*,const int,const int *,hipblasDoubleComplex *const*,const int,int *,const int) nogil> _hipblasZgetrsBatched__funptr)(handle,trans,n,nrhs,A,lda,ipiv,B,ldb,info,batchCount)


cdef void* _hipblasSgetrsStridedBatched__funptr = NULL
# @{
# \brief SOLVER API
# 
# \details
# getrsStridedBatched solves a batch of systems of n linear equations
# on n variables in its factorized forms.
# 
# For each instance i in the batch, it solves one of the following systems, depending on the value of trans:
# 
# \f[
#     \begin{array}{cl}
#     A_i X_i = B_i & \: \text{not transposed,}\\
#     A_i^T X_i = B_i & \: \text{transposed, or}\\
#     A_i^H X_i = B_i & \: \text{conjugate transposed.}
#     \end{array}
# \f]
# 
# Matrix \f$A_i\f$ is defined by its triangular factors as returned by \ref hipblasSgetrfStridedBatched "getrfStridedBatched".
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : No support
# 
# @param[in]
# handle      hipblasHandle_t.
# @param[in]
# trans       hipblasOperation_t.\n
#             Specifies the form of the system of equations of each instance in the batch.
# @param[in]
# n           int. n >= 0.\n
#             The order of the system, i.e. the number of columns and rows of all A_i matrices.
# @param[in]
# nrhs        int. nrhs >= 0.\n
#             The number of right hand sides, i.e., the number of columns
#             of all the matrices B_i.
# @param[in]
# A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
#             The factors L_i and U_i of the factorization A_i = P_i*L_i*U_i returned by \ref hipblasSgetrfStridedBatched "getrfStridedBatched".
# @param[in]
# lda         int. lda >= n.\n
#             The leading dimension of matrices A_i.
# @param[in]
# strideA     hipblasStride.\n
#             Stride from the start of one matrix A_i to the next one A_(i+1).
#             There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
# @param[in]
# ipiv        pointer to int. Array on the GPU (the size depends on the value of strideP).\n
#             Contains the vectors ipiv_i of pivot indices returned by \ref hipblasSgetrfStridedBatched "getrfStridedBatched".
# @param[in]
# strideP     hipblasStride.\n
#             Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
#             There is no restriction for the value of strideP. Normal use case is strideP >= n.
# @param[in,out]
# B           pointer to type. Array on the GPU (size depends on the value of strideB).\n
#             On entry, the right hand side matrices B_i.
#             On exit, the solution matrix X_i of each system in the batch.
# @param[in]
# ldb         int. ldb >= n.\n
#             The leading dimension of matrices B_i.
# @param[in]
# strideB     hipblasStride.\n
#             Stride from the start of one matrix B_i to the next one B_(i+1).
#             There is no restriction for the value of strideB. Normal use case is strideB >= ldb*nrhs.
# @param[out]
# info      pointer to a int on the host.\n
#           If info = 0, successful exit.
#           If info = j < 0, the j-th argument is invalid.
# @param[in]
# batchCount int. batchCount >= 0.\n
#             Number of instances (systems) in the batch.
#
cdef hipblasStatus_t hipblasSgetrsStridedBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,float * A,const int lda,const long strideA,const int * ipiv,const long strideP,float * B,const int ldb,const long strideB,int * info,const int batchCount) nogil:
    global _hipblasSgetrsStridedBatched__funptr
    __init_symbol(&_hipblasSgetrsStridedBatched__funptr,"hipblasSgetrsStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,float *,const int,const long,const int *,const long,float *,const int,const long,int *,const int) nogil> _hipblasSgetrsStridedBatched__funptr)(handle,trans,n,nrhs,A,lda,strideA,ipiv,strideP,B,ldb,strideB,info,batchCount)


cdef void* _hipblasDgetrsStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgetrsStridedBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,double * A,const int lda,const long strideA,const int * ipiv,const long strideP,double * B,const int ldb,const long strideB,int * info,const int batchCount) nogil:
    global _hipblasDgetrsStridedBatched__funptr
    __init_symbol(&_hipblasDgetrsStridedBatched__funptr,"hipblasDgetrsStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,double *,const int,const long,const int *,const long,double *,const int,const long,int *,const int) nogil> _hipblasDgetrsStridedBatched__funptr)(handle,trans,n,nrhs,A,lda,strideA,ipiv,strideP,B,ldb,strideB,info,batchCount)


cdef void* _hipblasCgetrsStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgetrsStridedBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasComplex * A,const int lda,const long strideA,const int * ipiv,const long strideP,hipblasComplex * B,const int ldb,const long strideB,int * info,const int batchCount) nogil:
    global _hipblasCgetrsStridedBatched__funptr
    __init_symbol(&_hipblasCgetrsStridedBatched__funptr,"hipblasCgetrsStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,hipblasComplex *,const int,const long,const int *,const long,hipblasComplex *,const int,const long,int *,const int) nogil> _hipblasCgetrsStridedBatched__funptr)(handle,trans,n,nrhs,A,lda,strideA,ipiv,strideP,B,ldb,strideB,info,batchCount)


cdef void* _hipblasZgetrsStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgetrsStridedBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,const long strideA,const int * ipiv,const long strideP,hipblasDoubleComplex * B,const int ldb,const long strideB,int * info,const int batchCount) nogil:
    global _hipblasZgetrsStridedBatched__funptr
    __init_symbol(&_hipblasZgetrsStridedBatched__funptr,"hipblasZgetrsStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,hipblasDoubleComplex *,const int,const long,const int *,const long,hipblasDoubleComplex *,const int,const long,int *,const int) nogil> _hipblasZgetrsStridedBatched__funptr)(handle,trans,n,nrhs,A,lda,strideA,ipiv,strideP,B,ldb,strideB,info,batchCount)


cdef void* _hipblasSgetriBatched__funptr = NULL
# @{
# \brief SOLVER API
# 
# \details
# getriBatched computes the inverse \f$C_i = A_i^{-1}\f$ of a batch of general n-by-n matrices \f$A_i\f$.
# 
# The inverse is computed by solving the linear system
# 
# \f[
#     A_i C_i = I
# \f]
# 
# where I is the identity matrix, and \f$A_i\f$ is factorized as \f$A_i = P_i  L_i  U_i\f$ as given by \ref hipblasSgetrfBatched "getrfBatched".
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# n         int. n >= 0.\n
#           The number of rows and columns of all matrices A_i in the batch.
# @param[in]
# A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
#           The factors L_i and U_i of the factorization A_i = P_i*L_i*U_i returned by \ref hipblasSgetrfBatched "getrfBatched".
# @param[in]
# lda       int. lda >= n.\n
#           Specifies the leading dimension of matrices A_i.
# @param[in]
# ipiv      pointer to int. Array on the GPU (the size depends on the value of strideP).\n
#           The pivot indices returned by \ref hipblasSgetrfBatched "getrfBatched".
#           ipiv can be passed in as a nullptr, this will assume that getrfBatched was called without partial pivoting.
# @param[out]
# C         array of pointers to type. Each pointer points to an array on the GPU of dimension ldc*n.\n
#           If info[i] = 0, the inverse of matrices A_i. Otherwise, undefined.
# @param[in]
# ldc       int. ldc >= n.\n
#           Specifies the leading dimension of C_i.
# @param[out]
# info      pointer to int. Array of batchCount integers on the GPU.\n
#           If info[i] = 0, successful exit for inversion of A_i.
#           If info[i] = j > 0, U_i is singular. U_i[j,j] is the first zero pivot.
# @param[in]
# batchCount int. batchCount >= 0.\n
#             Number of matrices in the batch.
#
cdef hipblasStatus_t hipblasSgetriBatched(void * handle,const int n,float *const* A,const int lda,int * ipiv,float *const* C,const int ldc,int * info,const int batchCount) nogil:
    global _hipblasSgetriBatched__funptr
    __init_symbol(&_hipblasSgetriBatched__funptr,"hipblasSgetriBatched")
    return (<hipblasStatus_t (*)(void *,const int,float *const*,const int,int *,float *const*,const int,int *,const int) nogil> _hipblasSgetriBatched__funptr)(handle,n,A,lda,ipiv,C,ldc,info,batchCount)


cdef void* _hipblasDgetriBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgetriBatched(void * handle,const int n,double *const* A,const int lda,int * ipiv,double *const* C,const int ldc,int * info,const int batchCount) nogil:
    global _hipblasDgetriBatched__funptr
    __init_symbol(&_hipblasDgetriBatched__funptr,"hipblasDgetriBatched")
    return (<hipblasStatus_t (*)(void *,const int,double *const*,const int,int *,double *const*,const int,int *,const int) nogil> _hipblasDgetriBatched__funptr)(handle,n,A,lda,ipiv,C,ldc,info,batchCount)


cdef void* _hipblasCgetriBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgetriBatched(void * handle,const int n,hipblasComplex *const* A,const int lda,int * ipiv,hipblasComplex *const* C,const int ldc,int * info,const int batchCount) nogil:
    global _hipblasCgetriBatched__funptr
    __init_symbol(&_hipblasCgetriBatched__funptr,"hipblasCgetriBatched")
    return (<hipblasStatus_t (*)(void *,const int,hipblasComplex *const*,const int,int *,hipblasComplex *const*,const int,int *,const int) nogil> _hipblasCgetriBatched__funptr)(handle,n,A,lda,ipiv,C,ldc,info,batchCount)


cdef void* _hipblasZgetriBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgetriBatched(void * handle,const int n,hipblasDoubleComplex *const* A,const int lda,int * ipiv,hipblasDoubleComplex *const* C,const int ldc,int * info,const int batchCount) nogil:
    global _hipblasZgetriBatched__funptr
    __init_symbol(&_hipblasZgetriBatched__funptr,"hipblasZgetriBatched")
    return (<hipblasStatus_t (*)(void *,const int,hipblasDoubleComplex *const*,const int,int *,hipblasDoubleComplex *const*,const int,int *,const int) nogil> _hipblasZgetriBatched__funptr)(handle,n,A,lda,ipiv,C,ldc,info,batchCount)


cdef void* _hipblasSgels__funptr = NULL
# @{
# \brief GELS solves an overdetermined (or underdetermined) linear system defined by an m-by-n
# matrix A, and a corresponding matrix B, using the QR factorization computed by \ref hipblasSgeqrf "GEQRF" (or the LQ
# factorization computed by "GELQF").
# 
# \details
# Depending on the value of trans, the problem solved by this function is either of the form
# 
# \f[
#     \begin{array}{cl}
#     A X = B & \: \text{not transposed, or}\\
#     A' X = B & \: \text{transposed if real, or conjugate transposed if complex}
#     \end{array}
# \f]
# 
# If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
# and a least-squares solution approximating X is found by minimizing
# 
# \f[
#     || B - A  X || \quad \text{(or} \: || B - A' X ||\text{)}
# \f]
# 
# If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
# and a unique solution for X is chosen such that \f$|| X ||\f$ is minimal.
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : currently unsupported
# 
# @param[in]
# handle      hipblasHandle_t.
# @param[in]
# trans       hipblasOperation_t.\n
#             Specifies the form of the system of equations.
# @param[in]
# m           int. m >= 0.\n
#             The number of rows of matrix A.
# @param[in]
# n           int. n >= 0.\n
#             The number of columns of matrix A.
# @param[in]
# nrhs        int. nrhs >= 0.\n
#             The number of columns of matrices B and X;
#             i.e., the columns on the right hand side.
# @param[inout]
# A           pointer to type. Array on the GPU of dimension lda*n.\n
#             On entry, the matrix A.
#             On exit, the QR (or LQ) factorization of A as returned by "GEQRF" (or "GELQF").
# @param[in]
# lda         int. lda >= m.\n
#             Specifies the leading dimension of matrix A.
# @param[inout]
# B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
#             On entry, the matrix B.
#             On exit, when info = 0, B is overwritten by the solution vectors (and the residuals in
#             the overdetermined cases) stored as columns.
# @param[in]
# ldb         int. ldb >= max(m,n).\n
#             Specifies the leading dimension of matrix B.
# @param[out]
# info        pointer to an int on the host.\n
#             If info = 0, successful exit.
#             If info = j < 0, the j-th argument is invalid.
# @param[out]
# deviceInfo  pointer to int on the GPU.\n
#             If info = 0, successful exit.
#             If info = i > 0, the solution could not be computed because input matrix A is
#             rank deficient; the i-th diagonal element of its triangular factor is zero.
cdef hipblasStatus_t hipblasSgels(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,float * A,const int lda,float * B,const int ldb,int * info,int * deviceInfo) nogil:
    global _hipblasSgels__funptr
    __init_symbol(&_hipblasSgels__funptr,"hipblasSgels")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,const int,float *,const int,float *,const int,int *,int *) nogil> _hipblasSgels__funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo)


cdef void* _hipblasDgels__funptr = NULL
cdef hipblasStatus_t hipblasDgels(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,double * A,const int lda,double * B,const int ldb,int * info,int * deviceInfo) nogil:
    global _hipblasDgels__funptr
    __init_symbol(&_hipblasDgels__funptr,"hipblasDgels")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,const int,double *,const int,double *,const int,int *,int *) nogil> _hipblasDgels__funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo)


cdef void* _hipblasCgels__funptr = NULL
cdef hipblasStatus_t hipblasCgels(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasComplex * A,const int lda,hipblasComplex * B,const int ldb,int * info,int * deviceInfo) nogil:
    global _hipblasCgels__funptr
    __init_symbol(&_hipblasCgels__funptr,"hipblasCgels")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,const int,hipblasComplex *,const int,hipblasComplex *,const int,int *,int *) nogil> _hipblasCgels__funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo)


cdef void* _hipblasZgels__funptr = NULL
cdef hipblasStatus_t hipblasZgels(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,hipblasDoubleComplex * B,const int ldb,int * info,int * deviceInfo) nogil:
    global _hipblasZgels__funptr
    __init_symbol(&_hipblasZgels__funptr,"hipblasZgels")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,const int,hipblasDoubleComplex *,const int,hipblasDoubleComplex *,const int,int *,int *) nogil> _hipblasZgels__funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo)


cdef void* _hipblasSgelsBatched__funptr = NULL
# @{
# \brief gelsBatched solves a batch of overdetermined (or underdetermined) linear systems
# defined by a set of m-by-n matrices \f$A_j\f$, and corresponding matrices \f$B_j\f$, using the
# QR factorizations computed by "GEQRF_BATCHED" (or the LQ factorizations computed by "GELQF_BATCHED").
# 
# \details
# For each instance in the batch, depending on the value of trans, the problem solved by this function is either of the form
# 
# \f[
#     \begin{array}{cl}
#     A_j X_j = B_j & \: \text{not transposed, or}\\
#     A_j' X_j = B_j & \: \text{transposed if real, or conjugate transposed if complex}
#     \end{array}
# \f]
# 
# If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
# and a least-squares solution approximating X_j is found by minimizing
# 
# \f[
#     || B_j - A_j  X_j || \quad \text{(or} \: || B_j - A_j' X_j ||\text{)}
# \f]
# 
# If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
# and a unique solution for X_j is chosen such that \f$|| X_j ||\f$ is minimal.
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# Note that cuBLAS backend supports only the non-transpose operation and only solves over-determined systems (m >= n).
# 
# @param[in]
# handle      hipblasHandle_t.
# @param[in]
# trans       hipblasOperation_t.\n
#             Specifies the form of the system of equations.
# @param[in]
# m           int. m >= 0.\n
#             The number of rows of all matrices A_j in the batch.
# @param[in]
# n           int. n >= 0.\n
#             The number of columns of all matrices A_j in the batch.
# @param[in]
# nrhs        int. nrhs >= 0.\n
#             The number of columns of all matrices B_j and X_j in the batch;
#             i.e., the columns on the right hand side.
# @param[inout]
# A           array of pointer to type. Each pointer points to an array on the GPU of dimension lda*n.\n
#             On entry, the matrices A_j.
#             On exit, the QR (or LQ) factorizations of A_j as returned by "GEQRF_BATCHED"
#             (or "GELQF_BATCHED").
# @param[in]
# lda         int. lda >= m.\n
#             Specifies the leading dimension of matrices A_j.
# @param[inout]
# B           array of pointer to type. Each pointer points to an array on the GPU of dimension ldb*nrhs.\n
#             On entry, the matrices B_j.
#             On exit, when info[j] = 0, B_j is overwritten by the solution vectors (and the residuals in
#             the overdetermined cases) stored as columns.
# @param[in]
# ldb         int. ldb >= max(m,n).\n
#             Specifies the leading dimension of matrices B_j.
# @param[out]
# info        pointer to an int on the host.\n
#             If info = 0, successful exit.
#             If info = j < 0, the j-th argument is invalid.
# @param[out]
# deviceInfo  pointer to int. Array of batchCount integers on the GPU.\n
#             If deviceInfo[j] = 0, successful exit for solution of A_j.
#             If deviceInfo[j] = i > 0, the solution of A_j could not be computed because input
#             matrix A_j is rank deficient; the i-th diagonal element of its triangular factor is zero.
# @param[in]
# batchCount  int. batchCount >= 0.\n
#             Number of matrices in the batch.
cdef hipblasStatus_t hipblasSgelsBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,float *const* A,const int lda,float *const* B,const int ldb,int * info,int * deviceInfo,const int batchCount) nogil:
    global _hipblasSgelsBatched__funptr
    __init_symbol(&_hipblasSgelsBatched__funptr,"hipblasSgelsBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,const int,float *const*,const int,float *const*,const int,int *,int *,const int) nogil> _hipblasSgelsBatched__funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo,batchCount)


cdef void* _hipblasDgelsBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgelsBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,double *const* A,const int lda,double *const* B,const int ldb,int * info,int * deviceInfo,const int batchCount) nogil:
    global _hipblasDgelsBatched__funptr
    __init_symbol(&_hipblasDgelsBatched__funptr,"hipblasDgelsBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,const int,double *const*,const int,double *const*,const int,int *,int *,const int) nogil> _hipblasDgelsBatched__funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo,batchCount)


cdef void* _hipblasCgelsBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgelsBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasComplex *const* A,const int lda,hipblasComplex *const* B,const int ldb,int * info,int * deviceInfo,const int batchCount) nogil:
    global _hipblasCgelsBatched__funptr
    __init_symbol(&_hipblasCgelsBatched__funptr,"hipblasCgelsBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,const int,hipblasComplex *const*,const int,hipblasComplex *const*,const int,int *,int *,const int) nogil> _hipblasCgelsBatched__funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo,batchCount)


cdef void* _hipblasZgelsBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgelsBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasDoubleComplex *const* A,const int lda,hipblasDoubleComplex *const* B,const int ldb,int * info,int * deviceInfo,const int batchCount) nogil:
    global _hipblasZgelsBatched__funptr
    __init_symbol(&_hipblasZgelsBatched__funptr,"hipblasZgelsBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,const int,hipblasDoubleComplex *const*,const int,hipblasDoubleComplex *const*,const int,int *,int *,const int) nogil> _hipblasZgelsBatched__funptr)(handle,trans,m,n,nrhs,A,lda,B,ldb,info,deviceInfo,batchCount)


cdef void* _hipblasSgelsStridedBatched__funptr = NULL
# @{
# \brief gelsStridedBatched solves a batch of overdetermined (or underdetermined) linear
# systems defined by a set of m-by-n matrices \f$A_j\f$, and corresponding matrices \f$B_j\f$,
# using the QR factorizations computed by "GEQRF_STRIDED_BATCHED"
# (or the LQ factorizations computed by "GELQF_STRIDED_BATCHED").
# 
# \details
# For each instance in the batch, depending on the value of trans, the problem solved by this function is either of the form
# 
# \f[
#     \begin{array}{cl}
#     A_j X_j = B_j & \: \text{not transposed, or}\\
#     A_j' X_j = B_j & \: \text{transposed if real, or conjugate transposed if complex}
#     \end{array}
# \f]
# 
# If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
# and a least-squares solution approximating X_j is found by minimizing
# 
# \f[
#     || B_j - A_j  X_j || \quad \text{(or} \: || B_j - A_j' X_j ||\text{)}
# \f]
# 
# If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
# and a unique solution for X_j is chosen such that \f$|| X_j ||\f$ is minimal.
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : currently unsupported
# 
# @param[in]
# handle      hipblasHandle_t.
# @param[in]
# trans       hipblasOperation_t.\n
#             Specifies the form of the system of equations.
# @param[in]
# m           int. m >= 0.\n
#             The number of rows of all matrices A_j in the batch.
# @param[in]
# n           int. n >= 0.\n
#             The number of columns of all matrices A_j in the batch.
# @param[in]
# nrhs        int. nrhs >= 0.\n
#             The number of columns of all matrices B_j and X_j in the batch;
#             i.e., the columns on the right hand side.
# @param[inout]
# A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
#             On entry, the matrices A_j.
#             On exit, the QR (or LQ) factorizations of A_j as returned by "GEQRF_STRIDED_BATCHED"
#             (or "GELQF_STRIDED_BATCHED").
# @param[in]
# lda         int. lda >= m.\n
#             Specifies the leading dimension of matrices A_j.
# @param[in]
# strideA     hipblasStride.\n
#             Stride from the start of one matrix A_j to the next one A_(j+1).
#             There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
# @param[inout]
# B           pointer to type. Array on the GPU (the size depends on the value of strideB).\n
#             On entry, the matrices B_j.
#             On exit, when info[j] = 0, each B_j is overwritten by the solution vectors (and the residuals in
#             the overdetermined cases) stored as columns.
# @param[in]
# ldb         int. ldb >= max(m,n).\n
#             Specifies the leading dimension of matrices B_j.
# @param[in]
# strideB     hipblasStride.\n
#             Stride from the start of one matrix B_j to the next one B_(j+1).
#             There is no restriction for the value of strideB. Normal use case is strideB >= ldb*nrhs
# @param[out]
# info        pointer to an int on the host.\n
#             If info = 0, successful exit.
#             If info = j < 0, the j-th argument is invalid.
# @param[out]
# deviceInfo  pointer to int. Array of batchCount integers on the GPU.\n
#             If deviceInfo[j] = 0, successful exit for solution of A_j.
#             If deviceInfo[j] = i > 0, the solution of A_j could not be computed because input
#             matrix A_j is rank deficient; the i-th diagonal element of its triangular factor is zero.
# @param[in]
# batchCount  int. batchCount >= 0.\n
#             Number of matrices in the batch.
cdef hipblasStatus_t hipblasSgelsStridedBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,float * A,const int lda,const long strideA,float * B,const int ldb,const long strideB,int * info,int * deviceInfo,const int batch_count) nogil:
    global _hipblasSgelsStridedBatched__funptr
    __init_symbol(&_hipblasSgelsStridedBatched__funptr,"hipblasSgelsStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,const int,float *,const int,const long,float *,const int,const long,int *,int *,const int) nogil> _hipblasSgelsStridedBatched__funptr)(handle,trans,m,n,nrhs,A,lda,strideA,B,ldb,strideB,info,deviceInfo,batch_count)


cdef void* _hipblasDgelsStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgelsStridedBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,double * A,const int lda,const long strideA,double * B,const int ldb,const long strideB,int * info,int * deviceInfo,const int batch_count) nogil:
    global _hipblasDgelsStridedBatched__funptr
    __init_symbol(&_hipblasDgelsStridedBatched__funptr,"hipblasDgelsStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,const int,double *,const int,const long,double *,const int,const long,int *,int *,const int) nogil> _hipblasDgelsStridedBatched__funptr)(handle,trans,m,n,nrhs,A,lda,strideA,B,ldb,strideB,info,deviceInfo,batch_count)


cdef void* _hipblasCgelsStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgelsStridedBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasComplex * A,const int lda,const long strideA,hipblasComplex * B,const int ldb,const long strideB,int * info,int * deviceInfo,const int batch_count) nogil:
    global _hipblasCgelsStridedBatched__funptr
    __init_symbol(&_hipblasCgelsStridedBatched__funptr,"hipblasCgelsStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,const int,hipblasComplex *,const int,const long,hipblasComplex *,const int,const long,int *,int *,const int) nogil> _hipblasCgelsStridedBatched__funptr)(handle,trans,m,n,nrhs,A,lda,strideA,B,ldb,strideB,info,deviceInfo,batch_count)


cdef void* _hipblasZgelsStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgelsStridedBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,const long strideA,hipblasDoubleComplex * B,const int ldb,const long strideB,int * info,int * deviceInfo,const int batch_count) nogil:
    global _hipblasZgelsStridedBatched__funptr
    __init_symbol(&_hipblasZgelsStridedBatched__funptr,"hipblasZgelsStridedBatched")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,const int,const int,const int,hipblasDoubleComplex *,const int,const long,hipblasDoubleComplex *,const int,const long,int *,int *,const int) nogil> _hipblasZgelsStridedBatched__funptr)(handle,trans,m,n,nrhs,A,lda,strideA,B,ldb,strideB,info,deviceInfo,batch_count)


cdef void* _hipblasSgeqrf__funptr = NULL
# @{
# \brief SOLVER API
# 
# \details
# geqrf computes a QR factorization of a general m-by-n matrix A.
# 
# The factorization has the form
# 
# \f[
#     A = Q\left[\begin{array}{c}
#     R\\
#     0
#     \end{array}\right]
# \f]
# 
# where R is upper triangular (upper trapezoidal if m < n), and Q is
# a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices
# 
# \f[
#     Q = H_1H_2\cdots H_k, \quad \text{with} \: k = \text{min}(m,n)
# \f]
# 
# Each Householder matrix \f$H_i\f$ is given by
# 
# \f[
#     H_i = I - \text{ipiv}[i] \cdot v_i v_i'
# \f]
# 
# where the first i-1 elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# m         int. m >= 0.\n
#           The number of rows of the matrix A.
# @param[in]
# n         int. n >= 0.\n
#           The number of columns of the matrix A.
# @param[inout]
# A         pointer to type. Array on the GPU of dimension lda*n.\n
#           On entry, the m-by-n matrix to be factored.
#           On exit, the elements on and above the diagonal contain the
#           factor R; the elements below the diagonal are the last m - i elements
#           of Householder vector v_i.
# @param[in]
# lda       int. lda >= m.\n
#           Specifies the leading dimension of A.
# @param[out]
# ipiv      pointer to type. Array on the GPU of dimension min(m,n).\n
#           The Householder scalars.
# @param[out]
# info      pointer to a int on the host.\n
#           If info = 0, successful exit.
#           If info = j < 0, the j-th argument is invalid.
#
cdef hipblasStatus_t hipblasSgeqrf(void * handle,const int m,const int n,float * A,const int lda,float * ipiv,int * info) nogil:
    global _hipblasSgeqrf__funptr
    __init_symbol(&_hipblasSgeqrf__funptr,"hipblasSgeqrf")
    return (<hipblasStatus_t (*)(void *,const int,const int,float *,const int,float *,int *) nogil> _hipblasSgeqrf__funptr)(handle,m,n,A,lda,ipiv,info)


cdef void* _hipblasDgeqrf__funptr = NULL
cdef hipblasStatus_t hipblasDgeqrf(void * handle,const int m,const int n,double * A,const int lda,double * ipiv,int * info) nogil:
    global _hipblasDgeqrf__funptr
    __init_symbol(&_hipblasDgeqrf__funptr,"hipblasDgeqrf")
    return (<hipblasStatus_t (*)(void *,const int,const int,double *,const int,double *,int *) nogil> _hipblasDgeqrf__funptr)(handle,m,n,A,lda,ipiv,info)


cdef void* _hipblasCgeqrf__funptr = NULL
cdef hipblasStatus_t hipblasCgeqrf(void * handle,const int m,const int n,hipblasComplex * A,const int lda,hipblasComplex * ipiv,int * info) nogil:
    global _hipblasCgeqrf__funptr
    __init_symbol(&_hipblasCgeqrf__funptr,"hipblasCgeqrf")
    return (<hipblasStatus_t (*)(void *,const int,const int,hipblasComplex *,const int,hipblasComplex *,int *) nogil> _hipblasCgeqrf__funptr)(handle,m,n,A,lda,ipiv,info)


cdef void* _hipblasZgeqrf__funptr = NULL
cdef hipblasStatus_t hipblasZgeqrf(void * handle,const int m,const int n,hipblasDoubleComplex * A,const int lda,hipblasDoubleComplex * ipiv,int * info) nogil:
    global _hipblasZgeqrf__funptr
    __init_symbol(&_hipblasZgeqrf__funptr,"hipblasZgeqrf")
    return (<hipblasStatus_t (*)(void *,const int,const int,hipblasDoubleComplex *,const int,hipblasDoubleComplex *,int *) nogil> _hipblasZgeqrf__funptr)(handle,m,n,A,lda,ipiv,info)


cdef void* _hipblasSgeqrfBatched__funptr = NULL
# @{
# \brief SOLVER API
# 
# \details
# geqrfBatched computes the QR factorization of a batch of general
# m-by-n matrices.
# 
# The factorization of matrix \f$A_i\f$ in the batch has the form
# 
# \f[
#     A_i = Q_i\left[\begin{array}{c}
#     R_i\\
#     0
#     \end{array}\right]
# \f]
# 
# where \f$R_i\f$ is upper triangular (upper trapezoidal if m < n), and \f$Q_i\f$ is
# a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices
# 
# \f[
#     Q_i = H_{i_1}H_{i_2}\cdots H_{i_k}, \quad \text{with} \: k = \text{min}(m,n)
# \f]
# 
# Each Householder matrix \f$H_{i_j}\f$ is given by
# 
# \f[
#     H_{i_j} = I - \text{ipiv}_i[j] \cdot v_{i_j} v_{i_j}'
# \f]
# 
# where the first j-1 elements of Householder vector \f$v_{i_j}\f$ are zero, and \f$v_{i_j}[j] = 1\f$.
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# m         int. m >= 0.\n
#           The number of rows of all the matrices A_i in the batch.
# @param[in]
# n         int. n >= 0.\n
#           The number of columns of all the matrices A_i in the batch.
# @param[inout]
# A         Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
#           On entry, the m-by-n matrices A_i to be factored.
#           On exit, the elements on and above the diagonal contain the
#           factor R_i. The elements below the diagonal are the last m - j elements
#           of Householder vector v_(i_j).
# @param[in]
# lda       int. lda >= m.\n
#           Specifies the leading dimension of matrices A_i.
# @param[out]
# ipiv      array of pointers to type. Each pointer points to an array on the GPU
#           of dimension min(m, n).\n
#           Contains the vectors ipiv_i of corresponding Householder scalars.
# @param[out]
# info      pointer to a int on the host.\n
#           If info = 0, successful exit.
#           If info = k < 0, the k-th argument is invalid.
# @param[in]
# batchCount  int. batchCount >= 0.\n
#              Number of matrices in the batch.
cdef hipblasStatus_t hipblasSgeqrfBatched(void * handle,const int m,const int n,float *const* A,const int lda,float *const* ipiv,int * info,const int batchCount) nogil:
    global _hipblasSgeqrfBatched__funptr
    __init_symbol(&_hipblasSgeqrfBatched__funptr,"hipblasSgeqrfBatched")
    return (<hipblasStatus_t (*)(void *,const int,const int,float *const*,const int,float *const*,int *,const int) nogil> _hipblasSgeqrfBatched__funptr)(handle,m,n,A,lda,ipiv,info,batchCount)


cdef void* _hipblasDgeqrfBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgeqrfBatched(void * handle,const int m,const int n,double *const* A,const int lda,double *const* ipiv,int * info,const int batchCount) nogil:
    global _hipblasDgeqrfBatched__funptr
    __init_symbol(&_hipblasDgeqrfBatched__funptr,"hipblasDgeqrfBatched")
    return (<hipblasStatus_t (*)(void *,const int,const int,double *const*,const int,double *const*,int *,const int) nogil> _hipblasDgeqrfBatched__funptr)(handle,m,n,A,lda,ipiv,info,batchCount)


cdef void* _hipblasCgeqrfBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgeqrfBatched(void * handle,const int m,const int n,hipblasComplex *const* A,const int lda,hipblasComplex *const* ipiv,int * info,const int batchCount) nogil:
    global _hipblasCgeqrfBatched__funptr
    __init_symbol(&_hipblasCgeqrfBatched__funptr,"hipblasCgeqrfBatched")
    return (<hipblasStatus_t (*)(void *,const int,const int,hipblasComplex *const*,const int,hipblasComplex *const*,int *,const int) nogil> _hipblasCgeqrfBatched__funptr)(handle,m,n,A,lda,ipiv,info,batchCount)


cdef void* _hipblasZgeqrfBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgeqrfBatched(void * handle,const int m,const int n,hipblasDoubleComplex *const* A,const int lda,hipblasDoubleComplex *const* ipiv,int * info,const int batchCount) nogil:
    global _hipblasZgeqrfBatched__funptr
    __init_symbol(&_hipblasZgeqrfBatched__funptr,"hipblasZgeqrfBatched")
    return (<hipblasStatus_t (*)(void *,const int,const int,hipblasDoubleComplex *const*,const int,hipblasDoubleComplex *const*,int *,const int) nogil> _hipblasZgeqrfBatched__funptr)(handle,m,n,A,lda,ipiv,info,batchCount)


cdef void* _hipblasSgeqrfStridedBatched__funptr = NULL
# @{
# \brief SOLVER API
# 
# \details
# geqrfStridedBatched computes the QR factorization of a batch of
# general m-by-n matrices.
# 
# The factorization of matrix \f$A_i\f$ in the batch has the form
# 
# \f[
#     A_i = Q_i\left[\begin{array}{c}
#     R_i\\
#     0
#     \end{array}\right]
# \f]
# 
# where \f$R_i\f$ is upper triangular (upper trapezoidal if m < n), and \f$Q_i\f$ is
# a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices
# 
# \f[
#     Q_i = H_{i_1}H_{i_2}\cdots H_{i_k}, \quad \text{with} \: k = \text{min}(m,n)
# \f]
# 
# Each Householder matrix \f$H_{i_j}\f$ is given by
# 
# \f[
#     H_{i_j} = I - \text{ipiv}_j[j] \cdot v_{i_j} v_{i_j}'
# \f]
# 
# where the first j-1 elements of Householder vector \f$v_{i_j}\f$ are zero, and \f$v_{i_j}[j] = 1\f$.
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : No support
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# m         int. m >= 0.\n
#           The number of rows of all the matrices A_i in the batch.
# @param[in]
# n         int. n >= 0.\n
#           The number of columns of all the matrices A_i in the batch.
# @param[inout]
# A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
#           On entry, the m-by-n matrices A_i to be factored.
#           On exit, the elements on and above the diagonal contain the
#           factor R_i. The elements below the diagonal are the last m - j elements
#           of Householder vector v_(i_j).
# @param[in]
# lda       int. lda >= m.\n
#           Specifies the leading dimension of matrices A_i.
# @param[in]
# strideA   hipblasStride.\n
#           Stride from the start of one matrix A_i to the next one A_(i+1).
#           There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
# @param[out]
# ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
#           Contains the vectors ipiv_i of corresponding Householder scalars.
# @param[in]
# strideP   hipblasStride.\n
#           Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
#           There is no restriction for the value
#           of strideP. Normal use is strideP >= min(m,n).
# @param[out]
# info      pointer to a int on the host.\n
#           If info = 0, successful exit.
#           If info = k < 0, the k-th argument is invalid.
# @param[in]
# batchCount  int. batchCount >= 0.\n
#              Number of matrices in the batch.
cdef hipblasStatus_t hipblasSgeqrfStridedBatched(void * handle,const int m,const int n,float * A,const int lda,const long strideA,float * ipiv,const long strideP,int * info,const int batchCount) nogil:
    global _hipblasSgeqrfStridedBatched__funptr
    __init_symbol(&_hipblasSgeqrfStridedBatched__funptr,"hipblasSgeqrfStridedBatched")
    return (<hipblasStatus_t (*)(void *,const int,const int,float *,const int,const long,float *,const long,int *,const int) nogil> _hipblasSgeqrfStridedBatched__funptr)(handle,m,n,A,lda,strideA,ipiv,strideP,info,batchCount)


cdef void* _hipblasDgeqrfStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasDgeqrfStridedBatched(void * handle,const int m,const int n,double * A,const int lda,const long strideA,double * ipiv,const long strideP,int * info,const int batchCount) nogil:
    global _hipblasDgeqrfStridedBatched__funptr
    __init_symbol(&_hipblasDgeqrfStridedBatched__funptr,"hipblasDgeqrfStridedBatched")
    return (<hipblasStatus_t (*)(void *,const int,const int,double *,const int,const long,double *,const long,int *,const int) nogil> _hipblasDgeqrfStridedBatched__funptr)(handle,m,n,A,lda,strideA,ipiv,strideP,info,batchCount)


cdef void* _hipblasCgeqrfStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasCgeqrfStridedBatched(void * handle,const int m,const int n,hipblasComplex * A,const int lda,const long strideA,hipblasComplex * ipiv,const long strideP,int * info,const int batchCount) nogil:
    global _hipblasCgeqrfStridedBatched__funptr
    __init_symbol(&_hipblasCgeqrfStridedBatched__funptr,"hipblasCgeqrfStridedBatched")
    return (<hipblasStatus_t (*)(void *,const int,const int,hipblasComplex *,const int,const long,hipblasComplex *,const long,int *,const int) nogil> _hipblasCgeqrfStridedBatched__funptr)(handle,m,n,A,lda,strideA,ipiv,strideP,info,batchCount)


cdef void* _hipblasZgeqrfStridedBatched__funptr = NULL
cdef hipblasStatus_t hipblasZgeqrfStridedBatched(void * handle,const int m,const int n,hipblasDoubleComplex * A,const int lda,const long strideA,hipblasDoubleComplex * ipiv,const long strideP,int * info,const int batchCount) nogil:
    global _hipblasZgeqrfStridedBatched__funptr
    __init_symbol(&_hipblasZgeqrfStridedBatched__funptr,"hipblasZgeqrfStridedBatched")
    return (<hipblasStatus_t (*)(void *,const int,const int,hipblasDoubleComplex *,const int,const long,hipblasDoubleComplex *,const long,int *,const int) nogil> _hipblasZgeqrfStridedBatched__funptr)(handle,m,n,A,lda,strideA,ipiv,strideP,info,batchCount)


cdef void* _hipblasGemmEx__funptr = NULL
# \brief BLAS EX API
# 
# \details
# gemmEx performs one of the matrix-matrix operations
# 
#     C = alpha*op( A )*op( B ) + beta*C,
# 
# where op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
# 
# alpha and beta are scalars, and A, B, and C are matrices, with
# op( A ) an m by k matrix, op( B ) a k by n matrix and C is a m by n matrix.
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# Note for int8 users - For rocBLAS backend, please read rocblas_gemm_ex documentation on int8
# data layout requirements. hipBLAS makes the assumption that the data layout is in the preferred
# format for a given device as documented in rocBLAS.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A ).
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B ).
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# k         [int]
#           matrix dimension k.
# @param[in]
# alpha     [const void *]
#           device pointer or host pointer specifying the scalar alpha. Same datatype as computeType.
# @param[in]
# A         [void *]
#           device pointer storing matrix A.
# @param[in]
# aType    [hipblasDatatype_t]
#           specifies the datatype of matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# B         [void *]
#           device pointer storing matrix B.
# @param[in]
# bType    [hipblasDatatype_t]
#           specifies the datatype of matrix B.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of B.
# @param[in]
# beta      [const void *]
#           device pointer or host pointer specifying the scalar beta. Same datatype as computeType.
# @param[in]
# C         [void *]
#           device pointer storing matrix C.
# @param[in]
# cType    [hipblasDatatype_t]
#           specifies the datatype of matrix C.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
# @param[in]
# computeType
#           [hipblasDatatype_t]
#           specifies the datatype of computation.
# @param[in]
# algo      [hipblasGemmAlgo_t]
#           enumerant specifying the algorithm type.
#
cdef hipblasStatus_t hipblasGemmEx(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const void * alpha,const void * A,hipblasDatatype_t aType,int lda,const void * B,hipblasDatatype_t bType,int ldb,const void * beta,void * C,hipblasDatatype_t cType,int ldc,hipblasDatatype_t computeType,hipblasGemmAlgo_t algo) nogil:
    global _hipblasGemmEx__funptr
    __init_symbol(&_hipblasGemmEx__funptr,"hipblasGemmEx")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,const void *,const void *,hipblasDatatype_t,int,const void *,hipblasDatatype_t,int,const void *,void *,hipblasDatatype_t,int,hipblasDatatype_t,hipblasGemmAlgo_t) nogil> _hipblasGemmEx__funptr)(handle,transA,transB,m,n,k,alpha,A,aType,lda,B,bType,ldb,beta,C,cType,ldc,computeType,algo)


cdef void* _hipblasGemmBatchedEx__funptr = NULL
# \brief BLAS EX API
# \details
# gemmBatchedEx performs one of the batched matrix-matrix operations
#     C_i = alpha*op(A_i)*op(B_i) + beta*C_i, for i = 1, ..., batchCount.
# where op( X ) is one of
#     op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
# alpha and beta are scalars, and A, B, and C are batched pointers to matrices, with
# op( A ) an m by k by batchCount batched matrix,
# op( B ) a k by n by batchCount batched matrix and
# C a m by n by batchCount batched matrix.
# The batched matrices are an array of pointers to matrices.
# The number of pointers to matrices is batchCount.
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# Note for int8 users - For rocBLAS backend, please read rocblas_gemm_batched_ex documentation on int8
# data layout requirements. hipBLAS makes the assumption that the data layout is in the preferred
# format for a given device as documented in rocBLAS.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A ).
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B ).
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# k         [int]
#           matrix dimension k.
# @param[in]
# alpha     [const void *]
#           device pointer or host pointer specifying the scalar alpha. Same datatype as computeType.
# @param[in]
# A         [void *]
#           device pointer storing array of pointers to each matrix A_i.
# @param[in]
# aType    [hipblasDatatype_t]
#           specifies the datatype of each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# B         [void *]
#           device pointer storing array of pointers to each matrix B_i.
# @param[in]
# bType    [hipblasDatatype_t]
#           specifies the datatype of each matrix B_i.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of each B_i.
# @param[in]
# beta      [const void *]
#           device pointer or host pointer specifying the scalar beta. Same datatype as computeType.
# @param[in]
# C         [void *]
#           device array of device pointers to each matrix C_i.
# @param[in]
# cType    [hipblasDatatype_t]
#           specifies the datatype of each matrix C_i.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of each C_i.
# @param[in]
# batchCount
#           [int]
#           number of gemm operations in the batch.
# @param[in]
# computeType
#           [hipblasDatatype_t]
#           specifies the datatype of computation.
# @param[in]
# algo      [hipblasGemmAlgo_t]
#           enumerant specifying the algorithm type.
#
cdef hipblasStatus_t hipblasGemmBatchedEx(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const void * alpha,const void ** A,hipblasDatatype_t aType,int lda,const void ** B,hipblasDatatype_t bType,int ldb,const void * beta,void ** C,hipblasDatatype_t cType,int ldc,int batchCount,hipblasDatatype_t computeType,hipblasGemmAlgo_t algo) nogil:
    global _hipblasGemmBatchedEx__funptr
    __init_symbol(&_hipblasGemmBatchedEx__funptr,"hipblasGemmBatchedEx")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,const void *,const void **,hipblasDatatype_t,int,const void **,hipblasDatatype_t,int,const void *,void **,hipblasDatatype_t,int,int,hipblasDatatype_t,hipblasGemmAlgo_t) nogil> _hipblasGemmBatchedEx__funptr)(handle,transA,transB,m,n,k,alpha,A,aType,lda,B,bType,ldb,beta,C,cType,ldc,batchCount,computeType,algo)


cdef void* _hipblasGemmStridedBatchedEx__funptr = NULL
# \brief BLAS EX API
# 
# \details
# gemmStridedBatchedEx performs one of the strided_batched matrix-matrix operations
# 
#     C_i = alpha*op(A_i)*op(B_i) + beta*C_i, for i = 1, ..., batchCount
# 
# where op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
# 
# alpha and beta are scalars, and A, B, and C are strided_batched matrices, with
# op( A ) an m by k by batchCount strided_batched matrix,
# op( B ) a k by n by batchCount strided_batched matrix and
# C a m by n by batchCount strided_batched matrix.
# 
# The strided_batched matrices are multiple matrices separated by a constant stride.
# The number of matrices is batchCount.
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# Note for int8 users - For rocBLAS backend, please read rocblas_gemm_strided_batched_ex documentation on int8
# data layout requirements. hipBLAS makes the assumption that the data layout is in the preferred
# format for a given device as documented in rocBLAS.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A ).
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B ).
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# k         [int]
#           matrix dimension k.
# @param[in]
# alpha     [const void *]
#           device pointer or host pointer specifying the scalar alpha. Same datatype as computeType.
# @param[in]
# A         [void *]
#           device pointer pointing to first matrix A_1.
# @param[in]
# aType    [hipblasDatatype_t]
#           specifies the datatype of each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# strideA  [hipblasStride]
#           specifies stride from start of one A_i matrix to the next A_(i + 1).
# @param[in]
# B         [void *]
#           device pointer pointing to first matrix B_1.
# @param[in]
# bType    [hipblasDatatype_t]
#           specifies the datatype of each matrix B_i.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of each B_i.
# @param[in]
# strideB  [hipblasStride]
#           specifies stride from start of one B_i matrix to the next B_(i + 1).
# @param[in]
# beta      [const void *]
#           device pointer or host pointer specifying the scalar beta. Same datatype as computeType.
# @param[in]
# C         [void *]
#           device pointer pointing to first matrix C_1.
# @param[in]
# cType    [hipblasDatatype_t]
#           specifies the datatype of each matrix C_i.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of each C_i.
# @param[in]
# strideC  [hipblasStride]
#           specifies stride from start of one C_i matrix to the next C_(i + 1).
# @param[in]
# batchCount
#           [int]
#           number of gemm operations in the batch.
# @param[in]
# computeType
#           [hipblasDatatype_t]
#           specifies the datatype of computation.
# @param[in]
# algo      [hipblasGemmAlgo_t]
#           enumerant specifying the algorithm type.
#
cdef hipblasStatus_t hipblasGemmStridedBatchedEx(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const void * alpha,const void * A,hipblasDatatype_t aType,int lda,long strideA,const void * B,hipblasDatatype_t bType,int ldb,long strideB,const void * beta,void * C,hipblasDatatype_t cType,int ldc,long strideC,int batchCount,hipblasDatatype_t computeType,hipblasGemmAlgo_t algo) nogil:
    global _hipblasGemmStridedBatchedEx__funptr
    __init_symbol(&_hipblasGemmStridedBatchedEx__funptr,"hipblasGemmStridedBatchedEx")
    return (<hipblasStatus_t (*)(void *,hipblasOperation_t,hipblasOperation_t,int,int,int,const void *,const void *,hipblasDatatype_t,int,long,const void *,hipblasDatatype_t,int,long,const void *,void *,hipblasDatatype_t,int,long,int,hipblasDatatype_t,hipblasGemmAlgo_t) nogil> _hipblasGemmStridedBatchedEx__funptr)(handle,transA,transB,m,n,k,alpha,A,aType,lda,strideA,B,bType,ldb,strideB,beta,C,cType,ldc,strideC,batchCount,computeType,algo)


cdef void* _hipblasTrsmEx__funptr = NULL
# BLAS EX API
# 
# \details
# trsmEx solves
# 
#     op(A)*X = alpha*B or X*op(A) = alpha*B,
# 
# where alpha is a scalar, X and B are m by n matrices,
# A is triangular matrix and op(A) is one of
# 
#     op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.
# 
# The matrix X is overwritten on B.
# 
# This function gives the user the ability to reuse the invA matrix between runs.
# If invA == NULL, hipblasTrsmEx will automatically calculate invA on every run.
# 
# Setting up invA:
# The accepted invA matrix consists of the packed 128x128 inverses of the diagonal blocks of
# matrix A, followed by any smaller diagonal block that remains.
# To set up invA it is recommended that hipblasTrtriBatched be used with matrix A as the input.
# 
# Device memory of size 128 x k should be allocated for invA ahead of time, where k is m when
# HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT. The actual number of elements in invA
# should be passed as invAsize.
# 
# To begin, hipblasTrtriBatched must be called on the full 128x128 sized diagonal blocks of
# matrix A. Below are the restricted parameters:
#   - n = 128
#   - ldinvA = 128
#   - stride_invA = 128x128
#   - batchCount = k / 128,
# 
# Then any remaining block may be added:
#   - n = k % 128
#   - invA = invA + stride_invA * previousBatchCount
#   - ldinvA = 128
#   - batchCount = 1
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#         HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_ON_C: op(A) = A^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B. n >= 0.
# 
# @param[in]
# alpha   [void *]
#         device pointer or host pointer specifying the scalar alpha. When alpha is
#         &zero then A is not referenced, and B need not be set before
#         entry.
# 
# @param[in]
# A       [void *]
#         device pointer storing matrix A.
#         of dimension ( lda, k ), where k is m
#         when HIPBLAS_SIDE_LEFT and
#         is n when HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[in, out]
# B       [void *]
#         device pointer storing matrix B.
#         B is of dimension ( ldb, n ).
#         Before entry, the leading m by n part of the array B must
#         contain the right-hand side matrix B, and on exit is
#         overwritten by the solution matrix X.
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of B. ldb >= max( 1, m ).
# 
# @param[in]
# invA    [void *]
#         device pointer storing the inverse diagonal blocks of A.
#         invA is of dimension ( ld_invA, k ), where k is m
#         when HIPBLAS_SIDE_LEFT and
#         is n when HIPBLAS_SIDE_RIGHT.
#         ld_invA must be equal to 128.
# 
# @param[in]
# invAsize [int]
#         invAsize specifies the number of elements of device memory in invA.
# 
# @param[in]
# computeType [hipblasDatatype_t]
#         specifies the datatype of computation
#
cdef hipblasStatus_t hipblasTrsmEx(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const void * alpha,void * A,int lda,void * B,int ldb,const void * invA,int invAsize,hipblasDatatype_t computeType) nogil:
    global _hipblasTrsmEx__funptr
    __init_symbol(&_hipblasTrsmEx__funptr,"hipblasTrsmEx")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const void *,void *,int,void *,int,const void *,int,hipblasDatatype_t) nogil> _hipblasTrsmEx__funptr)(handle,side,uplo,transA,diag,m,n,alpha,A,lda,B,ldb,invA,invAsize,computeType)


cdef void* _hipblasTrsmBatchedEx__funptr = NULL
# BLAS EX API
# 
# \details
# trsmBatchedEx solves
# 
#     op(A_i)*X_i = alpha*B_i or X_i*op(A_i) = alpha*B_i,
# 
# for i = 1, ..., batchCount; and where alpha is a scalar, X and B are arrays of m by n matrices,
# A is an array of triangular matrix and each op(A_i) is one of
# 
#     op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.
# 
# Each matrix X_i is overwritten on B_i.
# 
# This function gives the user the ability to reuse the invA matrix between runs.
# If invA == NULL, hipblasTrsmBatchedEx will automatically calculate each invA_i on every run.
# 
# Setting up invA:
# Each accepted invA_i matrix consists of the packed 128x128 inverses of the diagonal blocks of
# matrix A_i, followed by any smaller diagonal block that remains.
# To set up each invA_i it is recommended that hipblasTrtriBatched be used with matrix A_i as the input.
# invA is an array of pointers of batchCount length holding each invA_i.
# 
# Device memory of size 128 x k should be allocated for each invA_i ahead of time, where k is m when
# HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT. The actual number of elements in each invA_i
# should be passed as invAsize.
# 
# To begin, hipblasTrtriBatched must be called on the full 128x128 sized diagonal blocks of each
# matrix A_i. Below are the restricted parameters:
#   - n = 128
#   - ldinvA = 128
#   - stride_invA = 128x128
#   - batchCount = k / 128,
# 
# Then any remaining block may be added:
#   - n = k % 128
#   - invA = invA + stride_invA * previousBatchCount
#   - ldinvA = 128
#   - batchCount = 1
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#         HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  each A_i is a lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_OP_C: op(A) = A^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  each A_i is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of each B_i. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of each B_i. n >= 0.
# 
# @param[in]
# alpha   [void *]
#         device pointer or host pointer alpha specifying the scalar alpha. When alpha is
#         &zero then A is not referenced, and B need not be set before
#         entry.
# 
# @param[in]
# A       [void *]
#         device array of device pointers storing each matrix A_i.
#         each A_i is of dimension ( lda, k ), where k is m
#         when HIPBLAS_SIDE_LEFT and
#         is n when HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of each A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[in, out]
# B       [void *]
#         device array of device pointers storing each matrix B_i.
#         each B_i is of dimension ( ldb, n ).
#         Before entry, the leading m by n part of the array B_i must
#         contain the right-hand side matrix B_i, and on exit is
#         overwritten by the solution matrix X_i
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
# 
# @param[in]
# batchCount [int]
#         specifies how many batches.
# 
# @param[in]
# invA    [void *]
#         device array of device pointers storing the inverse diagonal blocks of each A_i.
#         each invA_i is of dimension ( ld_invA, k ), where k is m
#         when HIPBLAS_SIDE_LEFT and
#         is n when HIPBLAS_SIDE_RIGHT.
#         ld_invA must be equal to 128.
# 
# @param[in]
# invAsize [int]
#         invAsize specifies the number of elements of device memory in each invA_i.
# 
# @param[in]
# computeType [hipblasDatatype_t]
#         specifies the datatype of computation
#
cdef hipblasStatus_t hipblasTrsmBatchedEx(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const void * alpha,void * A,int lda,void * B,int ldb,int batchCount,const void * invA,int invAsize,hipblasDatatype_t computeType) nogil:
    global _hipblasTrsmBatchedEx__funptr
    __init_symbol(&_hipblasTrsmBatchedEx__funptr,"hipblasTrsmBatchedEx")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const void *,void *,int,void *,int,int,const void *,int,hipblasDatatype_t) nogil> _hipblasTrsmBatchedEx__funptr)(handle,side,uplo,transA,diag,m,n,alpha,A,lda,B,ldb,batchCount,invA,invAsize,computeType)


cdef void* _hipblasTrsmStridedBatchedEx__funptr = NULL
# BLAS EX API
# 
# \details
# trsmStridedBatchedEx solves
# 
#     op(A_i)*X_i = alpha*B_i or X_i*op(A_i) = alpha*B_i,
# 
# for i = 1, ..., batchCount; and where alpha is a scalar, X and B are strided batched m by n matrices,
# A is a strided batched triangular matrix and op(A_i) is one of
# 
#     op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.
# 
# Each matrix X_i is overwritten on B_i.
# 
# This function gives the user the ability to reuse each invA_i matrix between runs.
# If invA == NULL, hipblasTrsmStridedBatchedEx will automatically calculate each invA_i on every run.
# 
# Setting up invA:
# Each accepted invA_i matrix consists of the packed 128x128 inverses of the diagonal blocks of
# matrix A_i, followed by any smaller diagonal block that remains.
# To set up invA_i it is recommended that hipblasTrtriBatched be used with matrix A_i as the input.
# invA is a contiguous piece of memory holding each invA_i.
# 
# Device memory of size 128 x k should be allocated for each invA_i ahead of time, where k is m when
# HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT. The actual number of elements in each invA_i
# should be passed as invAsize.
# 
# To begin, hipblasTrtriBatched must be called on the full 128x128 sized diagonal blocks of each
# matrix A_i. Below are the restricted parameters:
#   - n = 128
#   - ldinvA = 128
#   - stride_invA = 128x128
#   - batchCount = k / 128,
# 
# Then any remaining block may be added:
#   - n = k % 128
#   - invA = invA + stride_invA * previousBatchCount
#   - ldinvA = 128
#   - batchCount = 1
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#         HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  each A_i is a lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_OP_C: op(A) = A^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  each A_i is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of each B_i. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of each B_i. n >= 0.
# 
# @param[in]
# alpha   [void *]
#         device pointer or host pointer specifying the scalar alpha. When alpha is
#         &zero then A is not referenced, and B need not be set before
#         entry.
# 
# @param[in]
# A       [void *]
#         device pointer storing matrix A.
#         of dimension ( lda, k ), where k is m
#         when HIPBLAS_SIDE_LEFT and
#         is n when HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[in]
# strideA [hipblasStride]
#         The stride between each A matrix.
# 
# @param[in, out]
# B       [void *]
#         device pointer pointing to first matrix B_i.
#         each B_i is of dimension ( ldb, n ).
#         Before entry, the leading m by n part of each array B_i must
#         contain the right-hand side of matrix B_i, and on exit is
#         overwritten by the solution matrix X_i.
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
# 
# @param[in]
# strideB [hipblasStride]
#         The stride between each B_i matrix.
# 
# @param[in]
# batchCount [int]
#         specifies how many batches.
# 
# @param[in]
# invA    [void *]
#         device pointer storing the inverse diagonal blocks of each A_i.
#         invA points to the first invA_1.
#         each invA_i is of dimension ( ld_invA, k ), where k is m
#         when HIPBLAS_SIDE_LEFT and
#         is n when HIPBLAS_SIDE_RIGHT.
#         ld_invA must be equal to 128.
# 
# @param[in]
# invAsize [int]
#         invAsize specifies the number of elements of device memory in each invA_i.
# 
# @param[in]
# strideInvA [hipblasStride]
#         The stride between each invA matrix.
# 
# @param[in]
# computeType [hipblasDatatype_t]
#         specifies the datatype of computation
#
cdef hipblasStatus_t hipblasTrsmStridedBatchedEx(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const void * alpha,void * A,int lda,long strideA,void * B,int ldb,long strideB,int batchCount,const void * invA,int invAsize,long strideInvA,hipblasDatatype_t computeType) nogil:
    global _hipblasTrsmStridedBatchedEx__funptr
    __init_symbol(&_hipblasTrsmStridedBatchedEx__funptr,"hipblasTrsmStridedBatchedEx")
    return (<hipblasStatus_t (*)(void *,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const void *,void *,int,long,void *,int,long,int,const void *,int,long,hipblasDatatype_t) nogil> _hipblasTrsmStridedBatchedEx__funptr)(handle,side,uplo,transA,diag,m,n,alpha,A,lda,strideA,B,ldb,strideB,batchCount,invA,invAsize,strideInvA,computeType)


cdef void* _hipblasAxpyEx__funptr = NULL
# \brief BLAS EX API
# 
# \details
# axpyEx computes constant alpha multiplied by vector x, plus vector y
# 
#     y := alpha * x + y
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[in]
# alpha     device pointer or host pointer to specify the scalar alpha.
# @param[in]
# alphaType [hipblasDatatype_t]
#           specifies the datatype of alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# yType [hipblasDatatype_t]
#       specifies the datatype of vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasAxpyEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,const void * x,hipblasDatatype_t xType,int incx,void * y,hipblasDatatype_t yType,int incy,hipblasDatatype_t executionType) nogil:
    global _hipblasAxpyEx__funptr
    __init_symbol(&_hipblasAxpyEx__funptr,"hipblasAxpyEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,const void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,int,hipblasDatatype_t) nogil> _hipblasAxpyEx__funptr)(handle,n,alpha,alphaType,x,xType,incx,y,yType,incy,executionType)


cdef void* _hipblasAxpyBatchedEx__funptr = NULL
# \brief BLAS EX API
# 
# \details
# axpyBatchedEx computes constant alpha multiplied by vector x, plus vector y over
#                   a set of batched vectors.
# 
#     y := alpha * x + y
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[in]
# alpha     device pointer or host pointer to specify the scalar alpha.
# @param[in]
# alphaType [hipblasDatatype_t]
#           specifies the datatype of alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# yType [hipblasDatatype_t]
#       specifies the datatype of each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasAxpyBatchedEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,const void * x,hipblasDatatype_t xType,int incx,void * y,hipblasDatatype_t yType,int incy,int batchCount,hipblasDatatype_t executionType) nogil:
    global _hipblasAxpyBatchedEx__funptr
    __init_symbol(&_hipblasAxpyBatchedEx__funptr,"hipblasAxpyBatchedEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,const void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,int,int,hipblasDatatype_t) nogil> _hipblasAxpyBatchedEx__funptr)(handle,n,alpha,alphaType,x,xType,incx,y,yType,incy,batchCount,executionType)


cdef void* _hipblasAxpyStridedBatchedEx__funptr = NULL
# \brief BLAS EX API
# 
# \details
# axpyStridedBatchedEx computes constant alpha multiplied by vector x, plus vector y over
#                   a set of strided batched vectors.
# 
#     y := alpha * x + y
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[in]
# alpha     device pointer or host pointer to specify the scalar alpha.
# @param[in]
# alphaType [hipblasDatatype_t]
#           specifies the datatype of alpha.
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) to the next one (x_i+1).
#           There are no restrictions placed on stridex, however the user should
#           take care to ensure that stridex is of appropriate size, for a typical
#           case this means stridex >= n * incx.
# @param[inout]
# y         device pointer to the first vector y_1.
# @param[in]
# yType [hipblasDatatype_t]
#       specifies the datatype of each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey   [hipblasStride]
#           stride from the start of one vector (y_i) to the next one (y_i+1).
#           There are no restrictions placed on stridey, however the user should
#           take care to ensure that stridey is of appropriate size, for a typical
#           case this means stridey >= n * incy.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasAxpyStridedBatchedEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,const void * x,hipblasDatatype_t xType,int incx,long stridex,void * y,hipblasDatatype_t yType,int incy,long stridey,int batchCount,hipblasDatatype_t executionType) nogil:
    global _hipblasAxpyStridedBatchedEx__funptr
    __init_symbol(&_hipblasAxpyStridedBatchedEx__funptr,"hipblasAxpyStridedBatchedEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,const void *,hipblasDatatype_t,int,long,void *,hipblasDatatype_t,int,long,int,hipblasDatatype_t) nogil> _hipblasAxpyStridedBatchedEx__funptr)(handle,n,alpha,alphaType,x,xType,incx,stridex,y,yType,incy,stridey,batchCount,executionType)


cdef void* _hipblasDotEx__funptr = NULL
# @{
# \brief BLAS EX API
# 
# \details
# dotEx  performs the dot product of vectors x and y
# 
#     result = x * y;
# 
# dotcEx  performs the dot product of the conjugate of complex vector x and complex vector y
# 
#     result = conjugate (x) * y;
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of y.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# yType [hipblasDatatype_t]
#       specifies the datatype of vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# result
#           device pointer or host pointer to store the dot product.
#           return is 0.0 if n <= 0.
# @param[in]
# resultType [hipblasDatatype_t]
#             specifies the datatype of the result.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasDotEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,const void * y,hipblasDatatype_t yType,int incy,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil:
    global _hipblasDotEx__funptr
    __init_symbol(&_hipblasDotEx__funptr,"hipblasDotEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,int,const void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasDotEx__funptr)(handle,n,x,xType,incx,y,yType,incy,result,resultType,executionType)


cdef void* _hipblasDotcEx__funptr = NULL
cdef hipblasStatus_t hipblasDotcEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,const void * y,hipblasDatatype_t yType,int incy,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil:
    global _hipblasDotcEx__funptr
    __init_symbol(&_hipblasDotcEx__funptr,"hipblasDotcEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,int,const void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasDotcEx__funptr)(handle,n,x,xType,incx,y,yType,incy,result,resultType,executionType)


cdef void* _hipblasDotBatchedEx__funptr = NULL
# @{
# \brief BLAS EX API
# 
# \details
# dotBatchedEx performs a batch of dot products of vectors x and y
# 
#     result_i = x_i * y_i;
# 
# dotcBatchedEx  performs a batch of dot products of the conjugate of complex vector x and complex vector y
# 
#     result_i = conjugate (x_i) * y_i;
# 
# where (x_i, y_i) is the i-th instance of the batch.
# x_i and y_i are vectors, for i = 1, ..., batchCount
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# yType [hipblasDatatype_t]
#       specifies the datatype of each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# @param[inout]
# result
#           device array or host array of batchCount size to store the dot products of each batch.
#           return 0.0 for each element if n <= 0.
# @param[in]
# resultType [hipblasDatatype_t]
#             specifies the datatype of the result.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasDotBatchedEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,const void * y,hipblasDatatype_t yType,int incy,int batchCount,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil:
    global _hipblasDotBatchedEx__funptr
    __init_symbol(&_hipblasDotBatchedEx__funptr,"hipblasDotBatchedEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,int,const void *,hipblasDatatype_t,int,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasDotBatchedEx__funptr)(handle,n,x,xType,incx,y,yType,incy,batchCount,result,resultType,executionType)


cdef void* _hipblasDotcBatchedEx__funptr = NULL
cdef hipblasStatus_t hipblasDotcBatchedEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,const void * y,hipblasDatatype_t yType,int incy,int batchCount,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil:
    global _hipblasDotcBatchedEx__funptr
    __init_symbol(&_hipblasDotcBatchedEx__funptr,"hipblasDotcBatchedEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,int,const void *,hipblasDatatype_t,int,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasDotcBatchedEx__funptr)(handle,n,x,xType,incx,y,yType,incy,batchCount,result,resultType,executionType)


cdef void* _hipblasDotStridedBatchedEx__funptr = NULL
# @{
# \brief BLAS EX API
# 
# \details
# dotStridedBatchedEx  performs a batch of dot products of vectors x and y
# 
#     result_i = x_i * y_i;
# 
# dotc_strided_batched_ex  performs a batch of dot products of the conjugate of complex vector x and complex vector y
# 
#     result_i = conjugate (x_i) * y_i;
# 
# where (x_i, y_i) is the i-th instance of the batch.
# x_i and y_i are vectors, for i = 1, ..., batchCount
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[in]
# x         device pointer to the first vector (x_1) in the batch.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex    [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1)
# @param[in]
# y         device pointer to the first vector (y_1) in the batch.
# @param[in]
# yType [hipblasDatatype_t]
#       specifies the datatype of each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey    [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch
# @param[inout]
# result
#           device array or host array of batchCount size to store the dot products of each batch.
#           return 0.0 for each element if n <= 0.
# @param[in]
# resultType [hipblasDatatype_t]
#             specifies the datatype of the result.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasDotStridedBatchedEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,long stridex,const void * y,hipblasDatatype_t yType,int incy,long stridey,int batchCount,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil:
    global _hipblasDotStridedBatchedEx__funptr
    __init_symbol(&_hipblasDotStridedBatchedEx__funptr,"hipblasDotStridedBatchedEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,int,long,const void *,hipblasDatatype_t,int,long,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasDotStridedBatchedEx__funptr)(handle,n,x,xType,incx,stridex,y,yType,incy,stridey,batchCount,result,resultType,executionType)


cdef void* _hipblasDotcStridedBatchedEx__funptr = NULL
cdef hipblasStatus_t hipblasDotcStridedBatchedEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,long stridex,const void * y,hipblasDatatype_t yType,int incy,long stridey,int batchCount,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil:
    global _hipblasDotcStridedBatchedEx__funptr
    __init_symbol(&_hipblasDotcStridedBatchedEx__funptr,"hipblasDotcStridedBatchedEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,int,long,const void *,hipblasDatatype_t,int,long,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasDotcStridedBatchedEx__funptr)(handle,n,x,xType,incx,stridex,y,yType,incy,stridey,batchCount,result,resultType,executionType)


cdef void* _hipblasNrm2Ex__funptr = NULL
# \brief BLAS_EX API
# 
# \details
# nrm2Ex computes the euclidean norm of a real or complex vector
# 
#           result := sqrt( x'*x ) for real vectors
#           result := sqrt( x**H*x ) for complex vectors
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of the vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# result
#           device pointer or host pointer to store the nrm2 product.
#           return is 0.0 if n, incx<=0.
# @param[in]
# resultType [hipblasDatatype_t]
#             specifies the datatype of the result.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
cdef hipblasStatus_t hipblasNrm2Ex(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil:
    global _hipblasNrm2Ex__funptr
    __init_symbol(&_hipblasNrm2Ex__funptr,"hipblasNrm2Ex")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasNrm2Ex__funptr)(handle,n,x,xType,incx,result,resultType,executionType)


cdef void* _hipblasNrm2BatchedEx__funptr = NULL
# \brief BLAS_EX API
# 
# \details
# nrm2BatchedEx computes the euclidean norm over a batch of real or complex vectors
# 
#           result := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batchCount
#           result := sqrt( x_i**H*x_i ) for complex vectors x, for i = 1, ..., batchCount
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each x_i.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device pointer or host pointer to array of batchCount size for nrm2 results.
#           return is 0.0 for each element if n <= 0, incx<=0.
# @param[in]
# resultType [hipblasDatatype_t]
#             specifies the datatype of the result.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasNrm2BatchedEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,int batchCount,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil:
    global _hipblasNrm2BatchedEx__funptr
    __init_symbol(&_hipblasNrm2BatchedEx__funptr,"hipblasNrm2BatchedEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,int,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasNrm2BatchedEx__funptr)(handle,n,x,xType,incx,batchCount,result,resultType,executionType)


cdef void* _hipblasNrm2StridedBatchedEx__funptr = NULL
# \brief BLAS_EX API
# 
# \details
# nrm2StridedBatchedEx computes the euclidean norm over a batch of real or complex vectors
# 
#           := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batchCount
#           := sqrt( x_i**H*x_i ) for complex vectors, for i = 1, ..., batchCount
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each x_i.
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
#           There are no restrictions placed on stride_x, however the user should
#           take care to ensure that stride_x is of appropriate size, for a typical
#           case this means stride_x >= n * incx.
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device pointer or host pointer to array for storing contiguous batchCount results.
#           return is 0.0 for each element if n <= 0, incx<=0.
# @param[in]
# resultType [hipblasDatatype_t]
#             specifies the datatype of the result.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasNrm2StridedBatchedEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,long stridex,int batchCount,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil:
    global _hipblasNrm2StridedBatchedEx__funptr
    __init_symbol(&_hipblasNrm2StridedBatchedEx__funptr,"hipblasNrm2StridedBatchedEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,int,long,int,void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasNrm2StridedBatchedEx__funptr)(handle,n,x,xType,incx,stridex,batchCount,result,resultType,executionType)


cdef void* _hipblasRotEx__funptr = NULL
# \brief BLAS EX API
# 
# \details
# rotEx applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
#     Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
# In the case where cs_type is real:
#     x := c * x + s * y
#         y := c * y - s * x
# 
# In the case where cs_type is complex, the imaginary part of c is ignored:
#     x := real(c) * x + s * y
#         y := real(c) * y - conj(s) * x
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in the x and y vectors.
# @param[inout]
# x       device pointer storing vector x.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of vector x.
# @param[in]
# incx    [int]
#         specifies the increment between elements of x.
# @param[inout]
# y       device pointer storing vector y.
# @param[in]
# yType [hipblasDatatype_t]
#        specifies the datatype of vector y.
# @param[in]
# incy    [int]
#         specifies the increment between elements of y.
# @param[in]
# c       device pointer or host pointer storing scalar cosine component of the rotation matrix.
# @param[in]
# s       device pointer or host pointer storing scalar sine component of the rotation matrix.
# @param[in]
# csType [hipblasDatatype_t]
#         specifies the datatype of c and s.
# @param[in]
# executionType [hipblasDatatype_t]
#                specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasRotEx(void * handle,int n,void * x,hipblasDatatype_t xType,int incx,void * y,hipblasDatatype_t yType,int incy,const void * c,const void * s,hipblasDatatype_t csType,hipblasDatatype_t executionType) nogil:
    global _hipblasRotEx__funptr
    __init_symbol(&_hipblasRotEx__funptr,"hipblasRotEx")
    return (<hipblasStatus_t (*)(void *,int,void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,int,const void *,const void *,hipblasDatatype_t,hipblasDatatype_t) nogil> _hipblasRotEx__funptr)(handle,n,x,xType,incx,y,yType,incy,c,s,csType,executionType)


cdef void* _hipblasRotBatchedEx__funptr = NULL
# \brief BLAS EX API
# 
# \details
# rotBatchedEx applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to batched vectors x_i and y_i, for i = 1, ..., batchCount.
#     Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
# In the case where cs_type is real:
#         x := c * x + s * y
#         y := c * y - s * x
# 
#     In the case where cs_type is complex, the imaginary part of c is ignored:
#         x := real(c) * x + s * y
#         y := real(c) * y - conj(s) * x
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in each x_i and y_i vectors.
# @param[inout]
# x       device array of deivce pointers storing each vector x_i.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx    [int]
#         specifies the increment between elements of each x_i.
# @param[inout]
# y       device array of device pointers storing each vector y_i.
# @param[in]
# yType [hipblasDatatype_t]
#        specifies the datatype of each vector y_i.
# @param[in]
# incy    [int]
#         specifies the increment between elements of each y_i.
# @param[in]
# c       device pointer or host pointer to scalar cosine component of the rotation matrix.
# @param[in]
# s       device pointer or host pointer to scalar sine component of the rotation matrix.
# @param[in]
# csType [hipblasDatatype_t]
#         specifies the datatype of c and s.
# @param[in]
# batchCount [int]
#             the number of x and y arrays, i.e. the number of batches.
# @param[in]
# executionType [hipblasDatatype_t]
#                specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasRotBatchedEx(void * handle,int n,void * x,hipblasDatatype_t xType,int incx,void * y,hipblasDatatype_t yType,int incy,const void * c,const void * s,hipblasDatatype_t csType,int batchCount,hipblasDatatype_t executionType) nogil:
    global _hipblasRotBatchedEx__funptr
    __init_symbol(&_hipblasRotBatchedEx__funptr,"hipblasRotBatchedEx")
    return (<hipblasStatus_t (*)(void *,int,void *,hipblasDatatype_t,int,void *,hipblasDatatype_t,int,const void *,const void *,hipblasDatatype_t,int,hipblasDatatype_t) nogil> _hipblasRotBatchedEx__funptr)(handle,n,x,xType,incx,y,yType,incy,c,s,csType,batchCount,executionType)


cdef void* _hipblasRotStridedBatchedEx__funptr = NULL
# \brief BLAS Level 1 API
# 
# \details
# rotStridedBatchedEx applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to strided batched vectors x_i and y_i, for i = 1, ..., batchCount.
#     Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
# In the case where cs_type is real:
#         x := c * x + s * y
#         y := c * y - s * x
# 
#     In the case where cs_type is complex, the imaginary part of c is ignored:
#         x := real(c) * x + s * y
#         y := real(c) * y - conj(s) * x
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in each x_i and y_i vectors.
# @param[inout]
# x       device pointer to the first vector x_1.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx    [int]
#         specifies the increment between elements of each x_i.
# @param[in]
# stridex [hipblasStride]
#          specifies the increment from the beginning of x_i to the beginning of x_(i+1)
# @param[inout]
# y       device pointer to the first vector y_1.
# @param[in]
# yType [hipblasDatatype_t]
#        specifies the datatype of each vector y_i.
# @param[in]
# incy    [int]
#         specifies the increment between elements of each y_i.
# @param[in]
# stridey [hipblasStride]
#          specifies the increment from the beginning of y_i to the beginning of y_(i+1)
# @param[in]
# c       device pointer or host pointer to scalar cosine component of the rotation matrix.
# @param[in]
# s       device pointer or host pointer to scalar sine component of the rotation matrix.
# @param[in]
# csType [hipblasDatatype_t]
#         specifies the datatype of c and s.
# @param[in]
# batchCount [int]
#         the number of x and y arrays, i.e. the number of batches.
# @param[in]
# executionType [hipblasDatatype_t]
#                specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasRotStridedBatchedEx(void * handle,int n,void * x,hipblasDatatype_t xType,int incx,long stridex,void * y,hipblasDatatype_t yType,int incy,long stridey,const void * c,const void * s,hipblasDatatype_t csType,int batchCount,hipblasDatatype_t executionType) nogil:
    global _hipblasRotStridedBatchedEx__funptr
    __init_symbol(&_hipblasRotStridedBatchedEx__funptr,"hipblasRotStridedBatchedEx")
    return (<hipblasStatus_t (*)(void *,int,void *,hipblasDatatype_t,int,long,void *,hipblasDatatype_t,int,long,const void *,const void *,hipblasDatatype_t,int,hipblasDatatype_t) nogil> _hipblasRotStridedBatchedEx__funptr)(handle,n,x,xType,incx,stridex,y,yType,incy,stridey,c,s,csType,batchCount,executionType)


cdef void* _hipblasScalEx__funptr = NULL
# \brief BLAS EX API
# 
# \details
# scalEx  scales each element of vector x with scalar alpha.
# 
#     x := alpha * x
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# alpha     device pointer or host pointer for the scalar alpha.
# @param[in]
# alphaType [hipblasDatatype_t]
#            specifies the datatype of alpha.
# @param[inout]
# x         device pointer storing vector x.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# executionType [hipblasDatatype_t]
#                specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasScalEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,void * x,hipblasDatatype_t xType,int incx,hipblasDatatype_t executionType) nogil:
    global _hipblasScalEx__funptr
    __init_symbol(&_hipblasScalEx__funptr,"hipblasScalEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,void *,hipblasDatatype_t,int,hipblasDatatype_t) nogil> _hipblasScalEx__funptr)(handle,n,alpha,alphaType,x,xType,incx,executionType)


cdef void* _hipblasScalBatchedEx__funptr = NULL
# \brief BLAS EX API
# 
# \details
# scalBatchedEx  scales each element of each vector x_i with scalar alpha.
# 
#     x_i := alpha * x_i
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# alpha     device pointer or host pointer for the scalar alpha.
# @param[in]
# alphaType [hipblasDatatype_t]
#            specifies the datatype of alpha.
# @param[inout]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# @param[in]
# executionType [hipblasDatatype_t]
#                specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasScalBatchedEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,void * x,hipblasDatatype_t xType,int incx,int batchCount,hipblasDatatype_t executionType) nogil:
    global _hipblasScalBatchedEx__funptr
    __init_symbol(&_hipblasScalBatchedEx__funptr,"hipblasScalBatchedEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,void *,hipblasDatatype_t,int,int,hipblasDatatype_t) nogil> _hipblasScalBatchedEx__funptr)(handle,n,alpha,alphaType,x,xType,incx,batchCount,executionType)


cdef void* _hipblasScalStridedBatchedEx__funptr = NULL
# \brief BLAS EX API
# 
# \details
# scalStridedBatchedEx  scales each element of vector x with scalar alpha over a set
#                          of strided batched vectors.
# 
#     x := alpha * x
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# alpha     device pointer or host pointer for the scalar alpha.
# @param[in]
# alphaType [hipblasDatatype_t]
#            specifies the datatype of alpha.
# @param[inout]
# x         device pointer to the first vector x_1.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) to the next one (x_i+1).
#           There are no restrictions placed on stridex, however the user should
#           take care to ensure that stridex is of appropriate size, for a typical
#           case this means stridex >= n * incx.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# @param[in]
# executionType [hipblasDatatype_t]
#                specifies the datatype of computation.
#
cdef hipblasStatus_t hipblasScalStridedBatchedEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,void * x,hipblasDatatype_t xType,int incx,long stridex,int batchCount,hipblasDatatype_t executionType) nogil:
    global _hipblasScalStridedBatchedEx__funptr
    __init_symbol(&_hipblasScalStridedBatchedEx__funptr,"hipblasScalStridedBatchedEx")
    return (<hipblasStatus_t (*)(void *,int,const void *,hipblasDatatype_t,void *,hipblasDatatype_t,int,long,int,hipblasDatatype_t) nogil> _hipblasScalStridedBatchedEx__funptr)(handle,n,alpha,alphaType,x,xType,incx,stridex,batchCount,executionType)


cdef void* _hipblasStatusToString__funptr = NULL
# HIPBLAS Auxiliary API
# 
# \details
# hipblasStatusToString
# 
# Returns string representing hipblasStatus_t value
# 
# @param[in]
# status  [hipblasStatus_t]
#         hipBLAS status to convert to string
cdef const char * hipblasStatusToString(hipblasStatus_t status) nogil:
    global _hipblasStatusToString__funptr
    __init_symbol(&_hipblasStatusToString__funptr,"hipblasStatusToString")
    return (<const char * (*)(hipblasStatus_t) nogil> _hipblasStatusToString__funptr)(status)
