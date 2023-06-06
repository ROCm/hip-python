# AMD_COPYRIGHT
cimport hip._util.posixloader as loader
cdef void* _lib_handle = NULL

cdef void __init() nogil:
    global _lib_handle
    if _lib_handle == NULL:
        with gil:
            _lib_handle = loader.open_library("librccl.so")

cdef void __init_symbol(void** result, const char* name) nogil:
    global _lib_handle
    if _lib_handle == NULL:
        __init()
    if result[0] == NULL:
        with gil:
            result[0] = loader.load_symbol(_lib_handle, name) 


cdef void* _ncclGetVersion__funptr = NULL
#  @brief Return the NCCL_VERSION_CODE of the NCCL library in the supplied integer.
# 
# @details This integer is coded with the MAJOR, MINOR and PATCH level of the
# NCCL library
cdef ncclResult_t ncclGetVersion(int * version) nogil:
    global _ncclGetVersion__funptr
    __init_symbol(&_ncclGetVersion__funptr,"ncclGetVersion")
    return (<ncclResult_t (*)(int *) nogil> _ncclGetVersion__funptr)(version)


cdef void* _pncclGetVersion__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclGetVersion(int * version) nogil:
    global _pncclGetVersion__funptr
    __init_symbol(&_pncclGetVersion__funptr,"pncclGetVersion")
    return (<ncclResult_t (*)(int *) nogil> _pncclGetVersion__funptr)(version)


cdef void* _ncclGetUniqueId__funptr = NULL
#    @brief Generates an ID for ncclCommInitRank
# 
#    @details
#    Generates an ID to be used in ncclCommInitRank. ncclGetUniqueId should be
#    called once and the Id should be distributed to all ranks in the
#    communicator before calling ncclCommInitRank.
# 
#    @param[in]
#    uniqueId     ncclUniqueId*
#                 pointer to uniqueId
# 
# /
cdef ncclResult_t ncclGetUniqueId(ncclUniqueId * uniqueId) nogil:
    global _ncclGetUniqueId__funptr
    __init_symbol(&_ncclGetUniqueId__funptr,"ncclGetUniqueId")
    return (<ncclResult_t (*)(ncclUniqueId *) nogil> _ncclGetUniqueId__funptr)(uniqueId)


cdef void* _pncclGetUniqueId__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclGetUniqueId(ncclUniqueId * uniqueId) nogil:
    global _pncclGetUniqueId__funptr
    __init_symbol(&_pncclGetUniqueId__funptr,"pncclGetUniqueId")
    return (<ncclResult_t (*)(ncclUniqueId *) nogil> _pncclGetUniqueId__funptr)(uniqueId)


cdef void* _ncclCommInitRank__funptr = NULL
# @brief Creates a new communicator (multi thread/process version).
# 
# @details
# rank must be between 0 and nranks-1 and unique within a communicator clique.
# Each rank is associated to a CUDA device, which has to be set before calling
# ncclCommInitRank.
# ncclCommInitRank implicitly syncronizes with other ranks, so it must be
# called by different threads/processes or use ncclGroupStart/ncclGroupEnd.
# 
# @param[in]
# comm        ncclComm_t*
#             communicator struct pointer
cdef ncclResult_t ncclCommInitRank(ncclComm_t* comm,int nranks,ncclUniqueId commId,int rank) nogil:
    global _ncclCommInitRank__funptr
    __init_symbol(&_ncclCommInitRank__funptr,"ncclCommInitRank")
    return (<ncclResult_t (*)(ncclComm_t*,int,ncclUniqueId,int) nogil> _ncclCommInitRank__funptr)(comm,nranks,commId,rank)


cdef void* _pncclCommInitRank__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclCommInitRank(ncclComm_t* comm,int nranks,ncclUniqueId commId,int rank) nogil:
    global _pncclCommInitRank__funptr
    __init_symbol(&_pncclCommInitRank__funptr,"pncclCommInitRank")
    return (<ncclResult_t (*)(ncclComm_t*,int,ncclUniqueId,int) nogil> _pncclCommInitRank__funptr)(comm,nranks,commId,rank)


cdef void* _ncclCommInitRankMulti__funptr = NULL
# @brief Creates a new communicator (multi thread/process version) allowing multiple ranks per device.
# 
# @details
# rank must be between 0 and nranks-1 and unique within a communicator clique.
# Each rank is associated to a HIP device, which has to be set before calling
# ncclCommInitRankMulti.
# Since this version of the function allows multiple ranks to utilize the same
# HIP device, a unique virtualId per device has to be provided by each calling
# rank.
# ncclCommInitRankMulti implicitly syncronizes with other ranks, so it must be
# called by different threads/processes or use ncclGroupStart/ncclGroupEnd.
# 
# @param[in]
# comm        ncclComm_t*
#             communicator struct pointer
cdef ncclResult_t ncclCommInitRankMulti(ncclComm_t* comm,int nranks,ncclUniqueId commId,int rank,int virtualId) nogil:
    global _ncclCommInitRankMulti__funptr
    __init_symbol(&_ncclCommInitRankMulti__funptr,"ncclCommInitRankMulti")
    return (<ncclResult_t (*)(ncclComm_t*,int,ncclUniqueId,int,int) nogil> _ncclCommInitRankMulti__funptr)(comm,nranks,commId,rank,virtualId)


cdef void* _pncclCommInitRankMulti__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclCommInitRankMulti(ncclComm_t* comm,int nranks,ncclUniqueId commId,int rank,int virtualId) nogil:
    global _pncclCommInitRankMulti__funptr
    __init_symbol(&_pncclCommInitRankMulti__funptr,"pncclCommInitRankMulti")
    return (<ncclResult_t (*)(ncclComm_t*,int,ncclUniqueId,int,int) nogil> _pncclCommInitRankMulti__funptr)(comm,nranks,commId,rank,virtualId)


cdef void* _ncclCommInitAll__funptr = NULL
#  @brief Creates a clique of communicators (single process version).
# 
# @details This is a convenience function to create a single-process communicator clique.
# Returns an array of ndev newly initialized communicators in comm.
# comm should be pre-allocated with size at least ndev*sizeof(ncclComm_t).
# If devlist is NULL, the first ndev HIP devices are used.
# Order of devlist defines user-order of processors within the communicator.
cdef ncclResult_t ncclCommInitAll(ncclComm_t* comm,int ndev,const int * devlist) nogil:
    global _ncclCommInitAll__funptr
    __init_symbol(&_ncclCommInitAll__funptr,"ncclCommInitAll")
    return (<ncclResult_t (*)(ncclComm_t*,int,const int *) nogil> _ncclCommInitAll__funptr)(comm,ndev,devlist)


cdef void* _pncclCommInitAll__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclCommInitAll(ncclComm_t* comm,int ndev,const int * devlist) nogil:
    global _pncclCommInitAll__funptr
    __init_symbol(&_pncclCommInitAll__funptr,"pncclCommInitAll")
    return (<ncclResult_t (*)(ncclComm_t*,int,const int *) nogil> _pncclCommInitAll__funptr)(comm,ndev,devlist)


cdef void* _ncclCommDestroy__funptr = NULL
# @brief Frees resources associated with communicator object, but waits for any operations that might still be running on the device */
cdef ncclResult_t ncclCommDestroy(ncclComm_t comm) nogil:
    global _ncclCommDestroy__funptr
    __init_symbol(&_ncclCommDestroy__funptr,"ncclCommDestroy")
    return (<ncclResult_t (*)(ncclComm_t) nogil> _ncclCommDestroy__funptr)(comm)


cdef void* _pncclCommDestroy__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclCommDestroy(ncclComm_t comm) nogil:
    global _pncclCommDestroy__funptr
    __init_symbol(&_pncclCommDestroy__funptr,"pncclCommDestroy")
    return (<ncclResult_t (*)(ncclComm_t) nogil> _pncclCommDestroy__funptr)(comm)


cdef void* _ncclCommAbort__funptr = NULL
# @brief Frees resources associated with communicator object and aborts any operations that might still be running on the device. */
cdef ncclResult_t ncclCommAbort(ncclComm_t comm) nogil:
    global _ncclCommAbort__funptr
    __init_symbol(&_ncclCommAbort__funptr,"ncclCommAbort")
    return (<ncclResult_t (*)(ncclComm_t) nogil> _ncclCommAbort__funptr)(comm)


cdef void* _pncclCommAbort__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclCommAbort(ncclComm_t comm) nogil:
    global _pncclCommAbort__funptr
    __init_symbol(&_pncclCommAbort__funptr,"pncclCommAbort")
    return (<ncclResult_t (*)(ncclComm_t) nogil> _pncclCommAbort__funptr)(comm)


cdef void* _ncclGetErrorString__funptr = NULL
# @brief Returns a string for each error code. */
cdef const char * ncclGetErrorString(ncclResult_t result) nogil:
    global _ncclGetErrorString__funptr
    __init_symbol(&_ncclGetErrorString__funptr,"ncclGetErrorString")
    return (<const char * (*)(ncclResult_t) nogil> _ncclGetErrorString__funptr)(result)


cdef void* _pncclGetErrorString__funptr = NULL
# @cond include_hidden
cdef const char * pncclGetErrorString(ncclResult_t result) nogil:
    global _pncclGetErrorString__funptr
    __init_symbol(&_pncclGetErrorString__funptr,"pncclGetErrorString")
    return (<const char * (*)(ncclResult_t) nogil> _pncclGetErrorString__funptr)(result)


cdef void* _ncclGetLastError__funptr = NULL
#  @brief Returns a human-readable message of the last error that occurred.
# comm is currently unused and can be set to NULL
cdef const char * ncclGetLastError(ncclComm_t comm) nogil:
    global _ncclGetLastError__funptr
    __init_symbol(&_ncclGetLastError__funptr,"ncclGetLastError")
    return (<const char * (*)(ncclComm_t) nogil> _ncclGetLastError__funptr)(comm)


cdef void* _pncclGetError__funptr = NULL
# @cond include_hidden
cdef const char * pncclGetError(ncclComm_t comm) nogil:
    global _pncclGetError__funptr
    __init_symbol(&_pncclGetError__funptr,"pncclGetError")
    return (<const char * (*)(ncclComm_t) nogil> _pncclGetError__funptr)(comm)


cdef void* _ncclCommGetAsyncError__funptr = NULL
# @endcond
cdef ncclResult_t ncclCommGetAsyncError(ncclComm_t comm,ncclResult_t * asyncError) nogil:
    global _ncclCommGetAsyncError__funptr
    __init_symbol(&_ncclCommGetAsyncError__funptr,"ncclCommGetAsyncError")
    return (<ncclResult_t (*)(ncclComm_t,ncclResult_t *) nogil> _ncclCommGetAsyncError__funptr)(comm,asyncError)


cdef void* _pncclCommGetAsyncError__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclCommGetAsyncError(ncclComm_t comm,ncclResult_t * asyncError) nogil:
    global _pncclCommGetAsyncError__funptr
    __init_symbol(&_pncclCommGetAsyncError__funptr,"pncclCommGetAsyncError")
    return (<ncclResult_t (*)(ncclComm_t,ncclResult_t *) nogil> _pncclCommGetAsyncError__funptr)(comm,asyncError)


cdef void* _ncclCommCount__funptr = NULL
# @brief Gets the number of ranks in the communicator clique. */
cdef ncclResult_t ncclCommCount(ncclComm_t comm,int * count) nogil:
    global _ncclCommCount__funptr
    __init_symbol(&_ncclCommCount__funptr,"ncclCommCount")
    return (<ncclResult_t (*)(ncclComm_t,int *) nogil> _ncclCommCount__funptr)(comm,count)


cdef void* _pncclCommCount__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclCommCount(ncclComm_t comm,int * count) nogil:
    global _pncclCommCount__funptr
    __init_symbol(&_pncclCommCount__funptr,"pncclCommCount")
    return (<ncclResult_t (*)(ncclComm_t,int *) nogil> _pncclCommCount__funptr)(comm,count)


cdef void* _ncclCommCuDevice__funptr = NULL
# @brief Returns the rocm device number associated with the communicator. */
cdef ncclResult_t ncclCommCuDevice(ncclComm_t comm,int * device) nogil:
    global _ncclCommCuDevice__funptr
    __init_symbol(&_ncclCommCuDevice__funptr,"ncclCommCuDevice")
    return (<ncclResult_t (*)(ncclComm_t,int *) nogil> _ncclCommCuDevice__funptr)(comm,device)


cdef void* _pncclCommCuDevice__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclCommCuDevice(ncclComm_t comm,int * device) nogil:
    global _pncclCommCuDevice__funptr
    __init_symbol(&_pncclCommCuDevice__funptr,"pncclCommCuDevice")
    return (<ncclResult_t (*)(ncclComm_t,int *) nogil> _pncclCommCuDevice__funptr)(comm,device)


cdef void* _ncclCommUserRank__funptr = NULL
# @brief Returns the user-ordered "rank" associated with the communicator. */
cdef ncclResult_t ncclCommUserRank(ncclComm_t comm,int * rank) nogil:
    global _ncclCommUserRank__funptr
    __init_symbol(&_ncclCommUserRank__funptr,"ncclCommUserRank")
    return (<ncclResult_t (*)(ncclComm_t,int *) nogil> _ncclCommUserRank__funptr)(comm,rank)


cdef void* _pncclCommUserRank__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclCommUserRank(ncclComm_t comm,int * rank) nogil:
    global _pncclCommUserRank__funptr
    __init_symbol(&_pncclCommUserRank__funptr,"pncclCommUserRank")
    return (<ncclResult_t (*)(ncclComm_t,int *) nogil> _pncclCommUserRank__funptr)(comm,rank)


cdef void* _ncclRedOpCreatePreMulSum__funptr = NULL
cdef ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t * op,void * scalar,ncclDataType_t datatype,ncclScalarResidence_t residence,ncclComm_t comm) nogil:
    global _ncclRedOpCreatePreMulSum__funptr
    __init_symbol(&_ncclRedOpCreatePreMulSum__funptr,"ncclRedOpCreatePreMulSum")
    return (<ncclResult_t (*)(ncclRedOp_t *,void *,ncclDataType_t,ncclScalarResidence_t,ncclComm_t) nogil> _ncclRedOpCreatePreMulSum__funptr)(op,scalar,datatype,residence,comm)


cdef void* _pncclRedOpCreatePreMulSum__funptr = NULL
cdef ncclResult_t pncclRedOpCreatePreMulSum(ncclRedOp_t * op,void * scalar,ncclDataType_t datatype,ncclScalarResidence_t residence,ncclComm_t comm) nogil:
    global _pncclRedOpCreatePreMulSum__funptr
    __init_symbol(&_pncclRedOpCreatePreMulSum__funptr,"pncclRedOpCreatePreMulSum")
    return (<ncclResult_t (*)(ncclRedOp_t *,void *,ncclDataType_t,ncclScalarResidence_t,ncclComm_t) nogil> _pncclRedOpCreatePreMulSum__funptr)(op,scalar,datatype,residence,comm)


cdef void* _ncclRedOpDestroy__funptr = NULL
cdef ncclResult_t ncclRedOpDestroy(ncclRedOp_t op,ncclComm_t comm) nogil:
    global _ncclRedOpDestroy__funptr
    __init_symbol(&_ncclRedOpDestroy__funptr,"ncclRedOpDestroy")
    return (<ncclResult_t (*)(ncclRedOp_t,ncclComm_t) nogil> _ncclRedOpDestroy__funptr)(op,comm)


cdef void* _pncclRedOpDestroy__funptr = NULL
cdef ncclResult_t pncclRedOpDestroy(ncclRedOp_t op,ncclComm_t comm) nogil:
    global _pncclRedOpDestroy__funptr
    __init_symbol(&_pncclRedOpDestroy__funptr,"pncclRedOpDestroy")
    return (<ncclResult_t (*)(ncclRedOp_t,ncclComm_t) nogil> _pncclRedOpDestroy__funptr)(op,comm)


cdef void* _ncclReduce__funptr = NULL
# 
# @brief Reduce
# 
# @details Reduces data arrays of length count in sendbuff into recvbuff using op
# operation.
# recvbuff may be NULL on all calls except for root device.
# root is the rank (not the CUDA device) where data will reside after the
# operation is complete.
# 
# In-place operation will happen if sendbuff == recvbuff.
cdef ncclResult_t ncclReduce(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,ncclRedOp_t op,int root,ncclComm_t comm,hipStream_t stream) nogil:
    global _ncclReduce__funptr
    __init_symbol(&_ncclReduce__funptr,"ncclReduce")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,ncclRedOp_t,int,ncclComm_t,hipStream_t) nogil> _ncclReduce__funptr)(sendbuff,recvbuff,count,datatype,op,root,comm,stream)


cdef void* _pncclReduce__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclReduce(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,ncclRedOp_t op,int root,ncclComm_t comm,hipStream_t stream) nogil:
    global _pncclReduce__funptr
    __init_symbol(&_pncclReduce__funptr,"pncclReduce")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,ncclRedOp_t,int,ncclComm_t,hipStream_t) nogil> _pncclReduce__funptr)(sendbuff,recvbuff,count,datatype,op,root,comm,stream)


cdef void* _ncclBcast__funptr = NULL
#  @brief (deprecated) Broadcast (in-place)
# 
# @details Copies count values from root to all other devices.
# root is the rank (not the CUDA device) where data resides before the
# operation is started.
# 
# This operation is implicitely in place.
cdef ncclResult_t ncclBcast(void * buff,unsigned long count,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil:
    global _ncclBcast__funptr
    __init_symbol(&_ncclBcast__funptr,"ncclBcast")
    return (<ncclResult_t (*)(void *,unsigned long,ncclDataType_t,int,ncclComm_t,hipStream_t) nogil> _ncclBcast__funptr)(buff,count,datatype,root,comm,stream)


cdef void* _pncclBcast__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclBcast(void * buff,unsigned long count,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil:
    global _pncclBcast__funptr
    __init_symbol(&_pncclBcast__funptr,"pncclBcast")
    return (<ncclResult_t (*)(void *,unsigned long,ncclDataType_t,int,ncclComm_t,hipStream_t) nogil> _pncclBcast__funptr)(buff,count,datatype,root,comm,stream)


cdef void* _ncclBroadcast__funptr = NULL
#  @brief Broadcast
# 
# @details Copies count values from root to all other devices.
# root is the rank (not the HIP device) where data resides before the
# operation is started.
# 
# In-place operation will happen if sendbuff == recvbuff.
cdef ncclResult_t ncclBroadcast(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil:
    global _ncclBroadcast__funptr
    __init_symbol(&_ncclBroadcast__funptr,"ncclBroadcast")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,int,ncclComm_t,hipStream_t) nogil> _ncclBroadcast__funptr)(sendbuff,recvbuff,count,datatype,root,comm,stream)


cdef void* _pncclBroadcast__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclBroadcast(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil:
    global _pncclBroadcast__funptr
    __init_symbol(&_pncclBroadcast__funptr,"pncclBroadcast")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,int,ncclComm_t,hipStream_t) nogil> _pncclBroadcast__funptr)(sendbuff,recvbuff,count,datatype,root,comm,stream)


cdef void* _ncclAllReduce__funptr = NULL
#  @brief All-Reduce
# 
# @details Reduces data arrays of length count in sendbuff using op operation, and
# leaves identical copies of result on each recvbuff.
# 
# In-place operation will happen if sendbuff == recvbuff.
cdef ncclResult_t ncclAllReduce(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,hipStream_t stream) nogil:
    global _ncclAllReduce__funptr
    __init_symbol(&_ncclAllReduce__funptr,"ncclAllReduce")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,ncclRedOp_t,ncclComm_t,hipStream_t) nogil> _ncclAllReduce__funptr)(sendbuff,recvbuff,count,datatype,op,comm,stream)


cdef void* _pncclAllReduce__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclAllReduce(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,hipStream_t stream) nogil:
    global _pncclAllReduce__funptr
    __init_symbol(&_pncclAllReduce__funptr,"pncclAllReduce")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,ncclRedOp_t,ncclComm_t,hipStream_t) nogil> _pncclAllReduce__funptr)(sendbuff,recvbuff,count,datatype,op,comm,stream)


cdef void* _ncclReduceScatter__funptr = NULL
# 
# @brief Reduce-Scatter
# 
# @details Reduces data in sendbuff using op operation and leaves reduced result
# scattered over the devices so that recvbuff on rank i will contain the i-th
# block of the result.
# Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
# should have a size of at least nranks*recvcount elements.
# 
# In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
cdef ncclResult_t ncclReduceScatter(const void * sendbuff,void * recvbuff,unsigned long recvcount,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,hipStream_t stream) nogil:
    global _ncclReduceScatter__funptr
    __init_symbol(&_ncclReduceScatter__funptr,"ncclReduceScatter")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,ncclRedOp_t,ncclComm_t,hipStream_t) nogil> _ncclReduceScatter__funptr)(sendbuff,recvbuff,recvcount,datatype,op,comm,stream)


cdef void* _pncclReduceScatter__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclReduceScatter(const void * sendbuff,void * recvbuff,unsigned long recvcount,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,hipStream_t stream) nogil:
    global _pncclReduceScatter__funptr
    __init_symbol(&_pncclReduceScatter__funptr,"pncclReduceScatter")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,ncclRedOp_t,ncclComm_t,hipStream_t) nogil> _pncclReduceScatter__funptr)(sendbuff,recvbuff,recvcount,datatype,op,comm,stream)


cdef void* _ncclAllGather__funptr = NULL
#  @brief All-Gather
# 
# @details Each device gathers sendcount values from other GPUs into recvbuff,
# receiving data from rank i at offset i*sendcount.
# Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
# should have a size of at least nranks*sendcount elements.
# 
# In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
cdef ncclResult_t ncclAllGather(const void * sendbuff,void * recvbuff,unsigned long sendcount,ncclDataType_t datatype,ncclComm_t comm,hipStream_t stream) nogil:
    global _ncclAllGather__funptr
    __init_symbol(&_ncclAllGather__funptr,"ncclAllGather")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,ncclComm_t,hipStream_t) nogil> _ncclAllGather__funptr)(sendbuff,recvbuff,sendcount,datatype,comm,stream)


cdef void* _pncclAllGather__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclAllGather(const void * sendbuff,void * recvbuff,unsigned long sendcount,ncclDataType_t datatype,ncclComm_t comm,hipStream_t stream) nogil:
    global _pncclAllGather__funptr
    __init_symbol(&_pncclAllGather__funptr,"pncclAllGather")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,ncclComm_t,hipStream_t) nogil> _pncclAllGather__funptr)(sendbuff,recvbuff,sendcount,datatype,comm,stream)


cdef void* _ncclSend__funptr = NULL
#  @brief Send
# 
# @details Send data from sendbuff to rank peer.
# Rank peer needs to call ncclRecv with the same datatype and the same count from this
# rank.
# 
# This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
# need to progress concurrently to complete, they must be fused within a ncclGroupStart/
# ncclGroupEnd section.
cdef ncclResult_t ncclSend(const void * sendbuff,unsigned long count,ncclDataType_t datatype,int peer,ncclComm_t comm,hipStream_t stream) nogil:
    global _ncclSend__funptr
    __init_symbol(&_ncclSend__funptr,"ncclSend")
    return (<ncclResult_t (*)(const void *,unsigned long,ncclDataType_t,int,ncclComm_t,hipStream_t) nogil> _ncclSend__funptr)(sendbuff,count,datatype,peer,comm,stream)


cdef void* _pncclSend__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclSend(const void * sendbuff,unsigned long count,ncclDataType_t datatype,int peer,ncclComm_t comm,hipStream_t stream) nogil:
    global _pncclSend__funptr
    __init_symbol(&_pncclSend__funptr,"pncclSend")
    return (<ncclResult_t (*)(const void *,unsigned long,ncclDataType_t,int,ncclComm_t,hipStream_t) nogil> _pncclSend__funptr)(sendbuff,count,datatype,peer,comm,stream)


cdef void* _ncclRecv__funptr = NULL
#  @brief Receive
# 
# @details Receive data from rank peer into recvbuff.
# Rank peer needs to call ncclSend with the same datatype and the same count to this
# rank.
# 
# This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
# need to progress concurrently to complete, they must be fused within a ncclGroupStart/
# ncclGroupEnd section.
cdef ncclResult_t ncclRecv(void * recvbuff,unsigned long count,ncclDataType_t datatype,int peer,ncclComm_t comm,hipStream_t stream) nogil:
    global _ncclRecv__funptr
    __init_symbol(&_ncclRecv__funptr,"ncclRecv")
    return (<ncclResult_t (*)(void *,unsigned long,ncclDataType_t,int,ncclComm_t,hipStream_t) nogil> _ncclRecv__funptr)(recvbuff,count,datatype,peer,comm,stream)


cdef void* _pncclRecv__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclRecv(void * recvbuff,unsigned long count,ncclDataType_t datatype,int peer,ncclComm_t comm,hipStream_t stream) nogil:
    global _pncclRecv__funptr
    __init_symbol(&_pncclRecv__funptr,"pncclRecv")
    return (<ncclResult_t (*)(void *,unsigned long,ncclDataType_t,int,ncclComm_t,hipStream_t) nogil> _pncclRecv__funptr)(recvbuff,count,datatype,peer,comm,stream)


cdef void* _ncclGather__funptr = NULL
#  @brief Gather
# 
# @details Root device gathers sendcount values from other GPUs into recvbuff,
# receiving data from rank i at offset i*sendcount.
# 
# Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
# should have a size of at least nranks*sendcount elements.
# 
# In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
cdef ncclResult_t ncclGather(const void * sendbuff,void * recvbuff,unsigned long sendcount,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil:
    global _ncclGather__funptr
    __init_symbol(&_ncclGather__funptr,"ncclGather")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,int,ncclComm_t,hipStream_t) nogil> _ncclGather__funptr)(sendbuff,recvbuff,sendcount,datatype,root,comm,stream)


cdef void* _pncclGather__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclGather(const void * sendbuff,void * recvbuff,unsigned long sendcount,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil:
    global _pncclGather__funptr
    __init_symbol(&_pncclGather__funptr,"pncclGather")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,int,ncclComm_t,hipStream_t) nogil> _pncclGather__funptr)(sendbuff,recvbuff,sendcount,datatype,root,comm,stream)


cdef void* _ncclScatter__funptr = NULL
#  @brief Scatter
# 
# @details Scattered over the devices so that recvbuff on rank i will contain the i-th
# block of the data on root.
# 
# Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
# should have a size of at least nranks*recvcount elements.
# 
# In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
cdef ncclResult_t ncclScatter(const void * sendbuff,void * recvbuff,unsigned long recvcount,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil:
    global _ncclScatter__funptr
    __init_symbol(&_ncclScatter__funptr,"ncclScatter")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,int,ncclComm_t,hipStream_t) nogil> _ncclScatter__funptr)(sendbuff,recvbuff,recvcount,datatype,root,comm,stream)


cdef void* _pncclScatter__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclScatter(const void * sendbuff,void * recvbuff,unsigned long recvcount,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil:
    global _pncclScatter__funptr
    __init_symbol(&_pncclScatter__funptr,"pncclScatter")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,int,ncclComm_t,hipStream_t) nogil> _pncclScatter__funptr)(sendbuff,recvbuff,recvcount,datatype,root,comm,stream)


cdef void* _ncclAllToAll__funptr = NULL
#  @brief All-To-All
# 
# @details Device (i) send (j)th block of data to device (j) and be placed as (i)th
# block. Each block for sending/receiving has count elements, which means
# that recvbuff and sendbuff should have a size of nranks*count elements.
# 
# In-place operation will happen if sendbuff == recvbuff.
cdef ncclResult_t ncclAllToAll(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,ncclComm_t comm,hipStream_t stream) nogil:
    global _ncclAllToAll__funptr
    __init_symbol(&_ncclAllToAll__funptr,"ncclAllToAll")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,ncclComm_t,hipStream_t) nogil> _ncclAllToAll__funptr)(sendbuff,recvbuff,count,datatype,comm,stream)


cdef void* _pncclAllToAll__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclAllToAll(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,ncclComm_t comm,hipStream_t stream) nogil:
    global _pncclAllToAll__funptr
    __init_symbol(&_pncclAllToAll__funptr,"pncclAllToAll")
    return (<ncclResult_t (*)(const void *,void *,unsigned long,ncclDataType_t,ncclComm_t,hipStream_t) nogil> _pncclAllToAll__funptr)(sendbuff,recvbuff,count,datatype,comm,stream)


cdef void* _ncclAllToAllv__funptr = NULL
#  @brief All-To-Allv
# 
# @details Device (i) sends sendcounts[j] of data from offset sdispls[j]
# to device (j). In the same time, device (i) receives recvcounts[j] of data
# from device (j) to be placed at rdispls[j].
# 
# sendcounts, sdispls, recvcounts and rdispls are all measured in the units
# of datatype, not bytes.
# 
# In-place operation will happen if sendbuff == recvbuff.
cdef ncclResult_t ncclAllToAllv(const void * sendbuff,const unsigned long* sendcounts,const unsigned long* sdispls,void * recvbuff,const unsigned long* recvcounts,const unsigned long* rdispls,ncclDataType_t datatype,ncclComm_t comm,hipStream_t stream) nogil:
    global _ncclAllToAllv__funptr
    __init_symbol(&_ncclAllToAllv__funptr,"ncclAllToAllv")
    return (<ncclResult_t (*)(const void *,const unsigned long*,const unsigned long*,void *,const unsigned long*,const unsigned long*,ncclDataType_t,ncclComm_t,hipStream_t) nogil> _ncclAllToAllv__funptr)(sendbuff,sendcounts,sdispls,recvbuff,recvcounts,rdispls,datatype,comm,stream)


cdef void* _pncclAllToAllv__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclAllToAllv(const void * sendbuff,const unsigned long* sendcounts,const unsigned long* sdispls,void * recvbuff,const unsigned long* recvcounts,const unsigned long* rdispls,ncclDataType_t datatype,ncclComm_t comm,hipStream_t stream) nogil:
    global _pncclAllToAllv__funptr
    __init_symbol(&_pncclAllToAllv__funptr,"pncclAllToAllv")
    return (<ncclResult_t (*)(const void *,const unsigned long*,const unsigned long*,void *,const unsigned long*,const unsigned long*,ncclDataType_t,ncclComm_t,hipStream_t) nogil> _pncclAllToAllv__funptr)(sendbuff,sendcounts,sdispls,recvbuff,recvcounts,rdispls,datatype,comm,stream)


cdef void* _ncclGroupStart__funptr = NULL
#  @brief Group Start
# 
# Start a group call. All calls to NCCL until ncclGroupEnd will be fused into
# a single NCCL operation. Nothing will be started on the CUDA stream until
# ncclGroupEnd.
cdef ncclResult_t ncclGroupStart() nogil:
    global _ncclGroupStart__funptr
    __init_symbol(&_ncclGroupStart__funptr,"ncclGroupStart")
    return (<ncclResult_t (*)() nogil> _ncclGroupStart__funptr)()


cdef void* _pncclGroupStart__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclGroupStart() nogil:
    global _pncclGroupStart__funptr
    __init_symbol(&_pncclGroupStart__funptr,"pncclGroupStart")
    return (<ncclResult_t (*)() nogil> _pncclGroupStart__funptr)()


cdef void* _ncclGroupEnd__funptr = NULL
#  @brief Group End
# 
# End a group call. Start a fused NCCL operation consisting of all calls since
# ncclGroupStart. Operations on the CUDA stream depending on the NCCL operations
# need to be called after ncclGroupEnd.
cdef ncclResult_t ncclGroupEnd() nogil:
    global _ncclGroupEnd__funptr
    __init_symbol(&_ncclGroupEnd__funptr,"ncclGroupEnd")
    return (<ncclResult_t (*)() nogil> _ncclGroupEnd__funptr)()


cdef void* _pncclGroupEnd__funptr = NULL
# @cond include_hidden
cdef ncclResult_t pncclGroupEnd() nogil:
    global _pncclGroupEnd__funptr
    __init_symbol(&_pncclGroupEnd__funptr,"pncclGroupEnd")
    return (<ncclResult_t (*)() nogil> _pncclGroupEnd__funptr)()
